"""SDK turn execution — streams responses and emits events.

One function: run_turn(). It takes a single on_event callback that receives
typed events instead of the old dual send_fn/working_fn pattern.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from typing import Callable

from .cards import tool_use_summary
from .types import JsonObject, TurnClient

log = logging.getLogger(__name__)

# Max turns we'll burn to drain stale TaskNotifications + retry the real turn.
# With drain-prompt strategy, each drain turn clears exactly one stale without
# spawning new tasks, so this caps the number of accumulated stales we can
# recover from in a single user turn. See SDK #788.
MAX_RETRIES = 20

# Drain prompt: deliberately minimal and tool-forbidding so it cannot
# trigger Agent/subagent calls that would spawn new background tasks (the
# amplification trap). The whole point is to consume a stale TaskNotification
# without adding state. See SDK #788 analysis.
#
# CRITICAL: Each drain prompt embeds a unique nonce, and asks the model to
# reply with that same nonce. Why: drain messages enter the SDK conversation
# history as a normal USER turn (the SDK has no API for system-privileged
# control messages — see #788 follow-up). If every drain were identical and
# the assistant always replied "ack", N drain rounds would form an N-shot
# pattern of `user: <fixed text> → assistant: ack`. The model then continues
# the pattern on the next *real* user message, replying "ack" to anything.
# Observed in production 2026-04-29: 13 stales drained in one turn, then
# every subsequent user message got "ack" until the daemon was restarted.
#
# Nonce-per-drain breaks the pattern: each `(prompt, response)` pair is
# unique, so there is no fixed answer template for the model to copy onto
# the user's next real prompt.
DRAIN_PROMPT_MARKER = "[NEMO_DRAIN"
_DRAIN_NONCE_BYTES = 4  # 8 hex chars — plenty unique across a turn's drains


def make_drain_prompt() -> tuple[str, str]:
  """Return ``(prompt, expected_reply)`` for one drain turn.

  Each call produces a unique nonce so the resulting user/assistant pair
  never matches the previous drain in conversation history — see the
  module-level note above for why this matters.
  """
  nonce = secrets.token_hex(_DRAIN_NONCE_BYTES)
  expected = f"NEMO_DRAIN_OK {nonce}"
  prompt = (
    f"{DRAIN_PROMPT_MARKER} {nonce}] Internal stale-notification drain. "
    f"Ignore any prior background-task notifications; they are stale "
    f"leftovers from earlier turns. Reply with exactly: {expected}. "
    f"Use NO tools. Do not spawn agents, read files, or run commands."
  )
  return prompt, expected


class TransientAPIError(RuntimeError):
  """claude CLI surfaced a recoverable API/network error as turn output.

  Raised from _single_turn when an AssistantMessage / ResultMessage body
  matches a transient-error pattern (e.g. ECONNRESET to api.anthropic.com).
  The claude subprocess is usually wedged in this state; SDKThread's
  reconnect loop catches this and spawns a fresh CLI.
  """


# The claude CLI surfaces API/network failures with a hard-coded "API Error:"
# prefix at the START of the result text. We only match that exact prefix to
# avoid false positives when the model is legitimately discussing these error
# codes (e.g. user asks "what does ECONNRESET mean?" — the model's answer
# wouldn't START with "API Error:").
_CLI_ERROR_PREFIX = "API Error:"

# Finer-grained body patterns; these are checked ONLY when the CLI-error
# prefix is present OR ResultMessage.is_error=True is set. They help us log
# a useful label; they are NOT used for standalone string matching.
_TRANSIENT_ERROR_SIGNALS: tuple[str, ...] = (
  "unable to connect",
  "econnreset",
  "econnrefused",
  "etimedout",
  "enetunreach",
  "eai_again",
  "socket hang up",
  "fetch failed",
  "timed out",
  "network",
)


def _looks_like_transient_api_error(text: str, *, is_error_flag: bool = False) -> bool:
  """Return True only when the body clearly came from the CLI error channel.

  Matches if EITHER:
    - the text starts with the CLI's 'API Error:' prefix, OR
    - ResultMessage.is_error was set AND at least one transient signal
      (ECONNRESET, timed out, etc) appears in the body.
  Prefix-anchored matching means user messages like
  "what does ECONNRESET mean?" will NOT trip the detector.
  """
  if not text:
    return False
  stripped = text.lstrip()
  if stripped.startswith(_CLI_ERROR_PREFIX):
    return True
  if is_error_flag:
    lowered = text.lower()
    return any(sig in lowered for sig in _TRANSIENT_ERROR_SIGNALS)
  return False

# If receive_response() yields nothing for this long, assume the turn is stuck.
# SDK docs: "If no ResultMessage is received, the iterator continues indefinitely."
# Must be generous — Agent spawning and complex edits can go quiet for minutes.
HEARTBEAT_TIMEOUT = 300  # seconds

# If the SDK keeps emitting non-progress messages (SystemMessage,
# RateLimitEvent) but never resumes real work, we'd wait forever because every
# such message resets HEARTBEAT_TIMEOUT above. PROGRESS_TIMEOUT caps the
# duration between two REAL progress events (AssistantMessage, tool use,
# ResultMessage, Task* messages). When exceeded we raise TimeoutError so the
# SDKThread reconnect loop spawns a fresh CLI. Observed case: claude CLI
# enters a rate-limit retry loop that emits SystemMessage every ~1 min
# indefinitely.
PROGRESS_TIMEOUT = 240  # seconds — 4 minutes without real work = stuck

# Messages that do NOT count as real progress. Anything outside this set
# refreshes the progress clock. Checked by class name to avoid importing
# optional SDK types (RateLimitEvent may not exist on older SDKs).
_NON_PROGRESS_MESSAGE_TYPES: frozenset[str] = frozenset({
  "SystemMessage",
  "RateLimitEvent",
})


# ---------------------------------------------------------------------------
# Turn events
# ---------------------------------------------------------------------------

@dataclass
class ProgressEvent:
  """Any intermediate step — thinking, tool use, reasoning.

  Feeds the Working card and the Done card's collapsible timeline.
  ``kind`` values: "thinking", "tool", "reasoning".
  ``first`` is True for the very first progress event in a turn (signals
  the consumer to create the Working card).
  """
  kind: str
  summary: str
  first: bool = False


@dataclass
class AnswerEvent:
  """Visible text answer from the agent.

  The last AnswerEvent in a turn becomes the Done card body.
  """
  text: str
  task_id: str | None = None
  pending_tasks: int = 0


@dataclass
class TaskStartedEvent:
  task_id: str


@dataclass
class TaskDoneEvent:
  task_id: str
  status: str


@dataclass
class DoneEvent:
  """Turn completed."""
  cost: float
  usage: JsonObject
  session_id: str = ""  # CLI session UUID — needed for --resume on model switch


@dataclass
class ErrorEvent:
  """Turn ended with error."""
  message: str


TurnEvent = (
  ProgressEvent | AnswerEvent |
  TaskStartedEvent | TaskDoneEvent | DoneEvent | ErrorEvent
)


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------

@dataclass
class _TurnResult:
  found_stale: bool
  cost: float
  usage: JsonObject
  session_id: str
  last_emitted: str
  last_thinking: str


_NOOP_EVENT: Callable[[TurnEvent], None] = lambda _e: None


async def _single_turn(
  client: TurnClient,
  prompt: str,
  on_event: Callable[[TurnEvent], None],
  stale_tasks: set[str],
  stop_task_disabled: list[bool],
) -> _TurnResult:
  """Issue one query() and consume its receive_response() stream.

  If a stale TaskNotification (id in ``stale_tasks``) appears at the front of
  the stream, set ``found_stale=True`` and suppress downstream events until
  ResultMessage. The outer ``run_turn`` uses this flag to decide whether to
  emit drain/retry turns.

  ``stop_task_disabled`` is a one-element mutable flag: once ``stop_task``
  returns "Control request timeout" the control channel is presumed wedged
  and we skip further stop_task calls for the remainder of this orchestration.
  """
  from claude_agent_sdk import (
    AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock, ResultMessage,
    TaskStartedMessage, TaskNotificationMessage, TaskProgressMessage,
  )

  import anyio as _anyio
  import time as _time
  log.info("query() prompt=%d chars", len(prompt))
  with _anyio.fail_after(15):
    await client.query(prompt)
  log.info("query() sent to CLI")

  cost = 0.0
  usage: JsonObject = {}
  sdk_session_id = ""
  pending_tasks: set[str] = set()
  progress_started = False
  found_stale = False
  timed_out = False
  last_emitted: str = ""
  last_thinking: str = ""
  transient_error_text: str = ""  # set if AssistantMessage body looked like API error

  FIRST_MSG_TIMEOUT = 30
  msg_count = 0
  last_progress_at = _time.monotonic()
  response = client.receive_response()
  while True:
    # Per-iteration timeout is the heartbeat budget, but we also cap it at
    # the remaining progress budget so SDK-internal retry loops (which emit
    # SystemMessage every ~1 min indefinitely) cannot keep a dead turn alive.
    heartbeat_budget = FIRST_MSG_TIMEOUT if msg_count == 0 else HEARTBEAT_TIMEOUT
    progress_budget = PROGRESS_TIMEOUT - (_time.monotonic() - last_progress_at)
    if progress_budget <= 0:
      log.error("no progress for %ds (msgs=%d) — forcing reconnect",
                PROGRESS_TIMEOUT, msg_count)
      timed_out = True
      break
    iter_timeout = max(1, min(heartbeat_budget, progress_budget))
    next_task = asyncio.ensure_future(response.__anext__())
    done, _ = await asyncio.wait({next_task}, timeout=iter_timeout)
    if not done:
      next_task.cancel()
      since_progress = _time.monotonic() - last_progress_at
      if since_progress >= PROGRESS_TIMEOUT:
        log.error("no progress for %.0fs (msgs=%d) — forcing reconnect",
                  since_progress, msg_count)
      else:
        log.error("receive_response() heartbeat timeout (%ds, msgs=%d) — forcing reconnect",
                  heartbeat_budget, msg_count)
      timed_out = True
      break
    try:
      message = next_task.result()
    except StopAsyncIteration:
      break
    msg_count += 1
    msg_type = type(message).__name__
    if msg_type in _NON_PROGRESS_MESSAGE_TYPES:
      since_progress = _time.monotonic() - last_progress_at
      log.warning("turn msg: %s (non-progress, idle=%.0fs/%ds)",
                  msg_type, since_progress, PROGRESS_TIMEOUT)
    else:
      log.info("turn msg: %s", msg_type)
      last_progress_at = _time.monotonic()

    # --- Stale task detection (SDK bug #788 workaround) ---
    if isinstance(message, TaskNotificationMessage) and message.task_id in stale_tasks:
      stale_tasks.discard(message.task_id)
      found_stale = True
      log.warning("Stale TaskNotification task=%s — will drain", message.task_id)
      continue

    if found_stale:
      if isinstance(message, ResultMessage):
        cost = getattr(message, "total_cost_usd", 0) or 0.0
        usage = getattr(message, "usage", None) or {}
        sdk_session_id = getattr(message, "session_id", "") or ""
        break
      continue

    # --- Normal message handling ---
    if isinstance(message, AssistantMessage):
      text_parts: list[str] = []
      thinking_parts: list[str] = []
      tool_summary = ""
      for block in message.content:
        if isinstance(block, ThinkingBlock) and block.thinking:
          thinking_parts.append(block.thinking)
        elif isinstance(block, TextBlock) and block.text:
          text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
          tool_summary = tool_use_summary(block.name, block.input)

      if thinking_parts:
        thinking_text = "\n".join(thinking_parts)
        is_first = not progress_started
        progress_started = True
        on_event(ProgressEvent(kind="thinking", summary=thinking_text, first=is_first))
        last_emitted = "progress"
        last_thinking = thinking_text

      text = "\n".join(text_parts)
      if not text and message.content:
        if tool_summary:
          is_first = not progress_started
          progress_started = True
          on_event(ProgressEvent(kind="tool", summary=tool_summary, first=is_first))
          last_emitted = "progress"
      elif text:
        # Detect claude-CLI-surfaced transient API errors BEFORE emitting as
        # a user-facing AnswerEvent. If we see this pattern, the subprocess
        # is wedged; suppress the event and raise after ResultMessage so
        # SDKThread.run_turn_with_reconnect can spawn a fresh CLI.
        if _looks_like_transient_api_error(text):
          log.warning("Transient API error in AssistantMessage: %r", text[:200])
          transient_error_text = text
        else:
          task_id = None
          parent = getattr(message, "parent_tool_use_id", None)
          if parent and pending_tasks:
            task_id = next(iter(pending_tasks))
          on_event(AnswerEvent(
            text=text, task_id=task_id,
            pending_tasks=len(pending_tasks),
          ))
          last_emitted = "answer"

    elif isinstance(message, TaskStartedMessage):
      pending_tasks.add(message.task_id)
      on_event(TaskStartedEvent(task_id=message.task_id))

    elif isinstance(message, TaskProgressMessage):
      desc = getattr(message, "description", "") or ""
      if desc and progress_started:
        on_event(ProgressEvent(kind="tool", summary=desc))

    elif isinstance(message, TaskNotificationMessage):
      pending_tasks.discard(message.task_id)
      on_event(TaskDoneEvent(
        task_id=message.task_id,
        status=getattr(message, "status", ""),
      ))

    elif isinstance(message, ResultMessage):
      cost = getattr(message, "total_cost_usd", 0) or 0.0
      usage = getattr(message, "usage", None) or {}
      sdk_session_id = getattr(message, "session_id", "") or ""

      # Transient API error detection:
      #   1) An AssistantMessage body earlier was recognised (strict prefix
      #      check), OR
      #   2) ResultMessage.is_error is set AND the result text contains a
      #      transient-signal keyword.
      # In either case the claude subprocess is wedged — bubble out as
      # TransientAPIError so SDKThread.run_turn_with_reconnect spawns a
      # fresh CLI for the retry.
      is_error_flag = bool(getattr(message, "is_error", False))
      result_text = getattr(message, "result", "") or ""
      if (transient_error_text
          or _looks_like_transient_api_error(result_text,
                                             is_error_flag=is_error_flag)):
        err_sample = (transient_error_text or result_text)[:200]
        log.warning("Transient API error in turn result — will reconnect: %r",
                    err_sample)
        # Clean up pending tasks before raising so stale bookkeeping stays
        # consistent.
        for tid in list(pending_tasks):
          stale_tasks.add(tid)
        pending_tasks.clear()
        raise TransientAPIError(err_sample)

      # Mark remaining pending tasks as stale for future turns.
      for tid in list(pending_tasks):
        stale_tasks.add(tid)
        if stop_task_disabled[0]:
          continue
        try:
          await client.stop_task(tid)
        except Exception as e:
          log.warning("Failed to stop stale task %s: %s", tid, e)
          if "control request timeout" in str(e).lower():
            stop_task_disabled[0] = True
            log.warning("stop_task control channel wedged — skipping further stops")
      pending_tasks.clear()
      break

  if timed_out:
    on_event(ErrorEvent(message="Turn timed out — SDK stopped responding"))
    raise TimeoutError("receive_response() heartbeat timeout")

  return _TurnResult(
    found_stale=found_stale,
    cost=cost,
    usage=usage,
    session_id=sdk_session_id,
    last_emitted=last_emitted,
    last_thinking=last_thinking,
  )


async def run_turn(
  client: TurnClient,
  prompt: str,
  on_event: Callable[[TurnEvent], None],
  stale_tasks: set[str] | None = None,
) -> tuple[float, JsonObject]:
  """Send prompt to SDK client, stream responses, emit events.

  Orchestrates the SDK #788 workaround: if a stale TaskNotification leaks
  into the front of a turn, we discard that turn's events and then alternate
  between *drain* turns (using a fresh nonce-bearing drain prompt that
  cannot spawn new tasks) and *real* retries of the user's prompt. A drain
  turn clears at most one stale notification at a time, but crucially never
  adds to the pending-task backlog, so the retry budget converges instead
  of amplifying.

  Returns (cost, usage_dict).
  """
  if stale_tasks is None:
    stale_tasks = set()
  stop_task_disabled: list[bool] = [False]

  # Snapshot the ids we inherited from prior turns. The ResultMessage
  # handler may promote fresh pending_tasks into ``stale_tasks`` mid-run;
  # those newcomers must NOT be swept by the "clean real turn" /
  # "drain clean but stales not surfaced" drop paths, otherwise we lose
  # the legitimately-pending task ids that the SDK will deliver on a
  # future turn.
  stale_at_start: set[str] = set(stale_tasks)

  # Always start in real mode: attempting the user's prompt first is the
  # minimum cost in the common case (stale_tasks populated but stop_task
  # succeeded → no notification ever arrives). If a stale does leak in at the
  # front of the stream, `found_stale` will trip and we'll switch to drain
  # mode for the retry.
  mode = "real"
  mode_final = "real"
  total_cost = 0.0
  total_usage: JsonObject = {}
  result: _TurnResult | None = None
  exhausted = False

  for attempt in range(MAX_RETRIES + 1):
    if mode == "drain":
      cur_prompt, _ = make_drain_prompt()
      cur_on_event = _NOOP_EVENT
    else:
      cur_prompt = prompt
      cur_on_event = on_event
    log.info("turn attempt=%d mode=%s pending_stales=%d",
             attempt, mode, len(stale_tasks))

    try:
      result = await _single_turn(
        client, cur_prompt, cur_on_event, stale_tasks, stop_task_disabled,
      )
    except TimeoutError:
      # _single_turn already emitted ErrorEvent on its on_event. In drain
      # mode that was _NOOP_EVENT — the user saw nothing. Surface it now
      # via the real on_event so the caller knows the turn failed.
      if mode == "drain":
        on_event(ErrorEvent(message="Turn timed out while draining stale notifications"))
      raise

    mode_final = mode
    total_cost += result.cost
    if result.usage:
      total_usage = result.usage

    if mode == "real" and not result.found_stale:
      # Clean real turn. Any stale ids that were INHERITED from prior
      # turns (stale_at_start) and still remain never arrived and likely
      # never will (bug #788 only leaks them at the front of the next
      # turn). Drop ONLY those — freshly promoted ids (added by this
      # turn's ResultMessage handler) must stay so the SDK's future
      # delivery is recognised as stale.
      never_surfaced = stale_at_start & stale_tasks
      if never_surfaced:
        log.info("Clean real turn: dropping %d never-surfaced inherited stales",
                 len(never_surfaced))
        stale_tasks -= never_surfaced
      break
    if result.found_stale:
      mode = "drain"
      continue
    # mode == "drain" and no found_stale.
    if not stale_tasks:
      mode = "real"
      continue
    # Drain clean but stale_tasks still populated — inherited ids that
    # never surfaced (likely dead-session). Drop ONLY inherited ids, not
    # newly promoted ones.
    inherited_remaining = stale_at_start & stale_tasks
    if inherited_remaining:
      log.info("Drain turn clean: dropping %d never-surfaced inherited stales",
               len(inherited_remaining))
      stale_tasks -= inherited_remaining
    mode = "real"
  else:
    exhausted = True
    log.warning("Exhausted MAX_RETRIES=%d draining stales; stale_tasks=%s",
                MAX_RETRIES, stale_tasks)
    on_event(ErrorEvent(
      message=(
        f"Stale-task drain exhausted after {MAX_RETRIES} retries — user "
        "prompt may not have reached the model. Consider /clear."
      )
    ))

  if result is None:
    raise RuntimeError("run_turn exited without executing any _single_turn")

  # Trailing thinking compensation only makes sense when the last turn was a
  # real turn whose thinking-only tail was shown to the user as a
  # ProgressEvent.  Drain-turn thinking was suppressed via _NOOP_EVENT, so
  # synthesizing it here would surface content the user never saw building up.
  if (mode_final == "real"
      and not exhausted
      and result.last_emitted == "progress"
      and result.last_thinking):
    log.info("Compensating trailing thinking (%d chars)",
             len(result.last_thinking))
    on_event(AnswerEvent(text=result.last_thinking))

  log.info("turn done (cost=%.4f cumulative, session=%s, final_mode=%s)",
           total_cost,
           result.session_id[:8] if result.session_id else "?",
           mode_final)
  on_event(DoneEvent(cost=total_cost, usage=total_usage,
                     session_id=result.session_id))
  return total_cost, total_usage
