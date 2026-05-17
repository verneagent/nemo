"""SDK turn execution — streams responses and emits events.

One function: run_turn(). It takes a single on_event callback that receives
typed events instead of the old dual send_fn/working_fn pattern.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from .cards import tool_use_summary
from .types import JsonObject, TurnClient

log = logging.getLogger(__name__)

# SDK #788 — stale TaskNotification leak.
#
# A background task spawned in an earlier turn can complete *after* that
# turn ends; the SDK then queues its TaskNotification and leaks it at the
# FRONT of the next turn's stream. The model sees it and answers the stale
# instead of the user's new prompt.
#
# The leak is a *stream/transport* artifact of the wedged claude CLI
# subprocess — NOT durable conversation history. We proved (scripts/
# poc_resume_recovery.py, 5/5 deterministic repro+recovery) that the clean,
# deterministic fix is: detect the leak via the ``task_id ∈ stale_tasks``
# check, raise StaleLeakError, and let SDKThread.run_turn_with_reconnect
# tear the subprocess down and reconnect with ``resume=<session_id>`` (NO
# interrupt — the wedged control channel is dead; the SIGKILL in
# _do_close is strictly more thorough). The resumed subprocess replays
# clean history from the session jsonl, so the leaked turn never recurs
# and conversation context is preserved.
#
# This replaces the earlier nonce-drain workaround, which polluted
# conversation history with synthetic user/assistant pairs and could
# pattern-train the model to reply "ack" to real prompts.


class TransientAPIError(RuntimeError):
  """claude CLI surfaced a recoverable API/network error as turn output.

  Raised from _single_turn when an AssistantMessage / ResultMessage body
  matches a transient-error pattern (e.g. ECONNRESET to api.anthropic.com).
  The claude subprocess is usually wedged in this state; SDKThread's
  reconnect loop catches this and spawns a fresh CLI.
  """


class StaleLeakError(TransientAPIError):
  """A prior turn's stale TaskNotification leaked into this turn's stream.

  See the module note above (SDK #788). Raised from ``_single_turn`` the
  moment a ``TaskNotificationMessage`` whose id is in ``stale_tasks``
  appears. Subclasses ``TransientAPIError`` so the existing
  ``SDKThread.run_turn_with_reconnect`` path catches it and recovers by
  reconnecting with ``resume=<session_id>`` (no interrupt) then retrying
  the *real* user prompt on the fresh, clean-history subprocess.
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


@dataclass
class CompactStartedEvent:
  """Conversation compaction is about to begin.

  Surfaced from the Claude Agent SDK's ``PreCompact`` hook, which fires
  before the CLI starts summarising. The compaction itself can take 10–60s
  during which the SDK emits no other messages — without this event the
  working card would silently stall. The matching completion signal is
  ``CompactNoticeEvent``.

  ``trigger`` — "auto" | "manual"; "auto" is what nemo will normally see.
  """
  trigger: str


@dataclass
class CompactNoticeEvent:
  """Conversation-compaction completed.

  The Claude CLI emits a SystemMessage with subtype="compact_boundary"
  once per compaction, AFTER the summarization has finished (the duration
  and post-tokens fields are populated only after the LLM call returns).
  There is no separate "compaction started" event on the stream, so this
  is the user's only signal that a multi-second silent stall on the
  working card was actually the CLI compacting context.

  ``trigger``      — "auto" | "manual"; "auto" is what nemo will normally see.
  ``pre_tokens``   — context-window size before compaction.
  ``post_tokens``  — context-window size after compaction (0 if absent).
  ``duration_ms``  — wall-clock cost of the summarization (0 if absent).
  """
  trigger: str
  pre_tokens: int
  post_tokens: int = 0
  duration_ms: int = 0


@dataclass
class RateLimitNoticeEvent:
  """Upstream rate-limit status surfaced from the SDK's RateLimitEvent.

  The CLI emits its own RateLimitEvent whenever the upstream rate-limit state
  changes (allowed → allowed_warning → rejected and back). Without this notice
  the user sees only a silent working card during a multi-minute retry loop;
  surfacing it lets them decide whether to wait or stop.

  ``status``           — "allowed" | "allowed_warning" | "rejected"; "allowed"
                          means the limit is no longer in effect.
  ``rate_limit_type``  — e.g. "five_hour", "seven_day"; may be empty.
  ``resets_at``        — unix timestamp; may be None.
  ``utilization``      — 0.0–1.0; may be None.
  """
  status: str
  rate_limit_type: str = ""
  resets_at: int | None = None
  utilization: float | None = None


@dataclass
class StaleLeakNoticeEvent:
  """A prior turn's stale TaskNotification leaked into this turn (SDK #788).

  Emitted right before ``StaleLeakError`` is raised, so the host can leave
  a visible breadcrumb explaining the (otherwise silent) reconnect-with-
  resume recovery and the brief delay it causes. ``task_id`` is the leaked
  task's id, for log/trace correlation.
  """
  task_id: str


TurnEvent = (
  ProgressEvent | AnswerEvent |
  TaskStartedEvent | TaskDoneEvent | DoneEvent | ErrorEvent |
  RateLimitNoticeEvent | CompactStartedEvent | CompactNoticeEvent |
  StaleLeakNoticeEvent
)


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------

@dataclass
class _TurnResult:
  cost: float
  usage: JsonObject
  session_id: str
  last_emitted: str
  last_thinking: str


async def _single_turn(
  client: TurnClient,
  prompt: str,
  on_event: Callable[[TurnEvent], None],
  stale_tasks: set[str],
  stop_task_disabled: list[bool],
) -> _TurnResult:
  """Issue one query() and consume its receive_response() stream.

  If a stale TaskNotification (id in ``stale_tasks``) appears anywhere in
  the stream, raise ``StaleLeakError`` immediately (SDK #788). It subclasses
  ``TransientAPIError`` so ``SDKThread.run_turn_with_reconnect`` recovers by
  reconnecting with ``resume=<session_id>`` and retrying the real prompt.

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
    sys_subtype = getattr(message, "subtype", "") if msg_type == "SystemMessage" else ""

    # Conversation compaction surfaces as SystemMessage(subtype=
    # "compact_boundary"). It's a real, multi-second piece of work the CLI
    # just did — count it as progress so back-to-back compactions can't
    # exhaust PROGRESS_TIMEOUT, and forward to the host as
    # CompactNoticeEvent so the user gets a card update during what would
    # otherwise be an unexplained working-card stall. ``microcompact_boundary``
    # is a smaller in-place compaction the CLI UI itself doesn't render —
    # we don't surface it either, but still tick the progress clock.
    if sys_subtype == "compact_boundary":
      data = getattr(message, "data", None) or {}
      meta = data.get("compact_metadata") if isinstance(data, dict) else None
      if not isinstance(meta, dict):
        meta = {}
      log.info("turn msg: SystemMessage[compact_boundary] trigger=%s pre=%s post=%s dur=%sms",
               meta.get("trigger"), meta.get("pre_tokens"),
               meta.get("post_tokens"), meta.get("duration_ms"))
      on_event(CompactNoticeEvent(
        trigger=str(meta.get("trigger") or ""),
        pre_tokens=int(meta.get("pre_tokens") or 0),
        post_tokens=int(meta.get("post_tokens") or 0),
        duration_ms=int(meta.get("duration_ms") or 0),
      ))
      last_progress_at = _time.monotonic()
      continue
    if sys_subtype == "microcompact_boundary":
      log.info("turn msg: SystemMessage[microcompact_boundary] (suppressed)")
      last_progress_at = _time.monotonic()
      continue

    if msg_type in _NON_PROGRESS_MESSAGE_TYPES:
      since_progress = _time.monotonic() - last_progress_at
      log.warning("turn msg: %s (non-progress, idle=%.0fs/%ds)",
                  msg_type, since_progress, PROGRESS_TIMEOUT)
      if msg_type == "RateLimitEvent":
        info = getattr(message, "rate_limit_info", None)
        if info is not None:
          on_event(RateLimitNoticeEvent(
            status=getattr(info, "status", "") or "",
            rate_limit_type=getattr(info, "rate_limit_type", "") or "",
            resets_at=getattr(info, "resets_at", None),
            utilization=getattr(info, "utilization", None),
          ))
    else:
      log.info("turn msg: %s", msg_type)
      last_progress_at = _time.monotonic()

    # --- Stale task leak detection (SDK bug #788) ---
    # Deterministic: a TaskNotification whose id we already know is stale
    # leaked from a prior turn into this stream. The wedged subprocess
    # cannot be salvaged in-stream; raise so run_turn_with_reconnect
    # rebuilds it with resume=<session_id> (no interrupt) and retries the
    # real prompt on clean replayed history. Clear stale_tasks: every id
    # we are tracking belongs to the subprocess we are about to discard;
    # the resumed session starts clean (proven by poc_resume_recovery.py).
    if isinstance(message, TaskNotificationMessage) and message.task_id in stale_tasks:
      log.warning(
        "Stale TaskNotification task=%s leaked into turn stream (#788) — "
        "raising StaleLeakError for reconnect-with-resume recovery",
        message.task_id)
      # Visible breadcrumb BEFORE the raise: the recovery (reconnect with
      # resume + silent retry) is otherwise invisible to the user — they'd
      # only see an unexplained delay. The same on_event closure is reused
      # across the reconnect retry, so this step persists onto the final
      # card's timeline.
      on_event(StaleLeakNoticeEvent(task_id=message.task_id))
      stale_tasks.clear()
      raise StaleLeakError(
        f"stale task {message.task_id} leaked into turn stream")

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

  Single pass. SDK #788 (a prior turn's stale TaskNotification leaking
  into this stream) is handled by ``_single_turn`` raising
  ``StaleLeakError``; that propagates out of here to
  ``SDKThread.run_turn_with_reconnect``, which reconnects with
  ``resume=<session_id>`` (no interrupt) and calls run_turn again with the
  same real prompt on a fresh, clean-history subprocess. We deliberately
  do NOT catch it here — recovery requires tearing down the wedged
  subprocess, which only the reconnect layer can do.

  Returns (cost, usage_dict).
  """
  if stale_tasks is None:
    stale_tasks = set()
  stop_task_disabled: list[bool] = [False]

  # _single_turn raises (StaleLeakError / TimeoutError / TransientAPIError)
  # straight through; the reconnect layer owns recovery.
  result = await _single_turn(
    client, prompt, on_event, stale_tasks, stop_task_disabled,
  )

  total_cost = result.cost
  total_usage: JsonObject = result.usage or {}

  # Trailing thinking compensation: if the turn ended on a thinking-only
  # tail shown as a ProgressEvent, surface it as the answer.
  if result.last_emitted == "progress" and result.last_thinking:
    log.info("Compensating trailing thinking (%d chars)",
             len(result.last_thinking))
    on_event(AnswerEvent(text=result.last_thinking))

  log.info("turn done (cost=%.4f, session=%s)",
           total_cost,
           result.session_id[:8] if result.session_id else "?")
  on_event(DoneEvent(cost=total_cost, usage=total_usage,
                     session_id=result.session_id))
  return total_cost, total_usage
