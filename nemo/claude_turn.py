"""Claude Agent SDK turn execution — consume the SDK message stream, emit
agent-agnostic turn events.

This is the Claude-SPECIFIC turn consumer, kept OUT of the neutral ``turn.py``
(which holds only the agent-agnostic event vocabulary + the shared usage
schema). It imports ``claude_agent_sdk`` message types and encodes the Claude
CLI's error-surfacing / stream conventions, so per AGENTS.md it belongs in the
concrete adapter layer, not in an agnostic module. ``SDKThread`` is its only
caller. (It lives in its own leaf module rather than inside ``claude_agent.py``
because ``sdk_thread`` imports ``run_turn`` while ``claude_agent`` imports
``SDKThread`` — colocating them there would create an import cycle.)

One entry point: run_turn().
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Callable

from .cards import tool_use_summary
from .turn import (
  AnswerEvent, CompactNoticeEvent, DoneEvent, ErrorEvent, ProgressEvent,
  RateLimitNoticeEvent, StaleLeakNoticeEvent, TaskDoneEvent, TaskStartedEvent,
  TurnEvent, canonical_usage,
)
from .types import JsonObject, TurnClient

log = logging.getLogger(__name__)


def _usage_int(usage: JsonObject, key: str) -> int:
  """Read a token count from a raw usage dict, coercing to a non-negative int."""
  value = usage.get(key)
  if isinstance(value, bool):  # bool is an int subclass — reject it explicitly
    return 0
  if isinstance(value, (int, float)):
    return max(0, int(value))
  return 0


def normalize_claude_usage(usage: JsonObject) -> JsonObject:
  """Map Claude's ResultMessage.usage (already per-turn) into canonical form.

  Claude's native keys (input_tokens / cache_read_input_tokens /
  cache_creation_input_tokens / output_tokens) line up directly; we just drop
  the extras (server_tool_use, service_tier, …) and add total_tokens. An empty
  usage stays empty so the card omits the line entirely.

  Inclusive-style upstream workaround — jundot/omlx#1487 (v0.3.8 through at
  least v0.3.12): omlx's Anthropic shim reports ``input_tokens`` INCLUSIVE of
  ``cache_read_input_tokens + cache_creation_input_tokens``, violating
  Anthropic's disjoint-triple contract. Without correction the canonical
  ``i`` and ``total_tokens`` end up roughly 2× the real values. Detect via
  exact equality (real Anthropic essentially never has new-uncached input
  that precisely matches the cached prefix sum — a typical Claude turn has
  ``input_tokens`` ≈ tens while ``cache_read`` ≈ tens of thousands) and
  subtract. Remove once omlx ships the fix.
  """
  if not usage:
    return {}
  raw_input = _usage_int(usage, "input_tokens")
  cache_read = _usage_int(usage, "cache_read_input_tokens")
  cache_creation = _usage_int(usage, "cache_creation_input_tokens")
  output_tokens = _usage_int(usage, "output_tokens")
  cache_sum = cache_read + cache_creation
  if cache_sum > 0 and raw_input == cache_sum:
    log.info(
      "Inclusive-usage upstream detected (omlx#1487 shape): "
      "raw_input=%d == cache_read+cache_creation=%d+%d — clamping i to 0",
      raw_input, cache_read, cache_creation,
    )
    input_tokens = 0
  else:
    input_tokens = raw_input
  return canonical_usage(
    input_tokens=input_tokens,
    cache_read=cache_read,
    cache_creation=cache_creation,
    output_tokens=output_tokens,
  )

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


class NonRetryableAPIError(RuntimeError):
  """claude CLI surfaced a provider/account error that retrying won't fix."""


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

_NON_RETRYABLE_ERROR_SIGNALS: tuple[str, ...] = (
  "402",
  "insufficient balance",
  "insufficient_quota",
  "quota exceeded",
  "billing",
  "payment required",
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


def _looks_like_non_retryable_api_error(
  text: str,
  *,
  is_error_flag: bool = False,
) -> bool:
  """Return True for provider/account failures that should not be retried."""
  if not text:
    return False
  stripped = text.lstrip()
  if not stripped.startswith(_CLI_ERROR_PREFIX) and not is_error_flag:
    return False
  lowered = text.lower()
  return any(sig in lowered for sig in _NON_RETRYABLE_ERROR_SIGNALS)


# Leaked tool-call markup.
#
# The model emitted an Anthropic function-call invocation as PLAIN TEXT —
# `<invoke name="...">...</invoke>` — instead of a structured tool_use block.
# The SDK only ever executes structured tool_use, so the intended tool never
# runs; when this is the turn's FINAL message the turn ends early and the raw
# markup would otherwise be rendered as a bogus "Done ✓" answer.
#
# Observed on Opus after auto-compaction in long, screenshot-heavy tool loops,
# e.g. the final message of a simulator-driving turn was the bare text:
#   court
#   <invoke name="Read">
#   <parameter name="file_path">/tmp/vf_agent.png</parameter>
#   </invoke>
# — so the screenshot was never read and the turn died with that markup as the
# card body. We can't make the tool run from here, but we MUST NOT present the
# markup as a successful answer; we wrap it in an explicit anomaly notice.
#
# Precision: require BOTH a `<invoke name=` opener and a matching `</invoke>`
# closer, and bail if the text contains a ``` fence (the model is then
# *displaying* the syntax in a code block — e.g. explaining tool-call format —
# not emitting a call). The `antml:` namespace variant is matched too.
_LEAKED_INVOKE_OPEN_RE = re.compile(r"<(?:antml:)?invoke\s+name\s*=", re.IGNORECASE)
_LEAKED_INVOKE_CLOSE_RE = re.compile(r"</(?:antml:)?invoke>", re.IGNORECASE)

_LEAKED_TOOL_CALL_NOTICE = (
  "⚠️ 模型把一个工具调用写成了纯文本（`<invoke …>`），该工具并未真正执行，"
  "本轮动作未完成。下面是模型的原始输出：\n\n"
)


def _looks_like_leaked_tool_call(text: str) -> bool:
  """True when an assistant TEXT block is actually an unparsed tool call.

  See the module note above. False (intentionally) when the markup sits inside
  a ``` fence, since that is the model showing the syntax rather than invoking
  it — the observed real leaks arrive as bare, un-fenced text.
  """
  if not text or "```" in text:
    return False
  return bool(_LEAKED_INVOKE_OPEN_RE.search(text)) and bool(
    _LEAKED_INVOKE_CLOSE_RE.search(text))

# If receive_messages() yields nothing for this long, assume the turn is stuck.
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

# After a ResultMessage arrives on a turn that had a steer injected, the CLI
# may process that steer as its OWN follow-on turn with its OWN ResultMessage.
# We must NOT stop reading at the first Result (that strands the steer's answer
# + Result in the receive buffer, where the NEXT turn drains it → permanent +1
# turn lag: every question gets the previous turn's answer). Instead we keep
# consuming until the CLI goes quiescent — i.e. no new message for
# QUIESCENCE_TIMEOUT. Any steer-continuation Result is thus folded into the
# SAME run_turn and nothing is left to strand. A steer that was *folded* (0
# extra Result) simply idles out this window; a *late* steer streams its
# continuation within it. This is Result-as-event, not Result-as-return.
QUIESCENCE_TIMEOUT = 8  # seconds of CLI silence after a Result = turn complete

# Messages that do NOT count as real progress. Anything outside this set
# refreshes the progress clock. Checked by class name to avoid importing
# optional SDK types (RateLimitEvent may not exist on older SDKs).
_NON_PROGRESS_MESSAGE_TYPES: frozenset[str] = frozenset({
  "SystemMessage",
  "RateLimitEvent",
})


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
  is_paused: Callable[[], bool] | None = None,
  steered: list[bool] | None = None,
  steer_probe: Callable[[str], int] | None = None,
) -> _TurnResult:
  """Issue one query() and consume the client's receive_messages() stream.

  If a stale TaskNotification (id in ``stale_tasks``) appears anywhere in
  the stream, raise ``StaleLeakError`` immediately (SDK #788). It subclasses
  ``TransientAPIError`` so ``SDKThread.run_turn_with_reconnect`` recovers by
  reconnecting with ``resume=<session_id>`` and retrying the real prompt.

  ``stop_task_disabled`` is a one-element mutable flag: once ``stop_task``
  returns "Control request timeout" the control channel is presumed wedged
  and we skip further stop_task calls for the remainder of this orchestration.

  ``is_paused`` (optional) returns True while an interactive prompt
  (AskUserQuestion / permission) is awaiting the user. During that time the
  SDK emits no messages because it is legitimately blocked inside a
  ``can_use_tool`` callback — that silence is expected, not a hang. While
  paused the progress/heartbeat watchdog is held off so it never force-
  reconnects mid-prompt (which would tear down the in-flight question and
  re-ask it, discarding the user's selections).

  ``steered`` is a one-element mutable flag set True by ``SDKThread.steer``
  when a user message is injected into THIS running turn. When set, we do not
  treat the first ResultMessage as end-of-turn: the CLI may run the steer as a
  separate turn with its own Result, so we keep consuming until the stream goes
  quiescent (QUIESCENCE_TIMEOUT). This folds the steer's answer into the same
  run_turn and prevents the "+1 turn lag" desync (see QUIESCENCE_TIMEOUT note).
  We stream ``receive_messages()`` (the raw stream) rather than
  ``receive_response()`` precisely because the latter returns at the first
  Result — which is what strands the steer continuation.

  ``steer_probe`` (optional) is ``session_id -> number of steer continuation
  batches`` derived from the session transcript (see
  ``ClaudeCodingAgent.steer_continuation_batches``). Quiescence alone is
  RACY: the CLI dequeues a steer at the Result boundary and its continuation
  may stay silent for longer than QUIESCENCE_TIMEOUT (observed 15s — model
  latency + tool call), so the 8s window declares the turn done and strands
  the whole continuation → permanent +1 turn lag (incident 2026-07-16
  18:33). At every quiescence expiry we therefore ask the transcript how
  many Results this turn owes (1 + started continuation batches) and, if
  Results are still outstanding, drop back to the normal heartbeat/progress
  budgets until the next Result instead of ending the turn.
  """
  from claude_agent_sdk import (
    AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock, ResultMessage,
    TaskStartedMessage, TaskNotificationMessage, TaskProgressMessage,
  )

  steered_flag: list[bool] = steered if steered is not None else [False]

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
  non_retryable_error_text: str = ""

  FIRST_MSG_TIMEOUT = 30
  msg_count = 0
  saw_result = False  # a ResultMessage arrived → in quiescence/steer-drain mode
  results_seen = 0  # ResultMessages consumed (steer continuations add more)
  last_progress_at = _time.monotonic()
  response = client.receive_messages()
  # The pending receive future persists across watchdog ticks. While a prompt
  # is awaiting the user (is_paused) we must NOT cancel it — the SDK is
  # legitimately mid-``can_use_tool`` and a fresh ``__anext__`` would race the
  # generator. We only recreate it after a message has been consumed.
  next_task: asyncio.Future[object] | None = None
  while True:
    # Per-iteration timeout is the heartbeat budget, but we also cap it at
    # the remaining progress budget so SDK-internal retry loops (which emit
    # SystemMessage every ~1 min indefinitely) cannot keep a dead turn alive.
    # While an interactive prompt (AskUserQuestion / permission) awaits the
    # user, the SDK emits nothing — expected, not a hang. Hold the progress
    # clock so the watchdog never force-reconnects mid-prompt.
    paused = bool(is_paused()) if is_paused is not None else False
    if paused:
      last_progress_at = _time.monotonic()
    if saw_result and not paused:
      # Quiescence / steer-drain mode: a Result already arrived on a steered
      # turn. We are only waiting to see whether the CLI emits a steer
      # continuation (its own follow-on turn). Bound the wait to
      # QUIESCENCE_TIMEOUT; an idle tick means the CLI has gone silent → the
      # turn (including any steer follow-on) is genuinely complete. Never force
      # a reconnect here — silence after a Result is the normal end of a turn.
      iter_timeout = QUIESCENCE_TIMEOUT
    else:
      heartbeat_budget = FIRST_MSG_TIMEOUT if msg_count == 0 else HEARTBEAT_TIMEOUT
      progress_budget = PROGRESS_TIMEOUT - (_time.monotonic() - last_progress_at)
      if progress_budget <= 0 and not paused:
        log.error("no progress for %ds (msgs=%d) — forcing reconnect",
                  PROGRESS_TIMEOUT, msg_count)
        timed_out = True
        break
      iter_timeout = max(1, min(heartbeat_budget, progress_budget))
    if next_task is None:
      next_task = asyncio.ensure_future(response.__anext__())
    done, _ = await asyncio.wait({next_task}, timeout=iter_timeout)
    if not done:
      # No message this tick. If a prompt is awaiting the user, keep the
      # receive future alive and just defer — do not cancel or reconnect.
      if is_paused is not None and is_paused():
        last_progress_at = _time.monotonic()
        continue
      if saw_result:
        # Quiescent after a Result. Before ending, ask the transcript whether
        # a steer continuation is still RUNNING: the CLI dequeues a steer at
        # the Result boundary (user-row fold) and runs it as a follow-on turn
        # with its own Result, whose first message can lag well past
        # QUIESCENCE_TIMEOUT (observed 15s). Ending here would strand that
        # continuation in the receive buffer → every later turn answers the
        # PREVIOUS message (+1 turn lag).
        expected_results = 1
        if steer_probe is not None and steered_flag[0]:
          try:
            expected_results = 1 + steer_probe(sdk_session_id)
          except Exception as exc:
            log.warning("steer continuation probe failed: %s", exc)
        if results_seen < expected_results:
          # Keep the pending receive future alive (like the paused path):
          # cancelling __anext__ mid-await can tear down the stream.
          log.info(
            "steer continuation pending (results=%d expected=%d) — "
            "waiting past quiescence", results_seen, expected_results)
          saw_result = False  # back to heartbeat/progress budgets
          last_progress_at = _time.monotonic()
          continue
        # Genuinely complete: any injected steer was folded (no extra
        # Result) or its continuation Result has been drained above. End
        # cleanly, NOT via reconnect. Nothing is left buffered to strand
        # into the next turn.
        next_task.cancel()
        next_task = None
        log.info("turn quiescent %ds after result (steer continuation drained)",
                 QUIESCENCE_TIMEOUT)
        break
      next_task.cancel()
      next_task = None
      since_progress = _time.monotonic() - last_progress_at
      if since_progress >= PROGRESS_TIMEOUT:
        log.error("no progress for %.0fs (msgs=%d) — forcing reconnect",
                  since_progress, msg_count)
      else:
        log.error("receive_messages() heartbeat timeout (%ds, msgs=%d) — forcing reconnect",
                  heartbeat_budget, msg_count)
      timed_out = True
      break
    try:
      message = next_task.result()
    except StopAsyncIteration:
      next_task = None
      break
    next_task = None
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
        # Detect claude-CLI-surfaced API errors BEFORE emitting as a
        # user-facing AnswerEvent. Provider/account failures such as 402
        # balance are not retryable; network-ish failures are.
        if _looks_like_non_retryable_api_error(text):
          log.warning("Non-retryable API error in AssistantMessage: %r", text[:200])
          non_retryable_error_text = text
        elif _looks_like_transient_api_error(text):
          log.warning("Transient API error in AssistantMessage: %r", text[:200])
          transient_error_text = text
        else:
          task_id = None
          parent = getattr(message, "parent_tool_use_id", None)
          if parent and pending_tasks:
            task_id = next(iter(pending_tasks))
          # Guard: the model wrote a tool call as plain text (tool never ran).
          # Wrap it in an anomaly notice so the Done card doesn't claim success.
          if _looks_like_leaked_tool_call(text):
            log.warning(
              "Leaked tool-call markup in AssistantMessage text "
              "(model wrote <invoke> as text — tool NOT executed): %r",
              text[:200])
            answer_text = _LEAKED_TOOL_CALL_NOTICE + "```\n" + text + "\n```"
          else:
            answer_text = text
          on_event(AnswerEvent(
            text=answer_text, task_id=task_id,
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
      # Accumulate cost across Results (a late steer adds a second Result whose
      # cost belongs to this same run_turn); keep the latest usage/session id.
      cost += getattr(message, "total_cost_usd", 0) or 0.0
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
      if (non_retryable_error_text
          or _looks_like_non_retryable_api_error(
            result_text, is_error_flag=is_error_flag)):
        err_sample = (non_retryable_error_text or result_text)[:500]
        log.warning("Non-retryable API error in turn result: %r", err_sample[:200])
        for tid in list(pending_tasks):
          stale_tasks.add(tid)
        pending_tasks.clear()
        on_event(ErrorEvent(message=err_sample))
        raise NonRetryableAPIError(err_sample)
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
      saw_result = True
      results_seen += 1
      if not steered_flag[0]:
        # No steer this turn → exactly one Result → done. (Unchanged behavior;
        # zero added latency for the common case.)
        break
      # A steer was injected this turn: do NOT break here or we strand the
      # steer's follow-on Result in the receive buffer. Keep looping — the
      # quiescence window (top of loop) ends the turn once the CLI is silent,
      # after consuming any steer continuation into THIS run_turn.
      last_progress_at = _time.monotonic()

  if timed_out:
    on_event(ErrorEvent(message="Turn timed out — SDK stopped responding"))
    raise TimeoutError("receive_messages() heartbeat timeout")

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
  is_paused: Callable[[], bool] | None = None,
  steered: list[bool] | None = None,
  steer_probe: Callable[[str], int] | None = None,
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
    client, prompt, on_event, stale_tasks, stop_task_disabled, is_paused,
    steered, steer_probe,
  )

  total_cost = result.cost
  # Normalize Claude's per-turn usage into the canonical schema (see
  # canonical_usage) so the done card and /context read the same shape across
  # all adapters.
  total_usage: JsonObject = normalize_claude_usage(result.usage or {})

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
