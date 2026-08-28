"""Agent-agnostic turn event vocabulary and the shared per-turn usage schema.

This module is deliberately AGENT-AGNOSTIC: it must NOT import any coding-agent
SDK (claude_agent_sdk, @openai/codex-sdk) or any concrete adapter. It defines
only the typed events every ``CodingAgent`` emits to the main loop, plus
``canonical_usage`` — the one per-turn usage shape all adapters normalize into.

The Claude-SDK stream consumer that PRODUCES these events lives in
``claude_turn.py`` (the Claude adapter layer); Codex/OpenCode produce them from
their own sidecars. ``tests/test_agnostic_imports.py`` is the guardrail that
keeps this and the other agnostic modules SDK-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import JsonObject


def canonical_usage(
  *,
  input_tokens: int,
  cache_read: int,
  cache_creation: int,
  output_tokens: int,
) -> JsonObject:
  """Build the unified PER-TURN usage dict that turn cards and /context read.

  Every CodingAgent adapter normalizes its native usage into this one schema
  so Claude, Codex and OpenCode report the SAME thing:
    - ``input_tokens``            new (uncached) input this turn
    - ``cache_read_input_tokens`` cached input reused this turn
    - ``cache_creation_input_tokens`` cache written this turn (Claude only)
    - ``output_tokens``           output this turn
    - ``total_tokens``            sum of the four above

  All figures are PER-TURN. This matters because Codex's SDK reports
  ``turn.completed.usage`` as the SESSION-CUMULATIVE total, so its adapter
  must difference successive totals before calling this — otherwise the card
  showed an ever-growing "in" while Claude showed a single turn (the bug this
  schema fixes).
  """
  return {
    "input_tokens": input_tokens,
    "cache_read_input_tokens": cache_read,
    "cache_creation_input_tokens": cache_creation,
    "output_tokens": output_tokens,
    "total_tokens": input_tokens + cache_read + cache_creation + output_tokens,
  }


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


@dataclass
class BackgroundTaskDoneEvent:
  """A background task (spawned in a prior turn) completed during idle.

  Surfaced by the between-turn idle stream drainer (``SDKThread``) when the
  CLI emits a ``TaskNotificationMessage`` while no turn is running. It is
  delivered through the SAME thread-safe callback contract as ``on_event``,
  but via the separate idle channel (``set_idle_notifier``) so the host
  renders it as its own notification card rather than touching the turn
  card. Without this the notification would sit unread in the SDK buffer
  and leak into the front of the next turn (SDK #788), and the user would
  never be told their background work finished.
  """
  task_id: str
  status: str = ""
  summary: str = ""


@dataclass
class BackgroundTurnDoneEvent:
  """The CLI ran a spontaneous turn during idle (Monitor fire / self-directed).

  The claude CLI pushes spontaneous turns onto the stream with zero stdin
  (e.g. a Monitor that fired, or a background job the model chose to follow
  up on). The idle drainer accumulates the assistant text and surfaces the
  final answer as a notification card when the turn's ResultMessage arrives.
  """
  text: str
  cost: float = 0.0


TurnEvent = (
  ProgressEvent | AnswerEvent |
  TaskStartedEvent | TaskDoneEvent | DoneEvent | ErrorEvent |
  RateLimitNoticeEvent | CompactStartedEvent | CompactNoticeEvent |
  StaleLeakNoticeEvent | BackgroundTaskDoneEvent | BackgroundTurnDoneEvent
)
