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

MAX_RETRIES = 5
# If receive_response() yields nothing for this long, assume the turn is stuck.
# SDK docs: "If no ResultMessage is received, the iterator continues indefinitely."
# Must be generous — Agent spawning and complex edits can go quiet for minutes.
HEARTBEAT_TIMEOUT = 300  # seconds


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

async def run_turn(
  client: TurnClient,
  prompt: str,
  on_event: Callable[[TurnEvent], None],
  stale_tasks: set[str] | None = None,
  _retry: int = 0,
) -> tuple[float, JsonObject]:
  """Send prompt to SDK client, stream responses, emit events.

  Returns (cost, usage_dict).
  """
  from claude_agent_sdk import (
    AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock, ResultMessage,
    TaskStartedMessage, TaskNotificationMessage, TaskProgressMessage,
  )

  if stale_tasks is None:
    stale_tasks = set()

  import anyio as _anyio
  log.info("query() prompt=%d chars retry=%d", len(prompt), _retry)
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
  last_emitted: str = ""  # "progress" or "answer" — for trailing thinking compensation
  last_thinking: str = ""  # last thinking text, used for compensation

  # The SDK uses anyio internally. Neither asyncio.wait_for nor anyio.fail_after
  # can reliably cancel stuck anyio operations from asyncio context.
  # Workaround: run __anext__() as a task and use asyncio.wait() with a
  # separate sleep task as the timeout.  If the timeout wins, we just abandon
  # the stuck task (it will be cleaned up when the client is closed).
  FIRST_MSG_TIMEOUT = 30  # seconds
  msg_count = 0

  response = client.receive_response()
  while True:
    timeout = FIRST_MSG_TIMEOUT if msg_count == 0 else HEARTBEAT_TIMEOUT
    next_task = asyncio.ensure_future(response.__anext__())
    done, _ = await asyncio.wait({next_task}, timeout=timeout)
    if not done:
      # Timeout — next_task is still pending
      next_task.cancel()
      log.error("receive_response() timeout (%ds, msgs=%d) — forcing reconnect",
                timeout, msg_count)
      timed_out = True
      break
    try:
      message = next_task.result()
    except StopAsyncIteration:
      break
    msg_count += 1
    log.info("turn msg: %s", type(message).__name__)

    # --- Stale task detection (SDK bug #788 workaround) ---
    if isinstance(message, TaskNotificationMessage) and message.task_id in stale_tasks:
      stale_tasks.discard(message.task_id)
      found_stale = True
      log.warning("Stale TaskNotification task=%s — will re-query", message.task_id)
      continue

    if found_stale:
      if isinstance(message, ResultMessage):
        cost = getattr(message, "total_cost_usd", 0) or 0.0
        usage = getattr(message, "usage", None) or {}
        sdk_session_id = getattr(message, "session_id", "") or ""
        break  # Don't wait for StopAsyncIteration in stale path either
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
        # Tool-only message → working card
        if tool_summary:
          is_first = not progress_started
          progress_started = True
          on_event(ProgressEvent(kind="tool", summary=tool_summary, first=is_first))
          last_emitted = "progress"
      elif text:
        # Text output → answer
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
      # Mark remaining pending tasks as stale
      for tid in list(pending_tasks):
        stale_tasks.add(tid)
        try:
          await client.stop_task(tid)
        except Exception as e:
          log.warning("Failed to stop stale task %s: %s", tid, e)
      pending_tasks.clear()
      break  # ResultMessage is the final message — don't wait for StopAsyncIteration

  if timed_out:
    on_event(ErrorEvent(message="Turn timed out — SDK stopped responding"))
    raise TimeoutError("receive_response() heartbeat timeout")

  # If stale notification contaminated this turn, re-query
  if found_stale and _retry < MAX_RETRIES:
    log.info("Stale turn — re-querying (retry %d/%d)", _retry + 1, MAX_RETRIES)
    return await run_turn(
      client, prompt, on_event,
      stale_tasks=stale_tasks, _retry=_retry + 1,
    )

  # --- Trailing thinking compensation ---
  # If the last emitted event was a ProgressEvent (thinking), the Done card
  # would miss the model's final reasoning.  Synthesize an AnswerEvent so
  # the consumer always has a complete final answer.
  if last_emitted == "progress" and last_thinking:
    log.info("Compensating trailing thinking (%d chars)", len(last_thinking))
    on_event(AnswerEvent(text=last_thinking))

  log.info("turn done (cost=%.4f, session=%s)", cost, sdk_session_id[:8] if sdk_session_id else "?")
  on_event(DoneEvent(cost=cost, usage=usage, session_id=sdk_session_id))
  return cost, usage
