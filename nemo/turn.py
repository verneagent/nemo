"""SDK turn execution — streams responses and emits events.

One function: run_turn(). It takes a single on_event callback that receives
typed events instead of the old dual send_fn/working_fn pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .cards import ToolRecord, tool_use_summary

log = logging.getLogger(__name__)

MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Turn events
# ---------------------------------------------------------------------------

@dataclass
class ToolStartEvent:
  """First tool use in a turn — create the Working card."""
  tool: ToolRecord


@dataclass
class ToolProgressEvent:
  """Subsequent tool use — update the Working card."""
  tool: ToolRecord


@dataclass
class TextEvent:
  """Agent produced text output."""
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
  usage: dict[str, Any]


@dataclass
class ErrorEvent:
  """Turn ended with error."""
  message: str


TurnEvent = (
  ToolStartEvent | ToolProgressEvent | TextEvent |
  TaskStartedEvent | TaskDoneEvent | DoneEvent | ErrorEvent
)


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------

async def run_turn(
  client: Any,
  prompt: str,
  on_event: Callable[[TurnEvent], None],
  stale_tasks: set[str] | None = None,
  _retry: int = 0,
) -> tuple[float, dict[str, Any]]:
  """Send prompt to SDK client, stream responses, emit events.

  Returns (cost, usage_dict).
  """
  from claude_agent_sdk import (
    AssistantMessage, TextBlock, ToolUseBlock, ResultMessage,
    TaskStartedMessage, TaskNotificationMessage, TaskProgressMessage,
  )

  if stale_tasks is None:
    stale_tasks = set()

  log.info("query() prompt=%d chars retry=%d", len(prompt), _retry)
  await client.query(prompt)

  cost = 0.0
  usage: dict[str, Any] = {}
  pending_tasks: set[str] = set()
  working_started = False
  found_stale = False

  async for message in client.receive_response():
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
      continue

    # --- Normal message handling ---
    if isinstance(message, AssistantMessage):
      text_parts = []
      tool_summary = ""
      for block in message.content:
        if isinstance(block, TextBlock) and block.text:
          text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
          tool_summary = tool_use_summary(block.name, block.input)

      text = "\n".join(text_parts)
      if not text and message.content:
        # Tool-only message → working card
        if tool_summary:
          record = ToolRecord(name="", summary=tool_summary)
          if not working_started:
            on_event(ToolStartEvent(tool=record))
            working_started = True
          else:
            on_event(ToolProgressEvent(tool=record))
      elif text:
        # Text output
        task_id = None
        parent = getattr(message, "parent_tool_use_id", None)
        if parent and pending_tasks:
          task_id = next(iter(pending_tasks))
        on_event(TextEvent(
          text=text, task_id=task_id,
          pending_tasks=len(pending_tasks),
        ))

    elif isinstance(message, TaskStartedMessage):
      pending_tasks.add(message.task_id)
      on_event(TaskStartedEvent(task_id=message.task_id))

    elif isinstance(message, TaskProgressMessage):
      desc = getattr(message, "description", "") or ""
      if desc and working_started:
        record = ToolRecord(name="", summary=desc)
        on_event(ToolProgressEvent(tool=record))

    elif isinstance(message, TaskNotificationMessage):
      pending_tasks.discard(message.task_id)
      on_event(TaskDoneEvent(
        task_id=message.task_id,
        status=getattr(message, "status", ""),
      ))

    elif isinstance(message, ResultMessage):
      cost = getattr(message, "total_cost_usd", 0) or 0.0
      usage = getattr(message, "usage", None) or {}
      # Mark remaining pending tasks as stale
      for tid in list(pending_tasks):
        stale_tasks.add(tid)
        try:
          await client.stop_task(tid)
        except Exception:
          pass
      pending_tasks.clear()

  # If stale notification contaminated this turn, re-query
  if found_stale and _retry < MAX_RETRIES:
    log.info("Stale turn — re-querying (retry %d/%d)", _retry + 1, MAX_RETRIES)
    return await run_turn(
      client, prompt, on_event,
      stale_tasks=stale_tasks, _retry=_retry + 1,
    )

  on_event(DoneEvent(cost=cost, usage=usage))
  return cost, usage
