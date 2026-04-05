"""Unified turn card — one Card V2 per turn, evolving via PATCH.

Lifecycle:
  1. Working phase  — grey header, current tool in body,
                      tool history in collapsible panel
  2. Response phase — response markdown in body, all tools in collapsible
  3. Done phase     — green header, response in body, tools + stats in collapsible

All phases update the SAME message via Lark PATCH API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ToolRecord:
  """One tool invocation recorded during a turn."""
  name: str
  summary: str
  ts: float = field(default_factory=time.time)


WORKING_TITLES = [
  (0, "Working..."),
  (20, "Working hard..."),
  (40, "Working really hard..."),
  (60, "Working super hard..."),
  (90, "Working incredibly hard..."),
  (120, "Working unreasonably hard..."),
]


def _elapsed_title(elapsed: int) -> str:
  title = "Working..."
  for threshold, t in WORKING_TITLES:
    if elapsed >= threshold:
      title = t
  return title


def _elapsed_text(elapsed: int) -> str:
  if elapsed < 60:
    return f"{elapsed}s"
  return f"{elapsed // 60}m {elapsed % 60}s"


def _usage_text(usage: dict) -> str:
  parts = []
  inp = usage.get("input_tokens")
  out = usage.get("output_tokens")
  if inp:
    parts.append(f"in: {inp:,}")
  if out:
    parts.append(f"out: {out:,}")
  return " | ".join(parts)


# ---------------------------------------------------------------------------
# Tool summary extraction
# ---------------------------------------------------------------------------

def tool_use_summary(tool_name: str, tool_input: dict) -> str:
  """Build a one-line summary from a ToolUseBlock."""
  if tool_name == "Bash":
    desc = tool_input.get("description", "")
    cmd = tool_input.get("command", "")
    label = desc or cmd
    if len(label) > 60:
      label = label[:57] + "..."
    return f"$ {label}"

  if tool_name in ("Edit", "Write"):
    fp = tool_input.get("file_path", "")
    name = os.path.basename(fp) if fp else "file"
    return f"{tool_name}: {name}"

  if tool_name == "Read":
    fp = tool_input.get("file_path", "")
    name = os.path.basename(fp) if fp else "file"
    return f"Read: {name}"

  if tool_name in ("Glob", "Grep"):
    pattern = tool_input.get("pattern", "")
    if len(pattern) > 40:
      pattern = pattern[:37] + "..."
    return f"{tool_name}: {pattern}"

  if tool_name == "Agent":
    desc = tool_input.get("description", "")
    return f"Agent: {desc}" if desc else "Agent"

  return tool_name


# ---------------------------------------------------------------------------
# Card V2 builders
# ---------------------------------------------------------------------------

def _collapsible_tools(tools: list[ToolRecord], label: str = "Tools") -> dict:
  """Build a collapsible_panel element with tool history."""
  lines = []
  for t in tools:
    lines.append(f"- `{t.summary}`")
  content = "\n".join(lines) if lines else "_none_"
  return {
    "tag": "collapsible_panel",
    "expanded": False,
    "header": {
      "tag": "markdown",
      "content": f"**{label} ({len(tools)})**",
    },
    "vertical_spacing": "8px",
    "elements": [
      {"tag": "markdown", "content": content},
    ],
  }


def _note_element(text: str) -> dict:
  return {
    "tag": "note",
    "elements": [{"tag": "plain_text", "content": text}],
  }


def build_turn_card(
  phase: str,
  *,
  body: str = "",
  tools: list[ToolRecord] | None = None,
  current_tool: str = "",
  elapsed: int = 0,
  usage: dict | None = None,
  chat_id: str = "",
) -> dict:
  """Build a unified turn card for any phase.

  phase: "working" | "response" | "done"
  """
  tools = tools or []
  elements: list[dict] = []

  if phase == "working":
    # Body: current tool action
    if current_tool:
      elements.append({"tag": "markdown", "content": f"`{current_tool}`"})
    # Collapsible tool history (only if >1 tool, current is always the last)
    past_tools = tools[:-1] if len(tools) > 1 else []
    if past_tools:
      elements.append(_collapsible_tools(past_tools, "Previous tools"))
    # Header
    title = _elapsed_title(elapsed)
    header: dict | None = {
      "title": {"tag": "plain_text", "content": title},
      "template": "grey",
    }

  elif phase == "response":
    # Body: response markdown
    if body:
      elements.append({"tag": "markdown", "content": body})
    # All tools in collapsible
    if tools:
      elements.append(_collapsible_tools(tools))
    header = None  # No header for response-only

  elif phase == "done":
    # Body: response markdown
    if body:
      elements.append({"tag": "markdown", "content": body})
    # Tools in collapsible
    if tools:
      elements.append(_collapsible_tools(tools))
    # Note: duration + tokens
    note_parts = [_elapsed_text(elapsed)]
    if usage:
      ut = _usage_text(usage)
      if ut:
        note_parts.append(ut)
    elements.append(_note_element(" | ".join(note_parts)))
    # Green header
    header = {
      "title": {"tag": "plain_text", "content": "Done ✓"},
      "template": "green",
    }

  else:
    raise ValueError(f"Unknown phase: {phase}")

  card: dict = {
    "schema": "2.0",
    "config": {"update_multi": True},
    "body": {"direction": "vertical", "elements": elements},
  }
  if header:
    card["header"] = header
  return card


# ---------------------------------------------------------------------------
# Simple card builders (for commands, errors, status)
# ---------------------------------------------------------------------------

def build_card(
  title: str,
  body: str = "",
  color: str = "blue",
  buttons: list[tuple[str, str, str]] | None = None,
  chat_id: str = "",
  note: str = "",
) -> dict:
  """Build a simple Card V1 for non-turn messages."""
  elements: list[dict] = []
  if body and body.strip():
    elements.append({
      "tag": "div",
      "text": {"content": body, "tag": "lark_md"},
    })
  if buttons:
    actions = []
    for label, action_value, button_type in buttons:
      actions.append({
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": button_type,
        "value": {"action": action_value, "chat_id": chat_id},
      })
    elements.append({"tag": "action", "actions": actions})
  if note:
    elements.append(_note_element(note))
  return {
    "header": {
      "title": {"tag": "plain_text", "content": title},
      "template": color,
    },
    "elements": elements,
  }


def build_markdown_card(content: str, title: str = "", color: str = "") -> dict:
  """Build a Card V2 with markdown content."""
  card: dict = {
    "schema": "2.0",
    "config": {"update_multi": True},
    "body": {
      "direction": "vertical",
      "elements": [{"tag": "markdown", "content": content}],
    },
  }
  if title:
    card["header"] = {"title": {"tag": "plain_text", "content": title}}
    if color:
      card["header"]["template"] = color
  return card
