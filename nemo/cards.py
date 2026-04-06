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
from typing import Any


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


def _usage_text(usage: dict[str, Any]) -> str:
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

def tool_use_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
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

def _collapsible_tools(tools: list[ToolRecord], label: str = "Tools") -> dict[str, Any]:
  """Build a collapsible_panel element with tool history."""
  lines = []
  for t in tools:
    lines.append(f"- `{t.summary}`")
  content = "\n".join(lines) if lines else "_none_"
  return {
    "tag": "collapsible_panel",
    "expanded": False,
    "header": {
      "title": {
        "tag": "plain_text",
        "content": f"{label} ({len(tools)})",
      },
    },
    "vertical_spacing": "8px",
    "elements": [
      {"tag": "markdown", "content": content},
    ],
  }


def _note_element(text: str) -> dict[str, Any]:
  """Small footer text. Uses markdown instead of deprecated 'note' tag (V2)."""
  return {
    "tag": "markdown",
    "content": f"<font color='grey'>{text}</font>",
    "text_size": "notation",
  }


def _stop_button(chat_id: str = "") -> dict[str, Any]:
  """Build a Stop button inside a column_set (Card V2 compatible)."""
  return {
    "tag": "column_set",
    "flex_mode": "none",
    "background_style": "default",
    "columns": [{
      "tag": "column",
      "width": "auto",
      "vertical_align": "top",
      "elements": [{
        "tag": "button",
        "text": {"tag": "plain_text", "content": "Stop"},
        "type": "danger",
        "value": {"action": "stop", "chat_id": chat_id},
      }],
    }],
  }


def build_turn_card(
  phase: str,
  *,
  body: str = "",
  tools: list[ToolRecord] | None = None,
  current_tool: str = "",
  elapsed: int = 0,
  usage: dict[str, Any] | None = None,
  chat_id: str = "",
) -> dict[str, Any]:
  """Build a unified turn card for any phase.

  phase: "working" | "response" | "done"
  """
  tools = tools or []
  elements: list[dict[str, Any]] = []

  if phase == "working":
    # Body: latest intermediate text (Claude's current thinking)
    if body:
      elements.append({"tag": "markdown", "content": body})
    # Current tool action
    if current_tool:
      elements.append({"tag": "markdown", "content": f"`{current_tool}`"})
    # All tools in collapsible
    if tools:
      elements.append(_collapsible_tools(tools))
    # Stop button
    elements.append(_stop_button(chat_id))
    # Header
    title = _elapsed_title(elapsed)
    header: dict[str, Any] | None = {
      "title": {"tag": "plain_text", "content": title},
      "template": "grey",
    }

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

  card: dict[str, Any] = {
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

def _buttons_row(
  buttons: list[tuple[str, str, str]],
  chat_id: str = "",
) -> dict[str, Any]:
  """Build a column_set row of buttons (Card V2 compatible)."""
  columns: list[dict[str, Any]] = []
  for label, action_value, button_type in buttons:
    columns.append({
      "tag": "column",
      "width": "auto",
      "vertical_align": "top",
      "elements": [{
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": button_type,
        "value": {"action": action_value, "chat_id": chat_id},
      }],
    })
  return {
    "tag": "column_set",
    "flex_mode": "none",
    "background_style": "default",
    "columns": columns,
  }


def build_card(
  title: str,
  body: str = "",
  color: str = "blue",
  buttons: list[tuple[str, str, str]] | None = None,
  chat_id: str = "",
  note: str = "",
) -> dict[str, Any]:
  """Build a simple Card V2 for non-turn messages."""
  elements: list[dict[str, Any]] = []
  if body and body.strip():
    elements.append({"tag": "markdown", "content": body})
  if buttons:
    elements.append(_buttons_row(buttons, chat_id))
  if note:
    elements.append(_note_element(note))
  card: dict[str, Any] = {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": title},
      "template": color,
    },
    "body": {"direction": "vertical", "elements": elements},
  }
  return card


def build_form_select(title: str, options: list[dict[str, str]],
                      chat_id: str = "") -> dict[str, Any]:
  """Build a card with select dropdown.

  Each option should have 'text' and 'value' keys.
  """
  select_options = [
    {"text": {"tag": "plain_text", "content": opt["text"]},
     "value": opt["value"]}
    for opt in options
  ]
  elements: list[dict[str, Any]] = [
    {
      "tag": "select_static",
      "placeholder": {"tag": "plain_text", "content": "Select..."},
      "options": select_options,
      "value": {"action": "form_select", "chat_id": chat_id},
    },
  ]
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": title},
      "template": "blue",
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_form_input(title: str, placeholder: str = "",
                     chat_id: str = "") -> dict[str, Any]:
  """Build a card with text input."""
  elements: list[dict[str, Any]] = [
    {
      "tag": "input",
      "name": "user_input",
      "placeholder": {"tag": "plain_text", "content": placeholder or "Type here..."},
      "value": {"action": "form_input", "chat_id": chat_id},
    },
  ]
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": title},
      "template": "blue",
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_markdown_card(content: str, title: str = "", color: str = "") -> dict[str, Any]:
  """Build a Card V2 with markdown content."""
  card: dict[str, Any] = {
    "schema": "2.0",
    "config": {"update_multi": True},
    "body": {
      "direction": "vertical",
      "elements": [{"tag": "markdown", "content": content}],
    },
  }
  if title:
    header: dict[str, Any] = {"title": {"tag": "plain_text", "content": title}}
    if color:
      header["template"] = color
    card["header"] = header
  return card
