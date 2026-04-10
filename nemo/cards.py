"""Unified turn card — one Card V2 per turn, evolving via PATCH.

Lifecycle:
  1. Working phase  — grey header, current action inline,
                      unified thinking timeline in collapsible
  2. Done phase     — green header, final response inline,
                      thinking timeline in collapsible
  3. Error phase    — red header, error inline, thinking in collapsible

All phases update the SAME message via Lark PATCH API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .types import JsonObject


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ToolRecord:
  """One tool invocation recorded during a turn."""
  name: str
  summary: str
  ts: float = field(default_factory=time.time)


@dataclass
class ThinkingStep:
  """One entry in the unified thinking timeline (text or tool)."""
  kind: str       # "text" | "tool"
  content: str    # text content or tool summary


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


def _usage_text(usage: JsonObject) -> str:
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

def tool_use_summary(tool_name: str, tool_input: JsonObject) -> str:
  """Build a short label for a tool call. Format: `{Type}: {detail}`.

  Type prefix lets _collapsible_thinking group consecutive same-type calls.
  """
  if tool_name == "Bash":
    desc = tool_input.get("description", "")
    cmd = tool_input.get("command", "")
    label = desc or cmd
    if len(label) > 60:
      label = label[:57] + "..."
    return f"Bash: {label}"

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

  if tool_name in ("Agent", "Task"):
    desc = tool_input.get("description", "")
    return f"Agent: {desc}" if desc else "Agent"

  if tool_name == "Skill":
    name = tool_input.get("skill", "")
    return f"Skill: {name}" if name else "Skill"

  return tool_name


# ---------------------------------------------------------------------------
# Card V2 builders
# ---------------------------------------------------------------------------

def _sanitize_markdown_keep_newlines(text: str) -> str:
  """Flatten markdown tables into bullet-like lines, preserving newlines.

  Lark cards have a hard limit (~3) on tables per card. Convert table
  rows to `col1 · col2 · col3` text so they still read naturally.
  """
  lines = []
  for line in text.split("\n"):
    stripped = line.lstrip()
    if stripped.startswith("|") or set(stripped.replace(" ", "").replace("|", "")) <= {"-", ":"}:
      cells = [c.strip() for c in stripped.strip("|").split("|")]
      cells = [c for c in cells if c and not set(c) <= {"-", ":"}]
      if cells:
        lines.append(" · ".join(cells))
    else:
      lines.append(line)
  return "\n".join(lines)


def _sanitize_markdown(text: str) -> str:
  """Same as above but collapses newlines (for inline truncated rendering)."""
  return " ".join(_sanitize_markdown_keep_newlines(text).split("\n"))


def _escape_md(text: str) -> str:
  """Escape markdown special chars that would break rendering."""
  # Lark markdown escapes < and > to &lt;/&gt; inside code blocks which looks
  # ugly. Replace with visually similar chars for patterns like `<<<<<<<`.
  return text.replace("<", "‹").replace(">", "›").replace("*", "\\*")


def _tool_type_and_detail(summary: str) -> tuple[str, str]:
  """Split a tool summary 'Type: detail' into (type, detail)."""
  if ":" in summary:
    t, d = summary.split(":", 1)
    return t.strip(), d.strip()
  return summary.strip(), ""


def _collapsible_thinking(steps: list[ThinkingStep]) -> JsonObject:
  """Build a collapsible_panel with narrative text + grouped tool lines.

  Consecutive tool calls of the same type are merged into one line,
  e.g. 7 Grep calls become `Grep: pat1, pat2, pat3, ...`.
  """
  lines: list[str] = []
  # Accumulator for consecutive same-type tool calls
  pending_type: str = ""
  pending_details: list[str] = []

  def _flush_tools() -> None:
    if not pending_type:
      return
    details = [d for d in pending_details if d]
    if details:
      joined = ", ".join(details)
      if len(joined) > 200:
        joined = joined[:197] + "..."
      lines.append(
        f"<font color='grey'>{pending_type}:</font> {_escape_md(joined)}")
    else:
      lines.append(
        f"<font color='grey'>{pending_type}</font> × {len(pending_details)}")

  for s in steps:
    if s.kind == "tool":
      ttype, detail = _tool_type_and_detail(s.content)
      if ttype == pending_type:
        pending_details.append(detail)
      else:
        _flush_tools()
        pending_type = ttype
        pending_details = [detail]
    else:
      # Narrative text or thinking — flush pending tools first
      _flush_tools()
      pending_type = ""
      pending_details = []
      text = _sanitize_markdown(s.content)
      max_len = 200 if s.kind == "thinking" else 300
      if len(text) > max_len:
        text = text[:max_len - 3] + "..."
      if s.kind == "thinking":
        lines.append(f"_{_escape_md(text)}_")
      else:
        lines.append(_escape_md(text))
  _flush_tools()

  content = "\n\n".join(lines) if lines else "_none_"
  return {
    "tag": "collapsible_panel",
    "expanded": False,
    "header": {
      "title": {
        "tag": "plain_text",
        "content": f"Thinking ({len(steps)})",
      },
    },
    "vertical_spacing": "8px",
    "elements": [
      {"tag": "markdown", "content": content},
    ],
  }


def _note_element(text: str) -> JsonObject:
  """Small footer text. Uses markdown instead of deprecated 'note' tag (V2)."""
  return {
    "tag": "markdown",
    "content": f"<font color='grey'>{text}</font>",
    "text_size": "notation",
  }


def _stop_button(chat_id: str = "") -> JsonObject:
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
        "value": {"action": "__stop__", "chat_id": chat_id},
      }],
    }],
  }


def _working_elements(
  *,
  steps: list[ThinkingStep],
  current_tool: str = "",
  include_stop_button: bool,
  chat_id: str = "",
) -> list[JsonObject]:
  """Build the shared body for working/stopping/stopped phases."""
  elements: list[JsonObject] = []
  if current_tool:
    elements.append({"tag": "markdown", "content": f"`{current_tool}`"})
  if steps:
    elements.append(_collapsible_thinking(steps))
  if include_stop_button:
    elements.append(_stop_button(chat_id))
  return elements


def build_turn_card(
  phase: str,
  *,
  body: str = "",
  steps: list[ThinkingStep] | None = None,
  current_tool: str = "",
  elapsed: int = 0,
  usage: JsonObject | None = None,
  chat_id: str = "",
  session_id: str = "",
) -> JsonObject:
  """Build a unified turn card for any phase.

  phase: "working" | "stopping" | "stopped" | "done" | "error"
  body:  for done/error — final response or error message
  steps: unified thinking timeline (text + tool entries in order)
  """
  steps = steps or []
  elements: list[JsonObject] = []

  if phase == "working":
    elements = _working_elements(
      steps=steps, current_tool=current_tool,
      include_stop_button=True, chat_id=chat_id,
    )
    title = _elapsed_title(elapsed)
    header: JsonObject | None = {
      "title": {"tag": "plain_text", "content": title},
      "template": "grey",
    }

  elif phase == "stopping":
    elements = _working_elements(
      steps=steps, current_tool=current_tool,
      include_stop_button=False,
    )
    header = {
      "title": {"tag": "plain_text", "content": "Stopping..."},
      "template": "orange",
    }

  elif phase == "stopped":
    elements = _working_elements(
      steps=steps, current_tool=current_tool,
      include_stop_button=False,
    )
    header = {
      "title": {"tag": "plain_text", "content": "Stopped"},
      "template": "grey",
    }

  elif phase == "done":
    # Final response (inline)
    if body:
      elements.append({"tag": "markdown", "content": body})
    # Thinking timeline
    if steps:
      elements.append(_collapsible_thinking(steps))
    # Note: duration + tokens + session
    note_parts = [_elapsed_text(elapsed)]
    if usage:
      ut = _usage_text(usage)
      if ut:
        note_parts.append(ut)
    if session_id:
      note_parts.append(f"session: {session_id[:8]}")
    elements.append(_note_element(" | ".join(note_parts)))
    # Green header
    header = {
      "title": {"tag": "plain_text", "content": "Done ✓"},
      "template": "green",
    }

  elif phase == "error":
    # Error message (inline)
    if body:
      elements.append({"tag": "markdown", "content": body})
    # Thinking timeline
    if steps:
      elements.append(_collapsible_thinking(steps))
    # Note: duration
    note_parts = [_elapsed_text(elapsed)]
    elements.append(_note_element(" | ".join(note_parts)))
    # Red header
    header = {
      "title": {"tag": "plain_text", "content": "Error"},
      "template": "red",
    }

  else:
    raise ValueError(f"Unknown phase: {phase}")

  card: JsonObject = {
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
) -> JsonObject:
  """Build a column_set row of buttons (Card V2 compatible)."""
  columns: list[JsonObject] = []
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
) -> JsonObject:
  """Build a simple Card V2 for non-turn messages."""
  elements: list[JsonObject] = []
  if body and body.strip():
    elements.append({"tag": "markdown", "content": body})
  if buttons:
    elements.append(_buttons_row(buttons, chat_id))
  if note:
    elements.append(_note_element(note))
  card: JsonObject = {
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
                      chat_id: str = "") -> JsonObject:
  """Build a card with select dropdown.

  Each option should have 'text' and 'value' keys.
  """
  select_options = [
    {"text": {"tag": "plain_text", "content": opt["text"]},
     "value": opt["value"]}
    for opt in options
  ]
  elements: list[JsonObject] = [
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
                     chat_id: str = "") -> JsonObject:
  """Build a card with text input."""
  elements: list[JsonObject] = [
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


def build_markdown_card(content: str, title: str = "", color: str = "") -> JsonObject:
  """Build a Card V2 with markdown content."""
  card: JsonObject = {
    "schema": "2.0",
    "config": {"update_multi": True},
    "body": {
      "direction": "vertical",
      "elements": [{"tag": "markdown", "content": content}],
    },
  }
  if title:
    header: JsonObject = {"title": {"tag": "plain_text", "content": title}}
    if color:
      header["template"] = color
    card["header"] = header
  return card
