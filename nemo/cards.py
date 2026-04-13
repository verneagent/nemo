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
  """One entry in the unified thinking timeline."""
  kind: str       # "thinking" | "tool" | "reasoning" | "answer"
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


MAX_TOOLS_PER_GROUP = 5


def _split_into_groups(steps: list[ThinkingStep]) -> list[tuple[str, list[ThinkingStep]]]:
  """Split steps into groups. Each group = (text or '', [tool/thinking steps]).

  A new group starts at each answer step. Tool/thinking/reasoning steps
  attach to the current group. If the first step isn't an answer, the
  first group has empty text.
  """
  groups: list[tuple[str, list[ThinkingStep]]] = []
  cur_text: str = ""
  cur_steps: list[ThinkingStep] = []
  started = False

  for s in steps:
    if s.kind == "answer":
      if started:
        groups.append((cur_text, cur_steps))
      cur_text = s.content
      cur_steps = []
      started = True
    else:
      if not started:
        started = True  # first group with no leading answer
      cur_steps.append(s)
  if started:
    groups.append((cur_text, cur_steps))
  return groups


def _render_tool_lines(tool_steps: list[ThinkingStep]) -> list[str]:
  """Coalesce consecutive same-type tool steps into rendered lines."""
  lines: list[str] = []
  pending_type: str = ""
  pending_details: list[str] = []

  def _flush() -> None:
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

  for s in tool_steps:
    ttype, detail = _tool_type_and_detail(s.content)
    if ttype == pending_type:
      pending_details.append(detail)
    else:
      _flush()
      pending_type = ttype
      pending_details = [detail]
  _flush()
  return lines


def _collapsible_thinking(steps: list[ThinkingStep]) -> JsonObject:
  """Build a collapsible_panel with grouped narrative text + tool lines.

  Steps are split into groups at each text step. Each group shows:
  - The group's text (default font)
  - Any thinking steps (default font)
  - The last MAX_TOOLS_PER_GROUP tool calls (small font, grey type)
  - A "+N earlier" indicator if tools were dropped

  Header shows the number of groups, not the total step count.
  Groups are separated by a horizontal rule.
  """
  groups = _split_into_groups(steps)

  # Each entry: ("text" | "tool" | "divider", content)
  entries: list[tuple[str, str]] = []
  for i, (text, grp_steps) in enumerate(groups):
    if i > 0:
      entries.append(("divider", "---"))

    if text:
      t = _sanitize_markdown(text)
      if len(t) > 300:
        t = t[:297] + "..."
      entries.append(("text", _escape_md(t)))

    # Separate thinking/reasoning and tool steps while preserving order
    thinkings = [s for s in grp_steps if s.kind in ("thinking", "reasoning")]
    tools = [s for s in grp_steps if s.kind == "tool"]

    # Render thinking blocks (all of them — not counted toward tool limit)
    for th in thinkings:
      tx = _sanitize_markdown(th.content)
      if len(tx) > 200:
        tx = tx[:197] + "..."
      entries.append(("text", _escape_md(tx)))

    # Apply 5-tool limit: keep the last MAX_TOOLS_PER_GROUP
    dropped = max(0, len(tools) - MAX_TOOLS_PER_GROUP)
    kept_tools = tools[dropped:]
    if dropped > 0:
      entries.append((
        "tool",
        f"<font color='grey'>… +{dropped} earlier tool call"
        f"{'s' if dropped > 1 else ''}</font>",
      ))
    for line in _render_tool_lines(kept_tools):
      entries.append(("tool", line))

  # Coalesce consecutive entries of the same kind into markdown blocks.
  elements: list[JsonObject] = []
  buf: list[str] = []
  buf_kind: str = ""

  def _emit() -> None:
    if not buf:
      return
    content = "  \n".join(buf)
    el: JsonObject = {"tag": "markdown", "content": content}
    if buf_kind == "tool":
      el["text_size"] = "notation"
    elements.append(el)

  for kind, content in entries:
    if kind == "divider":
      _emit()
      buf = []
      buf_kind = ""
      elements.append({"tag": "hr"})
      continue
    if kind != buf_kind:
      _emit()
      buf = []
      buf_kind = kind
    buf.append(content)
  _emit()

  if not elements:
    elements = [{"tag": "markdown", "content": "_none_"}]

  return {
    "tag": "collapsible_panel",
    "expanded": False,
    "header": {
      "title": {
        "tag": "plain_text",
        "content": f"Thinking ({len(groups)})",
      },
    },
    "vertical_spacing": "4px",
    "elements": elements,
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


def build_ask_user_question_card(
  questions: list[JsonObject],
  chat_id: str,
  nonce: str,
  answers: dict[int, object] | None = None,
) -> JsonObject:
  """Build a card that renders an AskUserQuestion tool call.

  Each question is a section with header text and one button per option.
  Action strings are `askq:{nonce}:{qidx}:{oidx}` for option clicks and
  `askq:{nonce}:done` for the "Submit" button (multi-select only).

  `answers` is the in-progress selection: maps question index → selected
  label (str) for single-select, or list[str] for multi-select. When
  rendering an "answered" view, render selected options with a check mark.
  """
  answers = answers or {}
  elements: list[JsonObject] = []

  for qidx, question in enumerate(questions):
    if qidx > 0:
      elements.append({"tag": "hr"})

    header = str(question.get("header") or question.get("question") or "Question")
    q_text = str(question.get("question", ""))
    multi_select = bool(question.get("multiSelect", False))
    options = question.get("options", []) or []

    elements.append({
      "tag": "markdown",
      "content": f"**{_escape_md(header)}**",
    })
    if q_text and q_text != header:
      elements.append({"tag": "markdown", "content": _escape_md(q_text)})

    selected = answers.get(qidx)
    selected_set: set[str] = set()
    if isinstance(selected, list):
      selected_set = {str(x) for x in selected}
    elif isinstance(selected, str):
      selected_set = {selected}

    # Render each option as a button. Lark cards lack visual radio
    # buttons; primary color marks selected ones.
    button_rows: list[tuple[str, str, str]] = []
    for oidx, opt in enumerate(options):
      label = str(opt.get("label") or opt.get("description") or f"Option {oidx + 1}")
      check = "✓ " if label in selected_set else ""
      btn_type = "primary" if label in selected_set else "default"
      button_rows.append((
        f"{check}{label}",
        f"askq:{nonce}:{qidx}:{oidx}",
        btn_type,
      ))
    # Always offer an "Other" → text input fallback
    button_rows.append((
      "Other (type below)",
      f"askq:{nonce}:{qidx}:other",
      "default",
    ))

    # Pack options into rows of 2 to keep mobile width manageable
    for start in range(0, len(button_rows), 2):
      elements.append(_buttons_row(button_rows[start:start + 2], chat_id))

    if multi_select:
      done_label = "Submit ✓" if selected_set else "Submit"
      elements.append(_buttons_row(
        [(done_label, f"askq:{nonce}:{qidx}:done", "primary")],
        chat_id,
      ))

  elements.append(_note_element(
    "Tap a button, or type your answer in chat. 10-min timeout."
  ))

  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": "Question from Claude"},
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
