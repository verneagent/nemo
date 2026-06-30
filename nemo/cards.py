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

import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import JsonObject


# The inline thinking timeline is trimmed (oldest entries dropped first) to
# this many characters so a very long turn — hundreds of thinking/tool steps,
# which for a done card collapse into a SINGLE uncapped group — cannot blow
# the Lark card size limit. The full, uncapped timeline is attached as a .md
# file by the done-card path when this trim kicks in, so no detail is lost.
# See _update_done_card_with_fallback in agent.py.
TIMELINE_CHAR_BUDGET = 18000

# Lark Card V2 content (the JSON string sent in the `content` field) has a
# hard size limit (~30 KB). A card that exceeds it is accepted by Lark with
# code=0 but renders an EMPTY body — no exception is raised — so callers must
# detect oversize proactively rather than rely on update_card to fail. Leave
# margin under the documented 30 KB ceiling.
LARK_CARD_BYTE_LIMIT = 28000

if TYPE_CHECKING:
  from .channel import AnsweredQuestion, PendingQuestion


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
  """One-line PER-TURN token breakdown shown on the done card.

  Reads the unified schema every adapter normalizes into (turn.canonical_usage)
  so Claude / Codex / OpenCode show the same five figures with the same meaning
  — all per-turn, none session-cumulative. Returns "" only when no usage was
  reported (the card then omits the line).
  """
  if not usage:
    return ""

  def _n(key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool):
      return 0
    if isinstance(value, (int, float)):
      return int(value)
    return 0

  # Compact labels (the grey note line is tight). The four buckets already
  # carry all the signal — i / cr / cw / o are disjoint per Anthropic's
  # contract, so their sum is just arithmetic and not informative on its own.
  # cw is omitted when 0 (Codex never has it; would be pure noise every turn).
  cw = _n("cache_creation_input_tokens")
  parts = [
    f"i {_n('input_tokens'):,}",
    f"cr {_n('cache_read_input_tokens'):,}",
  ]
  if cw:
    parts.append(f"cw {cw:,}")
  parts.append(f"o {_n('output_tokens'):,}")
  return " · ".join(parts)


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


def _timeline_entries(
  groups: list[tuple[str, list[ThinkingStep]]],
) -> list[tuple[str, str]]:
  """Build ordered ("text" | "tool" | "divider", content) timeline entries.

  Each group renders its text, then its thinking steps, then the last
  MAX_TOOLS_PER_GROUP tool calls (with a "+N earlier" indicator if dropped).
  Groups are separated by a horizontal rule.
  """
  entries: list[tuple[str, str]] = []
  for i, (text, grp_steps) in enumerate(groups):
    if i > 0:
      entries.append(("divider", "---"))

    if text:
      t = _sanitize_markdown(text)
      if len(t) > 300:
        t = t[:297] + "..."
      entries.append(("text", _escape_md(t)))

    # Separate thinking/reasoning/compact and tool steps while preserving
    # order. Compact-boundary steps render in the same "text" lane as
    # thinking — they're brief informational lines, not tool calls.
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
  return entries


def _trim_entries_to_budget(
  entries: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], int]:
  """Drop oldest entries until under TIMELINE_CHAR_BUDGET.

  Returns (kept_entries, omitted_count). A done card's whole timeline is a
  single uncapped group, so the per-group tool cap alone can't bound it —
  this byte budget is the backstop that guarantees the panel fits.
  """
  def _sz(e: tuple[str, str]) -> int:
    return len(e[1]) + 8  # + small per-element JSON overhead estimate

  total = sum(_sz(e) for e in entries)
  omitted = 0
  while entries and total > TIMELINE_CHAR_BUDGET:
    e = entries.pop(0)
    total -= _sz(e)
    if e[0] != "divider":
      omitted += 1
  # Trimming can leave a now-leading divider — drop it.
  while entries and entries[0][0] == "divider":
    entries.pop(0)
  return entries, omitted


def timeline_overflows(steps: list[ThinkingStep]) -> bool:
  """True if the timeline would be trimmed to fit (i.e. detail was dropped).

  The done-card path uses this to decide whether to attach the full,
  uncapped timeline as a .md file.
  """
  entries = _timeline_entries(_split_into_groups(steps))
  _, omitted = _trim_entries_to_budget(entries)
  return omitted > 0


def _collapsible_thinking(steps: list[ThinkingStep]) -> JsonObject:
  """Build a collapsible_panel with grouped narrative text + tool lines.

  Steps are split into groups at each text step. Each group shows:
  - The group's text (default font)
  - Any thinking steps (default font)
  - The last MAX_TOOLS_PER_GROUP tool calls (small font, grey type)
  - A "+N earlier" indicator if tools were dropped

  The whole timeline is then trimmed to TIMELINE_CHAR_BUDGET (oldest entries
  first) so it can never exceed Lark's card size limit. Header shows the
  number of groups, not the total step count. Groups are separated by an hr.
  """
  groups = _split_into_groups(steps)
  total_groups = len(groups)
  entries = _timeline_entries(groups)

  entries, omitted = _trim_entries_to_budget(entries)
  if omitted:
    entries.insert(0, (
      "text",
      f"<font color='grey'>… {omitted} earlier timeline "
      f"entr{'ies' if omitted != 1 else 'y'} omitted</font>",
    ))

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
        "content": f"Thinking ({total_groups})",
      },
    },
    "vertical_spacing": "4px",
    "elements": elements,
  }


def card_content_bytes(card: JsonObject) -> int:
  """Serialized size of a card's `content`, as Lark measures it (utf-8)."""
  return len(json.dumps(card, ensure_ascii=False).encode("utf-8"))


def is_card_oversized(card: JsonObject) -> bool:
  """True if the card exceeds Lark's content size limit.

  Lark accepts an oversized card with code=0 but renders an empty body, so
  callers must check this before PATCHing instead of relying on an exception.
  """
  return card_content_bytes(card) > LARK_CARD_BYTE_LIMIT


def render_timeline_markdown(steps: list[ThinkingStep]) -> str:
  """Render the FULL (uncapped) thinking timeline as plain markdown.

  Used to attach a long turn's complete timeline as a .md file when the
  inline card panel is capped to MAX_THINKING_GROUPS.
  """
  groups = _split_into_groups(steps)
  out: list[str] = []
  for i, (text, grp_steps) in enumerate(groups):
    out.append(f"## Step group {i + 1}/{len(groups)}")
    if text:
      out.append(text)
    for s in grp_steps:
      if s.kind in ("thinking", "reasoning"):
        out.append(f"> {s.content}")
      elif s.kind == "tool":
        out.append(f"- {s.content}")
    out.append("")
  return "\n".join(out).rstrip() + "\n"


def _note_element(text: str) -> JsonObject:
  """Small footer text. Uses markdown instead of deprecated 'note' tag (V2)."""
  return {
    "tag": "markdown",
    "content": f"<font color='grey'>{text}</font>",
    "text_size": "notation",
  }


def _stop_button(chat_id: str = "", action: str = "__stop__") -> JsonObject:
  """Build a Stop button inside a column_set (Card V2 compatible).

  ``action`` is the value the daemon routes on: ``__stop__`` for a main turn,
  ``fork_stop:<thread_id>`` for a /fork sub-thread so the click interrupts that
  fork's turn rather than the main one."""
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
        "value": {"action": action, "chat_id": chat_id},
      }],
    }],
  }


def _shell_abort_button(job_id: str, chat_id: str = "") -> JsonObject:
  """Build an Abort button for an operator-started shell job."""
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
        "text": {"tag": "plain_text", "content": "Abort"},
        "type": "danger",
        "value": {
          "action": "shell_abort",
          "job_id": job_id,
          "chat_id": chat_id,
        },
      }],
    }],
  }


def _format_answer_text(answer: object) -> str:
  """Format an AskUserQuestion answer for inline display."""
  if isinstance(answer, list):
    if not answer:
      return "_(none)_"
    return ", ".join(_escape_md(str(a)) for a in answer)
  text = str(answer or "")
  if not text:
    return "_(no answer)_"
  return _escape_md(text)


def _answered_questions_element(
  answered: list["AnsweredQuestion"],
) -> JsonObject:
  """Render previously-answered AskUserQuestion entries as a single
  markdown block: one ``❓ <header> → ✅ <answer>`` line per entry."""
  lines: list[str] = []
  for aq in answered:
    header = _escape_md(aq.header or aq.question or "Question")
    lines.append(f"❓ **{header}** → ✅ {_format_answer_text(aq.answer)}")
  return {"tag": "markdown", "content": "\n".join(lines)}


def _pending_question_elements(
  pending: "PendingQuestion",
  chat_id: str,
) -> list[JsonObject]:
  """Render an in-flight AskUserQuestion as inline card elements: each
  question's header + body, a row of option buttons, an "Other" fallback,
  and a Submit button for multi-select. Selected options get a leading
  ``✓`` and a primary button colour so the user can see their picks
  before the loop completes.

  Action strings match ``askq:{nonce}:{qidx}:{oidx}`` (or ``:other`` /
  ``:done``) so ``nemo.permissions._parse_askq_action`` parses them.
  """
  elements: list[JsonObject] = []
  questions = pending.questions or []
  answers = pending.answers or {}
  nonce = pending.nonce

  for qidx, question in enumerate(questions):
    if not isinstance(question, dict):
      continue
    if qidx > 0:
      elements.append({"tag": "hr"})

    header = str(question.get("header") or question.get("question") or "Question")
    q_text = str(question.get("question", ""))
    multi_select = bool(question.get("multiSelect", False))
    options = question.get("options", []) or []

    elements.append({
      "tag": "markdown",
      "content": f"❓ **{_escape_md(header)}**",
    })
    if q_text and q_text != header:
      elements.append({"tag": "markdown", "content": _escape_md(q_text)})

    selected = answers.get(qidx)
    selected_set: set[str] = set()
    if isinstance(selected, list):
      selected_set = {str(x) for x in selected}
    elif isinstance(selected, str) and selected:
      selected_set = {selected}

    # Collect option labels so we know which selected values are
    # free-text "Other" answers (typed by the user, not in the option
    # list) and need an explicit visual confirmation line — otherwise
    # the click + typed-answer flow has no visible feedback on the card
    # and looks like the bot silently ignored the reply.
    option_labels: set[str] = set()
    button_rows: list[tuple[str, str, str]] = []
    for oidx, opt in enumerate(options):
      if not isinstance(opt, dict):
        continue
      label = str(opt.get("label") or opt.get("description") or f"Option {oidx + 1}")
      option_labels.add(label)
      check = "✓ " if label in selected_set else ""
      btn_type = "primary" if label in selected_set else "default"
      button_rows.append((
        f"{check}{label}",
        f"askq:{nonce}:{qidx}:{oidx}",
        btn_type,
      ))
    button_rows.append((
      "Other (type below)",
      f"askq:{nonce}:{qidx}:other",
      "default",
    ))

    for start in range(0, len(button_rows), 2):
      elements.append(_buttons_row(button_rows[start:start + 2], chat_id))

    # Render any typed "Other" answer that is not in the option list so
    # the user gets immediate confirmation that their reply was received.
    free_text_answers = sorted(selected_set - option_labels)
    if free_text_answers:
      shown = ", ".join(_escape_md(a) for a in free_text_answers)
      elements.append({
        "tag": "markdown",
        "content": f"<font color='green'>✓ Your answer: {shown}</font>",
      })

    if multi_select:
      done_label = "Submit ✓" if selected_set else "Submit"
      elements.append(_buttons_row(
        [(done_label, f"askq:{nonce}:{qidx}:done", "primary")],
        chat_id,
      ))

  return elements


def _working_elements(
  *,
  steps: list[ThinkingStep],
  current_tool: str = "",
  include_stop_button: bool,
  chat_id: str = "",
  stop_action: str = "__stop__",
  status_notice: str = "",
  rate_limit_notice: str = "",
  compact_notice: str = "",
  answered_questions: list["AnsweredQuestion"] | None = None,
  pending_question: "PendingQuestion | None" = None,
) -> list[JsonObject]:
  """Build the shared body for working/stopping/stopped phases."""
  elements: list[JsonObject] = []
  if status_notice:
    # Live "what's happening right now" hint, shown above the timeline so
    # an otherwise-silent stretch (e.g. a recall turn reading a transcript
    # before first-token) doesn't look like the daemon has stalled. Blue
    # reads as in-progress info, distinct from rate-limit's orange warning
    # and compact's grey. Callers clear it once real progress streams.
    elements.append({
      "tag": "markdown",
      "content": f"<font color='blue'>{status_notice}</font>",
    })
  if rate_limit_notice:
    elements.append({
      "tag": "markdown",
      "content": f"<font color='orange'>{rate_limit_notice}</font>",
    })
  if compact_notice:
    # Same banner slot as rate_limit, different colour so they're
    # distinguishable when both fire in the same turn. Grey reads as
    # "informational status" vs rate-limit's orange "watch out".
    elements.append({
      "tag": "markdown",
      "content": f"<font color='grey'>{compact_notice}</font>",
    })
  # Answered questions sit just below status banners and above the
  # current-tool / thinking sections so the user keeps seeing what they
  # picked for the rest of the turn.
  if answered_questions:
    elements.append(_answered_questions_element(answered_questions))
  if pending_question is not None and pending_question.questions:
    elements.extend(_pending_question_elements(pending_question, chat_id))
  if current_tool:
    elements.append({"tag": "markdown", "content": f"`{current_tool}`"})
  if steps:
    elements.append(_collapsible_thinking(steps))
  if include_stop_button:
    elements.append(_stop_button(chat_id, stop_action))
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
  stop_action: str = "__stop__",
  status_notice: str = "",
  rate_limit_notice: str = "",
  compact_notice: str = "",
  answered_questions: list["AnsweredQuestion"] | None = None,
  pending_question: "PendingQuestion | None" = None,
  part_label: str = "",
) -> JsonObject:
  """Build a unified turn card for any phase.

  phase: "working" | "continued" | "stopping" | "stopped" | "done" | "error"
  body:  for done/error — final response or error message
  steps: unified thinking timeline (text + tool entries in order)
  status_notice: short blue banner shown above the working state as a
    live "what's happening now" hint (only rendered in working/stopping/
    stopped). Used to explain a silent stretch before first-token — e.g. a
    recall turn reading a past transcript. Callers clear it once real
    progress streams so it doesn't linger as stale.
  rate_limit_notice: short banner shown above the working state to flag
    upstream rate-limit pressure (only rendered in working/stopping/stopped).
  compact_notice: short banner explaining a context-compaction pause —
    rendered in the same banner slot as rate_limit_notice. Lives outside
    the collapsible thinking panel so a 10–60s silent compaction doesn't
    look like the daemon has stalled.
  answered_questions: previously-answered AskUserQuestion entries, shown
    as a compact ``❓ <header> → ✅ <answer>`` block near the top of the
    card in every phase. Persists through working/done so the user can
    see what they picked for the rest of the turn.
  pending_question: an in-flight AskUserQuestion, rendered inline with
    its option buttons in the working phase only. Stopping/stopped/done/
    error drop it (the loop has ended).
  part_label: optional title suffix for multi-card turn segments.
  """
  steps = steps or []
  answered_questions = answered_questions or []
  elements: list[JsonObject] = []

  if phase == "working":
    elements = _working_elements(
      steps=steps, current_tool=current_tool,
      include_stop_button=True, chat_id=chat_id,
      stop_action=stop_action,
      status_notice=status_notice,
      rate_limit_notice=rate_limit_notice,
      compact_notice=compact_notice,
      answered_questions=answered_questions,
      pending_question=pending_question,
    )
    title = _elapsed_title(elapsed)
    header: JsonObject | None = {
      "title": {"tag": "plain_text", "content": title},
      "template": "grey",
    }

  elif phase == "continued":
    elements = _working_elements(
      steps=steps,
      current_tool=current_tool,
      include_stop_button=False,
      status_notice=status_notice,
      rate_limit_notice=rate_limit_notice,
      compact_notice=compact_notice,
      answered_questions=answered_questions,
      pending_question=None,
    )
    if not elements:
      elements.append(_note_element("continued in the next card"))
    title = "Earlier progress"
    if part_label:
      title = f"{title} · {part_label}"
    header = {
      "title": {"tag": "plain_text", "content": title},
      "template": "grey",
    }

  elif phase == "stopping":
    elements = _working_elements(
      steps=steps, current_tool=current_tool,
      include_stop_button=False,
      status_notice=status_notice,
      rate_limit_notice=rate_limit_notice,
      compact_notice=compact_notice,
      answered_questions=answered_questions,
      pending_question=None,
    )
    header = {
      "title": {"tag": "plain_text", "content": "Stopping..."},
      "template": "orange",
    }

  elif phase == "stopped":
    elements = _working_elements(
      steps=steps, current_tool=current_tool,
      include_stop_button=False,
      status_notice=status_notice,
      rate_limit_notice=rate_limit_notice,
      compact_notice=compact_notice,
      answered_questions=answered_questions,
      pending_question=None,
    )
    header = {
      "title": {"tag": "plain_text", "content": "Stopped"},
      "template": "grey",
    }

  elif phase == "done":
    if compact_notice:
      elements.append({
        "tag": "markdown",
        "content": f"<font color='grey'>{compact_notice}</font>",
      })
    # Answered questions sit above the model's final response so the
    # decisions the user made stay visible in the terminal card.
    if answered_questions:
      elements.append(_answered_questions_element(answered_questions))
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
    # Answered questions sit above the error message — they explain what
    # the user picked before the failure.
    if answered_questions:
      elements.append(_answered_questions_element(answered_questions))
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


def _shell_status_title(phase: str) -> tuple[str, str]:
  if phase == "running":
    return "Shell running", "grey"
  if phase == "done":
    return "Shell done", "green"
  if phase == "failed":
    return "Shell failed", "red"
  if phase == "timeout":
    return "Shell timed out", "orange"
  if phase == "aborted":
    return "Shell aborted", "grey"
  return "Shell error", "red"


def _shell_tail(stdout: str, stderr: str, limit: int = 12_000) -> str:
  parts: list[str] = []
  if stdout:
    parts.append("stdout:\n" + stdout)
  if stderr:
    parts.append("stderr:\n" + stderr)
  text = "\n\n".join(parts)
  if not text:
    return ""
  if len(text) > limit:
    text = f"... {len(text) - limit} chars omitted ...\n{text[-limit:]}"
  return text


def build_shell_card(
  phase: str,
  *,
  job_id: str,
  command: str,
  cwd: str,
  elapsed: int,
  inject_context: bool,
  chat_id: str = "",
  exit_code: int | None = None,
  stdout: str = "",
  stderr: str = "",
) -> JsonObject:
  """Build a Card V2 shell job card with optional Abort action."""
  title, color = _shell_status_title(phase)
  elements: list[JsonObject] = []
  mode = "injects next turn" if inject_context else "no context injection"
  elements.append({
    "tag": "markdown",
    "content": (
      f"`$ {command}`\n\n"
      f"<font color='grey'>cwd: `{cwd}` · job: `{job_id}` · {mode}</font>"
    ),
  })
  status_parts = [_elapsed_text(elapsed)]
  if exit_code is not None:
    status_parts.append(f"exit {exit_code}")
  if phase == "timeout":
    status_parts.append("interactive commands are not supported")
  elements.append(_note_element(" | ".join(status_parts)))
  tail = _shell_tail(stdout, stderr)
  if tail:
    tail = tail.replace("```", "` ` `")
    elements.append({
      "tag": "markdown",
      "content": "```\n" + _escape_md(tail) + "\n```",
    })
  elif phase == "running":
    elements.append({
      "tag": "markdown",
      "content": "<font color='grey'>Waiting for output...</font>",
    })
  if phase == "running":
    elements.append(_shell_abort_button(job_id, chat_id))
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": title},
      "template": color,
    },
    "body": {"direction": "vertical", "elements": elements},
  }


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


def build_model_picker_card(
  options: list[tuple[str, str]],
  *,
  current_model: str,
  current_agent: str,
  chat_id: str = "",
  info: str = "",
  hint: str = "",
) -> JsonObject:
  """Build the interactive `/model` picker card.

  Renders a dropdown of available models inside a Lark V2 ``form`` so the
  selection only fires when the user clicks the Submit button — matching
  the user's "select model 然后 submit" UX request. Each option's value
  carries the ``model_switch:<name>`` discriminator so the daemon can
  route the resulting card.action.trigger event back to the model-switch
  flow without collisions with other actions.

  ``options`` is a list of ``(display_label, model_name)`` tuples in the
  order they should appear in the dropdown. ``display_label`` is what the
  user sees; ``model_name`` is the canonical slug passed to ``/model``.
  Empty option list still renders the card (the user is told there are
  no models to pick from) — same defensive shape as ``build_form_select``.

  ``info`` is the multi-line catalog listing (Available / API-only /
  aliases). It is rendered as a PLAIN markdown element. ``hint`` is a
  one-line usage tip rendered as a small grey footer note. They are
  kept separate on purpose: ``_note_element`` wraps its text in a
  single ``<font color='grey'>…</font>`` span, and a ``<font>`` tag
  cannot span a markdown paragraph break (``\\n\\n``). Stuffing the
  multi-line ``info`` into the note would split the span across blocks
  and leak a bare ``</font>`` into the rendered card (the bug the user
  hit). The footer ``hint`` must therefore stay single-line and free
  of raw ``<...>`` tags.
  """
  select_options = [
    {
      "text": {"tag": "plain_text", "content": label},
      "value": f"model_switch:{model_name}",
    }
    for label, model_name in options
  ]
  summary = (
    f"Current model: **{current_model}** "
    f"(agent **{current_agent}**)"
  )
  form_elements: list[JsonObject] = [
    {
      "tag": "select_static",
      "name": "model",
      "placeholder": {"tag": "plain_text", "content": "Pick a model..."},
      "options": select_options,
    },
    # Submit button — exact Card JSON 2.0 form-submit shape from the official
    # docs (feishu-cards/card-json-v2-components → button + form container):
    #   - ``form_action_type: "submit"`` is the field that submits the form
    #     (NOT ``action_type``);
    #   - the submit button is a DIRECT child of the form (nesting it in a
    #     column_set means Lark doesn't treat it as the form's submit
    #     trigger — clicking fires nothing → 200530 "callback not received");
    #   - the submit button MUST have a ``name``. The official callback
    #     payload puts the button's name in ``action.name`` and ONLY the data
    #     components (the select) in ``action.form_value``, so a named submit
    #     does NOT pollute form_value — the select still arrives as the single
    #     ``{"model": "model_switch:<name>"}`` entry the relay routes on.
    {
      "tag": "button",
      "text": {"tag": "plain_text", "content": "Submit"},
      "type": "primary",
      "form_action_type": "submit",
      "name": "submit",
      "value": {"action": "model_picker_submit", "chat_id": chat_id},
    },
  ]
  elements: list[JsonObject] = [
    {"tag": "markdown", "content": summary},
    {
      "tag": "form",
      "name": "model_picker_form",
      "elements": form_elements,
    },
  ]
  # Catalog listing as plain markdown (it is multi-line, so it must NOT
  # go through the <font>-wrapped note — see the docstring).
  if info:
    elements.append({"tag": "markdown", "content": info})
  # One-line footer hint — safe to grey-wrap.
  if hint:
    elements.append(_note_element(hint))
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": "Switch Model"},
      "template": "blue",
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_model_switched_card(
  *,
  agent: str,
  model: str,
  ok: bool = True,
  attempted: str = "",
  reason: str = "",
  info: str = "",
) -> JsonObject:
  """Build the locked, post-submit replacement for the /model picker.

  Once a model is picked and submitted the picker rewrites itself into
  this static card — no dropdown, no Submit button — so it can't be
  re-submitted with a now-stale model list (e.g. after the user has
  since switched agent). To change model again the user runs ``/model``
  for a fresh picker.

  ``ok=True``: the switch landed; prominently shows the current agent +
  model. ``ok=False``: the submitted ``attempted`` model was rejected
  (typically because the agent changed under a stale picker); shows the
  unchanged current agent + model plus ``reason``.
  """
  if ok:
    header_title = "✅ Model Switched"
    template = "green"
    lines = [
      f"**Agent:** {_escape_md(agent)}",
      f"**Model:** {_escape_md(model)}",
    ]
  else:
    header_title = "⚠️ Model Not Switched"
    template = "orange"
    lines = []
    if attempted:
      lines.append(f"Couldn't switch to **{_escape_md(attempted)}**.")
    if reason:
      lines.append(_escape_md(reason))
    lines.append(f"**Current agent:** {_escape_md(agent)}")
    lines.append(f"**Current model:** {_escape_md(model)}")
  elements: list[JsonObject] = [
    {"tag": "markdown", "content": "\n".join(lines)},
  ]
  # Keep the available-model catalog visible after the switch so the locked
  # card stays useful (the user can see what else they could switch to)
  # instead of being wiped down to just agent+model. Multi-line, so it goes
  # through plain markdown — NOT the <font>-wrapped note, which can't span a
  # \n\n paragraph break without leaking a bare </font>.
  if info:
    elements.append({"tag": "markdown", "content": info})
  elements.append(_note_element("Run `/model` for a fresh picker."))
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": header_title},
      "template": template,
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_agent_picker_card(
  options: list[tuple[str, str]],
  *,
  current_agent: str,
  current_model: str,
  chat_id: str = "",
  info: str = "",
  hint: str = "",
) -> JsonObject:
  """Build the interactive `/agent` picker card.

  Mirrors ``build_model_picker_card`` exactly: a Lark V2 ``form`` with a
  ``select_static`` dropdown of the three CodingAgent kinds plus a
  ``form_action_type: "submit"`` button. Option values carry the
  ``agent_switch:<name>`` discriminator so the daemon's card.action handler
  can route the form-submit back to the agent-switch flow without colliding
  with ``model_switch:`` (or any other) prefix.

  Same V2 / ``<font>`` constraints as the model picker — ``info`` goes
  through plain markdown (multi-line allowed), ``hint`` through the grey
  note element (single line, no raw HTML).
  """
  select_options = [
    {
      "text": {"tag": "plain_text", "content": label},
      "value": f"agent_switch:{agent_name}",
    }
    for label, agent_name in options
  ]
  summary = (
    f"Current agent: **{current_agent}** "
    f"(model **{current_model}**)"
  )
  form_elements: list[JsonObject] = [
    {
      "tag": "select_static",
      "name": "agent",
      "placeholder": {"tag": "plain_text", "content": "Pick an agent..."},
      "options": select_options,
    },
    # Same exact submit-button shape as the model picker — see that card's
    # docstring for why ``form_action_type``, the button ``name`` and the
    # direct-child placement all matter for Lark to fire the callback.
    {
      "tag": "button",
      "text": {"tag": "plain_text", "content": "Submit"},
      "type": "primary",
      "form_action_type": "submit",
      "name": "submit",
      "value": {"action": "agent_picker_submit", "chat_id": chat_id},
    },
  ]
  elements: list[JsonObject] = [
    {"tag": "markdown", "content": summary},
    {
      "tag": "form",
      "name": "agent_picker_form",
      "elements": form_elements,
    },
  ]
  if info:
    elements.append({"tag": "markdown", "content": info})
  if hint:
    elements.append(_note_element(hint))
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": "Switch Agent"},
      "template": "blue",
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_agent_switched_card(
  *,
  agent: str,
  model: str,
  ok: bool = True,
  attempted: str = "",
  reason: str = "",
  info: str = "",
) -> JsonObject:
  """Build the locked, post-submit replacement for the /agent picker.

  After the user submits the picker we PATCH the card into this static
  form — no dropdown, no Submit button — so the same picker can't be
  re-submitted with a now-stale selection. ``ok=False`` is used for
  unknown-agent rejections (the only invalid case for /agent, since the
  three valid kinds are fixed by ``agent_factory``).
  """
  if ok:
    header_title = "✅ Agent Switched"
    template = "green"
    lines = [
      f"**Agent:** {_escape_md(agent)}",
      f"**Model:** {_escape_md(model)}",
    ]
  else:
    header_title = "⚠️ Agent Not Switched"
    template = "orange"
    lines = []
    if attempted:
      lines.append(f"Couldn't switch to **{_escape_md(attempted)}**.")
    if reason:
      lines.append(_escape_md(reason))
    lines.append(f"**Current agent:** {_escape_md(agent)}")
    lines.append(f"**Current model:** {_escape_md(model)}")
  elements: list[JsonObject] = [
    {"tag": "markdown", "content": "\n".join(lines)},
  ]
  if info:
    elements.append({"tag": "markdown", "content": info})
  elements.append(_note_element("Run `/agent` for a fresh picker."))
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": header_title},
      "template": template,
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_session_picker_card(
  options: list[tuple[str, str]],
  *,
  chat_id: str = "",
  info: str = "",
  hint: str = "",
) -> JsonObject:
  """Build the interactive `/session recall` picker card.

  Feishu Card V2 has no radio-group component, and a ``select_static``
  dropdown collapses each session to a single cramped line. So instead of
  a form, each session is rendered as its own block — a multi-line
  markdown description followed by a dedicated **Recall** button — giving
  the user the full per-session detail (uuid · agent · model · age +
  prompt preview) before picking one. Clicking a button fires a plain
  ``card.action.trigger`` whose ``value.action`` is
  ``session_recall:<uuid>`` — the same discriminator the daemon already
  routes on (relay button_action → action_value['action']), so no separate
  submit step is needed. ``session_recall:`` is in the relay's
  BOT_OWNED_CARD_PREFIXES, so the click is toast-only (no "Selected:"
  flash) while the daemon PATCHes this card to its locked state.

  ``options`` is ``(description_markdown, uuid)`` pairs, newest first;
  ``description_markdown`` is the multi-line block shown above each
  button. Same V2 / ``<font>`` constraints as the other cards — ``info``
  is plain markdown (multi-line allowed); ``hint`` is the grey one-line note.
  """
  elements: list[JsonObject] = [
    {"tag": "markdown", "content": "Pick a past session to recall:"},
  ]
  for idx, (description, uuid) in enumerate(options):
    if idx > 0:
      elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": description})
    # One Recall button per session — a plain (non-form) button so the
    # click itself recalls; value.action carries the routing discriminator.
    elements.append(_buttons_row(
      [("Recall", f"session_recall:{uuid}", "primary")], chat_id))
  if info:
    elements.append({"tag": "markdown", "content": info})
  if hint:
    elements.append(_note_element(hint))
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": "Recall Session"},
      "template": "blue",
    },
    "body": {"direction": "vertical", "elements": elements},
  }


def build_session_recalled_card(
  *,
  uuid: str,
  agent: str = "",
  model: str = "",
) -> JsonObject:
  """Build the locked, post-submit replacement for the /session recall picker.

  After the user submits we PATCH the picker into this static card — no
  dropdown, no Submit button — so the same pick can't be re-submitted.
  Recall itself is read-only (the actual summary arrives as a separate
  turn), so this is purely a "you picked X, recalling now" confirmation.
  """
  lines = [f"📖 Recalling session **{_escape_md(uuid[:8])}**…"]
  meta_bits = []
  if agent:
    meta_bits.append(f"**Agent:** {_escape_md(agent)}")
  if model:
    meta_bits.append(f"**Model:** {_escape_md(model)}")
  if meta_bits:
    lines.append(" · ".join(meta_bits))
  elements: list[JsonObject] = [
    {"tag": "markdown", "content": "\n".join(lines)},
    _note_element("Run `/session recall` for a fresh picker."),
  ]
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": "Recall Session"},
      "template": "green",
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
  """Standalone AskUserQuestion card — used only as a fallback when the
  working turn card hasn't been created yet (or its creation failed).

  In the normal path the question is embedded directly into the working
  turn card via ``build_turn_card(..., pending_question=...)`` and there
  is no separate card. This builder exists so the askq handler can still
  deliver a usable UI when there is no turn card to embed into.

  Action strings: ``askq:{nonce}:{qidx}:{oidx}`` for option clicks,
  ``askq:{nonce}:{qidx}:other`` for the "Other" fallback, and
  ``askq:{nonce}:{qidx}:done`` for multi-select Submit.
  """
  from .channel import PendingQuestion

  pending = PendingQuestion(
    questions=[q for q in questions if isinstance(q, dict)],
    answers=dict(answers or {}),
    nonce=nonce,
  )
  elements = _pending_question_elements(pending, chat_id)
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
