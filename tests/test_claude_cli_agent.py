"""Unit tests for the claude-cli (pty-driven TUI) adapter's screen scraping.

The pty/subprocess machinery and live turns are exercised by
``scripts/cli_adapter_smoke.py`` and ``scripts/e2e_test.py --workflow`` (both
need a real account). These tests pin the *pure* scraping logic — the part most
sensitive to TUI layout — against synthetic buffers modeled on the real
claude-cli 2.1.175 render captured in development.
"""

from __future__ import annotations

from nemo.claude_cli_agent import (
  _extract_answer,
  _iter_assistant_blocks,
  _new_tool_summaries,
  _region_after_echo,
)


# A realistic post-turn screen: prompt → Write tool → "done" answer.
_TOOL_TURN = [
  "❯ create a file hello.txt in the current directory with the text hi, then reply: done",
  "⏺ Write(hello.txt)",
  "  ⎿  Wrote 1 lines to hello.txt",
  "      1 hi",
  "⏺ done",
  "✻ Crunched for 5s",
  "────────────────────────",
  "❯",
  "────────────────────────",
  "  ⏵⏵ accept edits on (shift+tab to cycle) · ← for agents",
]


def test_extract_answer_prose_after_tool() -> None:
  # Write(...) is a tool (matches Name(...) and is followed by ⎿); "done" is prose.
  assert _extract_answer(_TOOL_TURN, "create a file hello.txt") == "done"


def test_extract_answer_plain_reply() -> None:
  lines = ["❯ reply with exactly the word: pong", "⏺ pong", "✻ Cogitated for 3s", "❯"]
  assert _extract_answer(lines, "reply with exactly the word: pong") == "pong"


def test_extract_answer_multiline_block() -> None:
  lines = [
    "❯ summarize",
    "⏺ First line of the answer",
    "  continues on a second line",
    "  and a third",
    "────────",
    "❯",
  ]
  out = _extract_answer(lines, "summarize")
  assert "First line of the answer" in out and "third" in out


def test_extract_answer_anchors_on_current_prompt() -> None:
  # Two turns in scrollback: extraction must return THIS prompt's answer — the
  # desync class the region anchor prevents.
  lines = [
    "❯ first question", "⏺ first answer",
    "❯ second question", "⏺ second answer", "❯",
  ]
  assert _extract_answer(lines, "second question") == "second answer"
  assert _extract_answer(lines, "first question") == "first answer"


def test_extract_answer_tool_only_turn_is_empty() -> None:
  lines = [
    "❯ run the build", "⏺ Bash(make build)", "  ⎿  build succeeded", "❯",
  ]
  assert _extract_answer(lines, "run the build") == ""


def test_region_after_echo_slices_current_turn() -> None:
  region = _region_after_echo(_TOOL_TURN, "create a file hello.txt")
  assert region[0] == "⏺ Write(hello.txt)"
  assert "❯ create a file" not in "\n".join(region)


def test_iter_assistant_blocks_classifies_tool_vs_prose() -> None:
  region = _region_after_echo(_TOOL_TURN, "create a file hello.txt")
  blocks = _iter_assistant_blocks(region)
  texts = {t: is_tool for t, is_tool in blocks}
  assert texts["Write(hello.txt)"] is True
  assert texts["done"] is False


def test_new_tool_summaries_dedupes_within_turn() -> None:
  seen: set[str] = set()
  first = _new_tool_summaries(_TOOL_TURN, "create a file hello.txt", seen)
  assert first == ["Write(hello.txt)"]
  # Same screen again → already seen → nothing new.
  assert _new_tool_summaries(_TOOL_TURN, "create a file hello.txt", seen) == []


def test_new_tool_summaries_ignores_prose_and_other_turns() -> None:
  lines = [
    "❯ old question", "⏺ Read(old.py)", "  ⎿ ...",
    "❯ new question", "⏺ pong",  # prose, not a tool
    "❯",
  ]
  # Scoped to the NEW turn: the old Read(...) tool must not be surfaced.
  assert _new_tool_summaries(lines, "new question", set()) == []
