"""Unit tests for the claude-cli (pty-driven TUI) adapter's screen scraping.

The pty/subprocess machinery is exercised live by ``scripts/cli_adapter_smoke.py``
(needs a real account). These tests pin the *pure* scraping logic — the part
that is most fragile to TUI layout changes — against synthetic screen buffers
modeled on the real claude-cli 2.1.175 render captured during development.
"""

from __future__ import annotations

from nemo.claude_cli_agent import (
  _extract_answer,
  _latest_tool_summary,
  _region_after_echo,
)


# A realistic post-turn screen for: prompt → Write tool → "done" answer.
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
  # The answer is the ⏺ block that is NOT a tool call (Write(...) is, and it is
  # followed by a ⎿ result line); "done" is the prose tail.
  assert _extract_answer(_TOOL_TURN, "create a file hello.txt") == "done"


def test_extract_answer_plain_reply() -> None:
  lines = [
    "❯ reply with exactly the word: pong",
    "⏺ pong",
    "✻ Cogitated for 3s",
    "────────",
    "❯",
  ]
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
  assert "First line of the answer" in out
  assert "third" in out


def test_extract_answer_anchors_on_current_prompt() -> None:
  # Two turns in scrollback: extraction must return THIS prompt's answer, not
  # the earlier turn's — this is the desync class the region anchor prevents.
  lines = [
    "❯ first question",
    "⏺ first answer",
    "❯ second question",
    "⏺ second answer",
    "────────",
    "❯",
  ]
  assert _extract_answer(lines, "second question") == "second answer"
  assert _extract_answer(lines, "first question") == "first answer"


def test_extract_answer_tool_only_turn_returns_empty() -> None:
  # A turn that ran a tool but produced no prose tail ⇒ no prose answer.
  lines = [
    "❯ run the build",
    "⏺ Bash(make build)",
    "  ⎿  build succeeded",
    "────────",
    "❯",
  ]
  assert _extract_answer(lines, "run the build") == ""


def test_region_after_echo_slices_current_turn() -> None:
  region = _region_after_echo(_TOOL_TURN, "create a file hello.txt")
  assert region[0] == "⏺ Write(hello.txt)"
  assert "❯ create a file" not in "\n".join(region)


def test_latest_tool_summary_dedupes() -> None:
  seen: set[str] = set()
  region = _region_after_echo(_TOOL_TURN, "create a file hello.txt")
  first = _latest_tool_summary(region, seen)
  assert first == "Write(hello.txt)"
  # Same screen again → already seen → no re-emit.
  assert _latest_tool_summary(region, seen) is None


def test_latest_tool_summary_ignores_prose() -> None:
  seen: set[str] = set()
  lines = ["⏺ pong", "⏺ done"]
  assert _latest_tool_summary(lines, seen) is None
