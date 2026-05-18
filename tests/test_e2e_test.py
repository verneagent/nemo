from __future__ import annotations

from pathlib import Path

from scripts.e2e_test import LogAnalyzer, interactive_card_title, is_done_response


def test_interactive_card_title_extracts_done_title() -> None:
  msg = {
    "type": "interactive",
    "body": '{"title":"Done ✓","elements":[]}',
  }
  assert interactive_card_title(msg) == "Done ✓"


def test_interactive_card_title_extracts_card_v2_header() -> None:
  msg = {
    "type": "interactive",
    "body": '{"schema":"2.0","header":{"title":{"tag":"plain_text","content":"Shell done"}}}',
  }
  assert interactive_card_title(msg) == "Shell done"


def test_is_done_response_distinguishes_working_and_done_cards() -> None:
  working = {
    "type": "interactive",
    "body": '{"title":"Working...","elements":[]}',
  }
  done = {
    "type": "interactive",
    "body": '{"title":"Done ✓","elements":[]}',
  }
  text = {
    "type": "text",
    "body": '{"text":"pong"}',
  }
  assert not is_done_response(working)
  assert is_done_response(done)
  assert is_done_response(text)


def test_log_analyzer_read_since_and_wait_for_since(tmp_path: Path) -> None:
  log_path = tmp_path / "nemo.log"
  log_path.write_text("before\n")
  analyzer = LogAnalyzer(123)
  analyzer.path = str(log_path)
  mark = analyzer.mark()
  with log_path.open("a") as f:
    f.write("Turn response finalized transport=card\n")
  assert "Turn response finalized" in analyzer.read_since(mark)
  assert analyzer.wait_for_since("Turn response finalized", mark, timeout=1, poll=0.01)
