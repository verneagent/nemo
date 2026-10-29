from __future__ import annotations

import ast
import sys
from pathlib import Path

from scripts import e2e_test as e2e
from scripts.e2e_test import (
  LogAnalyzer,
  interactive_card_title,
  is_done_response,
  spawn_logged,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
E2E_PATH = REPO_ROOT / "scripts" / "e2e_test.py"


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


MARKER_CARD = {
  "type": "interactive",
  "body": '{"title":"👀 后台监控已挂上","elements":[]}',
  "time": "2000",
  "message_id": "om_marker",
}
DONE_CARD = {
  "type": "interactive",
  "body": '{"title":"Done ✓","elements":[]}',
  "time": "1000",
  "message_id": "om_done",
}


def test_wait_for_response_finds_the_done_card_under_a_newer_marker(
  monkeypatch,
) -> None:
  """A card posted AFTER the Done card must not hide it.

  Arming a background task posts a 👀 marker while the turn is still
  running, so the marker is often the newest bot message when the turn
  finishes. A newest-message-only search then never sees the Done card and
  reports a completed turn as failed (2026-10-29, ID01).
  """
  monkeypatch.setattr(
    e2e, "get_bot_msgs", lambda *a, **k: [MARKER_CARD, DONE_CARD])
  msg, _ = e2e.wait_for_response(
    "oc_test", after="0", timeout=1, poll=0, require_done=True)
  assert msg is not None
  assert msg["message_id"] == "om_done"


def test_control_newest_only_lookup_misses_the_done_card(monkeypatch) -> None:
  """Control: with the old newest-only lookup the same messages yield nothing.

  Pins the scan as the load-bearing part — if someone "simplifies"
  ``wait_for_response`` back to ``get_latest_bot_msg`` this assertion is what
  fails.
  """
  monkeypatch.setattr(
    e2e, "get_bot_msgs", lambda *a, **k: [MARKER_CARD, DONE_CARD])

  def _newest_only(chat_id: str, after: str, limit: int = 10) -> dict | None:
    msgs = e2e.get_bot_msgs(chat_id, after=after, limit=limit)
    return msgs[0] if msgs and e2e.is_done_response(msgs[0]) else None

  monkeypatch.setattr(e2e, "newest_completed_msg", _newest_only)
  msg, _ = e2e.wait_for_response(
    "oc_test", after="0", timeout=1, poll=0, require_done=True)
  assert msg is None


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


def test_spawn_logged_survives_a_stderr_flood(tmp_path: Path) -> None:
  """A child writing far more than a pipe buffer to stderr must not hang.

  Regression guard for the harness's old `stderr=subprocess.PIPE` + a single
  readline(): nothing drained the pipe, so a daemon that wrote >64KB to stderr
  blocked forever on its next write. That presented as a daemon deadlock —
  silent log, no heartbeat — and hid the traceback when the daemon really
  died. 1MB of stderr is ~16x the kernel pipe buffer.
  """
  script = (
    "import sys\n"
    "sys.stderr.write('x' * 1_000_000)\n"
    "sys.stderr.flush()\n"
  )
  proc, err_path = spawn_logged(
    [sys.executable, "-c", script], cwd=str(tmp_path), env={}, tag="flood")
  try:
    assert proc.wait(timeout=30) == 0
    assert Path(err_path).stat().st_size == 1_000_000
  finally:
    if proc.poll() is None:
      proc.kill()


def _piped_streams(path: Path) -> list[str]:
  """Every `subprocess.Popen(...)` in `path` whose stdout/stderr is a PIPE."""
  offenders: list[str] = []
  tree = ast.parse(path.read_text())
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "Popen"):
      continue
    for kw in node.keywords:
      if kw.arg not in ("stdout", "stderr", "stdin"):
        continue
      # Match `subprocess.PIPE` / bare `PIPE`, but not DEVNULL or a file.
      if isinstance(kw.value, ast.Attribute) and kw.value.attr == "PIPE":
        offenders.append(f"line {kw.value.lineno}: {kw.arg}=PIPE")
      elif isinstance(kw.value, ast.Name) and kw.value.id == "PIPE":
        offenders.append(f"line {kw.value.lineno}: {kw.arg}=PIPE")
  return offenders


def test_e2e_never_pipes_an_undrained_stream() -> None:
  """The suite must spawn children with stderr on a file, never a PIPE.

  A PIPE is only safe if something drains it for the child's whole lifetime;
  start_nemo's did not, which wedged the daemon. Route every spawn through
  `spawn_logged` (covered behaviourally above) instead.
  """
  assert _piped_streams(E2E_PATH) == []
