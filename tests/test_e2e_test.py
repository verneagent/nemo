from __future__ import annotations

import ast
import sys
from pathlib import Path

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
