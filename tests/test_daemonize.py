"""Regression tests for removed daemonization.

Nemo used to double-fork and set NEMO_FOREGROUND=1 in its own environ
to mark 'already daemonized'. That env var leaked into every
subprocess — including bash invocations inside the Claude SDK — so a
nested `nemo` launched from a parent nemo's bash tool would see the
marker, skip daemonization, run as a foreground child of bash, and die
silently when that bash invocation wrapped up.

We removed daemonization entirely. These tests make sure nothing
accidentally re-introduces the env pollution or the helper symbols.
"""

import os
import subprocess
import sys

import nemo.__main__ as nemo_main


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def test_daemonize_helpers_are_gone():
  """The daemonize helpers must not reappear — they are the thing that
  set and relied on the poisoned NEMO_FOREGROUND marker."""
  for name in ("_daemonize", "signal_ready", "signal_error", "_ready_fd"):
    assert not hasattr(nemo_main, name), (
      f"nemo.__main__.{name} should be gone after daemonize removal"
    )


def test_main_source_does_not_touch_nemo_foreground_env():
  """Regression: grep the module source for the leaked env var."""
  src_path = nemo_main.__file__
  with open(src_path) as f:
    src = f.read()
  assert "NEMO_FOREGROUND" not in src, (
    "NEMO_FOREGROUND env var reintroduced — it leaks into subprocess "
    "env and breaks nested nemo launches."
  )


def test_foreground_flag_is_gone():
  """--foreground / -f has no meaning and should be rejected by argparse."""
  result = subprocess.run(
    [sys.executable, "-m", "nemo", "--foreground"],
    capture_output=True, text=True, timeout=10,
    cwd=REPO_ROOT,
  )
  assert result.returncode != 0
  assert "unrecognized arguments" in result.stderr or "unrecognized" in result.stderr


def test_cli_subprocess_does_not_leak_nemo_foreground():
  """Run `python -m nemo --version` in a subprocess and verify it
  does not set NEMO_FOREGROUND in its own or descendant env.

  We check this by exporting env from a child that imports nemo.__main__
  at module load time (same import surface as the real CLI entry).
  """
  probe = (
    "import os, nemo.__main__;"
    "print('HAS_NEMO_FOREGROUND=' + str('NEMO_FOREGROUND' in os.environ))"
  )
  env = {k: v for k, v in os.environ.items() if k != "NEMO_FOREGROUND"}
  result = subprocess.run(
    [sys.executable, "-c", probe],
    capture_output=True, text=True, timeout=10,
    cwd=REPO_ROOT, env=env,
  )
  assert result.returncode == 0, result.stderr
  assert "HAS_NEMO_FOREGROUND=False" in result.stdout, (
    f"nemo import leaked NEMO_FOREGROUND into env: {result.stdout!r}"
  )


