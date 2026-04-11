"""Tests for _daemonize() double-fork."""

import os
import re
import subprocess
import sys
import time

import pytest


@pytest.mark.slow
def test_double_fork_reparents_to_pid1():
  """Daemon (grandchild) should be reparented to PID 1 after double-fork."""
  result = subprocess.run(
    [sys.executable, "-c", """
import os, sys, time
sys.path.insert(0, ".")
from nemo.__main__ import _daemonize, signal_ready
_daemonize()
signal_ready()
# We are the daemon — sleep so the test can inspect us
time.sleep(5)
"""],
    capture_output=True, text=True, timeout=10,
    cwd=os.path.dirname(os.path.dirname(__file__)),
  )
  assert result.returncode == 0
  m = re.search(r"PID (\d+)", result.stderr)
  assert m, f"Expected 'nemo started (PID N)' in stderr, got: {result.stderr!r}"
  daemon_pid = int(m.group(1))

  # Daemon should be alive
  try:
    os.kill(daemon_pid, 0)
  except ProcessLookupError:
    pytest.fail(f"Daemon PID {daemon_pid} is not alive")

  # Daemon PPID should be 1 (reparented to init/launchd)
  ps = subprocess.run(
    ["ps", "-p", str(daemon_pid), "-o", "ppid="],
    capture_output=True, text=True,
  )
  ppid = ps.stdout.strip()
  assert ppid == "1", f"Daemon PPID should be 1, got {ppid}"

  # Cleanup
  os.kill(daemon_pid, 15)


@pytest.mark.slow
def test_double_fork_daemon_survives_parent_kill():
  """Daemon should survive even if the launching shell is killed."""
  # Start nemo daemonize in a subprocess
  proc = subprocess.Popen(
    [sys.executable, "-c", """
import os, sys, time
sys.path.insert(0, ".")
from nemo.__main__ import _daemonize, signal_ready
_daemonize()
signal_ready()
time.sleep(10)
"""],
    stderr=subprocess.PIPE, stdout=subprocess.PIPE,
    cwd=os.path.dirname(os.path.dirname(__file__)),
  )
  stdout, stderr = proc.communicate(timeout=10)
  m = re.search(r"PID (\d+)", stderr.decode())
  assert m
  daemon_pid = int(m.group(1))

  # Give daemon a moment to settle
  time.sleep(0.3)

  # Daemon should be alive after parent exited
  try:
    os.kill(daemon_pid, 0)
  except ProcessLookupError:
    pytest.fail(f"Daemon PID {daemon_pid} died when parent exited")

  # Cleanup
  os.kill(daemon_pid, 15)


@pytest.mark.slow
def test_double_fork_reports_correct_pid():
  """The PID printed to stderr should match the actual daemon process."""
  result = subprocess.run(
    [sys.executable, "-c", """
import os, sys, time
sys.path.insert(0, ".")
from nemo.__main__ import _daemonize, signal_ready
_daemonize()
# Write our actual PID to a file for verification
with open("/tmp/.nemo_test_pid", "w") as f:
    f.write(str(os.getpid()))
signal_ready()
time.sleep(5)
"""],
    capture_output=True, text=True, timeout=10,
    cwd=os.path.dirname(os.path.dirname(__file__)),
  )
  m = re.search(r"PID (\d+)", result.stderr)
  assert m
  reported_pid = int(m.group(1))

  time.sleep(0.3)
  with open("/tmp/.nemo_test_pid") as f:
    actual_pid = int(f.read().strip())

  assert reported_pid == actual_pid, (
    f"Reported PID {reported_pid} != actual daemon PID {actual_pid}"
  )

  # Cleanup
  os.kill(reported_pid, 15)
  os.unlink("/tmp/.nemo_test_pid")
