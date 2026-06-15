#!/usr/bin/env python3
"""Exploration harness: run ONE real interactive `claude` turn under a pty and
dump the pyte screen buffer, so we can design completion-detection and
answer-extraction for the `claude-cli` adapter against the real TUI layout.

Uses the REAL endpoint (real subscription quota — one tiny turn). No
ANTHROPIC_BASE_URL override here; we want a genuine response on screen.

    python3 scripts/cli_tui_explore.py --prompt "reply with exactly: pong"
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import select
import signal
import struct
import shutil
import subprocess
import sys
import termios
import time

import pyte


def _set_winsize(fd: int, rows: int, cols: int) -> None:
  fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _display(screen: pyte.Screen) -> str:
  return "\n".join(screen.display)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--prompt", default="reply with exactly the word: pong")
  ap.add_argument("--project-dir", default=os.getcwd())
  ap.add_argument("--rows", type=int, default=40)
  ap.add_argument("--cols", type=int, default=120)
  ap.add_argument("--boot-wait", type=float, default=6.0)
  ap.add_argument("--turn-wait", type=float, default=45.0)
  ap.add_argument("--settle", type=float, default=2.5,
                  help="seconds of no screen change = turn done")
  args = ap.parse_args()

  claude = shutil.which("claude")
  if not claude:
    print("no claude on PATH", file=sys.stderr)
    return 1

  master, slave = pty.openpty()
  _set_winsize(master, args.rows, args.cols)
  env = dict(os.environ)
  env["TERM"] = "xterm-256color"
  env.pop("CLAUDE_CODE_ENTRYPOINT", None)

  proc = subprocess.Popen(
    [claude, "--permission-mode", "acceptEdits"],
    stdin=slave, stdout=slave, stderr=slave,
    cwd=args.project_dir, env=env, preexec_fn=os.setsid, close_fds=True,
  )
  os.close(slave)

  screen = pyte.HistoryScreen(args.cols, args.rows, history=4000, ratio=0.5)
  stream = pyte.ByteStream(screen)

  def pump(timeout: float) -> int:
    """Feed available pty bytes into pyte for up to `timeout`s; return bytes."""
    total = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      r, _, _ = select.select([master], [], [], 0.15)
      if master in r:
        try:
          data = os.read(master, 65536)
        except OSError:
          break
        if not data:
          break
        stream.feed(data)
        total += len(data)
    return total

  snapshots: list[str] = []
  try:
    pump(args.boot_wait)
    print("===== SCREEN AFTER BOOT =====")
    print(_display(screen))
    print("=============================\n")

    os.write(master, b"\r")           # dismiss any trust/theme dialog
    pump(0.8)
    os.write(master, args.prompt.encode())
    time.sleep(0.3)
    os.write(master, b"\r")           # submit

    # Poll until the screen stops changing for `settle` seconds.
    last_display = _display(screen)
    last_change = time.monotonic()
    deadline = time.monotonic() + args.turn_wait
    tick = 0
    while time.monotonic() < deadline:
      pump(0.4)
      cur = _display(screen)
      now = time.monotonic()
      if cur != last_display:
        last_display = cur
        last_change = now
        tick += 1
        if tick % 5 == 0:
          snapshots.append(cur)
      elif now - last_change >= args.settle:
        print(f"[settled after {now - last_change:.1f}s of no change]")
        break
  finally:
    try:
      os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
      pass
    try:
      os.close(master)
    except OSError:
      pass
    proc.wait(timeout=5)

  print("\n===== FINAL SCREEN (visible) =====")
  print(_display(screen))
  print("==================================\n")

  # HistoryScreen keeps wrapped-off lines in .history.top (a deque of rows).
  print("===== SCROLLBACK (history.top) =====")
  top = getattr(screen.history, "top", None)
  if top:
    for row in list(top):
      line = "".join(cell.data for _, cell in sorted(row.items())) \
        if isinstance(row, dict) else str(row)
      if line.strip():
        print(repr(line))
  else:
    print("(empty)")
  print("====================================")
  return 0


if __name__ == "__main__":
  sys.exit(main())
