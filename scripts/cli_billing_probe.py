#!/usr/bin/env python3
"""Billing-surface probe for the `claude-cli` (pty-driven TUI) experiment.

Goal (deliverables B + C of "pty 驱动 TUI 实验"): empirically confirm that an
*unmodified, interactive* `claude` TUI driven under a pty reports the
subscription surface tag

    User-Agent: claude-cli/<ver> (external, cli)

while the headless `claude -p ... --output-format stream-json` path reports

    User-Agent: claude-cli/<ver> (external, sdk-cli)

How it works, zero-cost and harmless:
  * Start a tiny local HTTP server that records every request's User-Agent and
    answers `/v1/messages` with 401 (so no tokens are ever spent — the request
    dies right after the header we care about is logged).
  * Point ANTHROPIC_BASE_URL at that server. We deliberately do NOT spend any
    real quota; we only read the request header the CLI self-reports.
  * Path A (interactive): spawn `claude` under a stdlib pty, type a prompt,
    submit, and wait for the first /v1/messages hit.
  * Path C (headless): run `claude -p "hi" --output-format stream-json` for the
    contrast tag.

This is the GATE for the whole experiment: if Path A reports `sdk-cli` (not
`cli`), driving the TUI buys nothing over the existing SDK adapter and we stop.

Usage:
    python3 scripts/cli_billing_probe.py            # run both paths
    python3 scripts/cli_billing_probe.py --interactive-only
    python3 scripts/cli_billing_probe.py --headless-only
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Requests recorded by the capture server: list of (path, user_agent, headers).
_CAPTURED: list[tuple[str, str, dict[str, str]]] = []
_CAPTURE_LOCK = threading.Lock()


class _CaptureHandler(BaseHTTPRequestHandler):
  """Log the request's User-Agent, then 401 so no real call ever completes."""

  def _record_and_401(self) -> None:
    ua = self.headers.get("User-Agent", "")
    hdrs = {k: v for k, v in self.headers.items()}
    with _CAPTURE_LOCK:
      _CAPTURED.append((self.path, ua, hdrs))
    body = b'{"type":"error","error":{"type":"authentication_error",' \
           b'"message":"probe: dummy key, intentional 401"}}'
    self.send_response(401)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    try:
      self.wfile.write(body)
    except BrokenPipeError:
      pass

  # The CLI may probe with GET/POST on various paths; record them all.
  def do_GET(self) -> None:  # noqa: N802
    self._record_and_401()

  def do_POST(self) -> None:  # noqa: N802
    self._record_and_401()

  def log_message(self, *_args: object) -> None:  # silence default stderr spam
    pass


def _start_capture_server() -> tuple[ThreadingHTTPServer, str]:
  server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
  port = server.server_address[1]
  t = threading.Thread(target=server.serve_forever, daemon=True)
  t.start()
  return server, f"http://127.0.0.1:{port}"


def _probe_env(base_url: str, entrypoint: str | None) -> dict[str, str]:
  """Env for a probe run: redirect API at the capture server, dummy key.

  We pass a dummy ANTHROPIC_API_KEY so even if the CLI falls back to API auth
  the call still 401s harmlessly. The User-Agent surface tag is what we read,
  and it is set by the CLI independent of which credential it picks.
  """
  env = dict(os.environ)
  env["ANTHROPIC_BASE_URL"] = base_url
  env["ANTHROPIC_API_KEY"] = "dummy-probe-key-do-not-use"
  env["TERM"] = "xterm-256color"
  # Make sure nothing in the inherited shell forces a non-interactive surface.
  env.pop("CLAUDE_CODE_ENTRYPOINT", None)
  if entrypoint is not None:
    env["CLAUDE_CODE_ENTRYPOINT"] = entrypoint
  return env


def _set_winsize(fd: int, rows: int = 40, cols: int = 120) -> None:
  winsize = struct.pack("HHHH", rows, cols, 0, 0)
  fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def _drain(fd: int, sink: list[bytes], timeout: float) -> None:
  """Read whatever is available from the pty master for up to `timeout`s."""
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    r, _, _ = select.select([fd], [], [], 0.2)
    if fd in r:
      try:
        chunk = os.read(fd, 65536)
      except OSError:
        break
      if not chunk:
        break
      sink.append(chunk)


def _first_messages_ua() -> str | None:
  with _CAPTURE_LOCK:
    for path, ua, _ in _CAPTURED:
      if "/v1/messages" in path:
        return ua
  return None


def run_interactive(base_url: str, project_dir: str, prompt: str,
                    boot_wait: float, total_wait: float) -> str | None:
  """Spawn interactive `claude` under a pty, submit `prompt`, return its
  /v1/messages User-Agent (or None if no such request was captured)."""
  claude = shutil.which("claude")
  if not claude:
    print("ERROR: `claude` not on PATH", file=sys.stderr)
    return None

  master, slave = pty.openpty()
  _set_winsize(master)
  env = _probe_env(base_url, entrypoint=None)  # let CLI choose its own (cli)

  proc = subprocess.Popen(
    [claude, "--permission-mode", "acceptEdits"],
    stdin=slave, stdout=slave, stderr=slave,
    cwd=project_dir, env=env, preexec_fn=os.setsid, close_fds=True,
  )
  os.close(slave)

  out: list[bytes] = []
  try:
    # Let the TUI boot (theme/trust prompts may appear on a fresh machine).
    _drain(master, out, boot_wait)
    # Nudge past any trust/theme dialog, then type the prompt and submit.
    os.write(master, b"\r")
    _drain(master, out, 0.8)
    os.write(master, prompt.encode())
    time.sleep(0.3)
    os.write(master, b"\r")
    # Wait for the first /v1/messages hit (or give up after total_wait).
    deadline = time.monotonic() + total_wait
    while time.monotonic() < deadline:
      _drain(master, out, 0.5)
      if _first_messages_ua() is not None:
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

  raw = b"".join(out)
  print(f"  [interactive] captured {len(raw)} bytes of TUI output")
  return _first_messages_ua()


def run_headless(base_url: str, project_dir: str) -> str | None:
  """Run `claude -p hi --output-format stream-json`, return its
  /v1/messages User-Agent for contrast."""
  claude = shutil.which("claude")
  if not claude:
    return None
  before = len(_CAPTURED)
  env = _probe_env(base_url, entrypoint=None)
  try:
    subprocess.run(
      [claude, "-p", "hi", "--output-format", "stream-json", "--verbose"],
      cwd=project_dir, env=env, timeout=30,
      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
  except subprocess.TimeoutExpired:
    pass
  with _CAPTURE_LOCK:
    for path, ua, _ in _CAPTURED[before:]:
      if "/v1/messages" in path:
        return ua
  return None


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--project-dir", default=os.getcwd())
  ap.add_argument("--prompt", default="say hi in one word")
  ap.add_argument("--boot-wait", type=float, default=6.0)
  ap.add_argument("--total-wait", type=float, default=40.0)
  ap.add_argument("--interactive-only", action="store_true")
  ap.add_argument("--headless-only", action="store_true")
  args = ap.parse_args()

  server, base_url = _start_capture_server()
  print(f"capture server: {base_url}  project: {args.project_dir}")
  inter_ua = head_ua = None
  try:
    if not args.headless_only:
      print("\n=== Path A: interactive TUI under pty ===")
      inter_ua = run_interactive(
        base_url, args.project_dir, args.prompt,
        args.boot_wait, args.total_wait)
      print(f"  interactive User-Agent: {inter_ua!r}")
    if not args.interactive_only:
      print("\n=== Path C: headless `claude -p ... stream-json` ===")
      head_ua = run_headless(base_url, args.project_dir)
      print(f"  headless User-Agent:    {head_ua!r}")
  finally:
    server.shutdown()

  print("\n=== all captured requests ===")
  with _CAPTURE_LOCK:
    for path, ua, _ in _CAPTURED:
      print(f"  {path}  <-  {ua}")

  print("\n=== verdict ===")
  ok = inter_ua is not None and "(external, cli)" in inter_ua
  if args.interactive_only or args.headless_only:
    print("  (single-path run — no comparison verdict)")
  elif ok and head_ua and "sdk-cli" in head_ua:
    print("  ✅ FEASIBLE: interactive=cli, headless=sdk-cli — surfaces differ as hoped")
  elif inter_ua is None:
    print("  ⚠️  INCONCLUSIVE: interactive path produced no /v1/messages request")
  elif "(external, cli)" not in (inter_ua or ""):
    print(f"  ❌ NOT FEASIBLE: interactive reported {inter_ua!r}, not (external, cli)")
  else:
    print("  ⚠️  partial: check the tags above")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
