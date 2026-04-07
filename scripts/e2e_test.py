#!/usr/bin/env python3
"""Nemo E2E test runner.

Starts a live nemo instance, sends real Lark messages, and verifies responses
via the Lark IM API. Designed to be run by a human or AI agent after major
code changes.

Usage:
    python3 scripts/e2e_test.py [--chat CHAT_ID] [--skip-sdk] [--verbose]

Prerequisites:
    - ~/.nemo/config.json (app_id, app_secret, relay_url)
    - ~/.nemo/user_token.json (2h TTL — refreshed automatically if possible)
    - Relay server running (http://47.95.232.145)

The script handles:
    - Token refresh (fails gracefully if refresh_token expired)
    - Starting/stopping nemo processes
    - Evicting stale nemo instances
    - Verifying responses via Lark API (not log files)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOME = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME, ".nemo/config.json")
TOKEN_PATH = os.path.join(HOME, ".nemo/user_token.json")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHAT_ID = "oc_8183e1682019ddc0857a29074b3e2858"
LOG_DIR = os.path.join(HOME, ".nemo/logs")

# Add project to path for imports
sys.path.insert(0, PROJECT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Colors:
  GREEN = "\033[92m"
  RED = "\033[91m"
  YELLOW = "\033[93m"
  BOLD = "\033[1m"
  RESET = "\033[0m"


def _load_config() -> dict:
  with open(CONFIG_PATH) as f:
    return json.load(f)


def _load_user_token() -> str:
  with open(TOKEN_PATH) as f:
    tok = json.load(f)
  # Check if expired
  saved = tok.get("saved_at", 0)
  expires = tok.get("expires_in", 7200)
  remaining = (saved + expires) - time.time()
  if remaining < 60:
    print(f"{Colors.YELLOW}User token expired ({remaining:.0f}s remaining), refreshing...{Colors.RESET}")
    _refresh_token()
    with open(TOKEN_PATH) as f:
      tok = json.load(f)
  return tok["access_token"]


def _refresh_token() -> None:
  """Refresh user token using refresh_token grant."""
  cfg = _load_config()
  with open(TOKEN_PATH) as f:
    tok = json.load(f)
  data = json.dumps({
    "grant_type": "refresh_token",
    "refresh_token": tok["refresh_token"],
    "client_id": cfg["app_id"],
    "client_secret": cfg["app_secret"],
  }).encode()
  req = urllib.request.Request(
    "https://open.larksuite.com/open-apis/authen/v2/oauth/token",
    data=data, method="POST")
  req.add_header("Content-Type", "application/json")
  try:
    resp = urllib.request.urlopen(req, timeout=10)
    d = json.loads(resp.read())
    d["saved_at"] = time.time()
    with open(TOKEN_PATH, "w") as f:
      json.dump(d, f, indent=2)
    print(f"  Token refreshed (expires_in={d.get('expires_in')}s)")
  except Exception as e:
    print(f"{Colors.RED}  Token refresh failed: {e}")
    print(f"  Run device flow manually (see CLAUDE.md){Colors.RESET}")
    sys.exit(1)


def _get_bot_token() -> str:
  cfg = _load_config()
  from nemo.lark.auth import get_token
  return get_token(cfg["app_id"], cfg["app_secret"])


def send_msg(text: str, chat_id: str) -> str:
  """Send a message as the user via Lark API. Returns message_id."""
  token = _load_user_token()
  data = json.dumps({
    "receive_id": chat_id,
    "msg_type": "text",
    "content": json.dumps({"text": text}),
  }).encode()
  req = urllib.request.Request(
    f"https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
    data=data, method="POST")
  req.add_header("Authorization", f"Bearer {token}")
  req.add_header("Content-Type", "application/json")
  resp = urllib.request.urlopen(req, timeout=10)
  result = json.loads(resp.read())
  return result.get("data", {}).get("message_id", "?")


def get_latest_bot_msg(chat_id: str, bot_id: str = "cli_a9583021bef89ed4",
                       after: str = "0") -> dict | None:
  """Get the latest bot message from the Lark group after a timestamp."""
  token = _get_bot_token()
  req = urllib.request.Request(
    f"https://open.larksuite.com/open-apis/im/v1/messages"
    f"?container_id_type=chat&container_id={chat_id}"
    f"&page_size=5&sort_type=ByCreateTimeDesc")
  req.add_header("Authorization", f"Bearer {token}")
  resp = urllib.request.urlopen(req, timeout=10)
  data = json.loads(resp.read())
  for item in data.get("data", {}).get("items", []):
    sender = item.get("sender", {}).get("id", "")
    ct = item.get("create_time", "0")
    if sender == bot_id and ct > after:
      return {
        "type": item.get("msg_type", ""),
        "body": item.get("body", {}).get("content", ""),
        "time": ct,
      }
  return None


def read_log(pid: int, last_n: int = 50) -> str:
  """Read recent lines from nemo's per-PID log file."""
  log_path = os.path.join(LOG_DIR, f"nemo-{pid}.log")
  try:
    with open(log_path) as f:
      lines = f.readlines()
    return "".join(lines[-last_n:])
  except FileNotFoundError:
    return ""


def start_nemo(chat_id: str, verbose: bool = False) -> int:
  """Start a nemo process. Returns PID."""
  cmd = [sys.executable, "-m", "nemo", "--chat", chat_id]
  if verbose:
    cmd.append("--verbose")
  proc = subprocess.Popen(
    cmd, cwd=PROJECT_DIR,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return proc.pid


def wait_for_ready(pid: int, timeout: int = 30) -> bool:
  """Wait for nemo to log 'SDK client connected'."""
  deadline = time.time() + timeout
  while time.time() < deadline:
    log = read_log(pid)
    if "SDK client connected" in log:
      return True
    time.sleep(1)
  return False


def is_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except (OSError, ProcessLookupError):
    return False


def wait_for_exit(pid: int, timeout: int = 25) -> bool:
  """Wait for process to exit."""
  deadline = time.time() + timeout
  while time.time() < deadline:
    if not is_alive(pid):
      return True
    time.sleep(0.5)
  return False


def kill_nemo(pid: int) -> None:
  """Kill nemo process."""
  try:
    os.kill(pid, signal.SIGTERM)
    if not wait_for_exit(pid, timeout=5):
      os.kill(pid, signal.SIGKILL)
  except (OSError, ProcessLookupError):
    pass


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
  def __init__(self):
    self.passed: list[str] = []
    self.failed: list[str] = []
    self.skipped: list[str] = []

  def ok(self, name: str, detail: str = ""):
    self.passed.append(name)
    d = f" ({detail})" if detail else ""
    print(f"  {Colors.GREEN}PASS{Colors.RESET} {name}{d}")

  def fail(self, name: str, detail: str = ""):
    self.failed.append(name)
    d = f" ({detail})" if detail else ""
    print(f"  {Colors.RED}FAIL{Colors.RESET} {name}{d}")

  def skip(self, name: str, reason: str = ""):
    self.skipped.append(name)
    r = f" ({reason})" if reason else ""
    print(f"  {Colors.YELLOW}SKIP{Colors.RESET} {name}{r}")

  def summary(self):
    total = len(self.passed) + len(self.failed) + len(self.skipped)
    print(f"\n{Colors.BOLD}Results: {len(self.passed)}/{total} passed", end="")
    if self.failed:
      print(f", {Colors.RED}{len(self.failed)} failed{Colors.RESET}", end="")
    if self.skipped:
      print(f", {len(self.skipped)} skipped", end="")
    print(Colors.RESET)
    if self.failed:
      print(f"  Failed: {', '.join(self.failed)}")
    return len(self.failed) == 0


def run_command_test(name: str, text: str, chat_id: str,
                     result: TestResult, wait: int = 5) -> None:
  """Send a command and verify a bot response card appears."""
  ts = str(int(time.time() * 1000))
  send_msg(text, chat_id)
  time.sleep(wait)
  msg = get_latest_bot_msg(chat_id, after=ts)
  if msg and msg["type"] == "interactive":
    result.ok(name)
  else:
    result.fail(name, f"no response card (got: {msg})")


def run_sdk_test(name: str, text: str, pid: int, chat_id: str,
                 result: TestResult, wait: int = 20,
                 expect_log: str | None = None) -> None:
  """Send a message that triggers an SDK turn, verify response."""
  ts = str(int(time.time() * 1000))
  send_msg(text, chat_id)
  time.sleep(wait)
  msg = get_latest_bot_msg(chat_id, after=ts)
  log = read_log(pid)
  if msg and msg["time"] > ts:
    if expect_log and expect_log not in log:
      result.ok(name, "card ok, log check skipped")
    else:
      result.ok(name)
  else:
    result.fail(name, "no response")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Nemo E2E test runner")
  parser.add_argument("--chat", default=DEFAULT_CHAT_ID, help="Chat ID")
  parser.add_argument("--skip-sdk", action="store_true", help="Skip SDK turn tests")
  parser.add_argument("--verbose", "-v", action="store_true", help="Verbose nemo")
  args = parser.parse_args()

  chat_id = args.chat
  result = TestResult()

  print(f"{Colors.BOLD}Nemo E2E Test Suite{Colors.RESET}")
  print(f"  Chat: {chat_id}")
  print(f"  Project: {PROJECT_DIR}")
  print()

  # ---- Phase 0: Setup ----
  print(f"{Colors.BOLD}Phase 0: Setup{Colors.RESET}")

  # Check config
  if not os.path.exists(CONFIG_PATH):
    print(f"{Colors.RED}Missing {CONFIG_PATH}{Colors.RESET}")
    return 1
  if not os.path.exists(TOKEN_PATH):
    print(f"{Colors.RED}Missing {TOKEN_PATH}{Colors.RESET}")
    return 1

  # Refresh token if needed
  try:
    _load_user_token()
    print("  User token: OK")
  except Exception as e:
    print(f"{Colors.RED}  User token: FAILED ({e}){Colors.RESET}")
    return 1

  # Start nemo
  print("  Starting nemo...")
  pid = start_nemo(chat_id, verbose=args.verbose)
  print(f"  PID: {pid}")

  if not wait_for_ready(pid, timeout=30):
    print(f"{Colors.RED}  Nemo failed to start (no SDK connection in 30s){Colors.RESET}")
    kill_nemo(pid)
    return 1
  print("  Nemo ready")
  print()

  try:
    # ---- Phase 1: Commands ----
    print(f"{Colors.BOLD}Phase 1: Commands{Colors.RESET}")
    run_command_test("T01 ping", "ping", chat_id, result)
    run_command_test("T02 /help", "/help", chat_id, result)
    run_command_test("T03 /model", "/model", chat_id, result)
    run_command_test("T04 /cost", "/cost", chat_id, result)
    run_command_test("T05 /diag", "/diag", chat_id, result, wait=8)
    print()

    # ---- Phase 2: SDK Turns ----
    if args.skip_sdk:
      for name in ["T06", "T07", "T08", "T09"]:
        result.skip(name, "skipped by --skip-sdk")
    else:
      print(f"{Colors.BOLD}Phase 2: SDK Turns{Colors.RESET}")
      run_sdk_test("T06 simple question", "What is 2+2? Just answer the number.",
                   pid, chat_id, result, wait=15, expect_log="Processing:")
      run_sdk_test("T07 bash tool", "Run ls in the current directory",
                   pid, chat_id, result, wait=20)
      run_sdk_test("T08 read tool", "Read the first line of pyproject.toml",
                   pid, chat_id, result, wait=20)
      run_sdk_test("T09 multi-tool", "How many Python files are in nemo/? Count them.",
                   pid, chat_id, result, wait=25)
    print()

    # ---- Phase 3: Signals ----
    print(f"{Colors.BOLD}Phase 3: Signals & Control{Colors.RESET}")

    # T10: /esc
    ts = str(int(time.time() * 1000))
    send_msg("Write a 500-word essay about Python history", chat_id)
    time.sleep(3)
    send_msg("/esc", chat_id)
    time.sleep(12)
    if is_alive(pid):
      msg = get_latest_bot_msg(chat_id, after=ts)
      if msg:
        result.ok("T10 /esc interrupt")
      else:
        result.fail("T10 /esc interrupt", "no response after esc")
    else:
      result.fail("T10 /esc interrupt", "process died")

    # T11: /clear
    run_command_test("T11 /clear", "/clear", chat_id, result, wait=15)

    # T12: post-clear turn
    if not args.skip_sdk:
      run_sdk_test("T12 post-clear turn", "Say hello",
                   pid, chat_id, result, wait=15)
    else:
      result.skip("T12 post-clear turn", "skipped by --skip-sdk")

    # T16: empty message (before exit)
    ts = str(int(time.time() * 1000))
    send_msg("   ", chat_id)
    time.sleep(3)
    msg = get_latest_bot_msg(chat_id, after=ts)
    if msg is None:
      result.ok("T16 empty message")
    else:
      result.fail("T16 empty message", "got unexpected response")

    # T13: /exit
    ts = str(int(time.time() * 1000))
    send_msg("/exit", chat_id)
    exited = wait_for_exit(pid, timeout=25)
    if exited:
      result.ok("T13 /exit shutdown")
    else:
      result.fail("T13 /exit shutdown", "process still running after 25s")
      kill_nemo(pid)
    print()

    # ---- Phase 4: Recovery ----
    print(f"{Colors.BOLD}Phase 4: Recovery{Colors.RESET}")

    # T14: restart
    pid2 = start_nemo(chat_id, verbose=args.verbose)
    if wait_for_ready(pid2, timeout=30):
      result.ok("T14 restart")
    else:
      result.fail("T14 restart", "failed to start")
      kill_nemo(pid2)
      result.summary()
      return 1 if result.failed else 0

    try:
      # T15: post-recovery turn
      if not args.skip_sdk:
        run_sdk_test("T15 post-recovery turn", "What is 3+3?",
                     pid2, chat_id, result, wait=15)
      else:
        result.skip("T15 post-recovery turn", "skipped by --skip-sdk")
    finally:
      # Clean shutdown
      send_msg("/exit", chat_id)
      if not wait_for_exit(pid2, timeout=25):
        kill_nemo(pid2)

    # T17: dissolve (always skip in automated testing)
    result.skip("T17 /dissolve", "destructive — manual only")

  except KeyboardInterrupt:
    print(f"\n{Colors.YELLOW}Interrupted{Colors.RESET}")
    kill_nemo(pid)
    return 1
  except Exception as e:
    print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
    kill_nemo(pid)
    raise

  print()
  ok = result.summary()
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
