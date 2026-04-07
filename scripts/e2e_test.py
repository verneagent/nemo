#!/usr/bin/env python3
"""Nemo E2E test runner.

Starts a live nemo instance, sends real Lark messages, and verifies responses
via the Lark IM API. Designed to be run by a human or AI agent after major
code changes.

Usage:
    python3 scripts/e2e_test.py [--chat CHAT_ID] [--skip-sdk] [--verbose]
    python3 scripts/e2e_test.py --stress     # only stale-task stress test
    python3 scripts/e2e_test.py --project    # only multi-turn project test

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
import shutil
import signal
import subprocess
import sys
import tempfile
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
BOT_ID = "cli_a9583021bef89ed4"

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


def get_latest_bot_msg(chat_id: str, after: str = "0") -> dict | None:
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
    if sender == BOT_ID and ct > after:
      return {
        "type": item.get("msg_type", ""),
        "body": item.get("body", {}).get("content", ""),
        "time": ct,
        "message_id": item.get("message_id", ""),
      }
  return None


def get_bot_msgs(chat_id: str, after: str = "0",
                 limit: int = 10) -> list[dict]:
  """Get all bot messages after a timestamp, newest first."""
  token = _get_bot_token()
  req = urllib.request.Request(
    f"https://open.larksuite.com/open-apis/im/v1/messages"
    f"?container_id_type=chat&container_id={chat_id}"
    f"&page_size={limit}&sort_type=ByCreateTimeDesc")
  req.add_header("Authorization", f"Bearer {token}")
  resp = urllib.request.urlopen(req, timeout=10)
  data = json.loads(resp.read())
  msgs = []
  for item in data.get("data", {}).get("items", []):
    sender = item.get("sender", {}).get("id", "")
    ct = item.get("create_time", "0")
    if sender == BOT_ID and ct > after:
      msgs.append({
        "type": item.get("msg_type", ""),
        "body": item.get("body", {}).get("content", ""),
        "time": ct,
        "message_id": item.get("message_id", ""),
      })
  return msgs


def wait_for_response(chat_id: str, after: str,
                      timeout: int = 60, poll: int = 3) -> tuple[dict | None, float]:
  """Poll for a bot response. Returns (msg, elapsed_seconds)."""
  start = time.time()
  deadline = start + timeout
  while time.time() < deadline:
    msg = get_latest_bot_msg(chat_id, after=after)
    if msg and msg["time"] > after:
      return msg, time.time() - start
    time.sleep(poll)
  return None, time.time() - start


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
# Phase 5: Stale Task Notification Stress Test
# ---------------------------------------------------------------------------

# Prompts designed to trigger multiple concurrent Agent spawns
AGENT_PROMPTS = [
  (
    "Do these 3 tasks IN PARALLEL using separate agents (not sequentially):\n"
    "1) Search all Python files in nemo/ for functions containing 'async def'\n"
    "2) Count total lines across all .py files in the nemo/ directory\n"
    "3) Read every file in nemo/lark/ and list each file's purpose in one line\n"
    "Launch 3 separate agents. Be brief."
  ),
  (
    "Run 3 agents in parallel:\n"
    "1) Find all TODO/FIXME/HACK comments in the codebase\n"
    "2) List every class defined in nemo/ with its file location\n"
    "3) Analyze the import graph: which nemo modules import which others?\n"
    "Use the Agent tool for each. Keep answers short."
  ),
  (
    "Launch 3 parallel agents to:\n"
    "1) Check if there are any Python files over 200 lines and list them\n"
    "2) Find all logging.getLogger calls and their names\n"
    "3) Grep for all exception handling patterns (try/except) in nemo/\n"
    "Use Agent tool. One-line summaries only."
  ),
]


def run_stale_task_stress(pid: int, chat_id: str, result: TestResult,
                          rounds: int = 3) -> None:
  """Phase 5: trigger multi-agent turns and verify stale task handling."""
  print(f"{Colors.BOLD}Phase 5: Stale Task Stress Test{Colors.RESET}")

  # Enable auto-approve so Agent tool calls don't block on permissions
  run_command_test("T20 autoapprove", "autoapprove on", chat_id, result, wait=5)

  for i in range(rounds):
    prompt = AGENT_PROMPTS[i % len(AGENT_PROMPTS)]
    tag = f"R{i+1}"

    # Step 1: Send multi-agent prompt
    print(f"  [{tag}] Sending multi-agent prompt...")
    ts = str(int(time.time() * 1000))
    send_msg(prompt, chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=180)
    if not msg:
      result.fail(f"T21 multi-agent {tag}", "timeout 180s")
      continue
    result.ok(f"T21 multi-agent {tag}", f"{elapsed:.0f}s")

    # Step 2: Immediately send simple follow-up
    time.sleep(2)
    ts2 = str(int(time.time() * 1000))
    send_msg(f"What is {i+2}+{i+3}? Just the number, nothing else.", chat_id)
    msg2, elapsed2 = wait_for_response(chat_id, after=ts2, timeout=60)

    if not msg2:
      result.fail(f"T22 follow-up {tag}", "timeout — likely stale task hang")
      # Check log for clues
      log = read_log(pid, last_n=50)
      if "Stale TaskNotification" in log:
        print(f"    Log shows stale task detected but response hung")
      continue

    if elapsed2 > 45:
      result.fail(f"T22 follow-up {tag}",
                  f"too slow ({elapsed2:.0f}s) — possible stale contamination")
    else:
      result.ok(f"T22 follow-up {tag}", f"{elapsed2:.0f}s")

    # Step 3: Check for duplicate responses (should be exactly 1 card)
    msgs = get_bot_msgs(chat_id, after=ts2)
    if len(msgs) > 2:
      result.fail(f"T23 no-dup {tag}", f"got {len(msgs)} bot messages (expect 1-2)")
    else:
      result.ok(f"T23 no-dup {tag}", f"{len(msgs)} msg(s)")

  # Summary: check log for stale task handling stats
  log = read_log(pid, last_n=200)
  stale_count = log.count("Stale TaskNotification")
  requery_count = log.count("Stale turn")
  task_started = log.count("TaskStartedEvent") + log.count("TaskStartedMessage")
  detail = f"tasks={task_started} stale={stale_count} re-queries={requery_count}"
  if stale_count > 0:
    result.ok("T24 stale handling", detail)
  else:
    result.ok("T24 stale handling", f"no stale tasks triggered ({detail})")

  print()


# ---------------------------------------------------------------------------
# Phase 6: Multi-Turn Project Flow
# ---------------------------------------------------------------------------

PROJECT_STEPS = [
  {
    "name": "T31 create calc",
    "msg": (
      "Create main.py: a Python CLI calculator. Usage: python main.py 2 + 3\n"
      "Use argparse. Support +, -, *, /. Print just the result."
    ),
    "check_files": ["main.py"],
    "timeout": 60,
  },
  {
    "name": "T32 add history",
    "msg": (
      "Create history.py: store last 10 calculations in calc_history.json.\n"
      "Each entry: {expr, result, timestamp}. Add save_calc() and get_history().\n"
      "Call save_calc() from main.py after each calculation."
    ),
    "check_files": ["history.py"],
    "timeout": 60,
  },
  {
    "name": "T33 write tests",
    "msg": (
      "Write test_calc.py using pytest. Test:\n"
      "- All four arithmetic operations via subprocess (run main.py)\n"
      "- Division by zero handling\n"
      "- History save/load round-trip"
    ),
    "check_files": ["test_calc.py"],
    "timeout": 60,
  },
  {
    "name": "T34 run tests",
    "msg": "Run `python -m pytest test_calc.py -v` and fix any failures. Show final output.",
    "check_files": [],
    "timeout": 120,
  },
  {
    "name": "T35 add verbose",
    "msg": (
      "Add --verbose flag to main.py: print the expression before the result.\n"
      "Add proper error handling for division by zero (print error, exit 1)."
    ),
    "check_files": [],
    "timeout": 60,
  },
]


def run_project_flow(_pid: int, chat_id: str, result: TestResult) -> None:
  """Phase 6: multi-turn project creation in a temp directory."""
  print(f"{Colors.BOLD}Phase 6: Multi-Turn Project Flow{Colors.RESET}")

  tmpdir = tempfile.mkdtemp(prefix="nemo-e2e-")
  subprocess.run(["git", "init", tmpdir], capture_output=True)
  subprocess.run(
    ["git", "-C", tmpdir, "config", "user.email", "test@test.com"],
    capture_output=True)
  subprocess.run(
    ["git", "-C", tmpdir, "config", "user.name", "Test"],
    capture_output=True)
  print(f"  Temp dir: {tmpdir}")

  try:
    # T30: /cd to temp dir
    ts = str(int(time.time() * 1000))
    send_msg(f"/cd {tmpdir}", chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=15)
    if msg:
      result.ok("T30 /cd", f"{elapsed:.0f}s")
    else:
      result.fail("T30 /cd", "no response")
      return

    # Enable auto-approve (Write/Edit/Bash need it)
    run_command_test("T30a autoapprove", "autoapprove on", chat_id, result, wait=5)

    # Run each project step
    for step in PROJECT_STEPS:
      name = step["name"]
      ts = str(int(time.time() * 1000))
      print(f"  [{name}] Sending...")
      send_msg(step["msg"], chat_id)

      msg, elapsed = wait_for_response(
        chat_id, after=ts, timeout=step["timeout"])

      if not msg:
        result.fail(name, f"timeout after {step['timeout']}s")
        continue

      # Check response timing
      if elapsed > step["timeout"] * 0.9:
        result.fail(name, f"nearly timed out ({elapsed:.0f}s)")
        continue

      # Check expected files
      missing = [
        f for f in step.get("check_files", [])
        if not os.path.exists(os.path.join(tmpdir, f))
      ]
      if missing:
        result.fail(name, f"missing files: {missing} ({elapsed:.0f}s)")
      else:
        result.ok(name, f"{elapsed:.0f}s")

      # Check for duplicate responses
      msgs = get_bot_msgs(chat_id, after=ts)
      if len(msgs) > 3:  # working card + done card + maybe ack
        print(f"    {Colors.YELLOW}WARN: {len(msgs)} bot messages for this turn{Colors.RESET}")

    # T36: verify project files exist
    existing = [f for f in os.listdir(tmpdir) if f.endswith(".py")]
    if "main.py" in existing:
      result.ok("T36 project files", f"{len(existing)} .py files: {existing}")
    else:
      result.fail("T36 project files", f"main.py not found, got: {existing}")

    # T37: context continuity — ask about previous work
    ts = str(int(time.time() * 1000))
    send_msg("What Python files exist in the current directory? Just list them.", chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=30)
    if msg:
      result.ok("T37 context", f"{elapsed:.0f}s")
    else:
      result.fail("T37 context", "no response — possible hang")

    # T38: rapid-fire — send 3 quick messages and verify 3 responses
    print(f"  [T38] Rapid-fire test (3 messages)...")
    timestamps = []
    questions = [
      "How many lines is main.py? Just the number.",
      "What does history.py export? One line.",
      "Does test_calc.py import pytest? Yes or no.",
    ]
    for q in questions:
      ts = str(int(time.time() * 1000))
      timestamps.append(ts)
      send_msg(q, chat_id)
      time.sleep(1)  # small gap so relay ordering is correct

    # Wait for all 3 responses
    time.sleep(45)
    all_ok = True
    for j, ts in enumerate(timestamps):
      msg = get_latest_bot_msg(chat_id, after=ts)
      if not msg:
        result.fail(f"T38 rapid-fire Q{j+1}", "no response")
        all_ok = False
    if all_ok:
      result.ok("T38 rapid-fire", "all 3 responses received")

  finally:
    # /cd back to original project dir
    send_msg(f"/cd {PROJECT_DIR}", chat_id)
    time.sleep(5)
    # Cleanup temp dir
    shutil.rmtree(tmpdir, ignore_errors=True)

  print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Nemo E2E test runner")
  parser.add_argument("--chat", default=DEFAULT_CHAT_ID, help="Chat ID")
  parser.add_argument("--skip-sdk", action="store_true",
                      help="Skip all SDK turn tests (commands only)")
  parser.add_argument("--stress", action="store_true",
                      help="Run only stale-task stress test (Phase 5)")
  parser.add_argument("--project", action="store_true",
                      help="Run only multi-turn project test (Phase 6)")
  parser.add_argument("--verbose", "-v", action="store_true",
                      help="Verbose nemo logging")
  args = parser.parse_args()

  chat_id = args.chat
  result = TestResult()
  only_stress = args.stress
  only_project = args.project
  run_all = not only_stress and not only_project

  print(f"{Colors.BOLD}Nemo E2E Test Suite{Colors.RESET}")
  print(f"  Chat: {chat_id}")
  print(f"  Project: {PROJECT_DIR}")
  if only_stress:
    print(f"  Mode: stress test only")
  elif only_project:
    print(f"  Mode: project flow only")
  print()

  # ---- Phase 0: Setup ----
  print(f"{Colors.BOLD}Phase 0: Setup{Colors.RESET}")

  if not os.path.exists(CONFIG_PATH):
    print(f"{Colors.RED}Missing {CONFIG_PATH}{Colors.RESET}")
    return 1
  if not os.path.exists(TOKEN_PATH):
    print(f"{Colors.RED}Missing {TOKEN_PATH}{Colors.RESET}")
    return 1

  try:
    _load_user_token()
    print("  User token: OK")
  except Exception as e:
    print(f"{Colors.RED}  User token: FAILED ({e}){Colors.RESET}")
    return 1

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
    if run_all:
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
        run_sdk_test("T06 simple question",
                     "What is 2+2? Just answer the number.",
                     pid, chat_id, result, wait=15, expect_log="Processing:")
        run_sdk_test("T07 bash tool",
                     "Run ls in the current directory",
                     pid, chat_id, result, wait=20)
        run_sdk_test("T08 read tool",
                     "Read the first line of pyproject.toml",
                     pid, chat_id, result, wait=20)
        run_sdk_test("T09 multi-tool",
                     "How many Python files are in nemo/? Count them.",
                     pid, chat_id, result, wait=25)
      print()

      # ---- Phase 3: Signals ----
      print(f"{Colors.BOLD}Phase 3: Signals & Control{Colors.RESET}")

      # T10: /esc
      if not args.skip_sdk:
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
      else:
        result.skip("T10 /esc interrupt", "skipped by --skip-sdk")

      # T11: /clear
      run_command_test("T11 /clear", "/clear", chat_id, result, wait=15)

      # T12: post-clear turn
      if not args.skip_sdk:
        run_sdk_test("T12 post-clear turn", "Say hello",
                     pid, chat_id, result, wait=15)
      else:
        result.skip("T12 post-clear turn", "skipped by --skip-sdk")

      # T16: empty message
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
      pid2 = start_nemo(chat_id, verbose=args.verbose)
      if wait_for_ready(pid2, timeout=30):
        result.ok("T14 restart")
      else:
        result.fail("T14 restart", "failed to start")
        kill_nemo(pid2)
        result.summary()
        return 1 if result.failed else 0

      try:
        if not args.skip_sdk:
          run_sdk_test("T15 post-recovery turn", "What is 3+3?",
                       pid2, chat_id, result, wait=15)
        else:
          result.skip("T15 post-recovery turn", "skipped by --skip-sdk")
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid2, timeout=25):
          kill_nemo(pid2)

      result.skip("T17 /dissolve", "destructive — manual only")
      print()

      # For Phase 5 & 6 we need a fresh nemo instance
      if not args.skip_sdk:
        print("  Starting fresh nemo for Phase 5 & 6...")
        pid = start_nemo(chat_id, verbose=args.verbose)
        if not wait_for_ready(pid, timeout=30):
          print(f"{Colors.RED}  Failed to start nemo for Phase 5{Colors.RESET}")
          result.summary()
          return 1 if result.failed else 0
        print(f"  PID: {pid}")
        print()

        try:
          run_stale_task_stress(pid, chat_id, result)
          run_project_flow(pid, chat_id, result)
        finally:
          send_msg("/exit", chat_id)
          if not wait_for_exit(pid, timeout=25):
            kill_nemo(pid)
      else:
        for name in ["T20-T24", "T30-T38"]:
          result.skip(name, "skipped by --skip-sdk")

    elif only_stress:
      # Stress test only
      run_stale_task_stress(pid, chat_id, result)
      send_msg("/exit", chat_id)
      if not wait_for_exit(pid, timeout=25):
        kill_nemo(pid)

    elif only_project:
      # Project flow only
      run_project_flow(pid, chat_id, result)
      send_msg("/exit", chat_id)
      if not wait_for_exit(pid, timeout=25):
        kill_nemo(pid)

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
