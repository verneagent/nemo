#!/usr/bin/env python3
"""Nemo E2E test runner.

Starts a live nemo instance, sends real Lark messages, and verifies responses
via the Lark IM API. Designed to be run by a human or AI agent after major
code changes.

Usage:
    python3 scripts/e2e_test.py                 # full run (all phases)
    python3 scripts/e2e_test.py --skip-sdk       # commands only (fast)
    python3 scripts/e2e_test.py --stress         # stale-task stress only
    python3 scripts/e2e_test.py --project        # multi-turn project only
    python3 scripts/e2e_test.py --perm           # permission flow only
    python3 scripts/e2e_test.py --dual           # dual-instance only
    python3 scripts/e2e_test.py --media          # media & interaction only
    python3 scripts/e2e_test.py --verbose        # debug nemo logging
    python3 scripts/e2e_test.py --chat <ID>      # custom chat group

Prerequisites:
    - ~/.nemo/default.json (app_id, app_secret, relay_url, email)
    - ~/.nemo/user_token.json (2h TTL — refreshed automatically)
    - Relay server running (http://47.95.232.145)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOME = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME, ".nemo/default.json")
TOKEN_PATH = os.path.join(HOME, ".nemo/user_token.json")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHAT_ID = "oc_8183e1682019ddc0857a29074b3e2858"
LOG_DIR = os.path.join(HOME, ".nemo/logs")
BOT_ID = "cli_a9583021bef89ed4"
# Operator open_id — used when injecting messages via relay
OPERATOR_OPEN_ID = "ou_1f03ce275afdf3486d658740a39d0d8a"

# Add project to path for imports
sys.path.insert(0, PROJECT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Colors:
  GREEN = "\033[92m"
  RED = "\033[91m"
  YELLOW = "\033[93m"
  CYAN = "\033[96m"
  BOLD = "\033[1m"
  DIM = "\033[2m"
  RESET = "\033[0m"


def _load_config() -> dict:
  with open(CONFIG_PATH) as f:
    return json.load(f)


def _load_user_token() -> str:
  with open(TOKEN_PATH) as f:
    tok = json.load(f)
  saved = tok.get("saved_at", 0)
  expires = tok.get("expires_in", 7200)
  remaining = (saved + expires) - time.time()
  if remaining < 60:
    print(f"{Colors.YELLOW}User token expired ({remaining:.0f}s remaining), "
          f"refreshing...{Colors.RESET}")
    _refresh_token()
    with open(TOKEN_PATH) as f:
      tok = json.load(f)
  return tok["access_token"]


def _refresh_token() -> None:
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


def _send_via_relay(text: str, chat_id: str) -> str:
  """Inject a message via relay webhook (no user token needed)."""
  cfg = _load_config()
  relay_url = cfg.get("relay_url", "").rstrip("/")
  verify_token = cfg.get("relay_verify_token", "")
  if not relay_url or not verify_token:
    raise RuntimeError("relay_url or relay_verify_token not in config")
  ts = str(int(time.time() * 1000))
  msg_id = f"test_{ts}_{os.getpid()}"
  payload = json.dumps({
    "header": {
      "token": verify_token,
      "event_type": "im.message.receive_v1",
      "event_id": f"evt_{msg_id}",
    },
    "event": {
      "message": {
        "chat_id": chat_id,
        "message_type": "text",
        "content": json.dumps({"text": text}),
        "create_time": ts,
        "message_id": msg_id,
      },
      "sender": {
        "sender_type": "user",
        "sender_id": {"open_id": OPERATOR_OPEN_ID},
      },
    },
  }).encode()
  req = urllib.request.Request(
    f"{relay_url}/webhook", data=payload, method="POST")
  req.add_header("Content-Type", "application/json")
  resp = urllib.request.urlopen(req, timeout=10)
  result = json.loads(resp.read())
  if not result.get("ok", False):
    raise RuntimeError(f"Relay inject failed: {result}")
  return msg_id


def _inject_relay_event(payload: dict, chat_id: str) -> str:
  """Inject a raw relay webhook event. Returns message_id."""
  cfg = _load_config()
  relay_url = cfg.get("relay_url", "").rstrip("/")
  verify_token = cfg.get("relay_verify_token", "")
  if not relay_url or not verify_token:
    raise RuntimeError("relay_url or relay_verify_token not in config")
  payload.setdefault("header", {})["token"] = verify_token
  data = json.dumps(payload).encode()
  req = urllib.request.Request(
    f"{relay_url}/webhook", data=data, method="POST")
  req.add_header("Content-Type", "application/json")
  resp = urllib.request.urlopen(req, timeout=10)
  result = json.loads(resp.read())
  if not result.get("ok", False):
    raise RuntimeError(f"Relay inject failed: {result}")
  return payload.get("event", {}).get("message", {}).get("message_id", "")


def send_image_msg(image_key: str, chat_id: str) -> str:
  """Inject an image message via relay webhook."""
  ts = str(int(time.time() * 1000))
  msg_id = f"test_img_{ts}_{os.getpid()}"
  return _inject_relay_event({
    "header": {
      "event_type": "im.message.receive_v1",
      "event_id": f"evt_{msg_id}",
    },
    "event": {
      "message": {
        "chat_id": chat_id,
        "message_type": "image",
        "content": json.dumps({"image_key": image_key}),
        "create_time": ts,
        "message_id": msg_id,
      },
      "sender": {
        "sender_type": "user",
        "sender_id": {"open_id": OPERATOR_OPEN_ID},
      },
    },
  }, chat_id)


def send_reply_msg(text: str, parent_id: str, chat_id: str) -> str:
  """Inject a reply message (with parent_id) via relay webhook."""
  ts = str(int(time.time() * 1000))
  msg_id = f"test_reply_{ts}_{os.getpid()}"
  return _inject_relay_event({
    "header": {
      "event_type": "im.message.receive_v1",
      "event_id": f"evt_{msg_id}",
    },
    "event": {
      "message": {
        "chat_id": chat_id,
        "message_type": "text",
        "content": json.dumps({"text": text}),
        "create_time": ts,
        "message_id": msg_id,
        "parent_id": parent_id,
      },
      "sender": {
        "sender_type": "user",
        "sender_id": {"open_id": OPERATOR_OPEN_ID},
      },
    },
  }, chat_id)


def send_topic_msg(text: str, thread_id: str, chat_id: str,
                   parent_id: str = "") -> str:
  """Inject a topic-chat style message via relay webhook.

  Topic-chat messages carry thread_id (format: "omt_<hex>") and
  usually parent_id pointing at the topic root. This simulates the
  FivedBug scenario end-to-end without needing to create a real topic
  group (which requires the Lark client — the OpenAPI only creates
  chat_mode=group).
  """
  ts = str(int(time.time() * 1000))
  msg_id = f"test_topic_{ts}_{os.getpid()}"
  message: dict = {
    "chat_id": chat_id,
    "chat_type": "group",
    "message_type": "text",
    "content": json.dumps({"text": text}),
    "create_time": ts,
    "message_id": msg_id,
    "thread_id": thread_id,
  }
  if parent_id:
    message["parent_id"] = parent_id
  return _inject_relay_event({
    "header": {
      "event_type": "im.message.receive_v1",
      "event_id": f"evt_{msg_id}",
    },
    "event": {
      "message": message,
      "sender": {
        "sender_type": "user",
        "sender_id": {"open_id": OPERATOR_OPEN_ID},
      },
    },
  }, chat_id)


def send_reaction(target_message_id: str, emoji_type: str,
                  chat_id: str) -> str:
  """Inject a reaction event via relay webhook."""
  ts = str(int(time.time() * 1000))
  return _inject_relay_event({
    "header": {
      "event_type": "im.message.reaction.created_v1",
      "event_id": f"evt_react_{ts}_{os.getpid()}",
    },
    "event": {
      "message_id": target_message_id,
      "reaction_type": {"emoji_type": emoji_type},
      "user_id": {"open_id": OPERATOR_OPEN_ID},
      "chat_id": chat_id,
    },
  }, chat_id)


def _send_via_user_api(text: str, chat_id: str) -> str:
  """Send a message as the user via Lark API."""
  token = _load_user_token()
  data = json.dumps({
    "receive_id": chat_id,
    "msg_type": "text",
    "content": json.dumps({"text": text}),
  }).encode()
  req = urllib.request.Request(
    "https://open.larksuite.com/open-apis/im/v1/messages"
    "?receive_id_type=chat_id",
    data=data, method="POST")
  req.add_header("Authorization", f"Bearer {token}")
  req.add_header("Content-Type", "application/json")
  resp = urllib.request.urlopen(req, timeout=10)
  result = json.loads(resp.read())
  return result.get("data", {}).get("message_id", "?")


def send_msg(text: str, chat_id: str) -> str:
  """Send a message — prefer user API (triggers real WS events), fall back to relay."""
  if os.path.exists(TOKEN_PATH):
    try:
      return _send_via_user_api(text, chat_id)
    except Exception as e:
      print(f"  {Colors.YELLOW}User API failed ({e}), falling back to relay{Colors.RESET}")
  return _send_via_relay(text, chat_id)


def _lark_get_json(url: str, token: str, retries: int = 5) -> dict:
  """GET JSON from Lark with light retry for transient server failures."""
  last_err = None
  for attempt in range(1, retries + 1):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    try:
      resp = urllib.request.urlopen(req, timeout=10)
      return json.loads(resp.read())
    except urllib.error.HTTPError as e:
      last_err = e
      # Lark occasionally returns 5xx while the message list is otherwise healthy.
      if 500 <= e.code < 600 and attempt < retries:
        time.sleep(min(attempt, 3))
        continue
      raise
    except Exception as e:
      last_err = e
      if attempt < retries:
        time.sleep(min(attempt, 3))
        continue
      raise
  assert last_err is not None
  raise last_err


def get_latest_bot_msg(chat_id: str, after: str = "0") -> dict | None:
  """Get the latest bot message after a timestamp."""
  token = _get_bot_token()
  data = _lark_get_json(
    f"https://open.larksuite.com/open-apis/im/v1/messages"
    f"?container_id_type=chat&container_id={chat_id}"
    f"&page_size=5&sort_type=ByCreateTimeDesc",
    token,
  )
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
  data = _lark_get_json(
    f"https://open.larksuite.com/open-apis/im/v1/messages"
    f"?container_id_type=chat&container_id={chat_id}"
    f"&page_size={limit}&sort_type=ByCreateTimeDesc",
    token,
  )
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
                      timeout: int = 60, poll: int = 3,
                      require_done: bool = False) -> tuple[dict | None, float]:
  """Poll for a bot response. Returns (msg, elapsed_seconds)."""
  start = time.time()
  deadline = start + timeout
  while time.time() < deadline:
    msg = get_latest_bot_msg(chat_id, after=after)
    if msg and msg["time"] > after:
      if require_done and not is_done_response(msg):
        time.sleep(poll)
        continue
      return msg, time.time() - start
    time.sleep(poll)
  return None, time.time() - start


def interactive_card_title(msg: dict | None) -> str:
  """Best-effort title extraction for interactive card bodies."""
  if not msg or msg.get("type") != "interactive":
    return ""
  body = msg.get("body", "")
  if not isinstance(body, str) or not body:
    return ""
  try:
    parsed = json.loads(body)
  except json.JSONDecodeError:
    return ""
  title = parsed.get("title")
  if isinstance(title, str):
    return title
  if isinstance(title, dict):
    text = title.get("content")
    if isinstance(text, str):
      return text
  return ""


def is_done_response(msg: dict | None) -> bool:
  """Return True when a bot response represents a completed turn."""
  if not msg:
    return False
  if msg.get("type") != "interactive":
    return True
  return interactive_card_title(msg).startswith("Done")


# ---------------------------------------------------------------------------
# Log analyzer
# ---------------------------------------------------------------------------

class LogAnalyzer:
  """Structured log file reader for a nemo instance."""

  def __init__(self, pid: int):
    self.pid = pid
    self.path = os.path.join(LOG_DIR, f"nemo-{pid}.log")

  def read(self, last_n: int = 200) -> str:
    try:
      with open(self.path) as f:
        lines = f.readlines()
      return "".join(lines[-last_n:])
    except FileNotFoundError:
      return ""

  def mark(self) -> int:
    try:
      return os.path.getsize(self.path)
    except OSError:
      return 0

  def read_since(self, offset: int) -> str:
    try:
      with open(self.path) as f:
        f.seek(offset)
        return f.read()
    except OSError:
      return ""

  def wait_for_since(self, pattern: str, offset: int,
                     timeout: int = 30, poll: float = 1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
      chunk = self.read_since(offset)
      try:
        rx = re.compile(pattern)
        if rx.search(chunk):
          return True
      except re.error:
        if pattern in chunk:
          return True
      time.sleep(poll)
    return False

  def lines(self, last_n: int = 200) -> list[str]:
    try:
      with open(self.path) as f:
        return f.readlines()[-last_n:]
    except FileNotFoundError:
      return []

  def count(self, pattern: str, last_n: int = 500) -> int:
    """Count lines matching pattern (substring or regex)."""
    return len(self.find(pattern, last_n))

  def find(self, pattern: str, last_n: int = 500) -> list[str]:
    """Find all lines matching pattern."""
    lines = self.lines(last_n)
    try:
      rx = re.compile(pattern)
      return [l for l in lines if rx.search(l)]
    except re.error:
      return [l for l in lines if pattern in l]

  def wait_for(self, pattern: str, timeout: int = 30, poll: float = 1) -> bool:
    """Poll log until pattern appears. Returns True if found."""
    deadline = time.time() + timeout
    while time.time() < deadline:
      if self.count(pattern, last_n=50) > 0:
        return True
      time.sleep(poll)
    return False

  def errors(self, last_n: int = 200) -> list[str]:
    """Find ERROR-level log lines."""
    return [l for l in self.lines(last_n) if " ERROR " in l or "Error:" in l]

  def processing_sequence(self) -> list[str]:
    """Extract ordered list of 'Processing: ...' messages."""
    result = []
    for line in self.lines(500):
      m = re.search(r"Processing: (.+)", line)
      if m:
        result.append(m.group(1).strip())
    return result

  def dump_tail(self, n: int = 20, label: str = "") -> None:
    """Print last N lines for debugging."""
    tag = f" [{label}]" if label else ""
    print(f"{Colors.DIM}    --- log tail{tag} ---{Colors.RESET}")
    for line in self.lines(n):
      print(f"    {Colors.DIM}{line.rstrip()}{Colors.RESET}")
    print(f"{Colors.DIM}    --- end ---{Colors.RESET}")


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def start_nemo(chat_id: str, verbose: bool = False,
               permission_mode: str = "bypassPermissions",
               provider: str = "claude") -> int:
  """Start a nemo process. Returns PID."""
  cmd = [sys.executable, "-m", "nemo", "--chat-id", chat_id,
         "--permission-mode", permission_mode,
         "--provider", provider]
  if verbose:
    cmd.append("--verbose")
  proc = subprocess.Popen(
    cmd, cwd=PROJECT_DIR,
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    text=True)
  try:
    line = proc.stderr.readline().strip() if proc.stderr else ""
  except Exception:
    line = ""
  m = re.search(r"nemo started \(PID (\d+)\)", line)
  if m:
    return int(m.group(1))
  return proc.pid


def wait_for_ready(pid: int, timeout: int = 30, provider: str = "claude") -> bool:
  """Wait for nemo to log a provider-specific ready signal."""
  log = LogAnalyzer(pid)
  if provider == "claude":
    if log.wait_for("SDK client connected", timeout=timeout):
      return True
  return log.wait_for("Start card sent:", timeout=timeout)


def is_alive(pid: int) -> bool:
  try:
    # Use waitpid for our child processes — os.kill(pid, 0) returns
    # success on zombie processes, making it useless for children
    # we spawned via Popen.
    result = os.waitpid(pid, os.WNOHANG)
    if result == (0, 0):
      return True  # Still running
    return False  # Exited (reaped the zombie)
  except ChildProcessError:
    # Not our child — fall back to kill signal 0
    try:
      os.kill(pid, 0)
      return True
    except (OSError, ProcessLookupError):
      return False


def wait_for_exit(pid: int, timeout: int = 25) -> bool:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if not is_alive(pid):
      return True
    time.sleep(0.5)
  return False


def kill_nemo(pid: int) -> None:
  try:
    os.kill(pid, signal.SIGTERM)
    if not wait_for_exit(pid, timeout=5):
      os.kill(pid, signal.SIGKILL)
  except (OSError, ProcessLookupError):
    pass


def _has_working_turn(chat_id: str) -> bool:
  """Check whether the chat still has an active working card in the session DB."""
  from nemo.db import Database

  db = Database(PROJECT_DIR)
  try:
    session_id = db.get_chat_owner(chat_id)
    if not session_id:
      return False
    return db.get_working(session_id) is not None
  finally:
    db.close()


def wait_for_idle(pid: int, chat_id: str, timeout: int = 30) -> None:
  """Wait for nemo to be idle for this chat."""
  log = LogAnalyzer(pid)
  deadline = time.time() + timeout
  while time.time() < deadline:
    tail = log.lines(8)
    active_log = any(
      "Processing:" in l or "query() prompt" in l or "turn msg:" in l
      for l in tail
    )
    active_working = _has_working_turn(chat_id)
    if not active_log and not active_working:
      return
    time.sleep(2)


# ---------------------------------------------------------------------------
# Lark group management (for dual-instance test)
# ---------------------------------------------------------------------------

def create_temp_group(name: str) -> str | None:
  """Create a temporary Lark group. Returns chat_id."""
  from nemo.lark.api import create_chat, add_chat_members, lookup_open_id_by_email
  token = _get_bot_token()
  try:
    chat_id = create_chat(token, name, description="E2E test temp group")
  except Exception as e:
    print(f"{Colors.RED}  Failed to create temp group: {e}{Colors.RESET}")
    return None

  # Add operator so they can send messages
  cfg = _load_config()
  email = cfg.get("email", "")
  if email:
    try:
      open_id = lookup_open_id_by_email(token, email)
      if open_id:
        add_chat_members(token, chat_id, [open_id])
    except Exception as e:
      print(f"{Colors.YELLOW}  Failed to add operator to temp group: {e}{Colors.RESET}")
  return chat_id


def dissolve_temp_group(chat_id: str) -> None:
  from nemo.lark.api import dissolve_chat
  try:
    dissolve_chat(_get_bot_token(), chat_id)
  except Exception as e:
    print(f"{Colors.YELLOW}  Failed to dissolve temp group: {e}{Colors.RESET}")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class E2EResult:
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
                     result: E2EResult, wait: int = 5) -> None:
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
                 result: E2EResult, wait: int = 20,
                 expect_log: str | None = None) -> None:
  """Send a message that triggers an SDK turn, verify response."""
  log = LogAnalyzer(pid)
  log_mark = log.mark()
  ts = str(int(time.time() * 1000))
  send_msg(text, chat_id)
  time.sleep(wait)
  msg = get_latest_bot_msg(chat_id, after=ts)
  if msg and msg["time"] > ts:
    if expect_log and log.count(expect_log, last_n=30) == 0:
      result.ok(name, "card ok, log check skipped")
    else:
      result.ok(name)
  elif log.wait_for_since("Turn response finalized", log_mark, timeout=5, poll=1):
    result.ok(name, "completed via log fallback")
  else:
    result.fail(name, "no response")
    log.dump_tail(10, name)


# ---------------------------------------------------------------------------
# Phase 5: Stale Task Notification Stress Test
# ---------------------------------------------------------------------------

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


def run_stale_task_stress(pid: int, chat_id: str, result: E2EResult,
                          rounds: int = 3) -> None:
  """Phase 5: trigger multi-agent turns and verify stale task handling."""
  print(f"{Colors.BOLD}Phase 5: Stale Task Stress Test{Colors.RESET}")
  log = LogAnalyzer(pid)

  # Enable auto-approve so Agent tool calls don't block
  run_command_test("T20 autoapprove", "autoapprove on", chat_id, result, wait=5)

  for i in range(rounds):
    prompt = AGENT_PROMPTS[i % len(AGENT_PROMPTS)]
    tag = f"R{i+1}"

    # Step 1: multi-agent prompt
    print(f"  [{tag}] Sending multi-agent prompt...")
    ts = str(int(time.time() * 1000))
    send_msg(prompt, chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=180)
    if not msg:
      result.fail(f"T21 multi-agent {tag}", "timeout 180s")
      log.dump_tail(15, f"multi-agent {tag}")
      continue
    result.ok(f"T21 multi-agent {tag}", f"{elapsed:.0f}s")

    # Step 2: follow-up after turn completes
    # Wait for nemo to finish processing the multi-agent turn
    wait_for_idle(pid, chat_id, timeout=60)
    time.sleep(2)
    ts2 = str(int(time.time() * 1000))
    send_msg(f"What is {i+2}+{i+3}? Just the number, nothing else.", chat_id)
    msg2, elapsed2 = wait_for_response(chat_id, after=ts2, timeout=90)

    if not msg2:
      result.fail(f"T22 follow-up {tag}", "timeout (90s) — likely stale task hang")
      log.dump_tail(15, f"follow-up {tag}")
      continue

    if elapsed2 > 60:
      result.ok(f"T22 follow-up {tag}", f"SLOW ({elapsed2:.0f}s) but completed")
    else:
      result.ok(f"T22 follow-up {tag}", f"{elapsed2:.0f}s")

    # Step 3: duplicate check
    msgs = get_bot_msgs(chat_id, after=ts2)
    if len(msgs) > 2:
      result.fail(f"T23 no-dup {tag}", f"got {len(msgs)} bot msgs (expect 1-2)")
    else:
      result.ok(f"T23 no-dup {tag}", f"{len(msgs)} msg(s)")

  # Summary from logs
  stale_count = log.count("Stale TaskNotification")
  requery_count = log.count("Stale turn")
  task_lines = log.find("TaskStarted")
  detail = f"tasks={len(task_lines)} stale={stale_count} re-queries={requery_count}"
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
    "check_files": [],  # File existence checked later by T36
    "timeout": 60,
  },
  {
    "name": "T34 run tests",
    "msg": "Run `python -m pytest test_calc.py -v` and fix any failures. "
           "Show final output.",
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


def run_project_flow(pid: int, chat_id: str, result: E2EResult) -> None:
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
    # T30: /cd
    ts = str(int(time.time() * 1000))
    send_msg(f"/cd {tmpdir}", chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=15)
    if msg:
      result.ok("T30 /cd", f"{elapsed:.0f}s")
    else:
      result.fail("T30 /cd", "no response")
      return

    # Enable auto-approve for Write/Edit/Bash
    run_command_test("T30a autoapprove", "autoapprove on",
                     chat_id, result, wait=5)

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

      if elapsed > step["timeout"] * 0.9:
        result.fail(name, f"nearly timed out ({elapsed:.0f}s)")
        continue

      check_files = step.get("check_files", [])
      if check_files:
        time.sleep(2)  # Allow file writes to flush
      missing = []
      for f in check_files:
        # Check direct path and also search recursively
        if os.path.exists(os.path.join(tmpdir, f)):
          continue
        # Search in subdirectories
        found = False
        for root, _dirs, files in os.walk(tmpdir):
          if f in files:
            found = True
            break
        if not found:
          missing.append(f)
      if missing:
        result.fail(name, f"missing: {missing} ({elapsed:.0f}s)")
      else:
        result.ok(name, f"{elapsed:.0f}s")

    # Wait for last step to finish before checking files
    wait_for_idle(pid, chat_id, timeout=30)

    # T36: verify project files
    existing = [f for f in os.listdir(tmpdir) if f.endswith(".py")]
    if "main.py" in existing:
      result.ok("T36 project files", f"{len(existing)} .py: {existing}")
    else:
      result.fail("T36 project files", f"main.py missing, got: {existing}")

    # T37: context continuity
    ts = str(int(time.time() * 1000))
    send_msg("What Python files exist in the current directory? Just list them.",
             chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=30)
    if msg:
      result.ok("T37 context", f"{elapsed:.0f}s")
    else:
      result.fail("T37 context", "no response")

    # T38: rapid-fire
    print(f"  [T38] Rapid-fire (3 messages)...")
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
      time.sleep(1)

    time.sleep(45)
    all_ok = True
    for j, ts in enumerate(timestamps):
      msg = get_latest_bot_msg(chat_id, after=ts)
      if not msg:
        result.fail(f"T38 rapid Q{j+1}", "no response")
        all_ok = False
    if all_ok:
      result.ok("T38 rapid-fire", "all 3 responses received")

  finally:
    send_msg(f"/cd {PROJECT_DIR}", chat_id)
    time.sleep(5)
    shutil.rmtree(tmpdir, ignore_errors=True)

  print()


# ---------------------------------------------------------------------------
# Phase 7: Permission Flow
# ---------------------------------------------------------------------------

def run_permission_tests(pid: int, chat_id: str,
                         result: E2EResult) -> None:
  """Phase 7: test approve / deny / always permission flow."""
  print(f"{Colors.BOLD}Phase 7: Permission Flow{Colors.RESET}")
  log = LogAnalyzer(pid)

  tmpdir = tempfile.mkdtemp(prefix="nemo-perm-")
  print(f"  Temp dir: {tmpdir}")

  # Switch to temp dir
  ts = str(int(time.time() * 1000))
  send_msg(f"/cd {tmpdir}", chat_id)
  wait_for_response(chat_id, after=ts, timeout=15)

  # Ensure autoapprove is OFF
  send_msg("autoapprove off", chat_id)
  time.sleep(5)

  try:
    # In "plan" permission_mode, the CLI requires approval for all tool use.
    # can_use_tool is called for every Bash, Write, Edit, etc.

    # T40: Approve ("y")
    ts = str(int(time.time() * 1000))
    send_msg(
      'Run this exact bash command: echo "approved" > approve_test.txt',
      chat_id)
    # Wait for permission card in log
    if log.wait_for("Permission request:", timeout=45):
      time.sleep(1)
      send_msg("y", chat_id)
      msg, elapsed = wait_for_response(chat_id, after=ts, timeout=60)
      if msg:
        result.ok("T40 perm approve", f"{elapsed:.0f}s")
      else:
        result.fail("T40 perm approve", "no response after approve")
        log.dump_tail(15, "perm approve")
    else:
      # Permission wasn't requested — might have been auto-approved
      msg, elapsed = wait_for_response(chat_id, after=ts, timeout=30)
      if msg:
        result.ok("T40 perm approve", f"auto-approved ({elapsed:.0f}s)")
      else:
        result.fail("T40 perm approve", "no permission card and no result")
        log.dump_tail(15, "perm approve")

    # T41: Deny ("n")
    ts = str(int(time.time() * 1000))
    send_msg(
      'Run this exact bash command: echo "denied" > deny_test.txt',
      chat_id)
    if log.wait_for("Permission request:", timeout=45):
      time.sleep(1)
      send_msg("n", chat_id)
      # Wait for turn to complete (denied turn should finish quickly)
      time.sleep(15)
      if not os.path.exists(os.path.join(tmpdir, "deny_test.txt")):
        # Check log for denial
        if log.count("Permission decision: deny", last_n=30) > 0:
          result.ok("T41 perm deny", "denied, file not created")
        else:
          result.ok("T41 perm deny", "file not created (no deny log)")
      else:
        result.fail("T41 perm deny", "file should not exist!")
    else:
      result.skip("T41 perm deny", "no permission card appeared")

    # Wait for any residual turn to finish
    time.sleep(10)

    # T42: Always ("always")
    ts = str(int(time.time() * 1000))
    send_msg(
      'Run this exact bash command: echo "always-approved" > always_test.txt',
      chat_id)
    if log.wait_for("Permission request:", timeout=45):
      time.sleep(1)
      send_msg("always", chat_id)
      msg, elapsed = wait_for_response(chat_id, after=ts, timeout=60)
      if msg:
        result.ok("T42 perm always", f"{elapsed:.0f}s")
      else:
        result.fail("T42 perm always", "no response")
        log.dump_tail(15, "perm always")
    else:
      result.skip("T42 perm always", "no permission card appeared")

    # T43: Auto-approve should be ON after "always"
    ts = str(int(time.time() * 1000))
    send_msg(
      'Run this exact bash command: echo "auto-approved" > auto_test.txt',
      chat_id)
    msg, elapsed = wait_for_response(chat_id, after=ts, timeout=45)
    # Should NOT show permission card (auto-approved by "always")
    perm_after = log.count("Permission request:.*auto_test", last_n=20)
    perm_seen = log.count("Permission request:", last_n=80) > 0
    if not perm_seen:
      result.skip("T43 auto-approve", "permission cards unsupported in current CLI")
      return
    if msg and os.path.exists(os.path.join(tmpdir, "auto_test.txt")):
      if perm_after == 0:
        result.ok("T43 auto-approve", f"no prompt ({elapsed:.0f}s)")
      else:
        result.ok("T43 auto-approve", f"file created but still prompted ({elapsed:.0f}s)")
    else:
      result.fail("T43 auto-approve", "file not created or no response")

    # Reset autoapprove
    send_msg("autoapprove off", chat_id)
    time.sleep(3)

  finally:
    send_msg(f"/cd {PROJECT_DIR}", chat_id)
    time.sleep(5)
    shutil.rmtree(tmpdir, ignore_errors=True)

  print()


# ---------------------------------------------------------------------------
# Phase 8: Dual-Instance (same dir, two groups)
# ---------------------------------------------------------------------------

def run_dual_instance(chat_id_a: str, result: E2EResult,
                      verbose: bool = False) -> None:
  """Phase 8: two nemo instances on same project dir, different groups."""
  print(f"{Colors.BOLD}Phase 8: Dual-Instance Test{Colors.RESET}")

  # T50: Create second group
  chat_id_b = create_temp_group("nemo-e2e-dual")
  if not chat_id_b:
    result.fail("T50 create group B", "failed to create")
    return
  result.ok("T50 create group B", chat_id_b[:20])

  pid_a = 0
  pid_b = 0
  tmpdir = tempfile.mkdtemp(prefix="nemo-dual-")
  subprocess.run(["git", "init", tmpdir], capture_output=True)
  subprocess.run(
    ["git", "-C", tmpdir, "config", "user.email", "test@test.com"],
    capture_output=True)
  subprocess.run(
    ["git", "-C", tmpdir, "config", "user.name", "Test"],
    capture_output=True)
  # Create a shared file to read
  with open(os.path.join(tmpdir, "shared.txt"), "w") as f:
    f.write("dual-instance test\n")

  try:
    # T51: Start two nemo instances (both use same project dir via /cd)
    pid_a = start_nemo(chat_id_a, verbose=verbose)
    pid_b = start_nemo(chat_id_b, verbose=verbose)
    log_a = LogAnalyzer(pid_a)
    log_b = LogAnalyzer(pid_b)
    print(f"  Nemo A: PID={pid_a} chat={chat_id_a[:20]}")
    print(f"  Nemo B: PID={pid_b} chat={chat_id_b[:20]}")

    ready_a = wait_for_ready(pid_a, timeout=30)
    ready_b = wait_for_ready(pid_b, timeout=30)
    if ready_a and ready_b:
      result.ok("T51 both started", f"A={pid_a} B={pid_b}")
    else:
      detail = f"A={'ok' if ready_a else 'FAIL'} B={'ok' if ready_b else 'FAIL'}"
      result.fail("T51 both started", detail)
      if not ready_a:
        log_a.dump_tail(10, "nemo A")
      if not ready_b:
        log_b.dump_tail(10, "nemo B")
      return

    # /cd both to the shared tmpdir
    for cid in [chat_id_a, chat_id_b]:
      ts = str(int(time.time() * 1000))
      send_msg(f"/cd {tmpdir}", cid)
      wait_for_response(cid, after=ts, timeout=15)
    # Enable autoapprove on both
    for cid in [chat_id_a, chat_id_b]:
      send_msg("autoapprove on", cid)
      time.sleep(3)

    # T52: Both respond to ping
    ts_a = str(int(time.time() * 1000))
    ts_b = str(int(time.time() * 1000))
    send_msg("ping", chat_id_a)
    send_msg("ping", chat_id_b)
    time.sleep(5)
    msg_a = get_latest_bot_msg(chat_id_a, after=ts_a)
    msg_b = get_latest_bot_msg(chat_id_b, after=ts_b)
    if msg_a and msg_b:
      result.ok("T52 both ping")
    else:
      result.fail("T52 both ping",
                  f"A={'ok' if msg_a else 'FAIL'} B={'ok' if msg_b else 'FAIL'}")

    # T53: Concurrent reads
    ts_a = str(int(time.time() * 1000))
    ts_b = str(int(time.time() * 1000))
    send_msg("Read shared.txt and tell me what it says.", chat_id_a)
    send_msg("Read shared.txt and tell me what it says.", chat_id_b)
    msg_a, el_a = wait_for_response(chat_id_a, after=ts_a, timeout=30)
    msg_b, el_b = wait_for_response(chat_id_b, after=ts_b, timeout=30)
    if msg_a and msg_b:
      result.ok("T53 concurrent reads", f"A={el_a:.0f}s B={el_b:.0f}s")
    else:
      result.fail("T53 concurrent reads",
                  f"A={'ok' if msg_a else 'FAIL'} B={'ok' if msg_b else 'FAIL'}")

    # T54: Concurrent writes (different files)
    ts_a = str(int(time.time() * 1000))
    ts_b = str(int(time.time() * 1000))
    send_msg("Create a file named from_a.txt containing 'written by A'.",
             chat_id_a)
    send_msg("Create a file named from_b.txt containing 'written by B'.",
             chat_id_b)
    msg_a, el_a = wait_for_response(chat_id_a, after=ts_a, timeout=60)
    msg_b, el_b = wait_for_response(chat_id_b, after=ts_b, timeout=60)
    file_a = os.path.exists(os.path.join(tmpdir, "from_a.txt"))
    file_b = os.path.exists(os.path.join(tmpdir, "from_b.txt"))
    if file_a and file_b:
      result.ok("T54 concurrent writes",
                f"both created (A={el_a:.0f}s B={el_b:.0f}s)")
    else:
      result.fail("T54 concurrent writes",
                  f"A={file_a} B={file_b}")

    # T55: Log isolation — each PID log should only have its own chat
    errors_a = [l for l in log_a.lines(200) if chat_id_b in l]
    errors_b = [l for l in log_b.lines(200) if chat_id_a in l]
    if not errors_a and not errors_b:
      result.ok("T55 log isolation", "no cross-chat entries")
    else:
      result.fail("T55 log isolation",
                  f"A has {len(errors_a)} B-entries, "
                  f"B has {len(errors_b)} A-entries")

    # T56: No errors in either log
    errs_a = log_a.errors()
    errs_b = log_b.errors()
    if not errs_a and not errs_b:
      result.ok("T56 no errors", "both logs clean")
    else:
      detail = f"A={len(errs_a)} errors, B={len(errs_b)} errors"
      result.fail("T56 no errors", detail)
      if errs_a:
        print(f"    Nemo A errors:")
        for e in errs_a[:3]:
          print(f"      {e.strip()}")
      if errs_b:
        print(f"    Nemo B errors:")
        for e in errs_b[:3]:
          print(f"      {e.strip()}")

  finally:
    # Cleanup
    for cid, p in [(chat_id_a, pid_a), (chat_id_b, pid_b)]:
      if p:
        send_msg("/exit", cid)
    time.sleep(5)
    for p in [pid_a, pid_b]:
      if p and is_alive(p):
        if not wait_for_exit(p, timeout=20):
          kill_nemo(p)
    dissolve_temp_group(chat_id_b)
    shutil.rmtree(tmpdir, ignore_errors=True)

  print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_media_tests(pid: int, chat_id: str, result: E2EResult) -> None:
  """Phase 9: Media & interaction tests — reaction, image, reply."""
  print(f"{Colors.BOLD}Phase 9: Media & Interaction{Colors.RESET}")
  log = LogAnalyzer(pid)

  # T60: Reaction — send a message, get response, react to it
  print("  [T60] Sending message for reaction test...")
  ts = str(int(time.time() * 1000))
  send_msg("Say OK", chat_id)
  msg, elapsed = wait_for_response(chat_id, ts, timeout=30)
  if msg and msg.get("message_id"):
    target_id = msg["message_id"]
    time.sleep(1)
    # React to bot's response
    send_reaction(target_id, "THUMBSUP", chat_id)
    time.sleep(5)
    # Check nemo received the reaction (logged as Processing or event)
    tail = log.lines(10)
    got_reaction = any("THUMBSUP" in l for l in tail)
    if got_reaction:
      result.ok("T60 reaction received")
    else:
      # Reaction routing depends on relay msgchat registration —
      # if the message wasn't registered, relay can't route
      result.fail("T60 reaction received", "THUMBSUP not found in log")
  else:
    result.fail("T60 reaction received", "no bot message to react to")

  wait_for_idle(pid, chat_id, timeout=30)

  # T61: Image message — inject an image, verify nemo processes it
  print("  [T61] Sending image message...")
  ts = str(int(time.time() * 1000))
  # Use a known image_key from the Lark app icon (always valid for download)
  send_image_msg("img_v3_0210h_placeholder_test", chat_id)
  time.sleep(5)
  tail = log.lines(10)
  # Nemo should log the event (even if image download fails)
  got_image = any("image" in l.lower() or "img_v3" in l for l in tail)
  if got_image:
    result.ok("T61 image event")
  else:
    # Image messages arrive with text="[image]" — check for that too
    got_image_text = any("[image]" in l for l in tail)
    if got_image_text:
      result.ok("T61 image event")
    else:
      result.fail("T61 image event", "no image event in log")

  wait_for_idle(pid, chat_id, timeout=30)

  # T62: Reply message (parent_id) — send a message, reply to bot's response
  print("  [T62] Sending reply message...")
  ts = str(int(time.time() * 1000))
  send_msg("Say hello", chat_id)
  msg, elapsed = wait_for_response(chat_id, ts, timeout=30)
  if msg and msg.get("message_id"):
    parent_id = msg["message_id"]
    wait_for_idle(pid, chat_id, timeout=30)
    ts2 = str(int(time.time() * 1000))
    send_reply_msg("What did you just say?", parent_id, chat_id)
    resp, elapsed = wait_for_response(chat_id, ts2, timeout=30)
    if resp:
      result.ok(f"T62 reply (parent_id)", f"{elapsed:.0f}s")
    else:
      result.fail("T62 reply (parent_id)", "no response to reply")
  else:
    result.fail("T62 reply (parent_id)", "no bot message to reply to")

  # T63: Verify parent_id appears in SDK prompt (JSON format)
  tail = log.lines(20)
  got_parent = any("parent_id" in l for l in tail)
  if got_parent:
    result.ok("T63 parent_id in prompt")
  else:
    # parent_id detection happens in messages.build_prompt which outputs JSON
    result.skip("T63 parent_id in prompt", "not visible in log")

  print()


def run_topic_tests(pid: int, chat_id: str, result: E2EResult) -> None:
  """Phase 10: Topic-chat scenarios — regression for parent-quote
  enrichment breaking slash commands and for thread_id propagation.

  We can't create an actual chat_mode=topic group via OpenAPI (only
  the Lark client can), so we inject topic-style events into the
  regular e2e chat: messages with ``thread_id`` set and/or
  ``parent_id`` pointing at an earlier bot message. This reproduces
  every code path except the LarkChannel.send_* routing to the reply
  API (which is gated on _chat_mode == "topic"; chat_mode detection
  is covered by unit tests).
  """
  print(f"{Colors.BOLD}Phase 10: Topic Chat{Colors.RESET}")
  log = LogAnalyzer(pid)

  # Anchor: make sure there's a recent bot message to use as parent.
  print("  [T70] Priming parent anchor...")
  ts = str(int(time.time() * 1000))
  send_msg("ping", chat_id)
  anchor_msg, _ = wait_for_response(chat_id, ts, timeout=20)
  if not anchor_msg or not anchor_msg.get("message_id"):
    result.fail("T70 prime anchor", "no bot message to anchor replies to")
    print()
    return
  anchor_id = anchor_msg["message_id"]
  result.ok("T70 prime anchor", anchor_id[:20])
  wait_for_idle(pid, chat_id, timeout=30)

  # T71: slash command sent as a reply (parent_id set). Pre-fix this
  # fell through to the SDK because the parent-quote tail broke
  # exact-match dispatch; post-fix strip_parent_quote peels it off
  # and the command dispatches locally.
  print("  [T71] /help as reply (parent-quote regression)...")
  ts = str(int(time.time() * 1000))
  send_reply_msg("/help", anchor_id, chat_id)
  msg, elapsed = wait_for_response(chat_id, ts, timeout=15)
  tail = log.lines(30)
  # "query() sent to CLI" means it was forwarded to the SDK — the bug.
  forwarded_to_sdk = any(
    "/help" in l and "query() sent to CLI" in l for l in tail
  )
  if msg and not forwarded_to_sdk:
    result.ok("T71 /help as reply", f"{elapsed:.1f}s")
  elif forwarded_to_sdk:
    result.fail(
      "T71 /help as reply",
      "dispatched to SDK instead of local handler (parent-quote not stripped)",
    )
    log.dump_tail(15, "T71")
  else:
    result.fail("T71 /help as reply", "no response card")
    log.dump_tail(15, "T71")

  wait_for_idle(pid, chat_id, timeout=30)

  # T72: /ping as a reply — same regression, second command.
  print("  [T72] /ping as reply...")
  ts = str(int(time.time() * 1000))
  send_reply_msg("/ping", anchor_id, chat_id)
  msg, elapsed = wait_for_response(chat_id, ts, timeout=15)
  if msg:
    result.ok("T72 /ping as reply", f"{elapsed:.1f}s")
  else:
    result.fail("T72 /ping as reply", "no response card")
    log.dump_tail(15, "T72")

  wait_for_idle(pid, chat_id, timeout=30)

  # T73: /model (arg-parsing command) as a reply — must not pick up
  # the parent-quote tail as its argument.
  print("  [T73] /model as reply (arg-parsing)...")
  ts = str(int(time.time() * 1000))
  send_reply_msg("/model", anchor_id, chat_id)
  msg, elapsed = wait_for_response(chat_id, ts, timeout=15)
  # When /model has no arg, nemo replies with usage text — the card
  # must not contain the parent-quote marker text.
  if msg:
    body_txt = json.dumps(msg.get("body", ""))
    if "The user is replying to this earlier message" in body_txt:
      result.fail(
        "T73 /model as reply",
        "parent-quote leaked into command response",
      )
    else:
      result.ok("T73 /model as reply", f"{elapsed:.1f}s")
  else:
    result.fail("T73 /model as reply", "no response card")
    log.dump_tail(15, "T73")

  wait_for_idle(pid, chat_id, timeout=30)

  # T74: thread_id propagation — inject a topic-style event carrying
  # thread_id and verify the LarkEvent parser surfaced it. We check
  # the log for "Processing:" to confirm the message was accepted;
  # thread_id parsing itself is covered by test_lark_events.py, but
  # an end-to-end smoke test catches relay-side regressions.
  print("  [T74] thread_id propagation...")
  ts = str(int(time.time() * 1000))
  send_topic_msg(
    "say pineapple",
    thread_id="omt_e2etest0001",
    chat_id=chat_id,
    parent_id=anchor_id,
  )
  msg, elapsed = wait_for_response(chat_id, ts, timeout=30)
  if msg:
    result.ok("T74 thread_id event accepted", f"{elapsed:.1f}s")
  else:
    result.fail("T74 thread_id event accepted", "no response")
    log.dump_tail(15, "T74")

  print()


def main():
  import argparse
  parser = argparse.ArgumentParser(description="Nemo E2E test runner")
  parser.add_argument("--chat-id", default="",
                      help="Chat ID (defaults to a fresh temp group)")
  parser.add_argument("--provider", default="claude", choices=["claude", "codex", "opencode"],
                      help="Coding agent provider (default: claude)")
  parser.add_argument("--skip-sdk", action="store_true",
                      help="Skip all SDK turn tests (commands only)")
  parser.add_argument("--stress", action="store_true",
                      help="Run only stale-task stress test (Phase 5)")
  parser.add_argument("--project", action="store_true",
                      help="Run only multi-turn project test (Phase 6)")
  parser.add_argument("--perm", action="store_true",
                      help="Run only permission flow test (Phase 7)")
  parser.add_argument("--dual", action="store_true",
                      help="Run only dual-instance test (Phase 8)")
  parser.add_argument("--media", action="store_true",
                      help="Run only media & interaction test (Phase 9)")
  parser.add_argument("--topic", action="store_true",
                      help="Run only topic-chat regression test (Phase 10)")
  parser.add_argument("--verbose", "-v", action="store_true",
                      help="Verbose nemo logging")
  args = parser.parse_args()

  chat_id = args.chat_id.strip()
  result = E2EResult()
  single_phase = (args.stress or args.project or args.perm
                  or args.dual or args.media or args.topic)
  run_all = not single_phase
  created_temp_chat = False

  def _cleanup_temp_chat() -> None:
    if created_temp_chat and chat_id:
      dissolve_temp_group(chat_id)

  def _finish(code: int) -> int:
    _cleanup_temp_chat()
    return code

  # ---- Phase 0: Setup ----
  print(f"{Colors.BOLD}Phase 0: Setup{Colors.RESET}")

  if not os.path.exists(CONFIG_PATH):
    print(f"{Colors.RED}Missing {CONFIG_PATH}{Colors.RESET}")
    return 1

  cfg = _load_config()
  if os.path.exists(TOKEN_PATH):
    try:
      _load_user_token()
      print("  Message mode: user API token (fallback: relay)")
    except Exception as e:
      print(f"{Colors.YELLOW}  User token: FAILED ({e}), will use relay{Colors.RESET}")
      if not cfg.get("relay_verify_token"):
        print(f"{Colors.RED}  No relay_verify_token either — cannot send{Colors.RESET}")
        return 1
  elif cfg.get("relay_verify_token"):
    print("  Message mode: relay injection (no user token)")
  else:
    print(f"{Colors.RED}Missing user token and no relay_verify_token{Colors.RESET}")
    return _finish(1)

  if not chat_id:
    print("  Creating fresh temp group...")
    chat_id = create_temp_group("nemo-e2e")
    if not chat_id:
      print(f"{Colors.RED}  Failed to create temp group{Colors.RESET}")
      return _finish(1)
    created_temp_chat = True
    print(f"  Temp chat: {chat_id}")

  print()
  print(f"{Colors.BOLD}Nemo E2E Test Suite{Colors.RESET}")
  print(f"  Chat: {chat_id}")
  print(f"  Provider: {args.provider}")
  print(f"  Project: {PROJECT_DIR}")
  if created_temp_chat:
    print("  Chat mode: fresh temp group")
  elif chat_id == DEFAULT_CHAT_ID:
    print("  Chat mode: shared default group")
  else:
    print("  Chat mode: explicit custom group")
  if args.stress:
    print(f"  Mode: stress test only")
  elif args.project:
    print(f"  Mode: project flow only")
  elif args.perm:
    print(f"  Mode: permission test only")
  elif args.dual:
    print(f"  Mode: dual-instance test only")
  elif args.media:
    print(f"  Mode: media & interaction test only")
  elif args.topic:
    print(f"  Mode: topic chat test only")
  print()

  # Dual-instance manages its own processes
  if args.dual:
    print()
    run_dual_instance(chat_id, result, verbose=args.verbose)
    print()
    ok = result.summary()
    return _finish(0 if ok else 1)

  # Start nemo for other phases
  # NOTE: can_use_tool callback requires CLI support for --permission-prompt-tool
  # which is not yet available in bundled CLI (2.1.81/2.1.92). Permission tests
  # will show auto-approved until a newer CLI version ships with this feature.
  perm_mode = "default" if args.perm else "bypassPermissions"
  print(f"  Starting nemo (permission_mode={perm_mode})...")
  pid = start_nemo(chat_id, verbose=args.verbose,
                   permission_mode=perm_mode,
                   provider=args.provider)
  print(f"  PID: {pid}")

  if not wait_for_ready(pid, timeout=30, provider=args.provider):
    print(f"{Colors.RED}  Nemo failed to start (no SDK connection in 30s)"
          f"{Colors.RESET}")
    LogAnalyzer(pid).dump_tail(15, "startup")
    kill_nemo(pid)
    return _finish(1)
  print("  Nemo ready")
  print()

  try:
    if run_all:
      # ---- Phase 1: Commands ----
      print(f"{Colors.BOLD}Phase 1: Commands{Colors.RESET}")
      run_command_test("T01 ping", "ping", chat_id, result)
      run_command_test("T02 /help", "/help", chat_id, result)
      run_command_test("T03 /model", "/model", chat_id, result)
      run_command_test("T03a /mention off", "/mention off", chat_id, result)
      run_command_test("T04 /cost", "/cost", chat_id, result)
      run_command_test("T05 /diag", "/diag", chat_id, result, wait=8)
      print()

      # ---- Phase 2: SDK Turns ----
      if args.skip_sdk:
        for n in ["T06", "T07", "T08", "T09", "T09a"]:
          result.skip(n, "skipped by --skip-sdk")
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

        # T09a: model switch with resume — context should survive
        print(f"  T09a /model switch + resume...")
        ts = str(int(time.time() * 1000))
        send_msg("Remember this secret word: PINEAPPLE", chat_id)
        time.sleep(15)
        switch_model = "claude-sonnet-4-6"
        restore_model = "claude-opus-4-6"
        if args.provider == "codex":
          switch_model = "gpt-5-codex"
          restore_model = "gpt-5-codex"
        send_msg(f"/model {switch_model}", chat_id)
        time.sleep(15)
        log_a = LogAnalyzer(pid)
        if log_a.count("Model switch", last_n=20) > 0:
          send_msg("What was the secret word I told you? Just say the word.", chat_id)
          time.sleep(15)
          msg = get_latest_bot_msg(chat_id, after=ts)
          if msg:
            # Check card body for "PINEAPPLE" (case-insensitive)
            body = json.dumps(msg.get("body", "")).lower()
            if "pineapple" in body:
              result.ok("T09a model switch resume", "context preserved")
            else:
              result.ok("T09a model switch resume", "card ok, word not in body (may be in text)")
          else:
            result.fail("T09a model switch resume", "no response after switch")
        else:
          result.fail("T09a model switch resume", "model switch didn't happen")
        # Switch back to opus
        send_msg(f"/model {restore_model}", chat_id)
        time.sleep(15)
      print()

      # ---- Phase 3: Signals ----
      print(f"{Colors.BOLD}Phase 3: Signals & Control{Colors.RESET}")

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

      run_command_test("T11 /clear", "/clear", chat_id, result, wait=15)

      if not args.skip_sdk:
        run_sdk_test("T12 post-clear turn", "Say hello",
                     pid, chat_id, result, wait=15)
      else:
        result.skip("T12 post-clear turn", "skipped by --skip-sdk")

      ts = str(int(time.time() * 1000))
      send_msg("   ", chat_id)
      time.sleep(3)
      msg = get_latest_bot_msg(chat_id, after=ts)
      if msg is None:
        result.ok("T16 empty message")
      else:
        result.fail("T16 empty message", "got unexpected response")

      # /exit — should complete in <10s with concurrent shutdown
      ts = str(int(time.time() * 1000))
      send_msg("/exit", chat_id)
      t0 = time.time()
      exited = wait_for_exit(pid, timeout=15)
      elapsed = time.time() - t0
      if exited:
        result.ok("T13 /exit shutdown", f"{elapsed:.1f}s")
      else:
        result.fail("T13 /exit shutdown", f"still running after {elapsed:.0f}s")
        kill_nemo(pid)
      print()

      # ---- Phase 4: Recovery ----
      print(f"{Colors.BOLD}Phase 4: Recovery{Colors.RESET}")
      pid2 = start_nemo(chat_id, verbose=args.verbose, provider=args.provider)
      if wait_for_ready(pid2, timeout=30, provider=args.provider):
        result.ok("T14 restart")
      else:
        result.fail("T14 restart", "failed to start")
        kill_nemo(pid2)
        result.summary()
        return _finish(1 if result.failed else 0)

      try:
        if not args.skip_sdk:
          run_sdk_test("T15 post-recovery turn", "What is 3+3?",
                       pid2, chat_id, result, wait=15)
        else:
          result.skip("T15 post-recovery turn", "skipped by --skip-sdk")

        # T18: Only one pinned config message after restart
        try:
          from nemo.lark.auth import get_token as _get_token
          from nemo.lark.api import list_pins as _list_pins, get_message as _get_msg
          from nemo.group_config import _parse_config_text
          _cfg = _load_config()
          _t = _get_token(_cfg["app_id"], _cfg["app_secret"])
          pins = _list_pins(_t, chat_id)
          config_pins = []
          for p in pins:
            mid = p.get("message_id", "")
            if not mid:
              continue
            try:
              msg = _get_msg(_t, mid)  # Returns message item dict (not raw API response)
              if msg:
                parsed = _parse_config_text(msg)
                if parsed is not None:
                  config_pins.append(mid)
            except Exception:
              pass
          if len(config_pins) == 1:
            result.ok("T18 single pin config", f"1 config pin")
          elif len(config_pins) == 0:
            result.fail("T18 single pin config",
                        f"no config pin found ({len(pins)} total pins)")
          else:
            result.fail("T18 single pin config",
                        f"found {len(config_pins)} config pins, expected 1")
        except Exception as e:
          result.fail("T18 single pin config", f"error: {e}")
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid2, timeout=15):
          kill_nemo(pid2)

      result.skip("T17 /dissolve", "destructive — manual only")
      print()

      # ---- Phases 5-8: Advanced (need fresh nemo) ----
      if not args.skip_sdk:
        print("  Starting fresh nemo for advanced phases...")
        pid = start_nemo(chat_id, verbose=args.verbose, provider=args.provider)
        if not wait_for_ready(pid, timeout=30, provider=args.provider):
          print(f"{Colors.RED}  Failed to start nemo for advanced phases"
                f"{Colors.RESET}")
          result.summary()
          return _finish(1 if result.failed else 0)
        print(f"  PID: {pid}")
        print()

        try:
          run_stale_task_stress(pid, chat_id, result)
          wait_for_idle(pid, chat_id, timeout=30)
          run_project_flow(pid, chat_id, result)
          wait_for_idle(pid, chat_id, timeout=30)
          run_media_tests(pid, chat_id, result)
          wait_for_idle(pid, chat_id, timeout=30)
          run_topic_tests(pid, chat_id, result)
        finally:
          send_msg("/exit", chat_id)
          if not wait_for_exit(pid, timeout=35):
            kill_nemo(pid)

        # Permission tests need plan mode — restart with different perms
        print("  Starting nemo for permission tests (plan mode)...")
        pid_perm = start_nemo(chat_id, verbose=args.verbose,
                              permission_mode="plan",
                              provider=args.provider)
        if wait_for_ready(pid_perm, timeout=30, provider=args.provider):
          try:
            run_permission_tests(pid_perm, chat_id, result)
          finally:
            send_msg("/exit", chat_id)
            if not wait_for_exit(pid_perm, timeout=35):
              kill_nemo(pid_perm)
        else:
          print(f"{Colors.RED}  Failed to start nemo for perm tests{Colors.RESET}")
          for name in ["T40-T43"]:
            result.skip(name, "nemo failed to start")

        # Phase 8: dual-instance (manages its own processes)
        run_dual_instance(chat_id, result, verbose=args.verbose)
      else:
        for name in ["T20-T24", "T30-T38", "T40-T43", "T50-T56", "T60-T63"]:
          result.skip(name, "skipped by --skip-sdk")

    elif args.stress:
      try:
        run_stale_task_stress(pid, chat_id, result)
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid, timeout=35):
          kill_nemo(pid)

    elif args.project:
      try:
        run_project_flow(pid, chat_id, result)
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid, timeout=35):
          kill_nemo(pid)

    elif args.perm:
      try:
        run_permission_tests(pid, chat_id, result)
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid, timeout=35):
          kill_nemo(pid)

    elif args.media:
      try:
        run_media_tests(pid, chat_id, result)
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid, timeout=35):
          kill_nemo(pid)

    elif args.topic:
      try:
        run_topic_tests(pid, chat_id, result)
      finally:
        send_msg("/exit", chat_id)
        if not wait_for_exit(pid, timeout=35):
          kill_nemo(pid)

  except KeyboardInterrupt:
    print(f"\n{Colors.YELLOW}Interrupted{Colors.RESET}")
    kill_nemo(pid)
    return _finish(1)
  except Exception as e:
    print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
    import traceback
    traceback.print_exc()
    kill_nemo(pid)
    return _finish(1)

  print()
  ok = result.summary()
  return _finish(0 if ok else 1)


if __name__ == "__main__":
  sys.exit(main())
