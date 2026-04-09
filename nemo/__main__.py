"""CLI entry point: python -m nemo"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import logging
import os
import signal
import shutil
import sys
import threading


def _ensure_claude_sdk():
  """Ensure we're running on a Python with claude_agent_sdk."""
  try:
    import claude_agent_sdk  # noqa: F401
    return
  except ImportError:
    pass
  import glob
  import subprocess
  candidates = []
  for d in os.environ.get("PATH", "").split(os.pathsep):
    candidates.extend(sorted(glob.glob(os.path.join(d, "python3.[0-9]*")), reverse=True))
  for fallback in ["/opt/homebrew/bin/python3", "/usr/local/bin/python3"]:
    if fallback not in candidates:
      candidates.append(fallback)
  for candidate in candidates:
    if not candidate or candidate == sys.executable:
      continue
    if not os.path.isfile(candidate):
      continue
    try:
      rc = subprocess.call(
        [candidate, "-c", "import claude_agent_sdk"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      )
      if rc == 0:
        os.execv(candidate, [candidate, "-m", "nemo"] + sys.argv[1:])
    except Exception:
      continue
  print("Error: claude_agent_sdk not found.", file=sys.stderr)
  sys.exit(1)


def _ensure_codex_cli() -> None:
  """Ensure the local codex CLI is available."""
  if shutil.which("codex"):
    return
  print("Error: codex CLI not found in PATH.", file=sys.stderr)
  sys.exit(1)


def _ensure_provider_runtime(provider: str) -> None:
  if provider == "claude":
    _ensure_claude_sdk()
    return
  if provider == "codex":
    _ensure_codex_cli()
    return
  print(f"Error: unsupported provider '{provider}'", file=sys.stderr)
  sys.exit(1)


_ready_fd: int | None = None  # write-end of readiness pipe (grandchild only)


def signal_ready() -> None:
  """Signal the waiting parent that the daemon is ready."""
  global _ready_fd
  if _ready_fd is not None:
    try:
      os.write(_ready_fd, f"ready:{os.getpid()}\n".encode())
    except OSError:
      pass  # Parent already exited — pipe broken, harmless
    try:
      os.close(_ready_fd)
    except OSError:
      pass
    _ready_fd = None


def signal_error(msg: str) -> None:
  """Signal the waiting parent that startup failed."""
  global _ready_fd
  if _ready_fd is not None:
    try:
      os.write(_ready_fd, f"error:{msg}\n".encode())
    except OSError:
      pass
    try:
      os.close(_ready_fd)
    except OSError:
      pass
    _ready_fd = None


def _daemonize():
  """Double-fork into background, fully detach from parent process tree.

  Single fork + setsid is not enough: the Claude SDK Bash tool tracks the
  child PID and kills it when the turn ends. Double-fork ensures the
  actual daemon (grandchild) is reparented to PID 1 and invisible to any
  parent process tree cleanup.

  The parent blocks until the daemon signals readiness (after the start
  card is sent) or the pipe closes on error.
  """
  # Pipe to communicate readiness back to original parent
  r_fd, w_fd = os.pipe()

  # First fork
  pid = os.fork()
  if pid > 0:
    # Original parent — wait for readiness signal from daemon
    os.close(w_fd)
    data = b""
    while True:
      chunk = os.read(r_fd, 256)
      if not chunk:
        break
      data += chunk
      if b"\n" in data:
        break
    os.close(r_fd)
    os.waitpid(pid, 0)
    msg = data.decode().strip()
    if msg.startswith("ready:"):
      daemon_pid = msg.split(":", 1)[1]
      print(f"nemo started (PID {daemon_pid})", file=sys.stderr)
      sys.exit(0)
    else:
      print("nemo failed to start", file=sys.stderr)
      if msg:
        print(msg, file=sys.stderr)
      sys.exit(1)

  # Intermediate child — new session, fork again, exit immediately
  os.close(r_fd)
  os.setsid()

  pid2 = os.fork()
  if pid2 > 0:
    os.close(w_fd)
    os._exit(0)

  # Grandchild — the actual daemon; keep w_fd for readiness signal
  global _ready_fd
  _ready_fd = w_fd
  devnull = os.open(os.devnull, os.O_RDWR)
  os.dup2(devnull, 0)
  os.dup2(devnull, 1)
  os.dup2(devnull, 2)
  os.close(devnull)
  os.environ["NEMO_FOREGROUND"] = "1"


def _setup_crash_diagnostics(log_path: str) -> None:
  """Set up crash diagnostics: faulthandler, signal logging, watchdog.

  Helps diagnose silent daemon deaths (SIGKILL, segfault, os._exit).
  """
  log = logging.getLogger("nemo")

  # faulthandler: writes traceback on SIGSEGV/SIGABRT/SIGBUS
  fh = open(log_path, "a")
  faulthandler.enable(file=fh, all_threads=True)
  # SIGUSR1: dump all thread tracebacks on demand (kill -USR1 <pid>)
  faulthandler.register(signal.SIGUSR1, file=fh, all_threads=True)

  # Log all catchable signals to detect what kills us
  for sname in ("SIGHUP", "SIGTERM", "SIGINT", "SIGUSR1", "SIGUSR2",
                "SIGPIPE", "SIGALRM", "SIGXCPU", "SIGXFSZ", "SIGVTALRM",
                "SIGPROF"):
    snum = getattr(signal, sname, None)
    if snum is None:
      continue
    def _make_handler(name, num):
      def _handler(signum, frame):
        log.warning("Signal %s (%d) received", name, num)
        if name in ("SIGTERM", "SIGHUP"):
          signal.signal(num, signal.SIG_DFL)
          os.kill(os.getpid(), num)
      return _handler
    signal.signal(snum, _make_handler(sname, snum))

  # Watchdog: periodic heartbeat with thread/child status
  import subprocess as _sp

  def _watchdog():
    import time
    while True:
      time.sleep(60)
      threads = [t.name for t in threading.enumerate()]
      try:
        result = _sp.run(
          ["pgrep", "-P", str(os.getpid())],
          capture_output=True, text=True, timeout=5,
        )
        children = result.stdout.strip() or "none"
      except Exception:
        children = "?"
      log.info("heartbeat pid=%d threads=%s children=%s",
               os.getpid(), threads, children)

  wd = threading.Thread(target=_watchdog, daemon=True, name="watchdog")
  wd.start()


def _cmd_list() -> int:
  """List all running nemo processes with their chat and project info."""
  import re
  import subprocess

  from .config import CONFIG_DIR, load_credentials
  from .lark.auth import get_token
  from .lark import api as lark_api

  # 1. Find all nemo daemon processes
  my_pid = os.getpid()
  try:
    result = subprocess.run(
      ["ps", "ax", "-o", "pid=,command="],
      capture_output=True, text=True, timeout=5,
    )
  except Exception as e:
    print(f"Error listing processes: {e}", file=sys.stderr)
    return 1

  nemo_procs: list[dict[str, str]] = []
  for line in result.stdout.splitlines():
    line = line.strip()
    if not line:
      continue
    parts = line.split(None, 1)
    if len(parts) < 2:
      continue
    try:
      pid = int(parts[0])
    except ValueError:
      continue
    cmd = parts[1]
    if pid == my_pid:
      continue
    # Match: command line contains /bin/nemo or /nemo as an argument
    if "/nemo" not in cmd:
      continue
    # Exclude grep, vim, etc. that happen to have 'nemo' in args
    if not any("/nemo" in part for part in cmd.split()[:3]):
      continue
    chat_match = re.search(r"--chat-id\s+(\S+)", cmd)
    nemo_procs.append({
      "pid": str(pid),
      "chat_id": chat_match.group(1) if chat_match else "",
    })

  # 2. For processes without --chat-id, scan their log files
  log_dir = os.path.join(CONFIG_DIR, "logs")
  for proc in nemo_procs:
    if proc["chat_id"]:
      continue
    log_path = os.path.join(log_dir, f"nemo-{proc['pid']}.log")
    try:
      with open(log_path) as f:
        for line in f:
          m = re.search(r"chat[=:](oc_[a-f0-9]{32})", line)
          if m:
            proc["chat_id"] = m.group(1)
            break
    except OSError:
      pass

  # Deduplicate by chat_id (parent + daemon may both appear)
  seen_chats: set[str] = set()
  unique_procs: list[dict[str, str]] = []
  for proc in nemo_procs:
    chat = proc["chat_id"]
    if chat and chat in seen_chats:
      continue
    if chat:
      seen_chats.add(chat)
    unique_procs.append(proc)

  if not unique_procs:
    print("No running nemo processes found.")
    return 0

  # 3. Resolve chat names and workspace tags via Lark API
  credentials = load_credentials()
  chat_info: dict[str, dict[str, str]] = {}
  if credentials:
    try:
      token = get_token(credentials["app_id"], credentials["app_secret"])
      for proc in unique_procs:
        chat = proc["chat_id"]
        if not chat or chat in chat_info:
          continue
        try:
          info = lark_api.get_chat_info(token, chat)
          name = str(info.get("name", "") or "")
          desc = str(info.get("description", "") or "")
          # Extract project path from workspace tag
          # New format: workspace:{machine}|{path}
          # Legacy format: workspace:{machine}-{path-with-dashes}
          project = ""
          wm = re.search(r"workspace:[^|\s]+\|(.+?)(?:\s|$)", desc)
          if wm:
            project = wm.group(1)
          else:
            wm = re.search(r"workspace:\S+?-((?:Users|home)-.+?)(?:\s|$)", desc)
            if wm:
              project = "/" + wm.group(1).replace("-", "/")
          chat_info[chat] = {"name": name, "project": project}
        except Exception:
          chat_info[chat] = {"name": "?", "project": "?"}
    except Exception as e:
      print(f"(Lark API unavailable: {e})", file=sys.stderr)

  # 4. Print table
  print(f"{'PID':<8} {'Chat':<25} {'Project'}")
  print("-" * 70)
  for proc in unique_procs:
    pid = proc["pid"]
    chat = proc["chat_id"] or "?"
    info = chat_info.get(chat, {})
    name = info.get("name", "")
    project = info.get("project", "")
    label = name or chat[:25]
    print(f"{pid:<8} {label:<25} {project}")

  return 0


def main():
  # Intercept subcommands before argparse
  if len(sys.argv) >= 2 and sys.argv[1] == "list":
    sys.exit(_cmd_list())

  parser = argparse.ArgumentParser(
    prog="nemo",
    description="Lark-connected coding agent daemon",
  )
  parser.add_argument("--chat-id", default="", help="Lark chat ID (auto-discovered if omitted)")
  parser.add_argument("--chat-name", default="", help="Find chat by name substring")
  parser.add_argument("--project-dir", default=".", help="Project directory (default: cwd)")
  parser.add_argument("--provider", default="claude", choices=["claude", "codex"],
                      help="Coding agent provider (default: claude)")
  parser.add_argument("--model", default="", help="Model to use (provider default if omitted)")
  parser.add_argument("--profile", default="default", help="Config profile name (default: default)")
  parser.add_argument("--permission-mode", default="bypassPermissions",
                      choices=["default", "acceptEdits", "plan", "bypassPermissions"],
                      help="SDK permission mode (default: bypassPermissions)")
  parser.add_argument("--foreground", "-f", action="store_true",
                      help="Run in foreground (default: daemonize)")
  parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
  args = parser.parse_args()

  _ensure_provider_runtime(args.provider)

  # Set active profile before any config loading
  from .config import set_profile, profile_path
  set_profile(args.profile)

  # Daemonize unless --foreground
  if not args.foreground and not os.environ.get("NEMO_FOREGROUND"):
    _daemonize()

  # Ignore SIGPIPE — broken pipe from parent exit is harmless
  signal.signal(signal.SIGPIPE, signal.SIG_IGN)

  # Log to both stderr and a persistent log file
  log_level = logging.DEBUG if args.verbose else logging.INFO
  log_format = "[nemo] [%(asctime)s] %(message)s"
  log_datefmt = "%H:%M:%S"
  logging.basicConfig(
    level=log_level,
    format=log_format,
    datefmt=log_datefmt,
  )
  # Flush all handlers after every log line — Python buffers stderr when
  # redirected to a file, making log-based monitoring unreliable.
  for h in logging.getLogger().handlers:
    _orig = h.emit
    def _make_flushing(orig, handler):
      def _flushing_emit(record):
        orig(record)
        handler.flush()
      return _flushing_emit
    h.emit = _make_flushing(_orig, h)

  # Per-process log file (created immediately at startup)
  from .config import CONFIG_DIR
  from logging.handlers import RotatingFileHandler
  log_dir = os.path.join(CONFIG_DIR, "logs")
  os.makedirs(log_dir, exist_ok=True)
  log_path = os.path.join(log_dir, f"nemo-{os.getpid()}.log")
  rfh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
  rfh.setLevel(log_level)
  rfh.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
  rfh.emit = _make_flushing(rfh.emit, rfh)
  logging.getLogger().addHandler(rfh)

  project_dir = os.path.abspath(args.project_dir)
  from .agent_factory import default_model_for_provider
  model = args.model or default_model_for_provider(args.provider)
  if not os.path.isdir(project_dir):
    print(f"Error: {project_dir} is not a directory", file=sys.stderr)
    return 1

  from .config import load_credentials
  credentials = load_credentials()
  if not credentials:
    p = profile_path()
    print(f"Error: No credentials configured ({p})", file=sys.stderr)
    return 1

  chat_id = args.chat_id

  # --chat-name: search by name substring
  if not chat_id and args.chat_name:
    from .lark.auth import get_token
    from .lark import api as lark_api
    token = get_token(credentials["app_id"], credentials["app_secret"])
    chats = lark_api.list_bot_chats(token)
    query = args.chat_name.lower()
    matches = [c for c in chats if query in (c.get("name") or "").lower()]
    if not matches:
      print(f"Error: No chat found matching '{args.chat_name}'", file=sys.stderr)
      return 1
    if len(matches) > 1:
      print(f"Multiple chats match '{args.chat_name}':", file=sys.stderr)
      for c in matches:
        print(f"  {c['chat_id']}  {c.get('name', '')}", file=sys.stderr)
      return 1
    chat_id = matches[0]["chat_id"]
    logging.getLogger("nemo").info("Found chat: %s (%s)", chat_id, matches[0].get("name", ""))

  if not chat_id:
    # Auto-discover idle chat or create a new one
    from .lark.auth import get_token
    from .workspace import discover_or_create_chat
    token = get_token(credentials["app_id"], credentials["app_secret"])
    chat_id = discover_or_create_chat(token, project_dir,
                                      email=credentials.get("email", ""))
    if not chat_id:
      print("Error: Failed to find or create Lark group", file=sys.stderr)
      return 1
    logging.getLogger("nemo").info("Using chat: %s", chat_id)

  # Preflight checks
  from .preflight import run_preflight
  preflight_errors = run_preflight(credentials, chat_id)
  if preflight_errors:
    for err in preflight_errors:
      print(f"Preflight error: {err}", file=sys.stderr)
    return 1

  # Crash diagnostics (faulthandler, signal logging, watchdog heartbeat)
  _setup_crash_diagnostics(log_path)

  from .agent import main_loop
  try:
    return asyncio.run(main_loop(chat_id, project_dir, model,
                                 provider=args.provider,
                                 permission_mode=args.permission_mode))
  except KeyboardInterrupt:
    return 0
  except BaseException as e:
    logging.getLogger("nemo").error("Fatal: %s: %s", type(e).__name__, e, exc_info=True)
    return 1


if __name__ == "__main__":
  sys.exit(main() or 0)
