"""CLI entry point: python -m nemo

Nemo always runs in the foreground. The caller is responsible for
detaching the process (tmux / `setsid nohup ... </dev/null >/dev/null 2>&1 &`).
We used to double-fork here, but the parent-set env marker leaked into
every subprocess — including bash invocations inside the Claude SDK —
and caused child nemos launched from those contexts to silently skip
daemonization and die with their spawning subprocess.
"""

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
    pass  # SDK not on this Python, try others
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
    except Exception as e:
      print(f"  Skipping {candidate}: {e}", file=sys.stderr)
      continue
  print("Error: claude_agent_sdk not found.", file=sys.stderr)
  sys.exit(1)


def _ensure_codex_cli() -> None:
  """Ensure the local codex CLI is available."""
  if shutil.which("codex"):
    return
  print("Error: codex CLI not found in PATH.", file=sys.stderr)
  sys.exit(1)


def _ensure_opencode_cli() -> None:
  """Ensure the local opencode CLI is available."""
  if shutil.which("opencode"):
    return
  print("Error: opencode CLI not found in PATH.", file=sys.stderr)
  sys.exit(1)


def _ensure_agent_runtime(agent: str) -> None:
  if agent == "claude":
    _ensure_claude_sdk()
    return
  if agent == "codex":
    _ensure_codex_cli()
    return
  if agent == "opencode":
    _ensure_opencode_cli()
    return
  print(f"Error: unsupported agent '{agent}'", file=sys.stderr)
  sys.exit(1)


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
      continue  # skip non-numeric PID
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
      pass  # log file missing or unreadable

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


def _get_version() -> str:
  """Return the installed captain-nemo version, or 'unknown' if unavailable."""
  try:
    from importlib.metadata import version, PackageNotFoundError
    try:
      return version("captain-nemo")
    except PackageNotFoundError:
      return "unknown"
  except Exception:
    return "unknown"


def main():
  # Intercept subcommands before argparse
  if len(sys.argv) >= 2 and sys.argv[1] == "list":
    sys.exit(_cmd_list())

  def _validate_chat_id(value: str) -> str:
    """Reject obviously-invalid Lark chat IDs at parse time.

    Lark group chat IDs always start with ``oc_`` followed by hex.
    Without this check, a typo like ``--chat-id 0`` propagates into
    workspace eviction logic — see _cmdline_targets_chat in workspace.py
    for the full failure mode (a stray chat_id="0" daemon SIGTERMs
    every other nemo on the host).
    """
    if not value:
      return value  # empty = auto-discover
    if not (value.startswith("oc_") and len(value) > 3):
      raise argparse.ArgumentTypeError(
        f"Invalid Lark chat_id {value!r}: must start with 'oc_' "
        f"(e.g. oc_da5004fb44ea33ce72ed90aabe2ab9dfe)"
      )
    return value

  parser = argparse.ArgumentParser(
    prog="nemo",
    description="Lark-connected coding agent daemon",
  )
  parser.add_argument("--version", "-V", action="version",
                      version=f"nemo {_get_version()}")
  parser.add_argument("--chat-id", default="", type=_validate_chat_id,
                      help="Lark chat ID (auto-discovered if omitted)")
  parser.add_argument("--chat-name", default="", help="Find chat by name substring")
  parser.add_argument("--project-dir", default=".", help="Project directory (default: cwd)")
  parser.add_argument("--agent", default="claude", choices=["claude", "codex", "opencode"],
                      help="Coding agent runtime (default: claude)")
  parser.add_argument("--model", default="", help="Model to use (agent default if omitted)")
  parser.add_argument("--effort", default="", choices=["", "low", "medium", "high", "max"],
                      help="Reasoning effort for the coding agent (default: agent default)")
  parser.add_argument("--profile", default="default", help="Config profile name (default: default)")
  parser.add_argument("--permission-mode", default="bypassPermissions",
                      choices=["default", "acceptEdits", "plan", "bypassPermissions"],
                      help="SDK permission mode (default: bypassPermissions)")
  parser.add_argument("--system-prompt-file", default="",
                      help="Path to a file whose contents are appended to the "
                           "agent's system prompt")
  parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
  args = parser.parse_args()

  system_prompt = ""
  if args.system_prompt_file:
    sp_path = os.path.expanduser(args.system_prompt_file)
    if not os.path.isfile(sp_path):
      print(f"Error: --system-prompt-file not found: {sp_path}", file=sys.stderr)
      return 1
    try:
      with open(sp_path, encoding="utf-8") as f:
        system_prompt = f.read().strip()
    except OSError as e:
      print(f"Error: cannot read --system-prompt-file: {e}", file=sys.stderr)
      return 1
    if not system_prompt:
      print(f"Error: --system-prompt-file is empty: {sp_path}", file=sys.stderr)
      return 1

  _ensure_agent_runtime(args.agent)

  # Set active profile before any config loading
  from .config import set_profile, profile_path
  set_profile(args.profile)

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

  # Beyond this point every failure path must reach the per-PID log file.
  # Background launchers (e.g. `nemobg`) redirect stderr to /dev/null, so
  # any error that only goes through `print(..., file=sys.stderr)` leaves
  # the daemon "didn't start" with a 0-byte log and no clue. Use
  # _startup_fail to dual-write each early failure: log.error (per-PID
  # file) + stderr (foreground users). The try/except below additionally
  # catches uncaught exceptions and routes them through the same logger
  # so tracebacks land in the file too.
  log = logging.getLogger("nemo")

  def _startup_fail(message: str) -> int:
    log.error("%s", message)
    print(message, file=sys.stderr)
    return 1

  try:
    project_dir = os.path.abspath(args.project_dir)
    from .agent_factory import default_model_for_agent
    model = args.model or default_model_for_agent(args.agent)
    if not os.path.isdir(project_dir):
      return _startup_fail(f"Error: {project_dir} is not a directory")

    from .config import load_credentials
    credentials = load_credentials()
    if not credentials:
      p = profile_path()
      return _startup_fail(f"Error: No credentials configured ({p})")

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
        return _startup_fail(f"Error: No chat found matching '{args.chat_name}'")
      if len(matches) > 1:
        match_lines = "\n".join(
          f"  {c['chat_id']}  {c.get('name', '')}" for c in matches)
        return _startup_fail(
          f"Multiple chats match '{args.chat_name}':\n{match_lines}")
      chat_id = matches[0]["chat_id"]
      log.info("Found chat: %s (%s)", chat_id, matches[0].get("name", ""))

    if not chat_id:
      # Auto-discover idle chat or create a new one
      from .lark.auth import get_token
      from .workspace import discover_or_create_chat
      token = get_token(credentials["app_id"], credentials["app_secret"])
      chat_id = discover_or_create_chat(token, project_dir,
                                        email=credentials.get("email", ""))
      if not chat_id:
        return _startup_fail("Error: Failed to find or create Lark group")
      log.info("Using chat: %s", chat_id)

    # Preflight checks
    from .preflight import run_preflight
    preflight_errors = run_preflight(credentials, chat_id)
    if preflight_errors:
      for err in preflight_errors:
        _startup_fail(f"Preflight error: {err}")
      return 1
  except Exception as e:
    log.error("Startup failed: %s: %s", type(e).__name__, e, exc_info=True)
    print(f"Error: nemo startup failed: {type(e).__name__}: {e}",
          file=sys.stderr)
    return 1

  # Crash diagnostics (faulthandler, signal logging, watchdog heartbeat)
  _setup_crash_diagnostics(log_path)

  # Resolve --model against the preset registry. If it matches, expand
  # to (endpoint, remote_model) so downstream code sees both the
  # routing config and the wire-format model id without the user having
  # to thread three flags. Unknown models pass through unchanged — they
  # might be a raw model slug like "claude-opus-4-7" or a custom
  # model the user added to ~/.nemo/models.json.
  from .coding_agent import EndpointConfig
  from .presets import resolve_preset
  endpoint = EndpointConfig()
  endpoint_key = ""
  preset = resolve_preset(model)
  if preset is not None:
    if not preset.supports(args.agent):
      return _startup_fail(
        f"Error: model {model!r} has no endpoint configured for "
        f"--agent {args.agent}. Add the protocol block to its provider "
        f"in ~/.nemo/models.json or pick a different agent.")
    if preset.api_key_env and not os.environ.get(preset.api_key_env):
      return _startup_fail(
        f"Error: model {model!r} requires ${preset.api_key_env}, "
        f"which is unset. Export it (e.g. `export {preset.api_key_env}=...`) "
        f"and re-run.")
    endpoint = preset.endpoint_for(args.agent)
    model = preset.remote_for(args.agent)
    # Key by the upstream URL, not the preset name — two presets that
    # hit the same gateway share signing keys and can safely share an
    # SDK session, while two presets at different URLs cannot.
    endpoint_key = endpoint.base_url
    log.info("Resolved preset %s → endpoint=%s model=%s",
             preset.name, endpoint.base_url, model)

  from .agent import main_loop
  try:
    return asyncio.run(main_loop(chat_id, project_dir, model,
                                 agent=args.agent,
                                 permission_mode=args.permission_mode,
                                 effort=args.effort,
                                 system_prompt=system_prompt,
                                 endpoint=endpoint,
                                 endpoint_key=endpoint_key))
  except KeyboardInterrupt:
    return 0
  except BaseException as e:
    logging.getLogger("nemo").error("Fatal: %s: %s", type(e).__name__, e, exc_info=True)
    return 1


if __name__ == "__main__":
  sys.exit(main() or 0)
