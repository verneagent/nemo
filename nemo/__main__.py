"""CLI entry point: python -m nemo"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def _ensure_sdk():
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


def main():
  _ensure_sdk()

  parser = argparse.ArgumentParser(
    prog="nemo",
    description="Lark-connected coding agent daemon",
  )
  parser.add_argument("--chat-id", default="", help="Lark chat ID (auto-discovered if omitted)")
  parser.add_argument("--project-dir", default=".", help="Project directory (default: cwd)")
  parser.add_argument("--model", default="claude-opus-4-6", help="Model to use")
  parser.add_argument("--sidecar", action="store_true", default=False,
                      help="Sidecar mode: only respond to @mentions, replies, reactions")
  parser.add_argument("--permission-mode", default="bypassPermissions",
                      choices=["default", "acceptEdits", "plan", "bypassPermissions"],
                      help="SDK permission mode (default: bypassPermissions)")
  parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
  args = parser.parse_args()

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
  fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
  fh.emit = _make_flushing(fh.emit, fh)
  logging.getLogger().addHandler(fh)

  project_dir = os.path.abspath(args.project_dir)
  if not os.path.isdir(project_dir):
    print(f"Error: {project_dir} is not a directory", file=sys.stderr)
    return 1

  from .config import load_credentials
  credentials = load_credentials()
  if not credentials:
    print("Error: No credentials configured (~/.nemo/config.json)", file=sys.stderr)
    return 1

  chat_id = args.chat_id
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

  from .agent import main_loop
  return asyncio.run(main_loop(chat_id, project_dir, args.model,
                               sidecar=args.sidecar,
                               permission_mode=args.permission_mode))


if __name__ == "__main__":
  sys.exit(main() or 0)
