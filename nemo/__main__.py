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
  parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="[nemo] [%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
  )

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
    # Auto-discover chat from workspace tag in group descriptions
    from .lark.auth import get_token
    from .workspace import discover_chat_id
    token = get_token(credentials["app_id"], credentials["app_secret"])
    chat_id = discover_chat_id(token, project_dir)
    if not chat_id:
      from .workspace import get_workspace_id
      ws_id = get_workspace_id(project_dir)
      print(f"Error: No Lark group found for workspace: {ws_id}", file=sys.stderr)
      print("Create a group with this in the description:", file=sys.stderr)
      print(f"  workspace:{ws_id}", file=sys.stderr)
      return 1

  # Preflight checks
  from .preflight import run_preflight
  preflight_errors = run_preflight(credentials, chat_id)
  if preflight_errors:
    for err in preflight_errors:
      print(f"Preflight error: {err}", file=sys.stderr)
    return 1

  from .agent import main_loop
  return asyncio.run(main_loop(chat_id, project_dir, args.model,
                               sidecar=args.sidecar))


if __name__ == "__main__":
  sys.exit(main() or 0)
