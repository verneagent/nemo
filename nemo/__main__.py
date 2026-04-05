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
  parser.add_argument("--chat-id", required=True, help="Lark chat ID")
  parser.add_argument("--project-dir", required=True, help="Project directory")
  parser.add_argument("--model", default="claude-opus-4-6", help="Model to use")
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

  from .agent import main_loop
  return asyncio.run(main_loop(args.chat_id, project_dir, args.model))


if __name__ == "__main__":
  sys.exit(main() or 0)
