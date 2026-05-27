"""CLI tool for SDK agent to send media to the Lark chat.

Usage (from Bash tool inside a nemo session):
  nemo-send image /path/to/screenshot.png
  nemo-send file /path/to/document.pdf

Reads NEMO_CHAT_ID from environment and credentials from ~/.nemo config.

Inside a `/fork` sub-thread the agent must post media into the fork's thread,
not the main chat. The fork adapter exports NEMO_REPLY_THREAD_FILE pointing at
a file holding the thread's anchor message id (written once Lark assigns the
thread — the SDK subprocess env is frozen before the thread exists, so the
anchor can't ride in on a plain env var). When that anchor is present we send
the media as a reply-in-thread; otherwise it goes to the chat root as before.
"""

from __future__ import annotations

import argparse
import os
import sys


def _reply_thread_anchor() -> str:
  """Return the fork sub-thread's anchor message id, or "" for the main chat.

  Reads NEMO_REPLY_THREAD_FILE (set only for read-only forks). Missing file /
  empty content (e.g. a main-chat turn, or before the thread is bound) → "",
  so media falls back to a plain chat send.
  """
  path = os.environ.get("NEMO_REPLY_THREAD_FILE", "")
  if not path:
    return ""
  try:
    with open(path, encoding="utf-8") as f:
      return f.read().strip()
  except OSError:
    return ""


def main() -> int:
  parser = argparse.ArgumentParser(
    prog="nemo-send",
    description="Send media to the current Lark chat",
  )
  sub = parser.add_subparsers(dest="command")

  img = sub.add_parser("image", help="Send an image")
  img.add_argument("path", help="Path to the image file")

  fil = sub.add_parser("file", help="Send a file")
  fil.add_argument("path", help="Path to the file")

  args = parser.parse_args()
  if not args.command:
    parser.print_help()
    return 1

  chat_id = os.environ.get("NEMO_CHAT_ID", "")
  if not chat_id:
    print("Error: NEMO_CHAT_ID not set", file=sys.stderr)
    return 1

  from .config import load_credentials
  from .lark import api as lark_api
  from .lark.auth import get_token

  credentials = load_credentials()
  if not credentials:
    print("Error: No credentials configured", file=sys.stderr)
    return 1

  token = get_token(credentials["app_id"], credentials["app_secret"])

  path = os.path.abspath(args.path)
  if not os.path.isfile(path):
    print(f"Error: File not found: {path}", file=sys.stderr)
    return 1

  thread_anchor = _reply_thread_anchor()

  if args.command == "image":
    image_key = lark_api.upload_image(token, path)
    if thread_anchor:
      msg_id = lark_api.reply_image(
        token, thread_anchor, image_key, reply_in_thread=True)
    else:
      msg_id = lark_api.send_image(token, chat_id, image_key)
    print(f"Sent image: {msg_id}")
  elif args.command == "file":
    file_key = lark_api.upload_file(token, path)
    if thread_anchor:
      msg_id = lark_api.reply_file(
        token, thread_anchor, file_key, reply_in_thread=True)
    else:
      msg_id = lark_api.send_file(token, chat_id, file_key)
    print(f"Sent file: {msg_id}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
