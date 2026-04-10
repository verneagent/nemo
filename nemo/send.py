"""CLI tool for SDK agent to send media to the Lark chat.

Usage (from Bash tool inside a nemo session):
  nemo-send image /path/to/screenshot.png
  nemo-send file /path/to/document.pdf

Reads NEMO_CHAT_ID from environment and credentials from ~/.nemo config.
"""

from __future__ import annotations

import argparse
import os
import sys


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

  if args.command == "image":
    image_key = lark_api.upload_image(token, path)
    msg_id = lark_api.send_image(token, chat_id, image_key)
    print(f"Sent image: {msg_id}")
  elif args.command == "file":
    file_key = lark_api.upload_file(token, path)
    msg_id = lark_api.send_file(token, chat_id, file_key)
    print(f"Sent file: {msg_id}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
