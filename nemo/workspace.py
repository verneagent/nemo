"""Workspace discovery — resolve chat_id from project directory.

The workspace ID is `{machine}-{folder}` and is stored in the Lark
group description as `workspace:{id}`. This lets nemo auto-discover
which chat to connect to based on the current project directory.
"""

from __future__ import annotations

import logging
import os
import platform
import socket

log = logging.getLogger(__name__)


def get_machine_name() -> str:
  """Get a stable machine identifier."""
  if platform.system() == "Darwin":
    try:
      import plistlib
      plist_path = "/Library/Preferences/SystemConfiguration/preferences.plist"
      with open(plist_path, "rb") as f:
        data = plistlib.load(f)
      name = data.get("System", {}).get("System", {}).get("ComputerName")
      if name:
        return name
    except Exception:
      pass
  return socket.gethostname().split(".")[0]


def get_workspace_id(project_dir: str) -> str:
  """Compute workspace ID: {machine}-{folder}."""
  machine = get_machine_name()
  folder = os.path.abspath(project_dir).replace("/", "-").strip("-")
  return f"{machine}-{folder}"


def _workspace_tag_matches(desc: str, tag: str) -> bool:
  """Check if description contains the exact workspace tag (not a prefix).

  The tag must be followed by whitespace, newline, or end of string.
  This prevents 'workspace:A-B' from matching 'workspace:A-B-C'.
  """
  start = 0
  while True:
    idx = desc.find(tag, start)
    if idx < 0:
      return False
    end = idx + len(tag)
    if end >= len(desc) or desc[end] in (" ", "\n", "\r", "\t"):
      return True
    start = end


def discover_chat_id(token: str, project_dir: str) -> str | None:
  """Find the Lark chat tagged with this project's workspace ID.

  Returns chat_id if found, None otherwise.
  """
  from .lark import api as lark_api

  workspace_id = get_workspace_id(project_dir)
  workspace_tag = f"workspace:{workspace_id}"
  log.info("Looking for workspace tag: %s", workspace_tag)

  try:
    chats = lark_api.list_bot_chats(token)
  except Exception as e:
    log.error("Failed to list bot chats: %s", e)
    return None

  for chat in chats:
    chat_id = chat.get("chat_id", "")
    if not chat_id:
      continue
    try:
      info = lark_api.get_chat_info(token, chat_id)
      desc = info.get("description") or ""
      if _workspace_tag_matches(desc, workspace_tag):
        log.info("Found chat %s (%s)", chat_id, info.get("name", ""))
        return chat_id
    except Exception as e:
      log.debug("Failed to inspect chat %s: %s", chat_id, e)
      continue

  log.warning("No chat found for workspace %s", workspace_id)
  return None
