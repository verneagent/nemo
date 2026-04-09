"""Workspace discovery — resolve chat_id from project directory.

The workspace ID is `{machine}-{folder}` and is stored in the Lark
group description as `workspace:{id}`. This lets nemo auto-discover
which chat to connect to based on the current project directory.

Groups are tracked as idle/occupied via relay heartbeat (cross-device)
or local process-table scan (fallback). On startup nemo picks an idle
group (or creates one).
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
  """Compute workspace ID: {machine}|{abspath}."""
  machine = get_machine_name()
  return f"{machine}|{os.path.abspath(project_dir)}"


def _get_legacy_workspace_id(project_dir: str) -> str:
  """Old format for backward compat: {machine}-{folder-with-dashes}."""
  machine = get_machine_name()
  folder = os.path.abspath(project_dir).replace("/", "-").strip("-")
  return f"{machine}-{folder}"


def _workspace_tag_matches(desc: str, tag: str) -> bool:
  """Check if description contains the exact workspace tag.

  The tag must be followed by whitespace, newline, or end of string.
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


def _is_pid_alive(pid: int) -> bool:
  """Check if a nemo process with the given PID is still running.

  Not just os.kill(pid, 0) — also verify the process is actually nemo,
  not an unrelated process that reused the PID after nemo was killed.
  """
  try:
    os.kill(pid, 0)
  except (OSError, ProcessLookupError):
    return False
  # PID exists — verify it's a nemo process
  try:
    import subprocess
    result = subprocess.run(
      ["ps", "-p", str(pid), "-o", "command="],
      capture_output=True, text=True, timeout=5,
    )
    cmdline = result.stdout.strip()
    return "nemo" in cmdline
  except Exception:
    return True  # Can't verify — assume alive to be safe


def _find_local_nemo_pids(chat_id: str) -> list[int]:
  """Find local nemo processes targeting the given chat_id.

  Scans the process table for nemo commands containing --chat-id <chat_id>.
  Returns PIDs excluding the current process.
  """
  import subprocess

  my_pid = os.getpid()
  pids: list[int] = []
  try:
    result = subprocess.run(
      ["ps", "ax", "-o", "pid=,command="],
      capture_output=True, text=True, timeout=5,
    )
    for line in result.stdout.splitlines():
      line = line.strip()
      if not line:
        continue
      if "nemo" not in line or chat_id not in line:
        continue
      parts = line.split(None, 1)
      if len(parts) < 2:
        continue
      try:
        pid = int(parts[0])
      except ValueError:
        continue
      if pid == my_pid:
        continue
      cmd = parts[1]
      if "--chat-id" in cmd and chat_id in cmd:
        pids.append(pid)
  except Exception:
    pass
  return pids


def _find_workspace_groups(token: str, project_dir: str) -> list[dict[str, str]]:
  """Find all Lark groups tagged with this project's workspace ID.

  Matches both new format (workspace:{machine}|{path}) and legacy
  format (workspace:{machine}-{folder-dashes}) for backward compat.
  Returns list of {"chat_id": ..., "name": ...}.
  """
  from .lark import api as lark_api

  workspace_tag = f"workspace:{get_workspace_id(project_dir)}"
  legacy_tag = f"workspace:{_get_legacy_workspace_id(project_dir)}"
  log.info("Looking for workspace tag: %s", workspace_tag)

  try:
    chats = lark_api.list_bot_chats(token)
  except Exception as e:
    log.error("Failed to list bot chats: %s", e)
    return []

  matches: list[dict[str, str]] = []
  for chat in chats:
    chat_id = chat.get("chat_id", "")
    if not chat_id:
      continue
    try:
      info = lark_api.get_chat_info(token, chat_id)
      desc = info.get("description") or ""
      if _workspace_tag_matches(desc, workspace_tag) or \
         _workspace_tag_matches(desc, legacy_tag):
        matches.append({"chat_id": chat_id, "name": info.get("name", "")})
    except Exception as e:
      log.debug("Failed to inspect chat %s: %s", chat_id, e)
      continue
  return matches


def _is_group_idle(token: str, chat_id: str) -> bool:
  """Check if a group is idle (no active nemo process occupying it).

  Strategy:
  1. If relay is configured, use heartbeat (works cross-device).
  2. Fall back to local process-table scan.
  """
  from .config import load_relay_config

  relay_url, _ = load_relay_config()
  if relay_url:
    from . import relay as relay_client
    alive = relay_client.is_alive(chat_id)
    log.debug("Relay heartbeat for %s: alive=%s", chat_id, alive)
    return not alive

  # Fallback: scan local process table for nemo targeting this chat
  pids = _find_local_nemo_pids(chat_id)
  return len(pids) == 0


def discover_chat_id(token: str, project_dir: str) -> str | None:
  """Find an idle Lark chat tagged with this project's workspace ID.

  Scans all matching groups, returns the first idle one (no active PID).
  Returns chat_id if found, None otherwise.
  """
  groups = _find_workspace_groups(token, project_dir)
  for g in groups:
    chat_id = g["chat_id"]
    if _is_group_idle(token, chat_id):
      log.info("Found idle chat %s (%s)", chat_id, g.get("name", ""))
      return chat_id
    else:
      log.info("Chat %s (%s) is occupied, skipping", chat_id, g.get("name", ""))
  if groups:
    log.info("All %d groups are occupied", len(groups))
  return None


def auto_create_chat(token: str, project_dir: str,
                     email: str = "",
                     existing_names: list[str] | None = None) -> str | None:
  """Create a new Lark group for this workspace.

  Creates the group, adds the operator (by email), writes workspace tag,
  and pins a default config. Returns the new chat_id or None on failure.
  """
  from .lark import api as lark_api

  workspace_id = get_workspace_id(project_dir)
  workspace_tag = f"workspace:{workspace_id}"
  folder_name = os.path.basename(os.path.abspath(project_dir))
  machine = get_machine_name()
  group_name = _compute_group_name(folder_name, machine, existing_names or [])

  # Create group (bot is automatically a member as creator)
  try:
    chat_id = lark_api.create_chat(token, group_name, description=workspace_tag)
    log.info("Created group %s (%s)", chat_id, group_name)
  except Exception as e:
    log.error("Failed to create group: %s", e)
    return None

  # Set group avatar
  try:
    avatar_path = os.path.join(os.path.dirname(__file__), "assets", "avatar.png")
    if os.path.isfile(avatar_path):
      image_key = lark_api.upload_image(token, avatar_path, image_type="avatar")
      lark_api.update_chat_info(token, chat_id, {"avatar": image_key})
  except Exception as e:
    log.warning("Failed to set group avatar: %s", e)

  # Add operator by email
  if email:
    try:
      open_id = lark_api.lookup_open_id_by_email(token, email)
      if open_id:
        lark_api.add_chat_members(token, chat_id, [open_id])
      else:
        log.warning("Could not resolve email %s to open_id", email)
    except Exception as e:
      log.warning("Failed to add operator to group: %s", e)

  # Pin default config
  try:
    from .group_config import DEFAULT_CONFIG, save_config
    save_config(token, chat_id, dict(DEFAULT_CONFIG))
    log.info("Pinned default config card in %s", chat_id)
  except Exception as e:
    log.warning("Failed to pin config card: %s", e)

  return chat_id


def _compute_group_name(folder_name: str, machine: str,
                        existing_names: list[str]) -> str:
  """Compute a numbered group name: foo@machine, foo2@machine, etc."""
  base = f"{folder_name}@{machine}"
  if base not in existing_names:
    return base
  # Find highest suffix
  max_n = 1
  for name in existing_names:
    if name == base:
      continue
    prefix = folder_name
    suffix = f"@{machine}"
    if name.startswith(prefix) and name.endswith(suffix):
      mid = name[len(prefix):-len(suffix)]
      try:
        n = int(mid)
        max_n = max(max_n, n)
      except ValueError:
        pass
  return f"{folder_name}{max_n + 1}@{machine}"


def discover_or_create_chat(token: str, project_dir: str,
                            email: str = "") -> str | None:
  """Find an idle group or create a new one.

  1. Find all groups matching this workspace tag
  2. Pick the first idle one (no active PID)
  3. If all occupied or none exist, create a new group
  """
  groups = _find_workspace_groups(token, project_dir)
  for g in groups:
    chat_id = g["chat_id"]
    if _is_group_idle(token, chat_id):
      log.info("Reusing idle chat %s (%s)", chat_id, g.get("name", ""))
      return chat_id
    else:
      log.info("Chat %s (%s) is occupied, skipping", chat_id, g.get("name", ""))

  # All occupied or none found → create new
  existing_names = [g.get("name", "") for g in groups]
  log.info("Creating new group (existing: %d, all occupied)", len(groups))
  return auto_create_chat(token, project_dir, email=email,
                          existing_names=existing_names)


def evict_existing(token: str, chat_id: str) -> None:
  """If another nemo process occupies this group, stop it before we start.

  Checks local process table and relay heartbeat. For local processes,
  sends SIGTERM. For remote (relay-only), sends a stop signal via relay.
  """
  from .config import load_relay_config

  # 1. Check local process table for nemo targeting this chat
  local_pids = _find_local_nemo_pids(chat_id)
  for old_pid in local_pids:
    if _is_pid_alive(old_pid):
      log.info("Stopping existing nemo process (pid=%d)", old_pid)
      import signal as _signal
      try:
        os.kill(old_pid, _signal.SIGTERM)
        import time
        for _ in range(30):
          if not _is_pid_alive(old_pid):
            log.info("Previous process (pid=%d) stopped", old_pid)
            break
          time.sleep(0.5)
        else:
          log.warning("Previous process (pid=%d) did not exit, killing", old_pid)
          os.kill(old_pid, _signal.SIGKILL)
      except OSError:
        pass  # Already exited

  # 2. Check relay heartbeat (covers remote processes)
  relay_url, _ = load_relay_config()
  if relay_url:
    from . import relay as relay_client
    if relay_client.is_alive(chat_id):
      log.info("Relay shows chat occupied — sending stop signal")
      try:
        relay_client.send_stop(chat_id)
        import time
        for _ in range(10):
          time.sleep(1)
          if not relay_client.is_alive(chat_id):
            log.info("Previous agent released heartbeat")
            break
        else:
          log.warning("Previous agent did not release heartbeat, proceeding anyway")
      except Exception as e:
        log.warning("Stop signal failed: %s", e)


def claim_group(token: str, chat_id: str,
                model: str = "", machine: str = "") -> None:
  """Mark this group as occupied via relay heartbeat."""
  from .config import load_relay_config

  relay_url, _ = load_relay_config()
  if relay_url:
    from . import relay as relay_client
    if not machine:
      machine = get_machine_name()
    relay_client.send_heartbeat(chat_id, pid=os.getpid(),
                                model=model, machine=machine)
  log.info("Claimed group %s (pid=%d)", chat_id, os.getpid())


def release_group(token: str, chat_id: str) -> None:
  """Mark this group as idle by releasing relay heartbeat."""
  from .config import load_relay_config

  relay_url, _ = load_relay_config()
  if relay_url:
    from . import relay as relay_client
    relay_client.release_heartbeat(chat_id)
  log.info("Released group %s", chat_id)


def ensure_workspace_tag(token: str, chat_id: str, project_dir: str) -> None:
  """Ensure the chat description contains the workspace tag.

  If the new-format tag is present, do nothing. If only legacy tag exists,
  replace it. Otherwise append new-format tag.
  """
  from .lark import api as lark_api

  workspace_tag = f"workspace:{get_workspace_id(project_dir)}"
  legacy_tag = f"workspace:{_get_legacy_workspace_id(project_dir)}"

  try:
    info = lark_api.get_chat_info(token, chat_id)
    desc = info.get("description") or ""
    if _workspace_tag_matches(desc, workspace_tag):
      return  # Already tagged with new format
    # Append tag to existing description
    new_desc = f"{desc}\n{workspace_tag}".strip() if desc.strip() else workspace_tag
    lark_api.update_chat_info(token, chat_id, {"description": new_desc})
    log.info("Tagged chat %s with %s", chat_id, workspace_tag)
  except Exception as e:
    log.warning("Failed to tag chat description: %s", e)
