"""Manual cleanup for idle Nemo Lark groups."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .types import JsonObject


@dataclass(frozen=True)
class GcChat:
  chat_id: str
  name: str
  description: str
  is_nemo: bool
  alive: bool
  heartbeat_error: str
  machine: str
  model: str
  is_owner: bool

  @property
  def status(self) -> str:
    if self.heartbeat_error:
      return "UNKNOWN"
    return "ACTIVE" if self.alive else "IDLE"


def _str_value(value: object) -> str:
  return value if isinstance(value, str) else ""


def _bool_value(value: object) -> bool:
  return value if isinstance(value, bool) else False


def _nested_str_value(value: object, key: str) -> str:
  if not isinstance(value, dict):
    return ""
  nested = value.get(key)
  return nested if isinstance(nested, str) else ""


def _chat_owner_id(info: JsonObject) -> str:
  """Return the chat owner's open_id when Lark includes it."""
  for key in ("owner_id", "owner_open_id", "chat_owner_id"):
    value = info.get(key)
    if isinstance(value, str) and value:
      return value
    nested = _nested_str_value(value, "open_id")
    if nested:
      return nested
  return ""


def _is_config_message(msg: JsonObject) -> bool:
  from .group_config import _parse_config_text

  return _parse_config_text(msg) is not None


def _has_nemo_config_pin(token: str, chat_id: str) -> bool:
  from .lark import api as lark_api

  try:
    pins = lark_api.list_pins(token, chat_id)
  except Exception:
    return False
  for pin in pins:
    msg_id = _str_value(pin.get("message_id"))
    if not msg_id:
      continue
    try:
      if _is_config_message(lark_api.get_message(token, msg_id)):
        return True
    except Exception:
      continue
  return False


def _is_nemo_chat(token: str, chat_id: str, description: str) -> bool:
  if "workspace:" in description:
    return True
  return _has_nemo_config_pin(token, chat_id)


def _bot_open_id(token: str) -> str:
  from .lark import api as lark_api

  try:
    bot_info = lark_api.get_bot_info(token)
  except Exception:
    return ""
  return _str_value(bot_info.get("open_id"))


def _heartbeat_status(chat_id: str) -> tuple[bool, str, str, str]:
  from . import relay

  status = relay.heartbeat_status(chat_id)
  error = _str_value(status.get("error"))
  return (
    _bool_value(status.get("alive")),
    error,
    _str_value(status.get("machine")),
    _str_value(status.get("model")),
  )


def collect_gc_chats(token: str, *, include_unknown: bool = True) -> list[GcChat]:
  """Collect Nemo-managed chats with heartbeat status."""
  from .lark import api as lark_api

  rows: list[GcChat] = []
  bot_open_id = _bot_open_id(token)
  if not bot_open_id:
    return rows
  for chat in lark_api.list_bot_chats(token):
    chat_id = _str_value(chat.get("chat_id"))
    if not chat_id:
      continue
    try:
      info = lark_api.get_chat_info(token, chat_id)
    except Exception:
      continue
    name = _str_value(info.get("name")) or _str_value(chat.get("name")) or chat_id
    description = _str_value(info.get("description"))
    is_nemo = _is_nemo_chat(token, chat_id, description)
    if not is_nemo:
      continue
    alive, error, machine, model = _heartbeat_status(chat_id)
    if error and not include_unknown:
      continue
    owner_id = _chat_owner_id(info)
    rows.append(GcChat(
      chat_id=chat_id,
      name=name,
      description=description,
      is_nemo=is_nemo,
      alive=alive,
      heartbeat_error=error,
      machine=machine,
      model=model,
      is_owner=owner_id == bot_open_id,
    ))
  rows.sort(key=lambda r: (r.status, not r.is_owner, r.name.lower(), r.chat_id))
  return rows


def _short(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  return text[:max(0, limit - 3)] + "..."


def format_gc_table(rows: list[GcChat]) -> str:
  if not rows:
    return "No Nemo Lark groups found."
  lines = [f"{'#':>3} {'STATUS':<8} {'OWNER':<5} {'CHAT_ID':<36} {'NAME':<28} DETAILS"]
  for idx, row in enumerate(rows, start=1):
    details = ""
    if row.heartbeat_error:
      details = f"relay error: {row.heartbeat_error}"
    elif row.alive:
      parts = []
      if row.machine:
        parts.append(f"machine={row.machine}")
      if row.model:
        parts.append(f"model={row.model}")
      details = " ".join(parts)
    lines.append(
      f"{idx:>3} {row.status:<8} {'yes' if row.is_owner else 'no':<5} "
      f"{row.chat_id:<36} "
      f"{_short(row.name, 28):<28} {details}"
    )
  return "\n".join(lines)


def _parse_selection(selection: str, rows: list[GcChat]) -> list[GcChat]:
  text = selection.strip().lower()
  if text == "all":
    return list(rows)
  chosen: list[GcChat] = []
  seen: set[int] = set()
  for raw in text.replace(",", " ").split():
    try:
      idx = int(raw)
    except ValueError:
      continue
    if idx < 1 or idx > len(rows) or idx in seen:
      continue
    seen.add(idx)
    chosen.append(rows[idx - 1])
  return chosen


def _confirm(prompt: str) -> bool:
  print(prompt, end="", flush=True)
  return sys.stdin.readline().strip() == "dissolve"


def _safe_to_dissolve(chat_id: str) -> tuple[bool, str]:
  alive, error, _machine, _model = _heartbeat_status(chat_id)
  if error:
    return False, f"relay heartbeat check failed: {error}"
  if alive:
    return False, "heartbeat is alive"
  return True, ""


def dissolve_chats(token: str, rows: list[GcChat]) -> tuple[list[str], list[str]]:
  from .lark import api as lark_api

  completed: list[str] = []
  skipped: list[str] = []
  bot_open_id = ""
  for row in rows:
    ok, reason = _safe_to_dissolve(row.chat_id)
    if not ok:
      skipped.append(f"{row.name} ({row.chat_id}): {reason}")
      continue
    try:
      if row.is_owner:
        lark_api.dissolve_chat(token, row.chat_id)
        completed.append(f"Dissolved {row.name} ({row.chat_id})")
      else:
        if not bot_open_id:
          bot_open_id = _bot_open_id(token)
        if not bot_open_id:
          skipped.append(f"{row.name} ({row.chat_id}): bot open_id unavailable")
          continue
        lark_api.remove_chat_members(token, row.chat_id, [bot_open_id])
        completed.append(f"Left {row.name} ({row.chat_id})")
    except Exception as exc:
      skipped.append(f"{row.name} ({row.chat_id}): {exc}")
  return completed, skipped


def gc_list(token: str) -> int:
  print(format_gc_table(collect_gc_chats(token)))
  return 0


def gc_clean(token: str, *, chat_id: str = "", yes: bool = False) -> int:
  if chat_id:
    rows = [row for row in collect_gc_chats(token) if row.chat_id == chat_id]
    if not rows:
      print(f"No Nemo Lark group found for chat_id {chat_id}.", file=sys.stderr)
      return 1
    selected = rows
  else:
    candidates = [
      row for row in collect_gc_chats(token, include_unknown=False)
      if not row.alive and not row.heartbeat_error
    ]
    if not candidates:
      print("No idle Nemo Lark groups found.")
      return 0
    print(format_gc_table(candidates))
    if yes:
      selected = candidates
    else:
      print('\nSelect groups to dissolve (numbers separated by spaces, or "all"): ', end="", flush=True)
      selected = _parse_selection(sys.stdin.readline(), candidates)
      if not selected:
        print("No groups selected.")
        return 0
  if not yes and not _confirm(
      f'About to dissolve {len(selected)} Lark group(s). Type "dissolve" to continue: '):
    print("Cancelled.")
    return 1
  dissolved, skipped = dissolve_chats(token, selected)
  for item in dissolved:
    print(item)
  for item in skipped:
    print(f"Skipped {item}", file=sys.stderr)
  return 1 if skipped and not dissolved else 0
