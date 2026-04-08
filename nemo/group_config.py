"""Group configuration via pinned text message.

Stores persistent group-level config (guests, autoapprove, rules)
as YAML inside a pinned text message. The marker prefix "__nemo_config__"
identifies it.

NOTE: We use text messages instead of cards because Lark's get_message API
returns degraded content for interactive (card) messages, losing the original
body. Text messages preserve their content reliably.

Config schema:
  guests:
    - open_id: ou_xxx
      name: Alice
      role: coowner
  autoapprove: false
  rules: {}
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import yaml

log = logging.getLogger(__name__)

CONFIG_MARKER = "__nemo_config__"

DEFAULT_CONFIG: dict[str, Any] = {
  "guests": [],
  "autoapprove": False,
  "rules": {},
}

_VALID_KEYS = {"guests", "autoapprove", "rules", "active_pid", "need_mention"}
_save_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Text message format: "__nemo_config__\n<yaml>"
# ---------------------------------------------------------------------------

def _build_config_text(config: dict[str, Any]) -> str:
  """Encode config as a text message string (YAML)."""
  return f"{CONFIG_MARKER}\n{yaml.dump(config, default_flow_style=False, allow_unicode=True).rstrip()}"


def _parse_config_text(msg: dict[str, Any]) -> dict[str, Any] | None:
  """Extract config from a pinned text message (YAML).

  Expected msg format from get_message API:
    {"msg_type": "text", "body": {"content": '{"text": "..."}' }}
  """
  try:
    if msg.get("msg_type") != "text":
      return None
    content_raw = msg.get("body", {}).get("content", "")
    if not content_raw:
      return None
    # Content is JSON-encoded: {"text": "actual text"}
    text = json.loads(content_raw).get("text", "")
    if not text.startswith(CONFIG_MARKER):
      return None
    payload = text[len(CONFIG_MARKER):].strip()
    config = yaml.safe_load(payload)
    if not isinstance(config, dict):
      return None
    if not _VALID_KEYS & set(config.keys()):
      return None
    return config
  except Exception:
    return None


# ---------------------------------------------------------------------------
# Pin management
# ---------------------------------------------------------------------------

def _find_config_pin(token: str, chat_id: str) -> tuple[str, dict[str, Any]] | None:
  """Find the pinned config message. Returns (message_id, config) or None.

  If multiple config pins exist (from past failures), keeps the first and
  cleans up the rest.
  """
  from .lark import api as lark_api

  try:
    pins = lark_api.list_pins(token, chat_id)
  except Exception as e:
    log.warning("Failed to list pins: %s", e)
    return None

  found: tuple[str, dict[str, Any]] | None = None
  for pin in pins:
    msg_id = pin.get("message_id", "")
    if not msg_id:
      continue
    try:
      msg = lark_api.get_message(token, msg_id)
      config = _parse_config_text(msg)
      if config is not None:
        if found is None:
          found = (msg_id, config)
        else:
          # Duplicate — clean it up
          log.info("Removing duplicate config pin: %s", msg_id)
          try:
            lark_api.delete_pin(token, msg_id)
            lark_api.delete_message(token, msg_id)
          except Exception:
            pass
    except Exception:
      continue
  return found


def _create_config_pin(token: str, chat_id: str,
                       config: dict[str, Any]) -> str:
  """Create a new config text message and pin it. Returns message_id."""
  from .lark import api as lark_api

  text = _build_config_text(config)
  msg_id = lark_api.send_text(token, chat_id, text)
  lark_api.create_pin(token, msg_id)
  return msg_id


def _update_config_pin(token: str, chat_id: str,
                       pin_msg_id: str,
                       config: dict[str, Any]) -> str:
  """Update existing config message. Deletes old and creates new."""
  from .lark import api as lark_api

  # Text messages can't be PATCHed, so replace: unpin+delete old, create new
  try:
    lark_api.delete_pin(token, pin_msg_id)
    lark_api.delete_message(token, pin_msg_id)
  except Exception:
    pass
  return _create_config_pin(token, chat_id, config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(token: str, chat_id: str) -> dict[str, Any]:
  """Load group config from pinned message. Returns default if none found."""
  result = _find_config_pin(token, chat_id)
  if result is not None:
    _msg_id, config = result
    # Merge with defaults for any missing keys
    merged = {**DEFAULT_CONFIG, **config}
    return merged
  return dict(DEFAULT_CONFIG)


def save_config(token: str, chat_id: str, config: dict[str, Any]) -> str:
  """Save config to pinned message. Creates or replaces. Returns message_id."""
  with _save_lock:
    result = _find_config_pin(token, chat_id)
    if result is not None:
      pin_msg_id, _old = result
      return _update_config_pin(token, chat_id, pin_msg_id, config)
    return _create_config_pin(token, chat_id, config)


def get_autoapprove(token: str, chat_id: str) -> bool:
  """Check if autoapprove is enabled."""
  config = load_config(token, chat_id)
  return bool(config.get("autoapprove"))


def set_autoapprove(token: str, chat_id: str, enabled: bool) -> None:
  """Toggle autoapprove."""
  config = load_config(token, chat_id)
  config["autoapprove"] = enabled
  save_config(token, chat_id, config)
