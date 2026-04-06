"""Group configuration via pinned Lark card.

Stores persistent group-level config (guests, autoapprove, filter, rules)
as JSON inside a pinned interactive card. The card title "__nemo_config__"
identifies it.

Config schema:
{
  "guests": [{"open_id": "ou_xxx", "name": "Alice", "role": "coowner"}],
  "autoapprove": false,
  "filter": "concise",
  "rules": {}
}
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

CONFIG_TITLE = "__nemo_config__"

DEFAULT_CONFIG: dict[str, Any] = {
  "guests": [],
  "autoapprove": False,
  "filter": "concise",
  "rules": {},
}

_VALID_KEYS = {"guests", "autoapprove", "filter", "rules"}


# ---------------------------------------------------------------------------
# Card builders / parsers
# ---------------------------------------------------------------------------

def _build_config_card(config: dict[str, Any]) -> dict[str, Any]:
  """Build an interactive card containing the config JSON."""
  body = f"```json\n{json.dumps(config, indent=2)}\n```"
  return {
    "schema": "2.0",
    "config": {"update_multi": True},
    "header": {
      "title": {"tag": "plain_text", "content": CONFIG_TITLE},
      "template": "indigo",
    },
    "body": {
      "direction": "vertical",
      "elements": [{"tag": "markdown", "content": body}],
    },
  }


def _parse_config_from_card(msg: dict[str, Any]) -> dict[str, Any] | None:
  """Extract config JSON from a message's card content.

  Handles two formats:
  1. Direct card dict (body.elements[0].content)
  2. get_message API response (content is a JSON string containing the card)

  Returns the config dict if valid, None otherwise.
  """
  try:
    content_str = ""

    # Path 1: get_message API — content is a JSON string wrapping the card
    card_content = msg.get("content", "")
    if isinstance(card_content, str) and card_content.strip():
      try:
        card_json = json.loads(card_content)
        elements = card_json.get("body", {}).get("elements", [])
        if elements:
          content_str = elements[0].get("content", "")
      except (json.JSONDecodeError, KeyError):
        pass

    # Path 2: direct card dict — body.elements[0].content
    if not content_str:
      body = msg.get("body", {})
      if isinstance(body, str):
        body = json.loads(body)
      elements = body.get("elements", []) if isinstance(body, dict) else []
      if elements:
        content_str = elements[0].get("content", "")

    if not content_str:
      return None

    # Strip markdown code fence
    text = content_str.strip()
    if text.startswith("```"):
      lines = text.split("\n")
      # Remove first and last line (fences)
      lines = lines[1:]
      if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
      text = "\n".join(lines)

    config = json.loads(text)
    if not isinstance(config, dict):
      return None
    # Validate: must have at least one known key
    if not _VALID_KEYS & set(config.keys()):
      return None
    return config

  except Exception:
    return None


# ---------------------------------------------------------------------------
# Pin management
# ---------------------------------------------------------------------------

def _find_config_pin(token: str, chat_id: str) -> tuple[str, dict[str, Any]] | None:
  """Find the pinned config card. Returns (message_id, config) or None."""
  from .lark import api as lark_api

  try:
    pins = lark_api.list_pins(token, chat_id)
  except Exception as e:
    log.warning("Failed to list pins: %s", e)
    return None

  for pin in pins:
    msg_id = pin.get("pin", {}).get("message_id", "")
    if not msg_id:
      continue
    try:
      msg = lark_api.get_message(token, msg_id)
      config = _parse_config_from_card(msg)
      if config is not None:
        return msg_id, config
    except Exception:
      continue
  return None


def _create_config_pin(token: str, chat_id: str,
                       config: dict[str, Any]) -> str:
  """Create a new config card and pin it. Returns message_id."""
  from .lark import api as lark_api

  card = _build_config_card(config)
  msg_id = lark_api.send_card(token, chat_id, card)
  lark_api.create_pin(token, msg_id)
  return msg_id


def _update_config_pin(token: str, chat_id: str,
                       pin_msg_id: str,
                       config: dict[str, Any]) -> str:
  """Update existing config card. Recreates if PATCH fails (14-day expiry)."""
  from .lark import api as lark_api

  card = _build_config_card(config)
  try:
    lark_api.update_card(token, pin_msg_id, card)
    return pin_msg_id
  except RuntimeError:
    # Card expired, recreate
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
  """Load group config from pinned card. Returns default if none found."""
  result = _find_config_pin(token, chat_id)
  if result is not None:
    _msg_id, config = result
    # Merge with defaults for any missing keys
    merged = {**DEFAULT_CONFIG, **config}
    return merged
  return dict(DEFAULT_CONFIG)


def save_config(token: str, chat_id: str, config: dict[str, Any]) -> str:
  """Save config to pinned card. Creates or updates. Returns message_id."""
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
