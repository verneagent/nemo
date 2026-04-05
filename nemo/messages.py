"""Message filtering and prompt building.

Works with both LarkEvent objects and plain dicts via _get() helper.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _get(obj: Any, key: str, default: Any = "") -> Any:
  """Get attribute or dict key — works with LarkEvent or dict."""
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)


def build_prompt(replies: list[Any]) -> str:
  """Build a prompt string from messages.

  If any reply has media (image/file) or parent_id, return JSON.
  Otherwise, join plain text.
  """
  has_media = any(
    _get(r, "image_key") or _get(r, "file_key")
    or _get(r, "msg_type") in ("image", "file")
    for r in replies
  )
  if has_media or any(_get(r, "parent_id") for r in replies):
    # Convert to dicts for JSON serialization
    dicts: list[dict[str, Any]] = []
    for r in replies:
      if isinstance(r, dict):
        dicts.append(r)
      elif hasattr(r, "__dict__"):
        dicts.append({k: v for k, v in r.__dict__.items() if v})
      else:
        dicts.append({"text": str(r)})
    return json.dumps(dicts, ensure_ascii=False, indent=2)

  texts = [_get(r, "text", "").strip() for r in replies if _get(r, "text")]
  if texts:
    return "\n".join(texts)
  # Fallback to JSON
  dicts: list[dict[str, Any]] = []
  for r in replies:
    if isinstance(r, dict):
      dicts.append(r)
    elif hasattr(r, "__dict__"):
      dicts.append({k: v for k, v in r.__dict__.items() if v})
    else:
      dicts.append({"text": str(r)})
  return json.dumps(dicts, ensure_ascii=False)


def strip_mentions(text: str, replies: list[Any]) -> str:
  """Remove @-mention markers from text."""
  for r in replies:
    for m in (_get(r, "mentions") or []):
      key = m.get("key", "") if isinstance(m, dict) else getattr(m, "key", "")
      if key:
        text = text.replace(key, "")
  return re.sub(r"\s+", " ", text).strip()


def filter_self_bot(replies: list[Any], bot_open_id: str) -> list[Any]:
  """Remove messages sent by the bot itself."""
  if not bot_open_id:
    return replies
  return [r for r in replies if _get(r, "sender_id") != bot_open_id]


def filter_by_operator(replies: list[Any], operator_open_id: str) -> list[Any]:
  """Keep only messages from the operator."""
  if not operator_open_id:
    return replies
  return [r for r in replies if _get(r, "sender_id") == operator_open_id]


def filter_by_allowed_senders(
  replies: list[Any],
  operator_open_id: str,
  member_roles: dict[str, str],
) -> list[Any]:
  """Keep messages from operator, coowners, and guests."""
  if not operator_open_id:
    return replies
  result = []
  for r in replies:
    sid = _get(r, "sender_id", "")
    if sid == operator_open_id:
      result.append(r)
    elif sid in member_roles:
      result.append(r)
  return result


def filter_bot_interactions(replies: list[Any], bot_open_id: str) -> list[Any]:
  """In need_mention mode, keep only bot-directed messages."""
  if not bot_open_id:
    return replies
  result = []
  for r in replies:
    mentions = _get(r, "mentions") or []
    if any(
      (m.get("id") if isinstance(m, dict) else getattr(m, "id", "")) == bot_open_id
      for m in mentions
    ):
      result.append(r)
      continue
    if _get(r, "parent_id"):
      result.append(r)
      continue
    if _get(r, "msg_type") in ("reaction", "sticker"):
      result.append(r)
      continue
  return result
