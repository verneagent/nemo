"""Message filtering and prompt building."""

from __future__ import annotations

import json
import re


def build_prompt(replies: list[dict]) -> str:
  """Build a prompt string from Lark reply dicts.

  If any reply has media (image/file) or parent_id, return JSON.
  Otherwise, join plain text.
  """
  has_media = any(
    r.get("image_key") or r.get("file_key")
    or r.get("msg_type") in ("image", "file")
    for r in replies
  )
  if has_media or any(r.get("parent_id") for r in replies):
    return json.dumps(replies, ensure_ascii=False, indent=2)

  texts = [r.get("text", "").strip() for r in replies if r.get("text")]
  return "\n".join(texts) if texts else json.dumps(replies, ensure_ascii=False)


def strip_mentions(text: str, replies: list[dict]) -> str:
  """Remove @-mention markers from text."""
  for r in replies:
    for m in (r.get("mentions") or []):
      key = m.get("key", "")
      if key:
        text = text.replace(key, "")
  return re.sub(r"\s+", " ", text).strip()


def filter_self_bot(replies: list[dict], bot_open_id: str) -> list[dict]:
  """Remove messages sent by the bot itself."""
  if not bot_open_id:
    return replies
  return [r for r in replies if r.get("sender_id") != bot_open_id]


def filter_by_operator(replies: list[dict], operator_open_id: str) -> list[dict]:
  """Keep only messages from the operator."""
  if not operator_open_id:
    return replies
  return [r for r in replies if r.get("sender_id") == operator_open_id]


def filter_by_allowed_senders(
  replies: list[dict],
  operator_open_id: str,
  member_roles: dict[str, str],
) -> list[dict]:
  """Keep messages from operator, coowners, and guests."""
  if not operator_open_id:
    return replies
  result = []
  for r in replies:
    sid = r.get("sender_id", "")
    if sid == operator_open_id:
      result.append(r)
    elif sid in member_roles:
      result.append(r)
  return result


def filter_bot_interactions(replies: list[dict], bot_open_id: str) -> list[dict]:
  """In need_mention mode, keep only bot-directed messages."""
  if not bot_open_id:
    return replies
  result = []
  for r in replies:
    # @-mention
    mentions = r.get("mentions") or []
    if any(m.get("id") == bot_open_id for m in mentions):
      result.append(r)
      continue
    # Reply to bot message
    if r.get("parent_id"):
      result.append(r)
      continue
    # Reaction/sticker
    if r.get("msg_type") in ("reaction", "sticker"):
      result.append(r)
      continue
  return result
