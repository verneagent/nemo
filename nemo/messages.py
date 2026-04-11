"""Message filtering and prompt building.

Works with both LarkEvent objects and plain dicts via _get() helper.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .types import JsonObject


def _get(obj: object, key: str, default: object = "") -> object:
  """Get attribute or dict key — works with LarkEvent or dict."""
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)


def build_prompt(replies: list[object]) -> str:
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
    dicts: list[JsonObject] = []
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
  dicts: list[JsonObject] = []
  for r in replies:
    if isinstance(r, dict):
      dicts.append(r)
    elif hasattr(r, "__dict__"):
      dicts.append({k: v for k, v in r.__dict__.items() if v})
    else:
      dicts.append({"text": str(r)})
  return json.dumps(dicts, ensure_ascii=False)


# Marker appended by LarkChannel._to_incoming when a message has a
# parent_id / thread root. The tail is useful as SDK context but breaks
# slash-command dispatch (which relies on exact or prefix matches on the
# whole string). strip_parent_quote() peels it off for command dispatch
# while leaving the full enriched text intact for the SDK prompt.
_PARENT_QUOTE_MARKER = "\n\n(The user is replying to this earlier message"


def strip_parent_quote(text: str) -> str:
  """Remove the LarkChannel parent-quote tail from a user message.

  In Lark topic groups every message carries a root_id pointing at the
  topic root, so the parent-quote enrichment fires on every message —
  including plain slash commands. Without stripping, `"/mention"` looks
  like `"/mention\\n\\n(The user is replying to this earlier message…)"`
  to ``commands.try_dispatch`` and never matches the command table.
  Return only the portion before the marker; leave plain text alone.
  """
  idx = text.find(_PARENT_QUOTE_MARKER)
  if idx >= 0:
    return text[:idx].rstrip()
  return text


def strip_mentions(text: str, replies: list[object],
                   bot_open_id: str = "") -> str:
  """Strip bot @-mentions; replace other @-mentions with the person's name."""
  for r in replies:
    for m in (_get(r, "mentions") or []):
      key = m.get("key", "") if isinstance(m, dict) else getattr(m, "key", "")
      if not key:
        continue
      mid = m.get("id", "") if isinstance(m, dict) else getattr(m, "id", "")
      name = m.get("name", "") if isinstance(m, dict) else getattr(m, "name", "")
      if bot_open_id and mid == bot_open_id:
        text = text.replace(key, "")
      elif name:
        text = text.replace(key, name)
      else:
        text = text.replace(key, "")
  return re.sub(r"\s+", " ", text).strip()


def filter_self_bot(replies: list[object], bot_open_id: str) -> list[object]:
  """Remove messages sent by the bot itself."""
  if not bot_open_id:
    return replies
  return [r for r in replies if _get(r, "sender_id") != bot_open_id]


def filter_by_operator(replies: list[object], operator_open_id: str) -> list[object]:
  """Keep only messages from the operator."""
  if not operator_open_id:
    return replies
  return [r for r in replies if _get(r, "sender_id") == operator_open_id]


def filter_by_allowed_senders(
  replies: list[object],
  operator_open_id: str,
  member_roles: dict[str, str],
) -> list[object]:
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


def filter_bot_interactions(
  replies: list[object],
  bot_open_id: str,
  is_own_message: Callable[[str], bool] | None = None,
) -> list[object]:
  """In need_mention mode, keep only bot-directed messages.

  A message is considered bot-directed when:
    - it @-mentions the bot, OR
    - it's a reply whose parent was sent by the bot (text/card reply),
      OR
    - it's a reaction whose target message was sent by the bot.

  Reactions are a form of reply — reacting to one of the bot's own
  messages counts as addressing the bot. Reacting to someone else's
  message does not.

  Replying to *other* messages (e.g. quoting a teammate's card while
  @-mentioning another teammate) is NOT considered bot-directed.

  `is_own_message(message_id)` should return True iff the given id
  corresponds to a message the bot itself sent. If not supplied, any
  parent_id/reaction is treated as implicit mention — the permissive
  legacy behavior kept for test fixtures that don't model ownership.
  """
  if not bot_open_id:
    return replies
  result = []
  for r in replies:
    mentions = _get(r, "mentions") or []
    if any(
      (m.get("id") if isinstance(m, dict) else getattr(m, "id", "")) == bot_open_id
      for m in mentions  # type: ignore[union-attr]
    ):
      result.append(r)
      continue

    # Reactions: target is in message_id (no parent_id).
    if _get(r, "event_type") == "im.message.reaction.created_v1":
      target = str(_get(r, "message_id") or "")
      if not target:
        continue
      if is_own_message is None:
        result.append(r)
        continue
      try:
        if is_own_message(target):
          result.append(r)
      except Exception:
        pass
      continue

    # Regular replies: parent_id points at the quoted message.
    parent_id = _get(r, "parent_id")
    if parent_id:
      if is_own_message is None:
        result.append(r)
        continue
      try:
        if is_own_message(str(parent_id)):
          result.append(r)
          continue
      except Exception:
        pass
  return result
