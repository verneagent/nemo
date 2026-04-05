"""Signal watcher during SDK turns.

During a turn, monitors the Lark event stream for interrupt signals
(/esc, handback, takeover). In the Worker-free architecture, this
watches the same LarkEventStream used by the main loop.

For the initial implementation (Phase 1 pending), this module provides
the signal detection logic that will be called by agent.py's concurrent
event reader.
"""

from __future__ import annotations

import re


def is_esc(text: str, mentions: list[dict[str, str]] | None = None) -> bool:
  """Check if a message is an /esc command."""
  t = _strip_mentions(text, mentions)
  return t in ("/esc", "esc", "cancel", "取消")


def is_handback(text: str, mentions: list[dict[str, str]] | None = None) -> bool:
  """Check if a message is a handback command."""
  t = _strip_mentions(text, mentions)
  return t in ("handback", "hand back", "handback dissolve", "hand back dissolve")


def is_permission_reply(text: str) -> str | None:
  """Check if a message is a permission decision.

  Returns "allow", "always", "deny", or None.
  """
  t = text.strip().lower()
  if t in ("y", "yes", "ok", "approve", "允许", "是"):
    return "allow"
  if t in ("always", "全部允许"):
    return "always"
  if t in ("n", "no", "deny", "拒绝", "否"):
    return "deny"
  return None


def is_authorized(
  sender_id: str,
  operator_open_id: str,
  member_roles: dict[str, str] | None = None,
) -> bool:
  """Check if a sender is authorized (operator or coowner)."""
  if not operator_open_id:
    return True
  if sender_id == operator_open_id:
    return True
  if member_roles and member_roles.get(sender_id) == "coowner":
    return True
  return False


def _strip_mentions(text: str, mentions: list[dict[str, str]] | None = None) -> str:
  t = text.strip().lower()
  for m in (mentions or []):
    key = m.get("key", "")
    if key:
      t = t.replace(key.lower(), "").strip()
  return re.sub(r"\s+", " ", t).strip()
