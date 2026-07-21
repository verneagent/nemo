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
  """Check if a message is an /esc command (with or without follow-up text)."""
  return parse_esc(text, mentions) is not None


def parse_esc(
  text: str, mentions: list[dict[str, str]] | None = None,
) -> str | None:
  """Parse `/esc` or `/esc <text>`.

  Returns:
    - None  : not an esc command
    - ""    : bare `/esc`
    - "<x>" : `/esc <x>` — original case preserved
  """
  stripped = _strip_mentions_preserving_case(text, mentions)
  m = _ESC_PATTERN.match(stripped)
  if not m:
    return None
  return (m.group(1) or "").strip()


_ESC_PATTERN = re.compile(
  r"^/esc(?:\s+(.+))?$",
  re.IGNORECASE | re.DOTALL,
)


def _strip_mentions_preserving_case(
  text: str, mentions: list[dict[str, str]] | None = None,
) -> str:
  t = text.strip()
  for m in (mentions or []):
    key = m.get("key", "")
    if key:
      t = re.sub(re.escape(key), "", t, flags=re.IGNORECASE)
  return re.sub(r"\s+", " ", t).strip()


def is_exit(text: str, mentions: list[dict[str, str]] | None = None) -> bool:
  """Check if a message is an /exit command."""
  t = _strip_mentions(text, mentions)
  return t == "/exit"


def is_dissolve(text: str, mentions: list[dict[str, str]] | None = None) -> bool:
  """Check if a message is a /dissolve command."""
  t = _strip_mentions(text, mentions)
  return t == "/dissolve"


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


def is_privileged(
  sender_id: str,
  operator_open_id: str,
  member_roles: dict[str, str] | None = None,
) -> bool:
  """Check if a sender has privileged rights (operator or coowner).

  Used for actions that guests should not perform (e.g. stopping a turn).
  For ordinary message-relay authorization see ``guests.is_authorized_sender``.
  """
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
