"""Text-based permission bridge — no card buttons needed.

When the SDK wants to run a tool that needs approval:
1. Send a read-only card showing the tool description
2. Wait for the user's text reply: "y"/"n"/"always"
3. PATCH card to show decision
4. Return Allow/Deny to the SDK

No card action callbacks, no Worker, no polling.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

log = logging.getLogger(__name__)

# Tools that are always auto-approved (internal operations)
AUTO_APPROVE_PATTERNS = {
  "handoff_ops.py", "send_to_group.py", "wait_for_reply.py",
  "send_and_wait.py", "iterm2_silence.py", "end_and_cleanup.py",
  "start_and_wait.py", "preflight.py", "enter_handoff.py",
}


def is_auto_approve(tool_name: str, tool_input: dict[str, Any]) -> bool:
  """Check if a tool call should be auto-approved."""
  if tool_name != "Bash":
    return False
  cmd = tool_input.get("command", "")
  return any(pat in cmd for pat in AUTO_APPROVE_PATTERNS)


def format_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
  """Format tool for permission card body."""
  if tool_name == "Bash":
    desc = tool_input.get("description", "")
    cmd = tool_input.get("command", "")
    label = desc or cmd
    if len(label) > 200:
      label = label[:197] + "..."
    return f"**Bash**: `{label}`"
  if tool_name in ("Edit", "Write", "Read"):
    fp = tool_input.get("file_path", "")
    name = os.path.basename(fp) if fp else "file"
    return f"**{tool_name}**: `{name}`"
  return f"**{tool_name}**"


def build_permission_handler(
  credentials: dict[str, str],
  chat_id: str,
  db: Any,
  events_source: Any,
) -> Callable[..., Any]:
  """Build an async can_use_tool handler for the SDK.

  events_source: an object with async next_message() that returns the next
  Lark message from the operator. This is how we receive "y"/"n" replies
  without card buttons.
  """
  from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

  async def can_use_tool(tool_name: str, tool_input: dict[str, Any], _context: Any) -> Any:
    # Auto-approve internals
    if is_auto_approve(tool_name, tool_input):
      return PermissionResultAllow()

    # Check autoapprove setting
    session = db.get_current_session()
    if session and session.get("autoapprove"):
      return PermissionResultAllow()

    # Send permission info card (read-only, no buttons)
    from .lark.auth import get_token
    from .lark.api import send_card, update_card
    from .cards import build_card

    token = get_token(credentials["app_id"], credentials["app_secret"])
    body = format_tool(tool_name, tool_input)

    card = build_card(
      "⚠️ Permission Request",
      body=f"{body}\n\nReply **y** to approve, **n** to deny, **always** to approve all",
      color="yellow",
    )
    msg_id = send_card(token, chat_id, card)

    # Wait for text reply (skip non-message events like card actions)
    import time as _time
    from .monitor import is_permission_reply
    decision = None
    deadline = _time.time() + 300
    _pending: list[Any] = []
    while decision is None:
      remaining = deadline - _time.time()
      if remaining <= 0:
        break
      reply = await events_source.next_message(timeout=remaining)
      if reply is None:
        break
      reply_text = getattr(reply, "text", "")
      event_type = getattr(reply, "event_type", "")
      reply_chat = getattr(reply, "chat_id", "")
      # Scope to this session's chat
      if reply_chat and reply_chat != chat_id:
        _pending.append(reply)
        continue
      # Skip card action events — only process text messages
      if event_type == "card.action.trigger":
        continue
      decision = is_permission_reply(reply_text)
      if decision is None:
        # Not a permission reply — re-queue so it isn't lost
        _pending.append(reply)

    # Re-queue any consumed non-permission messages
    for msg in _pending:
      events_source.push_back(msg)

    if decision is None:
      decision = "deny"  # Timeout or unrecognized = deny

    # Update card with decision
    try:
      token = get_token(credentials["app_id"], credentials["app_secret"])
      if decision in ("allow", "always"):
        update_card(token, msg_id, build_card(
          "✓ Approved", body=body, color="green"))
        if decision == "always":
          db.set_autoapprove(chat_id, True)
      else:
        update_card(token, msg_id, build_card(
          "✗ Denied", body=body, color="red"))
    except Exception:
      pass

    if decision in ("allow", "always"):
      return PermissionResultAllow()
    return PermissionResultDeny()

  return can_use_tool
