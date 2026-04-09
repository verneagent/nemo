"""Button-based permission bridge — Approve / Approve All / Deny.

When the SDK wants to run a tool that needs approval:
1. Send a card with 3 buttons (Approve, Approve All, Deny)
2. Wait for: button click (card action), text reply, or THUMBSUP reaction
3. PATCH card to show decision
4. Return Allow/Deny to the SDK

Supports:
- Card action callbacks (button clicks via relay)
- Text replies (y/n/always — backward compatible)
- THUMBSUP reaction on the permission card = approve
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Callable

from .channel import Channel
from .db import Database
from .types import JsonObject

log = logging.getLogger(__name__)

# Tools that are always auto-approved (internal operations)
AUTO_APPROVE_PATTERNS: set[str] = set()

# Reaction emoji types that mean "approve"
APPROVE_REACTIONS = {"THUMBSUP", "OK", "YES", "APPROVE", "LIKESMILEY"}


def is_auto_approve(tool_name: str, tool_input: JsonObject) -> bool:
  """Check if a tool call should be auto-approved."""
  if tool_name != "Bash":
    return False
  cmd = tool_input.get("command", "")
  return any(pat in cmd for pat in AUTO_APPROVE_PATTERNS)


def format_tool(tool_name: str, tool_input: JsonObject) -> str:
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


def _build_permission_card(
  body: str,
  chat_id: str,
  nonce: str,
) -> JsonObject:
  """Build a permission request card with Approve/Approve All/Deny buttons."""
  from .cards import build_card

  buttons = [
    ("Approve", f"perm_approve:{nonce}", "primary"),
    ("Approve All", f"perm_always:{nonce}", "default"),
    ("Deny", f"perm_deny:{nonce}", "danger"),
  ]
  return build_card(
    "Permission Request",
    body=body,
    color="yellow",
    buttons=buttons,
    chat_id=chat_id,
  )


def _classify_action(action_value: JsonObject, nonce: str) -> str | None:
  """Classify a card action event as a permission decision.

  Returns "allow", "always", "deny", or None if not a permission action.
  """
  action = action_value.get("action", "")
  if not isinstance(action, str):
    return None
  if action == f"perm_approve:{nonce}":
    return "allow"
  if action == f"perm_always:{nonce}":
    return "always"
  if action == f"perm_deny:{nonce}":
    return "deny"
  return None


def _classify_reaction(emoji_type: str) -> str | None:
  """Classify a reaction emoji as a permission decision."""
  if emoji_type.upper() in APPROVE_REACTIONS:
    return "allow"
  return None


def build_permission_handler(
  credentials: dict[str, str],
  chat_id: str,
  db: Database,
  events_source: Channel,
) -> Callable[[str, JsonObject, object], object]:
  """Build an async can_use_tool handler for the SDK.

  events_source: a Channel that returns the next operator event via receive().

  IMPORTANT: can_use_tool runs on the SDK thread's event loop (not the main
  loop). The events_source queue is bound to the main loop, so we must
  bridge calls via run_coroutine_threadsafe to avoid cross-loop hangs.
  """
  import asyncio as _asyncio

  from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

  # Capture the main loop at build time — this is called from the main loop.
  _main_loop = _asyncio.get_event_loop()

  async def _read_from_main_loop(timeout: float) -> object:
    """Read next message from events_source on the main loop."""
    return await events_source.receive(timeout=timeout)

  def _set_permission_flag(active: bool) -> None:
    events_source.permission_active = active

  def _push_back_on_main(msg: object) -> None:
    from .channel import IncomingMessage

    if isinstance(msg, IncomingMessage):
      events_source.push_back(msg)

  async def can_use_tool(
    tool_name: str,
    tool_input: JsonObject,
    _context: object,
  ) -> object:
    log.debug("can_use_tool: %s %s", tool_name,
              {k: str(v)[:80] for k, v in tool_input.items()})

    # Auto-approve internals
    if is_auto_approve(tool_name, tool_input):
      return PermissionResultAllow()

    # Check autoapprove setting (db uses check_same_thread=False, safe here)
    session = db.get_current_session()
    if session and session.get("autoapprove"):
      return PermissionResultAllow()

    # Generate nonce for this permission request
    nonce = uuid.uuid4().hex[:12]

    # Send permission card with buttons
    from .lark.auth import get_token
    from .lark.api import send_card, update_card
    from .cards import build_card

    token = get_token(credentials["app_id"], credentials["app_secret"])
    body = format_tool(tool_name, tool_input)

    card = _build_permission_card(body, chat_id, nonce)
    msg_id = send_card(token, chat_id, card)
    log.info("Permission request: %s (card=%s, nonce=%s)", tool_name, msg_id, nonce)

    # Wait for button click, text reply, or reaction
    import time as _time
    from .monitor import is_permission_reply
    decision = None
    deadline = _time.time() + 300
    _pending: list[object] = []

    # Detect if we're on the main loop (tests) vs SDK thread (production)
    try:
      _current_loop = _asyncio.get_running_loop()
    except RuntimeError:
      _current_loop = None
    _on_main_loop = _current_loop is _main_loop

    if _on_main_loop:
      _set_permission_flag(True)
    else:
      _main_loop.call_soon_threadsafe(_set_permission_flag, True)

    try:
      while decision is None:
        remaining = deadline - _time.time()
        if remaining <= 0:
          break
        timeout = min(remaining, 30)
        try:
          if _on_main_loop:
            reply = await _read_from_main_loop(timeout)
          else:
            future = _asyncio.run_coroutine_threadsafe(
              _read_from_main_loop(timeout), _main_loop)
            reply = await _asyncio.wrap_future(future)
        except Exception:
          break
        if reply is None:
          if remaining <= 30:
            break
          continue

        event_type = getattr(reply, "event_type", "")
        reply_chat = getattr(reply, "chat_id", "")

        # Scope to this session's chat
        if reply_chat and reply_chat != chat_id:
          _pending.append(reply)
          continue

        # 1. Card action callback (button click)
        if event_type == "card.action.trigger":
          action_value = getattr(reply, "action_value", {})
          decision = _classify_action(action_value, nonce)
          if decision is None:
            # Not our permission card — re-queue
            _pending.append(reply)
          continue

        # 2. Reaction on the permission card
        if event_type == "im.message.reaction.created_v1":
          reaction_target = getattr(reply, "message_id", "")
          emoji = getattr(reply, "text", "")
          if reaction_target == msg_id:
            decision = _classify_reaction(emoji)
          if decision is None:
            _pending.append(reply)
          continue

        # 3. Text reply (backward compatible)
        reply_text = getattr(reply, "text", "")
        decision = is_permission_reply(reply_text)
        if decision is None:
          # Not a permission reply — re-queue so it isn't lost
          _pending.append(reply)

    finally:
      if _on_main_loop:
        _set_permission_flag(False)
      else:
        _main_loop.call_soon_threadsafe(_set_permission_flag, False)

    # Re-queue any consumed non-permission messages
    for msg in _pending:
      if _on_main_loop:
        _push_back_on_main(msg)
      else:
        _main_loop.call_soon_threadsafe(_push_back_on_main, msg)

    if decision is None:
      decision = "deny"  # Timeout or unrecognized = deny

    log.info("Permission decision: %s for %s", decision, tool_name)

    # Update card with decision
    try:
      token = get_token(credentials["app_id"], credentials["app_secret"])
      if decision in ("allow", "always"):
        update_card(token, msg_id, build_card(
          "Approved ✓", body=body, color="green"))
        if decision == "always":
          db.set_autoapprove(chat_id, True)
      else:
        update_card(token, msg_id, build_card(
          "Denied ✗", body=body, color="red"))
    except Exception as e:
      log.warning("Failed to update permission card: %s", e)

    if decision in ("allow", "always"):
      return PermissionResultAllow()
    return PermissionResultDeny()

  return can_use_tool
