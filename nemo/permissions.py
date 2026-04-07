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

  IMPORTANT: can_use_tool runs on the SDK thread's event loop (not the main
  loop). The events_source queue is bound to the main loop, so we must
  bridge calls via run_coroutine_threadsafe to avoid cross-loop hangs.
  """
  import asyncio as _asyncio

  from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

  # Capture the main loop at build time — this is called from the main loop.
  _main_loop = _asyncio.get_event_loop()

  async def _read_from_main_loop(timeout: float) -> Any:
    """Read next message from events_source on the main loop."""
    return await events_source.next_message(timeout=timeout)

  def _set_permission_flag(active: bool) -> None:
    events_source.permission_active = active

  def _push_back_on_main(msg: Any) -> None:
    events_source.push_back(msg)

  async def can_use_tool(tool_name: str, tool_input: dict[str, Any], _context: Any) -> Any:
    # Auto-approve internals
    if is_auto_approve(tool_name, tool_input):
      return PermissionResultAllow()

    # Check autoapprove setting (db uses check_same_thread=False, safe here)
    session = db.get_current_session()
    if session and session.get("autoapprove"):
      return PermissionResultAllow()

    # Send permission info card (synchronous HTTP, safe from any thread)
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

    # Wait for text reply — bridge to main loop for queue reads
    import time as _time
    from .monitor import is_permission_reply
    decision = None
    deadline = _time.time() + 300
    _pending: list[Any] = []

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
            # Same loop — direct await
            reply = await _read_from_main_loop(timeout)
          else:
            # Cross-thread — bridge via run_coroutine_threadsafe
            future = _asyncio.run_coroutine_threadsafe(
              _read_from_main_loop(timeout), _main_loop)
            reply = await _asyncio.wrap_future(future)
        except Exception:
          break
        if reply is None:
          if remaining <= 30:
            break
          continue
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
