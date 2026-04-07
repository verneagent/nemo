"""Tests for nemo.permissions — auto-approve, formatting, button cards, reactions."""

from nemo.permissions import (
  is_auto_approve, format_tool, _classify_action, _classify_reaction,
)


def test_auto_approve_handoff_ops():
  assert is_auto_approve("Bash", {"command": "python3 handoff_ops.py download"})
  assert is_auto_approve("Bash", {"command": "python3 send_to_group.py hello"})


def test_auto_approve_non_bash():
  assert not is_auto_approve("Read", {"file_path": "/tmp/x"})
  assert not is_auto_approve("Edit", {"file_path": "/tmp/x"})


def test_auto_approve_random_bash():
  assert not is_auto_approve("Bash", {"command": "rm -rf /"})
  assert not is_auto_approve("Bash", {"command": "ls -la"})


def test_format_tool_bash():
  result = format_tool("Bash", {"command": "ls -la", "description": "List files"})
  assert "Bash" in result
  assert "List files" in result


def test_format_tool_bash_long():
  long_cmd = "x" * 300
  result = format_tool("Bash", {"command": long_cmd})
  assert len(result) < 250  # truncated


def test_format_tool_edit():
  result = format_tool("Edit", {"file_path": "/a/b/main.py"})
  assert "Edit" in result
  assert "main.py" in result


def test_format_tool_read():
  result = format_tool("Read", {"file_path": "/x/y/config.json"})
  assert "Read" in result
  assert "config.json" in result


def test_format_tool_unknown():
  result = format_tool("CustomTool", {})
  assert "CustomTool" in result


def test_auto_approve_non_bash_tools():
  """is_auto_approve returns False for non-Bash tools regardless of input."""
  assert not is_auto_approve("Write", {"file_path": "/tmp/x"})
  assert not is_auto_approve("Agent", {"description": "handoff_ops.py"})
  assert not is_auto_approve("Glob", {"pattern": "*.py"})
  assert not is_auto_approve("Grep", {"pattern": "handoff_ops.py"})


def test_format_tool_bash_long_description():
  """Bash with description >200 chars should be truncated."""
  long_desc = "A" * 250
  result = format_tool("Bash", {"command": "ls", "description": long_desc})
  assert len(result) < 250
  assert "..." in result


# ---------------------------------------------------------------------------
# classify_action tests
# ---------------------------------------------------------------------------

def test_classify_action_approve():
  nonce = "abc123"
  assert _classify_action({"action": f"perm_approve:{nonce}"}, nonce) == "allow"


def test_classify_action_always():
  nonce = "abc123"
  assert _classify_action({"action": f"perm_always:{nonce}"}, nonce) == "always"


def test_classify_action_deny():
  nonce = "abc123"
  assert _classify_action({"action": f"perm_deny:{nonce}"}, nonce) == "deny"


def test_classify_action_wrong_nonce():
  assert _classify_action({"action": "perm_approve:wrong"}, "abc123") is None


def test_classify_action_unrelated():
  assert _classify_action({"action": "__stop__"}, "abc123") is None


def test_classify_action_empty():
  assert _classify_action({}, "abc123") is None


# ---------------------------------------------------------------------------
# classify_reaction tests
# ---------------------------------------------------------------------------

def test_classify_reaction_thumbsup():
  assert _classify_reaction("THUMBSUP") == "allow"


def test_classify_reaction_ok():
  assert _classify_reaction("OK") == "allow"


def test_classify_reaction_case_insensitive():
  assert _classify_reaction("thumbsup") == "allow"


def test_classify_reaction_unrelated():
  assert _classify_reaction("HEART") is None
  assert _classify_reaction("FROWN") is None


# ---------------------------------------------------------------------------
# Tests for build_permission_handler / can_use_tool
# ---------------------------------------------------------------------------
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

from nemo.permissions import build_permission_handler


def _make_fixtures(reply_event=None, timeout=False, autoapprove=False):
  """Create mock credentials, chat_id, db, events_source, and SDK results.

  reply_event: a MagicMock LarkEvent to return from next_message.
    If a str, creates a text reply event with that text.
  """
  # Mock SDK permission result classes
  sdk_mod = types.ModuleType("claude_agent_sdk")
  sdk_mod.PermissionResultAllow = type("PermissionResultAllow", (), {})
  sdk_mod.PermissionResultDeny = type("PermissionResultDeny", (), {})
  sys.modules["claude_agent_sdk"] = sdk_mod

  credentials = {"app_id": "test_app", "app_secret": "test_secret"}
  chat_id = "oc_test_chat"

  db = MagicMock()
  session = {"autoapprove": autoapprove}
  db.get_current_session.return_value = session

  events = MagicMock()
  events.permission_active = False

  if timeout:
    events.next_message = AsyncMock(return_value=None)
  elif reply_event is not None:
    if isinstance(reply_event, str):
      reply = MagicMock()
      reply.text = reply_event
      reply.event_type = "im.message.receive_v1"
      reply.chat_id = chat_id
      events.next_message = AsyncMock(return_value=reply)
    else:
      events.next_message = AsyncMock(return_value=reply_event)
  else:
    events.next_message = AsyncMock(return_value=None)

  return credentials, chat_id, db, events, sdk_mod


def _run_with_handler(fixtures, tool_name="Bash", tool_input=None):
  """Build handler and run can_use_tool inside an event loop."""
  creds, chat_id, db, events, sdk = fixtures
  if tool_input is None:
    tool_input = {"command": "ls"}

  async def _go():
    handler = build_permission_handler(creds, chat_id, db, events)
    return handler, await handler(tool_name, tool_input, None)

  return asyncio.run(_go())


def test_build_permission_handler_returns_callable():
  """build_permission_handler returns an async callable."""
  creds, chat_id, db, events, _sdk = _make_fixtures()
  import inspect
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    handler = build_permission_handler(creds, chat_id, db, events)
    assert callable(handler)
    assert inspect.iscoroutinefunction(handler)
  finally:
    asyncio.set_event_loop(None)
    loop.close()


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_approve_text(_get_tok, _send, _update, _build, _pcard):
  """Replying 'y' approves the tool (returns PermissionResultAllow)."""
  fixtures = _make_fixtures(reply_event="y")
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_deny_text(_get_tok, _send, _update, _build, _pcard):
  """Replying 'n' denies the tool (returns PermissionResultDeny)."""
  fixtures = _make_fixtures(reply_event="n")
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultDeny"


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_timeout(_get_tok, _send, _update, _build, _pcard):
  """When next_message returns None (timeout), deny by default."""
  import time as _time
  real_time = _time.time
  call_count = 0

  def fast_time():
    nonlocal call_count
    call_count += 1
    return real_time() + (400 if call_count > 2 else 0)

  fixtures = _make_fixtures(timeout=True)
  with patch("time.time", side_effect=fast_time):
    _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultDeny"


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_always(_get_tok, _send, _update, _build, _pcard):
  """Replying 'always' approves and sets autoapprove in db."""
  fixtures = _make_fixtures(reply_event="always")
  creds, chat_id, db, events, sdk = fixtures
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"
  db.set_autoapprove.assert_called_once_with(chat_id, True)


# ---------------------------------------------------------------------------
# Button card action tests
# ---------------------------------------------------------------------------

def _make_button_event(action_str, chat_id="oc_test_chat"):
  """Create a mock card action event for a button click."""
  event = MagicMock()
  event.event_type = "card.action.trigger"
  event.chat_id = chat_id
  event.action_value = {"action": action_str}
  event.text = ""
  event.message_id = ""
  return event


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
@patch("nemo.permissions.uuid")
def test_can_use_tool_button_approve(mock_uuid, _tok, _send, _upd, _build, _pcard):
  """Clicking Approve button approves the tool."""
  mock_uuid.uuid4.return_value = MagicMock(hex="abcdef123456aaaa")
  nonce = "abcdef123456"

  event = _make_button_event(f"perm_approve:{nonce}")
  fixtures = _make_fixtures(reply_event=event)
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
@patch("nemo.permissions.uuid")
def test_can_use_tool_button_deny(mock_uuid, _tok, _send, _upd, _build, _pcard):
  """Clicking Deny button denies the tool."""
  mock_uuid.uuid4.return_value = MagicMock(hex="abcdef123456aaaa")
  nonce = "abcdef123456"

  event = _make_button_event(f"perm_deny:{nonce}")
  fixtures = _make_fixtures(reply_event=event)
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultDeny"


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
@patch("nemo.permissions.uuid")
def test_can_use_tool_button_always(mock_uuid, _tok, _send, _upd, _build, _pcard):
  """Clicking Approve All button approves and sets autoapprove."""
  mock_uuid.uuid4.return_value = MagicMock(hex="abcdef123456aaaa")
  nonce = "abcdef123456"

  event = _make_button_event(f"perm_always:{nonce}")
  fixtures = _make_fixtures(reply_event=event)
  creds, chat_id, db, events, sdk = fixtures
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"
  db.set_autoapprove.assert_called_once_with(chat_id, True)


# ---------------------------------------------------------------------------
# Reaction tests
# ---------------------------------------------------------------------------

def _make_reaction_event(emoji_type, target_msg_id, chat_id="oc_test_chat"):
  """Create a mock reaction event."""
  event = MagicMock()
  event.event_type = "im.message.reaction.created_v1"
  event.chat_id = chat_id
  event.message_id = target_msg_id
  event.text = emoji_type
  event.action_value = {}
  return event


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_thumbsup_reaction(_tok, _send, _upd, _build, _pcard):
  """THUMBSUP reaction on the permission card approves the tool."""
  event = _make_reaction_event("THUMBSUP", "msg_001")
  fixtures = _make_fixtures(reply_event=event)
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"


@patch("nemo.permissions._build_permission_card", return_value={})
@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_reaction_wrong_message_ignored(_tok, _send, _upd, _build, _pcard):
  """Reaction on a different message is not treated as permission decision."""
  import time as _time
  real_time = _time.time
  call_count = 0

  def fast_time():
    nonlocal call_count
    call_count += 1
    return real_time() + (400 if call_count > 4 else 0)

  event = _make_reaction_event("THUMBSUP", "msg_other")
  fixtures = _make_fixtures(reply_event=event)
  with patch("time.time", side_effect=fast_time):
    _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultDeny"  # timeout → deny


# ---------------------------------------------------------------------------
# Autoapprove bypass test
# ---------------------------------------------------------------------------

@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_autoapprove_skips_card(_tok):
  """When autoapprove is set, no card is sent."""
  fixtures = _make_fixtures(autoapprove=True)
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"


# ---------------------------------------------------------------------------
# Test embedded image exists
# ---------------------------------------------------------------------------

import os

def test_embedded_test_image_exists():
  """Verify test fixture image exists and is a valid PNG."""
  img_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_image.png")
  assert os.path.isfile(img_path)
  with open(img_path, "rb") as f:
    header = f.read(8)
  assert header[:4] == b'\x89PNG'
