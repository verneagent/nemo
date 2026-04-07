"""Tests for nemo.permissions — auto-approve and tool formatting."""

from nemo.permissions import is_auto_approve, format_tool


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
  # The label uses description when present, and gets truncated to 200 chars
  assert len(result) < 250
  assert "..." in result


# ---------------------------------------------------------------------------
# Tests for build_permission_handler / can_use_tool
# ---------------------------------------------------------------------------
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

from nemo.permissions import build_permission_handler


def _make_fixtures(reply_text=None, timeout=False, autoapprove=False):
  """Create mock credentials, chat_id, db, events_source, and SDK results."""

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
  elif reply_text is not None:
    reply = MagicMock()
    reply.text = reply_text
    reply.event_type = "im.message.receive_v1"
    reply.chat_id = chat_id
    events.next_message = AsyncMock(return_value=reply)
  else:
    events.next_message = AsyncMock(return_value=None)

  return credentials, chat_id, db, events, sdk_mod


def _run_with_handler(fixtures, tool_name="Bash", tool_input=None):
  """Build handler and run can_use_tool inside an event loop.

  build_permission_handler captures the current event loop at build time,
  so both build and invocation must happen inside the same loop.
  """
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


@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_approve(_get_tok, _send, _update, _build):
  """Replying 'y' approves the tool (returns PermissionResultAllow)."""
  fixtures = _make_fixtures(reply_text="y")
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"


@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_deny(_get_tok, _send, _update, _build):
  """Replying 'n' denies the tool (returns PermissionResultDeny)."""
  fixtures = _make_fixtures(reply_text="n")
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultDeny"


@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_timeout(_get_tok, _send, _update, _build):
  """When next_message returns None (timeout), deny by default."""
  import time as _time
  real_time = _time.time
  call_count = 0

  def fast_time():
    nonlocal call_count
    call_count += 1
    # After a few calls, jump past the 300s deadline
    return real_time() + (400 if call_count > 2 else 0)

  fixtures = _make_fixtures(timeout=True)
  with patch("time.time", side_effect=fast_time):
    _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultDeny"


@patch("nemo.cards.build_card", return_value={})
@patch("nemo.lark.api.update_card")
@patch("nemo.lark.api.send_card", return_value="msg_001")
@patch("nemo.lark.auth.get_token", return_value="tok_test")
def test_can_use_tool_always(_get_tok, _send, _update, _build):
  """Replying 'always' approves and sets autoapprove in db."""
  fixtures = _make_fixtures(reply_text="always")
  creds, chat_id, db, events, sdk = fixtures
  _, result = _run_with_handler(fixtures)
  assert type(result).__name__ == "PermissionResultAllow"
  db.set_autoapprove.assert_called_once_with(chat_id, True)
