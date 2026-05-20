"""Tests for nemo.permissions — auto-approve, formatting, button cards, reactions."""

import sys as _sys

import pytest as _pytest

from nemo.permissions import (
  is_auto_approve, format_tool, _classify_action, _classify_reaction,
  _parse_askq_action,
)


@_pytest.fixture(autouse=True)
def _restore_claude_agent_sdk():
  """Several helpers here swap a stub into ``sys.modules['claude_agent_sdk']``
  and never restore it. That leaks the stub into any later test file that
  imports the real SDK (e.g. test_claude_agent's /btw side_question tests
  saw ``module has no attribute 'query'``). Snapshot + restore around every
  test in this module so the pollution can't escape, regardless of order.
  """
  saved = _sys.modules.get("claude_agent_sdk")
  yield
  if saved is None:
    _sys.modules.pop("claude_agent_sdk", None)
  else:
    _sys.modules["claude_agent_sdk"] = saved


def test_auto_approve_empty_patterns():
  """No hardcoded auto-approve patterns — handoff remnants removed."""
  assert not is_auto_approve("Bash", {"command": "python3 handoff_ops.py download"})
  assert not is_auto_approve("Bash", {"command": "python3 send_to_group.py hello"})


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
from typing import Awaitable, Protocol, cast
from unittest.mock import AsyncMock, MagicMock, patch

from nemo.permissions import build_permission_handler
from nemo.types import JsonObject


class _AllowResult(Protocol):
  """Minimal protocol for the fake PermissionResultAllow created in tests."""
  updated_input: JsonObject


def _make_fixtures(reply_event=None, timeout=False, autoapprove=False):
  """Create mock credentials, chat_id, db, events_source, and SDK results.

  reply_event: a MagicMock LarkEvent to return from receive().
    If a str, creates a text reply event with that text.
  """
  # Mock SDK permission result classes
  sdk_mod = types.ModuleType("claude_agent_sdk")
  setattr(sdk_mod, "PermissionResultAllow", type("PermissionResultAllow", (), {}))
  setattr(sdk_mod, "PermissionResultDeny", type("PermissionResultDeny", (), {}))
  sys.modules["claude_agent_sdk"] = sdk_mod

  credentials = {"app_id": "test_app", "app_secret": "test_secret"}
  chat_id = "oc_test_chat"

  db = MagicMock()
  session = {"autoapprove": autoapprove}
  db.get_current_session.return_value = session

  events = MagicMock()
  events.permission_active = False
  events.push_back = MagicMock()

  if timeout:
    events.receive = AsyncMock(return_value=None)
  elif reply_event is not None:
    if isinstance(reply_event, str):
      reply = MagicMock()
      reply.text = reply_event
      reply.event_type = "im.message.receive_v1"
      reply.chat_id = chat_id
      events.receive = AsyncMock(return_value=reply)
    else:
      events.receive = AsyncMock(return_value=reply_event)
  else:
    events.receive = AsyncMock(return_value=None)

  return credentials, chat_id, db, events, sdk_mod


def _run_with_handler(fixtures, tool_name="Bash", tool_input=None):
  """Build handler and run can_use_tool inside an event loop."""
  creds, chat_id, db, events, _sdk = fixtures
  ti: JsonObject = tool_input if tool_input is not None else {"command": "ls"}

  async def _go():
    handler = build_permission_handler(creds, chat_id, db, events)
    return handler, await cast(Awaitable[object], handler(tool_name, ti, None))

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
  """When receive() returns None (timeout), deny by default."""
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


# ---------------------------------------------------------------------------
# _parse_askq_action — parse action strings from card button clicks
# ---------------------------------------------------------------------------

def test_parse_askq_action_option():
  result = _parse_askq_action({"action": "askq:N1:0:2"}, "N1")
  assert result == ("option", 0, "2")


def test_parse_askq_action_other():
  result = _parse_askq_action({"action": "askq:N1:1:other"}, "N1")
  assert result == ("other", 1, "")


def test_parse_askq_action_done():
  result = _parse_askq_action({"action": "askq:N1:0:done"}, "N1")
  assert result == ("done", 0, "")


def test_parse_askq_action_wrong_nonce():
  assert _parse_askq_action({"action": "askq:WRONG:0:0"}, "N1") is None


def test_parse_askq_action_unrelated_prefix():
  assert _parse_askq_action({"action": "perm_approve:N1"}, "N1") is None


def test_parse_askq_action_empty():
  assert _parse_askq_action({}, "N1") is None


def test_parse_askq_action_malformed_qidx():
  assert _parse_askq_action({"action": "askq:N1:notanint:0"}, "N1") is None


def test_parse_askq_action_malformed_payload():
  # Payload not "other"/"done"/int → invalid
  assert _parse_askq_action({"action": "askq:N1:0:bogus"}, "N1") is None


# ---------------------------------------------------------------------------
# build_ask_user_question_handler — interactive AskUserQuestion bridge
# ---------------------------------------------------------------------------

from nemo.permissions import build_ask_user_question_handler


def _make_askq_fixtures(events_seq=None, *, embed_card: bool = False):
  """Like _make_fixtures but for the askq handler.

  events_seq: list of LarkEvent-shaped MagicMocks to return from receive(),
    in order. After exhausting the list, receive() returns None (timeout).
  embed_card: if True, simulate the agent successfully creating the
    working turn card whenever the askq handler calls redraw(). This
    drives the embedded code path. When False (default), redraw() is a
    no-op so the handler falls back to a standalone card (the case
    where AskUserQuestion fires before any working card exists).
  """
  from nemo.channel import TurnCardCtx

  sdk_mod = types.ModuleType("claude_agent_sdk")
  # Capture updated_input on PermissionResultAllow so tests can introspect it.
  class _Allow:
    def __init__(self, updated_input=None):
      self.updated_input = updated_input
  class _Deny:
    def __init__(self, message=""):
      self.message = message
  setattr(sdk_mod, "PermissionResultAllow", _Allow)
  setattr(sdk_mod, "PermissionResultDeny", _Deny)
  sys.modules["claude_agent_sdk"] = sdk_mod

  credentials = {"app_id": "test_app", "app_secret": "test_secret"}
  chat_id = "oc_test_chat"

  events = MagicMock()
  events.permission_active = False
  events.push_back = MagicMock()

  # Stand in for the per-turn TurnCardCtx the agent wires up. Tests that
  # need to drive the embedded path pass embed_card=True so the redraw
  # callback simulates the agent creating the working card.
  ctx = TurnCardCtx()
  redraw_calls: list[dict] = []

  def _redraw():
    snapshot = {
      "turn_card_id": ctx.turn_card_id,
      "pending_question": ctx.pending_question,
      "answered_questions": list(ctx.answered_questions),
    }
    if ctx.pending_question is not None:
      snapshot["pending_answers"] = dict(ctx.pending_question.answers)
      snapshot["pending_multi_done"] = set(ctx.pending_question.multi_done)
    redraw_calls.append(snapshot)
    if embed_card and not ctx.turn_card_id:
      # Mirror what agent._ensure_card does once the working card lands.
      ctx.turn_card_id = "turn_card_001"

  ctx.redraw = _redraw
  events.turn_ctx = ctx
  events.redraw_calls = redraw_calls

  seq = list(events_seq or [])

  async def _receive(timeout=None):
    if seq:
      return seq.pop(0)
    return None

  events.receive = AsyncMock(side_effect=_receive)
  return credentials, chat_id, events, sdk_mod


def _askq_event(action_str, chat_id="oc_test_chat"):
  # Use real IncomingMessage so _push_back_on_main's isinstance check passes.
  from nemo.channel import IncomingMessage
  return IncomingMessage(
    event_type="card.action.trigger",
    chat_id=chat_id,
    action_value={"action": action_str},
  )


def _text_event(text, chat_id="oc_test_chat"):
  from nemo.channel import IncomingMessage
  return IncomingMessage(
    event_type="im.message.receive_v1",
    chat_id=chat_id,
    text=text,
  )


def _run_askq(fixtures, questions, nonce_patcher_value="abc123") -> _AllowResult:
  creds, chat_id, events, _sdk = fixtures
  # Patch nonce to a fixed value so the action strings in the test events line up.
  with patch("nemo.permissions.uuid.uuid4") as uu, \
       patch("nemo.lark.api.send_card", return_value="msg_001") as send_card_mock, \
       patch("nemo.lark.api.update_card") as update_card_mock, \
       patch("nemo.lark.api.send_text"), \
       patch("nemo.lark.auth.get_token", return_value="tok"):
    # Handler does `nonce = uuid.uuid4().hex[:12]`, and test events use
    # action strings with nonce=nonce_patcher_value. Give the mock a hex
    # value that slices to exactly nonce_patcher_value.
    uu.return_value = MagicMock(hex=nonce_patcher_value)

    # Expose the patched lark API calls on the events MagicMock so tests
    # can assert against them (e.g. "did the standalone fallback path
    # call send_card?").
    events.send_card_mock = send_card_mock
    events.update_card_mock = update_card_mock

    async def _go():
      handler = build_ask_user_question_handler(creds, chat_id, events)
      ti: JsonObject = {"questions": questions}
      return await cast(Awaitable[object],
                        handler("AskUserQuestion", ti, None))

    return cast(_AllowResult, asyncio.run(_go()))


def test_askq_handler_returns_callable():
  creds, chat_id, events, _sdk = _make_askq_fixtures()
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    handler = build_ask_user_question_handler(creds, chat_id, events)
    import inspect
    assert callable(handler)
    assert inspect.iscoroutinefunction(handler)
  finally:
    asyncio.set_event_loop(None)
    loop.close()


def test_askq_handler_non_askq_tool_passes_through():
  """Handler called with a non-AskUserQuestion tool returns Allow without UI."""
  fixtures = _make_askq_fixtures()
  creds, chat_id, events, _sdk = fixtures
  with patch("nemo.lark.api.send_card") as send:
    async def _go():
      handler = build_ask_user_question_handler(creds, chat_id, events)
      ti: JsonObject = {"command": "ls"}
      return await cast(Awaitable[object], handler("Bash", ti, None))
    result = asyncio.run(_go())
  assert type(result).__name__ == "_Allow"
  assert send.call_count == 0  # no card sent for non-askq


def test_askq_handler_empty_questions_returns_empty_answers():
  """Malformed call with no questions returns empty answers without prompting."""
  fixtures = _make_askq_fixtures()
  result = _run_askq(fixtures, questions=[])
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {}
  metadata = cast(JsonObject, result.updated_input["metadata"])
  assert metadata["error"] == "no_questions"


def test_askq_handler_single_button_click_resolves():
  """One button click on a single-question card produces the right answer."""
  questions = [{
    "question": "Where?",
    "header": "Screen",
    "options": [{"label": "Login"}, {"label": "Match"}],
    "multiSelect": False,
  }]
  fixtures = _make_askq_fixtures(events_seq=[_askq_event("askq:abc123:0:0")])
  result = _run_askq(fixtures, questions)
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {"Where?": "Login"}
  assert result.updated_input["questions"] == questions


def test_askq_handler_text_reply_routes_to_first_unanswered():
  """A free-text reply (no button click) is routed to the first unanswered question."""
  questions = [{
    "question": "What?",
    "header": "What",
    "options": [{"label": "a"}],
    "multiSelect": False,
  }]
  fixtures = _make_askq_fixtures(events_seq=[_text_event("custom answer")])
  result = _run_askq(fixtures, questions)
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {"What?": "custom answer"}


def test_askq_handler_other_button_then_text():
  """Clicking 'Other' followed by a text reply uses the text as the answer."""
  questions = [{
    "question": "Where?",
    "header": "Screen",
    "options": [{"label": "Login"}],
    "multiSelect": False,
  }]
  fixtures = _make_askq_fixtures(events_seq=[
    _askq_event("askq:abc123:0:other"),
    _text_event("Settings page"),
  ])
  result = _run_askq(fixtures, questions)
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {"Where?": "Settings page"}


def test_askq_handler_multi_question_two_clicks():
  """Two questions resolved by two clicks."""
  questions = [
    {
      "question": "Where?", "header": "Screen",
      "options": [{"label": "Login"}, {"label": "Match"}],
      "multiSelect": False,
    },
    {
      "question": "Severity?", "header": "How bad?",
      "options": [{"label": "P0"}, {"label": "P3"}],
      "multiSelect": False,
    },
  ]
  fixtures = _make_askq_fixtures(events_seq=[
    _askq_event("askq:abc123:0:1"),  # q0 → Match
    _askq_event("askq:abc123:1:0"),  # q1 → P0
  ])
  result = _run_askq(fixtures, questions)
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {"Where?": "Match", "Severity?": "P0"}


def test_askq_handler_multi_select_accumulates_then_done():
  """multiSelect=True: clicking two options + done returns a list."""
  questions = [{
    "question": "Pick?",
    "header": "Tags",
    "options": [{"label": "a"}, {"label": "b"}, {"label": "c"}],
    "multiSelect": True,
  }]
  fixtures = _make_askq_fixtures(events_seq=[
    _askq_event("askq:abc123:0:0"),  # toggle a
    _askq_event("askq:abc123:0:2"),  # toggle c
    _askq_event("askq:abc123:0:done"),
  ])
  result = _run_askq(fixtures, questions)
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {"Pick?": ["a", "c"]}


def test_askq_handler_wrong_chat_pushed_back():
  """Card actions from a different chat are re-queued, not consumed."""
  questions = [{
    "question": "Q?", "header": "Q",
    "options": [{"label": "x"}],
    "multiSelect": False,
  }]
  wrong = _askq_event("askq:abc123:0:0", chat_id="oc_other")
  right = _askq_event("askq:abc123:0:0", chat_id="oc_test_chat")
  fixtures = _make_askq_fixtures(events_seq=[wrong, right])
  result = _run_askq(fixtures, questions)
  assert type(result).__name__ == "_Allow"
  assert result.updated_input["answers"] == {"Q?": "x"}
  # The wrong-chat event was pushed back at end
  creds, chat_id, events, _sdk = fixtures
  events.push_back.assert_called()


# ---------------------------------------------------------------------------
# Embedded turn-card path (the normal case): askq updates the working
# card in place via channel.turn_ctx, no separate card sent.
# ---------------------------------------------------------------------------


def test_askq_handler_embedded_writes_pending_question():
  """In the embedded path the handler publishes the in-flight question on
  channel.turn_ctx.pending_question and triggers a redraw before any
  click arrives, so the user can see the buttons in the working card."""
  questions = [{
    "question": "Where?", "header": "Screen",
    "options": [{"label": "Login"}],
    "multiSelect": False,
  }]
  fixtures = _make_askq_fixtures(
    events_seq=[_askq_event("askq:abc123:0:0")], embed_card=True)
  creds, chat_id, events, _sdk = fixtures
  result = _run_askq(fixtures, questions)
  # No standalone card was sent — the question was embedded.
  assert events.send_card_mock.call_count == 0
  # First redraw published the pending question before any click.
  initial = events.redraw_calls[0]
  assert initial["pending_question"] is not None
  assert initial["pending_question"].questions == questions
  assert initial["pending_answers"] == {}
  assert result.updated_input["answers"] == {"Where?": "Login"}


def test_askq_handler_embedded_moves_to_answered_history():
  """After all questions are answered, the handler moves them out of
  pending_question and into turn_ctx.answered_questions so the turn card
  keeps showing them as history for the rest of the turn."""
  questions = [{
    "question": "Where?", "header": "Screen",
    "options": [{"label": "Login"}, {"label": "Match"}],
    "multiSelect": False,
  }]
  fixtures = _make_askq_fixtures(
    events_seq=[_askq_event("askq:abc123:0:1")], embed_card=True)
  creds, chat_id, events, _sdk = fixtures
  result = _run_askq(fixtures, questions)
  assert result.updated_input["answers"] == {"Where?": "Match"}
  # Pending cleared and the question landed in answered history.
  ctx = events.turn_ctx
  assert ctx.pending_question is None
  assert len(ctx.answered_questions) == 1
  assert ctx.answered_questions[0].header == "Screen"
  assert ctx.answered_questions[0].question == "Where?"
  assert ctx.answered_questions[0].answer == "Match"


def test_askq_handler_embedded_multi_question_no_lockout():
  """Regression: with multiple questions in one card the user must be
  able to answer them independently. Each click must update only its
  own question and trigger a redraw; the loop must wait for ALL
  answers before clearing pending_question. (The original bug was that
  the answered state appeared to lock out the remaining questions.)"""
  questions = [
    {
      "question": "Where?", "header": "Screen",
      "options": [{"label": "Login"}, {"label": "Match"}],
      "multiSelect": False,
    },
    {
      "question": "Severity?", "header": "How bad?",
      "options": [{"label": "P0"}, {"label": "P3"}],
      "multiSelect": False,
    },
  ]
  fixtures = _make_askq_fixtures(events_seq=[
    _askq_event("askq:abc123:0:1"),  # q0 → Match
    _askq_event("askq:abc123:1:0"),  # q1 → P0  (must still work!)
  ], embed_card=True)
  creds, chat_id, events, _sdk = fixtures
  result = _run_askq(fixtures, questions)
  assert result.updated_input["answers"] == {
    "Where?": "Match", "Severity?": "P0"}
  # After the first click, pending_question must still be present
  # (loop has not concluded) — confirmed by a redraw that still has
  # pending_question != None.
  intermediate = events.redraw_calls[1]
  assert intermediate["pending_question"] is not None
  assert intermediate["pending_answers"] == {0: "Match"}
  # After both clicks resolved, the final redraw drops pending and
  # carries both questions in the answered history.
  final = events.redraw_calls[-1]
  assert final["pending_question"] is None
  headers = {aq.header for aq in final["answered_questions"]}
  assert headers == {"Screen", "How bad?"}


def test_askq_handler_fallback_to_standalone_card_without_working_card():
  """When AskUserQuestion fires before any working card exists (and the
  agent loop can't create one — e.g. transient Lark error), the handler
  must fall back to sending a standalone card so the user can still
  answer instead of silently hanging."""
  questions = [{
    "question": "Q?", "header": "Q",
    "options": [{"label": "x"}],
    "multiSelect": False,
  }]
  # embed_card=False → redraw never sets turn_card_id, simulating a
  # turn with no working card yet.
  fixtures = _make_askq_fixtures(
    events_seq=[_askq_event("askq:abc123:0:0")], embed_card=False)
  creds, chat_id, events, _sdk = fixtures
  result = _run_askq(fixtures, questions)
  events.send_card_mock.assert_called_once()
  assert result.updated_input["answers"] == {"Q?": "x"}
