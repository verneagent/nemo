"""Tests for nemo.lark.events — event parsing and stream behavior."""

import asyncio
import json

from nemo.lark.events import (
  LarkEvent, LarkEventStream, parse_event, _model_to_dict,
  _parse_message_event, _parse_card_action, _parse_reaction_event,
)


# ---------------------------------------------------------------------------
# parse_event / _parse_message_event
# ---------------------------------------------------------------------------

def test_parse_message_event():
  payload = {
    "schema": "2.0",
    "header": {"event_type": "im.message.receive_v1", "app_id": "cli_test"},
    "event": {
      "message": {
        "chat_id": "oc_123",
        "chat_type": "p2p",
        "content": json.dumps({"text": "hello"}),
        "message_id": "om_abc",
        "message_type": "text",
        "create_time": "1234567890",
        "parent_id": "om_parent",
      },
      "sender": {
        "sender_id": {"open_id": "ou_user1"},
        "sender_type": "user",
      },
    },
  }
  ev = parse_event(payload)
  assert ev.event_type == "im.message.receive_v1"
  assert ev.chat_id == "oc_123"
  assert ev.chat_type == "p2p"
  assert ev.sender_id == "ou_user1"
  assert ev.message_id == "om_abc"
  assert ev.msg_type == "text"
  assert ev.text == "hello"
  assert ev.parent_id == "om_parent"
  assert ev.create_time == "1234567890"


def test_parse_message_image():
  payload = {
    "header": {"event_type": "im.message.receive_v1"},
    "event": {
      "message": {
        "chat_id": "oc_1",
        "content": json.dumps({"image_key": "img_abc"}),
        "message_type": "image",
        "message_id": "om_1",
      },
      "sender": {"sender_id": {"open_id": "ou_1"}},
    },
  }
  ev = parse_event(payload)
  assert ev.image_key == "img_abc"
  assert ev.msg_type == "image"


def test_parse_message_bad_content():
  """Malformed content JSON should not crash."""
  payload = {
    "header": {"event_type": "im.message.receive_v1"},
    "event": {
      "message": {
        "chat_id": "oc_1",
        "content": "not json",
        "message_id": "om_1",
      },
      "sender": {"sender_id": {}},
    },
  }
  ev = parse_event(payload)
  assert ev.text == ""
  assert ev.chat_id == "oc_1"


# ---------------------------------------------------------------------------
# _parse_card_action
# ---------------------------------------------------------------------------

def test_parse_card_action():
  payload = {
    "header": {"event_type": "card.action.trigger"},
    "event": {
      "operator": {"open_id": "ou_op1", "tenant_key": "tk"},
      "action": {"value": {"action": "approve", "nonce": "abc"}, "tag": "button"},
      "context": {
        "open_message_id": "om_card1",
        "open_chat_id": "oc_chat1",
      },
    },
  }
  ev = parse_event(payload)
  assert ev.event_type == "card.action.trigger"
  assert ev.action_value == {"action": "approve", "nonce": "abc"}
  assert ev.action_tag == "button"
  assert ev.operator_id == "ou_op1"
  assert ev.chat_id == "oc_chat1"
  assert ev.message_id == "om_card1"


# ---------------------------------------------------------------------------
# _parse_reaction_event
# ---------------------------------------------------------------------------

def test_parse_reaction():
  payload = {
    "header": {"event_type": "im.message.reaction.created_v1"},
    "event": {
      "message_id": "om_react1",
      "user_id": {"open_id": "ou_reactor"},
    },
  }
  ev = parse_event(payload)
  assert ev.event_type == "im.message.reaction.created_v1"
  assert ev.message_id == "om_react1"
  assert ev.sender_id == "ou_reactor"


# ---------------------------------------------------------------------------
# Unknown event type
# ---------------------------------------------------------------------------

def test_parse_unknown_event():
  payload = {
    "header": {"event_type": "some.unknown.event"},
    "event": {"foo": "bar"},
  }
  ev = parse_event(payload)
  assert ev.event_type == "some.unknown.event"
  assert ev.raw == payload


# ---------------------------------------------------------------------------
# LarkEventStream async behavior
# ---------------------------------------------------------------------------

def test_stream_next_event_timeout():
  """next_event should return None on timeout."""
  async def _run():
    stream = LarkEventStream("app", "secret")
    stream._running = True
    result = await stream.next_event(timeout=0.05)
    assert result is None
  asyncio.run(_run())


def test_stream_stop_sentinel():
  """Stopping the stream should unblock consumers."""
  async def _run():
    stream = LarkEventStream("app", "secret")
    stream._running = True
    # Put sentinel
    await stream.stop()
    result = await stream.next_event(timeout=1)
    assert result is None
  asyncio.run(_run())


def test_stream_iteration():
  """Test async iteration over queued events."""
  async def _run():
    stream = LarkEventStream("app", "secret")
    stream._running = True
    # Enqueue events
    ev1 = LarkEvent(event_type="im.message.receive_v1", text="hi")
    ev2 = LarkEvent(event_type="card.action.trigger", action_value={"a": 1})
    stream._queue.put_nowait(ev1)
    stream._queue.put_nowait(ev2)
    stream._queue.put_nowait(LarkEvent(event_type="_stop"))
    collected = []
    async for ev in stream:
      collected.append(ev)
    assert len(collected) == 2
    assert collected[0].text == "hi"
    assert collected[1].action_value == {"a": 1}
  asyncio.run(_run())


# ---------------------------------------------------------------------------
# _model_to_dict — SDK model object conversion
# ---------------------------------------------------------------------------

class FakeSenderId:
  def __init__(self):
    self.open_id = "ou_user1"

class FakeSender:
  def __init__(self):
    self.sender_id = FakeSenderId()
    self.sender_type = "user"

class FakeMessage:
  def __init__(self):
    self.chat_id = "oc_123"
    self.chat_type = "p2p"
    self.content = '{"text": "hello"}'
    self.message_id = "om_abc"
    self.message_type = "text"
    self.create_time = "1234567890"
    self.parent_id = "om_parent"
    self.mentions = None

class FakeEvent:
  def __init__(self):
    self.message = FakeMessage()
    self.sender = FakeSender()


def test_model_to_dict_primitives():
  assert _model_to_dict(None) is None
  assert _model_to_dict("hello") == "hello"
  assert _model_to_dict(42) == 42
  assert _model_to_dict(True) is True


def test_model_to_dict_plain_dict():
  d = {"a": 1, "b": "x"}
  assert _model_to_dict(d) == {"a": 1, "b": "x"}


def test_model_to_dict_list():
  assert _model_to_dict([1, "a", None]) == [1, "a", None]


def test_model_to_dict_sdk_objects():
  """Simulate lark-oapi SDK model objects (nested, no .get())."""
  event = FakeEvent()
  result = _model_to_dict(event)
  assert isinstance(result, dict)
  assert result["message"]["chat_id"] == "oc_123"
  assert result["message"]["message_type"] == "text"
  assert result["sender"]["sender_id"]["open_id"] == "ou_user1"


def test_model_to_dict_full_parse():
  """Converted SDK objects should be parseable by parse_event."""
  event = FakeEvent()
  payload = {
    "header": {"event_type": "im.message.receive_v1"},
    "event": _model_to_dict(event),
  }
  ev = parse_event(payload)
  assert ev.event_type == "im.message.receive_v1"
  assert ev.chat_id == "oc_123"
  assert ev.sender_id == "ou_user1"
  assert ev.text == "hello"
  assert ev.msg_type == "text"


def test_model_to_dict_skips_private():
  """Private attributes (starting with _) should be excluded."""
  class Obj:
    def __init__(self):
      self.public = "yes"
      self._private = "no"
  result = _model_to_dict(Obj())
  assert "public" in result
  assert "_private" not in result


def test_push_back():
  """push_back should re-queue an event so it can be retrieved next."""
  async def _run():
    stream = LarkEventStream("app", "secret")
    stream._running = True
    ev = LarkEvent(event_type="im.message.receive_v1", text="pushed back")
    stream.push_back(ev)
    result = await stream.next_event(timeout=1)
    assert result is not None
    assert result.text == "pushed back"
    assert result.event_type == "im.message.receive_v1"
  asyncio.run(_run())
