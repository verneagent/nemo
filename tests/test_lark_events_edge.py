"""Tests for nemo.lark.events — push_back ordering, stop behavior."""

import asyncio

from nemo.lark.events import LarkEvent, LarkEventStream, parse_event


# ---------------------------------------------------------------------------
# push_back ordering
# ---------------------------------------------------------------------------

def _run(coro):
  loop = asyncio.new_event_loop()
  try:
    return loop.run_until_complete(coro)
  finally:
    loop.close()


def test_push_back_goes_to_back():
  """push_back uses put_nowait, so pushed events go to the BACK of the queue.

  This means if A is in queue and we push_back B, next_event returns A first.
  This is a known limitation — push_back doesn't actually push to front.
  """
  async def run():
    stream = LarkEventStream("id", "secret")
    stream._running = True
    # Put A in queue
    event_a = LarkEvent(event_type="test", text="A")
    event_b = LarkEvent(event_type="test", text="B")
    stream._queue.put_nowait(event_a)
    # Push back B
    stream.push_back(event_b)
    # A comes first (FIFO), then B
    first = await stream.next_event(timeout=1)
    second = await stream.next_event(timeout=1)
    assert first.text == "A"
    assert second.text == "B"
    return True

  assert _run(run())


def test_next_event_timeout():
  """next_event returns None on timeout."""
  async def run():
    stream = LarkEventStream("id", "secret")
    stream._running = True
    result = await stream.next_event(timeout=0.1)
    assert result is None

  _run(run())


def test_stop_unblocks_consumer():
  """stop() puts sentinel that causes __anext__ to raise StopAsyncIteration."""
  async def run():
    stream = LarkEventStream("id", "secret")
    stream._running = True
    await stream.stop()
    events = []
    async for e in stream:
      events.append(e)
    assert events == []

  _run(run())


def test_next_event_returns_none_on_stop():
  """next_event returns None when stop sentinel is received."""
  async def run():
    stream = LarkEventStream("id", "secret")
    stream._running = True
    await stream.stop()
    result = await stream.next_event(timeout=5)
    assert result is None

  _run(run())


# ---------------------------------------------------------------------------
# parse_event
# ---------------------------------------------------------------------------

def test_parse_message_event():
  payload = {
    "header": {"event_type": "im.message.receive_v1"},
    "event": {
      "message": {
        "chat_id": "oc_123",
        "chat_type": "group",
        "message_id": "om_456",
        "message_type": "text",
        "content": '{"text": "hello"}',
        "create_time": "1234567890",
      },
      "sender": {"sender_id": {"open_id": "ou_789"}},
    },
  }
  e = parse_event(payload)
  assert e.event_type == "im.message.receive_v1"
  assert e.chat_id == "oc_123"
  assert e.sender_id == "ou_789"
  assert e.text == "hello"


def test_parse_card_action():
  payload = {
    "header": {"event_type": "card.action.trigger"},
    "event": {
      "operator": {"open_id": "ou_111"},
      "action": {"value": {"action": "approve"}, "tag": "button"},
      "context": {"open_chat_id": "oc_222", "open_message_id": "om_333"},
    },
  }
  e = parse_event(payload)
  assert e.event_type == "card.action.trigger"
  assert e.chat_id == "oc_222"
  assert e.action_value == {"action": "approve"}


def test_parse_reaction_event():
  payload = {
    "header": {"event_type": "im.message.reaction.created_v1"},
    "event": {
      "message_id": "om_999",
      "user_id": {"open_id": "ou_aaa"},
    },
  }
  e = parse_event(payload)
  assert e.event_type == "im.message.reaction.created_v1"
  assert e.message_id == "om_999"
  assert e.sender_id == "ou_aaa"


def test_parse_unknown_event():
  payload = {"header": {"event_type": "unknown.event"}, "event": {}}
  e = parse_event(payload)
  assert e.event_type == "unknown.event"


def test_parse_malformed_content():
  """Non-JSON content string should not crash."""
  payload = {
    "header": {"event_type": "im.message.receive_v1"},
    "event": {
      "message": {"content": "not-json", "chat_id": "oc_1"},
      "sender": {"sender_id": {}},
    },
  }
  e = parse_event(payload)
  assert e.text == ""
  assert e.chat_id == "oc_1"
