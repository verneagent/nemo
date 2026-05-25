"""Tests for nemo.relay_events — relay message conversion and event stream."""

import asyncio
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from nemo.lark.events import LarkEvent
from nemo.relay_events import RelayEventStream, _relay_msg_to_event


# ---------------------------------------------------------------------------
# _relay_msg_to_event — message type conversion
# ---------------------------------------------------------------------------

def test_convert_text_message():
    msg = {
        "text": "hello world",
        "msg_type": "text",
        "sender_id": "ou_user1",
        "message_id": "om_msg1",
        "create_time": "1700000000000",
        "parent_id": "om_parent1",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "im.message.receive_v1"
    assert ev.chat_id == "oc_chat1"
    assert ev.text == "hello world"
    assert ev.msg_type == "text"
    assert ev.sender_id == "ou_user1"
    assert ev.message_id == "om_msg1"
    assert ev.create_time == "1700000000000"
    assert ev.parent_id == "om_parent1"


def test_convert_thread_id():
    """A message that belongs to a Lark thread carries thread_id; the
    converter must forward it so /fork sub-thread routing works on the
    relay path (the direct-Lark path already keeps it)."""
    msg = {
        "text": "back in the fork thread",
        "msg_type": "text",
        "sender_id": "ou_user1",
        "message_id": "om_msg9",
        "create_time": "1700000099000",
        "thread_id": "omt_fork_abc",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.thread_id == "omt_fork_abc"


def test_convert_no_thread_id_defaults_empty():
    """Top-level (non-threaded) messages have no thread_id."""
    msg = {"text": "top level", "msg_type": "text", "sender_id": "ou_u"}
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.thread_id == ""


def test_convert_image_message():
    msg = {
        "text": "[image]",
        "msg_type": "image",
        "image_key": "img_abc123",
        "sender_id": "ou_user2",
        "create_time": "1700000001000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "im.message.receive_v1"
    assert ev.msg_type == "image"
    assert ev.image_key == "img_abc123"


def test_convert_file_message():
    msg = {
        "text": "[file: readme.txt]",
        "msg_type": "file",
        "file_key": "file_xyz",
        "file_name": "readme.txt",
        "sender_id": "ou_user3",
        "create_time": "1700000002000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.msg_type == "file"
    assert ev.file_key == "file_xyz"
    assert ev.file_name == "readme.txt"


def test_convert_post_message():
    msg = {
        "text": "rich text\nwith lines",
        "msg_type": "post",
        "sender_id": "ou_user4",
        "create_time": "1700000003000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.msg_type == "post"
    assert "rich text" in ev.text


def test_convert_with_mentions():
    msg = {
        "text": "@Bot hello",
        "msg_type": "text",
        "sender_id": "ou_user5",
        "create_time": "1700000004000",
        "mentions": [{"key": "@_user_1", "id": "ou_bot1", "name": "Bot"}],
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert len(ev.mentions) == 1
    assert ev.mentions[0]["id"] == "ou_bot1"
    assert ev.mentions[0]["key"] == "@_user_1"


def test_convert_mentions_invalid():
    """Non-list mentions should be treated as empty."""
    msg = {
        "text": "no mentions",
        "msg_type": "text",
        "sender_id": "ou_user6",
        "create_time": "1700000005000",
        "mentions": "invalid",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.mentions == []


# ---------------------------------------------------------------------------
# Card action conversion
# ---------------------------------------------------------------------------

def test_convert_button_action():
    msg = {
        "text": "approve",
        "msg_type": "button_action",
        "sender_id": "ou_op1",
        "create_time": "1700000010000",
        "message_id": "om_card1",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "card.action.trigger"
    assert ev.action_value == {"action": "approve"}
    assert ev.operator_id == "ou_op1"
    assert ev.sender_id == "ou_op1"


def test_convert_select_action():
    msg = {
        "text": "option_a",
        "msg_type": "select_action",
        "sender_id": "ou_op2",
        "create_time": "1700000011000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "card.action.trigger"
    assert ev.action_value == {"action": "option_a"}


def test_convert_form_action():
    msg = {
        "text": "form_data",
        "msg_type": "form_action",
        "sender_id": "ou_op3",
        "create_time": "1700000012000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "card.action.trigger"


def test_convert_form_action_for_model_picker_keeps_prefix():
    """End-to-end shape check: the relay pushes the picker's
    ``model_switch:<name>`` form_value as the reply text; the
    converter must land it in action_value['action'] unchanged so the
    daemon's ``startswith("model_switch:")`` routing still fires.

    The picker card's own message_id rides along via reply.message_id
    so daemon-side handlers can PATCH the originating card after the
    switch completes."""
    msg = {
        "text": "model_switch:claude-sonnet-4-6",
        "msg_type": "form_action",
        "sender_id": "ou_picker_clicker",
        "create_time": "1700000020000",
        "message_id": "om_picker_xyz",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "card.action.trigger"
    assert ev.action_value == {"action": "model_switch:claude-sonnet-4-6"}
    assert ev.operator_id == "ou_picker_clicker"
    assert ev.message_id == "om_picker_xyz"


def test_convert_input_action():
    msg = {
        "text": "user input text",
        "msg_type": "input_action",
        "sender_id": "ou_op4",
        "create_time": "1700000013000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "card.action.trigger"
    assert ev.action_value == {"action": "user input text"}


# ---------------------------------------------------------------------------
# Stop signal conversion
# ---------------------------------------------------------------------------

def test_convert_stop_signal():
    msg = {
        "text": "__stop__",
        "msg_type": "stop_signal",
        "create_time": "1700000020000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "card.action.trigger"
    assert ev.action_value == {"action": "stop"}
    assert ev.chat_id == "oc_chat1"


# ---------------------------------------------------------------------------
# Reaction conversion
# ---------------------------------------------------------------------------

def test_convert_reaction():
    msg = {
        "text": "THUMBSUP",
        "msg_type": "reaction",
        "sender_id": "ou_reactor1",
        "target_message_id": "om_target1",
        "create_time": "1700000030000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "im.message.reaction.created_v1"
    assert ev.text == "THUMBSUP"
    assert ev.message_id == "om_target1"
    assert ev.sender_id == "ou_reactor1"


# ---------------------------------------------------------------------------
# Recall conversion
# ---------------------------------------------------------------------------

def test_convert_recall():
    msg = {
        "msg_type": "recall",
        "message_id": "om_recalled1",
        "create_time": "1700000040000",
    }
    ev = _relay_msg_to_event(msg, "oc_chat1")
    assert ev.event_type == "im.message.recalled_v1"
    assert ev.message_id == "om_recalled1"
    assert ev.chat_id == "oc_chat1"


# ---------------------------------------------------------------------------
# Missing fields — graceful defaults
# ---------------------------------------------------------------------------

def test_convert_minimal_message():
    """Message with only msg_type should not crash."""
    ev = _relay_msg_to_event({"msg_type": "text"}, "oc_1")
    assert ev.event_type == "im.message.receive_v1"
    assert ev.text == ""
    assert ev.sender_id == ""
    assert ev.chat_id == "oc_1"


def test_convert_empty_dict():
    """Empty dict should produce a valid LarkEvent."""
    ev = _relay_msg_to_event({}, "oc_1")
    assert ev.event_type == "im.message.receive_v1"


# ---------------------------------------------------------------------------
# RelayEventStream — async behavior
# ---------------------------------------------------------------------------

def test_stream_next_message_timeout():
    """next_message should return None on timeout."""
    async def _run():
        stream = RelayEventStream("http://localhost:9999", "key", "oc_test")
        stream._running = True
        result = await stream.next_message(timeout=0.05)
        assert result is None
    asyncio.run(_run())


def test_stream_push_back():
    """push_back should re-queue an event."""
    async def _run():
        stream = RelayEventStream("http://localhost:9999", "key", "oc_test")
        stream._running = True
        ev = LarkEvent(event_type="im.message.receive_v1", text="pushed")
        stream.push_back(ev)
        result = await stream.next_message(timeout=1)
        assert result is not None
        assert result.text == "pushed"
    asyncio.run(_run())


def test_stream_close_unblocks():
    """close() should unblock next_message."""
    async def _run():
        stream = RelayEventStream("http://localhost:9999", "key", "oc_test")
        stream._running = True
        await stream.close()
        result = await stream.next_message(timeout=1)
        assert result is None
    asyncio.run(_run())


def test_stream_permission_active_flag():
    """permission_active should default to False."""
    stream = RelayEventStream("http://localhost:9999", "key", "oc_test")
    assert stream.permission_active is False
    stream.permission_active = True
    assert stream.permission_active is True


def test_ws_loop_passes_proxy_none(monkeypatch):
    """The WS connect must pass proxy=None so HTTP_PROXY / HTTPS_PROXY /
    ALL_PROXY env vars (e.g. ClashX in China) don't get auto-applied to
    relay connections. Pre-fix, websockets >=15.0 default proxy=True
    auto-pulled the local proxy and every relay WS connect failed with
    'did not receive a valid HTTP response from proxy'."""
    captured: dict = {}

    class _FakeWS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def recv(self, timeout=None):
            raise StopIteration  # break the inner loop

    def _fake_connect(uri, **kwargs):
        captured.update(kwargs)
        captured["uri"] = uri
        # Stop the outer reconnect loop after one call
        stream._running = False
        return _FakeWS()

    import websockets.sync.client as ws_client
    monkeypatch.setattr(ws_client, "connect", _fake_connect)

    stream = RelayEventStream("http://relay.example.com", "k", "oc_test")
    stream._running = True
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        stream._ws_loop(loop)
    finally:
        loop.close()

    assert captured.get("uri") == "ws://relay.example.com/ws/chat:oc_test"
    assert "proxy" in captured, "proxy kwarg must be set explicitly"
    assert captured["proxy"] is None, (
        f"proxy must be None to bypass env-var proxies; got {captured['proxy']!r}"
    )


# ---------------------------------------------------------------------------
# RelayEventStream + relay server integration
# ---------------------------------------------------------------------------

# These tests start a real relay server and verify end-to-end behavior.

def _start_relay_server():
    """Start the relay test server in background, return (loop, runner)."""
    import sys
    sys.path.insert(0, "/private/tmp/claude/nemo-relay")
    os.environ["RELAY_PORT"] = "19802"
    os.environ["RELAY_DB"] = "/private/tmp/claude/test_relay_events.db"
    os.environ["RELAY_API_KEY"] = "test-key"
    os.environ["VERIFY_TOKENS"] = "tok1"

    import importlib
    import relay as relay_mod
    importlib.reload(relay_mod)

    if os.path.exists("/private/tmp/claude/test_relay_events.db"):
        os.remove("/private/tmp/claude/test_relay_events.db")
    relay_mod._init_db()

    loop = asyncio.new_event_loop()
    app = relay_mod.create_app()

    async def start():
        runner = relay_mod.web.AppRunner(app)
        await runner.setup()
        site = relay_mod.web.TCPSite(runner, "127.0.0.1", 19802)
        await site.start()
        return runner

    runner = loop.run_until_complete(start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, runner, thread, relay_mod


class RelayIntegrationTest(unittest.TestCase):
    """Integration tests: RelayEventStream ↔ relay server."""

    relay_mod = None
    server_loop = None
    server_runner = None
    server_thread = None

    @classmethod
    def setUpClass(cls):
        cls.server_loop, cls.server_runner, cls.server_thread, cls.relay_mod = \
            _start_relay_server()

    @classmethod
    def tearDownClass(cls):
        if cls.server_runner and cls.server_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    cls.server_runner.cleanup(), cls.server_loop
                ).result(10)
            except Exception:
                pass
        if cls.server_loop:
            cls.server_loop.call_soon_threadsafe(cls.server_loop.stop)
            if cls.server_thread:
                cls.server_thread.join(3)
        db_path = "/private/tmp/claude/test_relay_events.db"
        if os.path.exists(db_path):
            os.remove(db_path)

    def _push_webhook(self, chat_id, text, create_time, evt_id=None):
        """Push a message through the webhook."""
        import urllib.request
        data = {
            "header": {
                "token": "tok1",
                "event_type": "im.message.receive_v1",
                "event_id": evt_id or f"evt_{create_time}",
            },
            "event": {
                "message": {
                    "chat_id": chat_id,
                    "message_type": "text",
                    "content": json.dumps({"text": text}),
                    "create_time": create_time,
                    "message_id": f"msg_{create_time}",
                },
                "sender": {
                    "sender_type": "user",
                    "sender_id": {"open_id": "ou_tester"},
                },
            },
        }
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:19802/webhook",
            data=body, method="POST",
        )
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=5)

    def test_ws_receives_message(self):
        """Message pushed via webhook should arrive on WS event stream."""
        chat_id = "oc_ws_test1"

        async def run():
            stream = RelayEventStream(
                "http://127.0.0.1:19802", "test-key", chat_id)
            await stream.connect()
            # Give WS time to connect
            await asyncio.sleep(0.3)

            # Push a message through webhook
            self._push_webhook(chat_id, "hello ws", "1700100000000")

            # Should receive it via WS
            ev = await stream.next_message(timeout=5)
            assert ev is not None, "Expected message, got None"
            assert ev.text == "hello ws"
            assert ev.event_type == "im.message.receive_v1"
            assert ev.sender_id == "ou_tester"
            assert ev.chat_id == chat_id

            await stream.close()

        asyncio.run(run())

    def test_ws_receives_multiple(self):
        """Multiple messages should all arrive."""
        chat_id = "oc_ws_test2"

        async def run():
            stream = RelayEventStream(
                "http://127.0.0.1:19802", "test-key", chat_id)
            await stream.connect()
            await asyncio.sleep(0.3)

            self._push_webhook(chat_id, "msg1", "1700200001000", "e1")
            self._push_webhook(chat_id, "msg2", "1700200002000", "e2")

            ev1 = await stream.next_message(timeout=5)
            assert ev1 is not None
            # Both messages may arrive in a single WS frame
            ev2 = await stream.next_message(timeout=5)
            assert ev2 is not None
            texts = {ev1.text, ev2.text}
            assert texts == {"msg1", "msg2"}

            await stream.close()

        asyncio.run(run())

    def test_ws_stop_signal(self):
        """POST /stop should deliver stop event via WS."""
        chat_id = "oc_ws_test3"

        async def run():
            stream = RelayEventStream(
                "http://127.0.0.1:19802", "test-key", chat_id)
            await stream.connect()
            await asyncio.sleep(0.3)

            # Send stop signal
            import urllib.request
            req = urllib.request.Request(
                f"http://127.0.0.1:19802/stop/chat:{chat_id}",
                data=b"", method="POST",
            )
            req.add_header("Authorization", "Bearer test-key")
            urllib.request.urlopen(req, timeout=5)

            # Should receive stop event
            ev = await stream.next_message(timeout=5)
            assert ev is not None
            assert ev.event_type == "card.action.trigger"
            assert ev.action_value.get("action") == "stop"

            await stream.close()

        asyncio.run(run())

    def test_ws_card_action(self):
        """Card action should arrive as card.action.trigger event."""
        chat_id = "oc_ws_test4"

        async def run():
            stream = RelayEventStream(
                "http://127.0.0.1:19802", "test-key", chat_id)
            await stream.connect()
            await asyncio.sleep(0.3)

            # Push card action through webhook
            data = {
                "header": {
                    "token": "tok1",
                    "event_type": "card.action.trigger",
                    "event_id": "evt_card1",
                },
                "event": {
                    "operator": {"open_id": "ou_op1"},
                    "action": {
                        "value": {
                            "action": "approve",
                            "chat_id": chat_id,
                        },
                    },
                },
            }
            body = json.dumps(data).encode()
            req = __import__("urllib.request", fromlist=["Request"]).Request(
                "http://127.0.0.1:19802/webhook",
                data=body, method="POST",
            )
            req.add_header("Content-Type", "application/json")
            __import__("urllib.request", fromlist=["urlopen"]).urlopen(req, timeout=5)

            ev = await stream.next_message(timeout=5)
            assert ev is not None
            assert ev.event_type == "card.action.trigger"
            assert ev.action_value.get("action") == "approve"

            await stream.close()

        asyncio.run(run())

    def test_no_cross_chat_leak(self):
        """Messages for another chat should not appear."""
        chat_id = "oc_ws_test5"
        other_chat = "oc_ws_other"

        async def run():
            stream = RelayEventStream(
                "http://127.0.0.1:19802", "test-key", chat_id)
            await stream.connect()
            await asyncio.sleep(0.3)

            # Push to different chat
            self._push_webhook(other_chat, "wrong chat", "1700500001000")
            # Push to our chat
            self._push_webhook(chat_id, "right chat", "1700500002000")

            ev = await stream.next_message(timeout=5)
            assert ev is not None
            assert ev.text == "right chat"

            await stream.close()

        asyncio.run(run())
