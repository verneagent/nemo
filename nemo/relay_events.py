"""Relay-based event stream — connects to the nemo relay via WebSocket.

Replaces LarkEventStream (lark-oapi WebSocket) when relay is configured.
Same interface: next_message(), push_back(), permission_active flag.

The relay receives Lark webhooks and card actions, parses them into a
flat message format, and stores them. This module connects to the relay's
/ws/chat:{chatId} WebSocket and receives real-time push notifications.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.request

from .lark.events import LarkEvent
from .types import JsonObject

log = logging.getLogger(__name__)


def _relay_msg_to_event(msg: JsonObject, chat_id: str) -> LarkEvent:
    """Convert a relay message dict to a LarkEvent."""
    msg_type = msg.get("msg_type", "")

    # Card action messages from relay
    if msg_type in ("button_action", "select_action", "input_action", "form_action"):
        return LarkEvent(
            event_type="card.action.trigger",
            chat_id=chat_id,
            sender_id=msg.get("sender_id", ""),
            operator_id=msg.get("sender_id", ""),
            action_value={"action": msg.get("text", "")},
            create_time=msg.get("create_time", ""),
            message_id=msg.get("message_id", ""),
        )

    # Stop signal
    if msg_type == "stop_signal":
        return LarkEvent(
            event_type="card.action.trigger",
            chat_id=chat_id,
            sender_id=msg.get("sender_id", ""),
            operator_id=msg.get("sender_id", ""),
            action_value={"action": "stop"},
            create_time=msg.get("create_time", ""),
        )

    # Reaction
    if msg_type == "reaction":
        return LarkEvent(
            event_type="im.message.reaction.created_v1",
            chat_id=chat_id,
            sender_id=msg.get("sender_id", ""),
            message_id=msg.get("target_message_id", ""),
            text=msg.get("text", ""),
            create_time=msg.get("create_time", ""),
        )

    # Regular message
    mentions = msg.get("mentions", [])
    return LarkEvent(
        event_type="im.message.receive_v1",
        chat_id=chat_id,
        sender_id=msg.get("sender_id", ""),
        message_id=msg.get("message_id", ""),
        msg_type=msg_type,
        text=msg.get("text", ""),
        mentions=mentions if isinstance(mentions, list) else [],
        image_key=msg.get("image_key", ""),
        file_key=msg.get("file_key", ""),
        file_name=msg.get("file_name", ""),
        parent_id=msg.get("parent_id", ""),
        create_time=msg.get("create_time", ""),
    )


class RelayEventStream:
    """Event stream backed by relay WebSocket.

    Connects to /ws/chat:{chatId} for real-time push.
    Falls back to long-poll if websockets library is unavailable.
    Converts relay messages to LarkEvent objects.
    """

    def __init__(self, relay_url: str, api_key: str, chat_id: str):
        self._relay_url = relay_url.rstrip("/")
        self._api_key = api_key
        self._chat_id = chat_id
        self._key = f"chat:{chat_id}"
        self._queue: asyncio.Queue[LarkEvent] = asyncio.Queue()
        self._running = False
        self._bg_thread: threading.Thread | None = None
        self._since = ""
        self._ws: object = None  # Current WS connection (for force-close)
        self.permission_active: bool = False

    # --- WebSocket approach (preferred) ---

    def _ws_url(self) -> str:
        """Build WebSocket URL from HTTP relay URL."""
        url = self._relay_url.replace("http://", "ws://").replace("https://", "wss://")
        ws = f"{url}/ws/{self._key}"
        if self._since:
            ws += f"?since={self._since}"
        return ws

    def _ws_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Background thread: connect via websockets, push events to queue."""
        import websockets.sync.client as ws_client

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        while self._running:
            try:
                with ws_client.connect(self._ws_url(),
                                       additional_headers=headers,
                                       close_timeout=2) as ws:
                    self._ws = ws  # Save for force-close on shutdown
                    log.info("WS connected to relay")
                    # Ping every 30s to keep alive
                    ws.recv_bufsize = 1024 * 1024
                    while self._running:
                        try:
                            raw = ws.recv(timeout=30)
                        except TimeoutError:
                            # Send ping
                            ws.send(json.dumps({"ping": True}))
                            continue

                        data = json.loads(raw)
                        replies = data.get("replies", [])

                        if data.get("takeover"):
                            continue  # nemo doesn't use takeover

                        if data.get("stop"):
                            event = LarkEvent(
                                event_type="card.action.trigger",
                                chat_id=self._chat_id,
                                action_value={"action": "stop"},
                            )
                            loop.call_soon_threadsafe(self._queue.put_nowait, event)
                            continue

                        max_ct = ""
                        for msg in replies:
                            ct = msg.get("create_time", "")
                            if ct > max_ct:
                                max_ct = ct
                            event = _relay_msg_to_event(msg, self._chat_id)
                            loop.call_soon_threadsafe(self._queue.put_nowait, event)

                        if max_ct:
                            # Ack via WS
                            ws.send(json.dumps({"ack": max_ct}))
                            self._since = max_ct

            except Exception as e:
                if self._running:
                    log.warning("WS error, reconnecting in 3s: %s", e)
                    time.sleep(3)

    # --- Long-poll fallback ---

    def _poll_once(self, timeout: int = 25) -> JsonObject:
        url = f"{self._relay_url}/poll/{self._key}?timeout={timeout}"
        if self._since:
            url += f"&since={self._since}"
        req = urllib.request.Request(url)
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
            return json.loads(resp.read())

    def _ack_http(self, before: str) -> None:
        url = f"{self._relay_url}/replies/{self._key}/ack?before={before}"
        req = urllib.request.Request(url, data=b"", method="POST")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            log.debug("Ack failed: %s", e)

    def _poll_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        while self._running:
            try:
                data = self._poll_once()
                replies = data.get("replies", [])
                if data.get("stop"):
                    event = LarkEvent(
                        event_type="card.action.trigger",
                        chat_id=self._chat_id,
                        action_value={"action": "stop"},
                    )
                    loop.call_soon_threadsafe(self._queue.put_nowait, event)
                    continue

                if not replies:
                    continue

                max_ct = ""
                for msg in replies:
                    ct = msg.get("create_time", "")
                    if ct > max_ct:
                        max_ct = ct
                    event = _relay_msg_to_event(msg, self._chat_id)
                    loop.call_soon_threadsafe(self._queue.put_nowait, event)

                if max_ct:
                    self._ack_http(max_ct)
                    self._since = max_ct

            except Exception as e:
                if self._running:
                    log.warning("Poll error: %s", e)
                    time.sleep(3)

    # --- Public interface ---

    async def connect(self) -> None:
        """Start WS connection (or long-poll fallback) in a background thread."""
        self._running = True
        loop = asyncio.get_event_loop()

        # Prefer websockets library, fall back to long-poll
        try:
            import websockets.sync.client  # noqa: F401
            target = self._ws_loop
            mode = "ws"
        except ImportError:
            target = self._poll_loop
            mode = "poll"

        self._bg_thread = threading.Thread(
            target=target,
            args=(loop,),
            daemon=True,
            name=f"relay-{mode}",
        )
        self._bg_thread.start()
        log.info("RelayEventStream started (%s %s)", mode, self._key)

    async def close(self) -> None:
        """Stop the background connection."""
        self._running = False
        # Force-close WebSocket to unblock recv()
        ws = self._ws
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        await self._queue.put(LarkEvent(event_type="_stop"))
        if self._bg_thread:
            self._bg_thread.join(timeout=1)
        log.info("RelayEventStream stopped")

    async def next_message(self, timeout: float = 300) -> LarkEvent | None:
        """Wait for the next event with timeout."""
        try:
            event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            if event.event_type == "_stop":
                return None
            return event
        except asyncio.TimeoutError:
            return None

    def push_back(self, event: LarkEvent) -> None:
        """Re-queue an event so it will be consumed next."""
        self._queue.put_nowait(event)

    # Aliases
    async def start(self) -> None:
        return await self.connect()

    async def stop(self) -> None:
        return await self.close()
