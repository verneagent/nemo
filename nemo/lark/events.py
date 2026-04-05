"""Lark WebSocket event subscription (长连接).

Connects directly to Lark's WebSocket gateway to receive events.
No Cloudflare Worker needed — events arrive directly from Lark.

This is the core architectural difference from handoff_agent.py.

TODO (Phase 1): Implement the actual WebSocket connection.
Research needed:
- Lark WebSocket gateway endpoint and auth flow
- Reference: lark-cli event +subscribe implementation
- Event format and acknowledgment protocol
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class LarkEvent:
  """Parsed Lark event from WebSocket."""
  event_type: str = ""       # "im.message.receive_v1"
  chat_id: str = ""
  sender_id: str = ""
  message_id: str = ""
  msg_type: str = ""         # "text", "image", "file", "reaction", "sticker"
  text: str = ""
  mentions: list = field(default_factory=list)
  image_key: str = ""
  file_key: str = ""
  file_name: str = ""
  parent_id: str = ""
  create_time: str = ""
  raw: dict = field(default_factory=dict)


class LarkEventStream:
  """Async iterator over Lark events via WebSocket 长连接.

  Usage:
    stream = LarkEventStream(app_id, app_secret)
    await stream.connect()
    async for event in stream:
      if event.event_type == "im.message.receive_v1":
        handle_message(event)

  The stream auto-reconnects on disconnect.
  """

  def __init__(self, app_id: str, app_secret: str):
    self._app_id = app_id
    self._app_secret = app_secret
    self._queue: asyncio.Queue[LarkEvent] = asyncio.Queue()
    self._running = False
    self._task: asyncio.Task | None = None

  async def connect(self) -> None:
    """Establish WebSocket connection to Lark gateway."""
    self._running = True
    self._task = asyncio.create_task(self._listen_loop())
    log.info("LarkEventStream connected")

  async def close(self) -> None:
    """Disconnect from Lark gateway."""
    self._running = False
    if self._task:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
    log.info("LarkEventStream closed")

  def __aiter__(self):
    return self

  async def __anext__(self) -> LarkEvent:
    if not self._running and self._queue.empty():
      raise StopAsyncIteration
    return await self._queue.get()

  async def next_message(self, timeout: float = 300) -> dict | None:
    """Wait for the next message event. Returns reply dict or None on timeout.

    Convenience method for permission bridge and signal monitoring.
    """
    try:
      event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
      return {
        "text": event.text,
        "sender_id": event.sender_id,
        "message_id": event.message_id,
        "msg_type": event.msg_type,
        "mentions": event.mentions,
        "parent_id": event.parent_id,
        "create_time": event.create_time,
      }
    except asyncio.TimeoutError:
      return None

  async def _listen_loop(self) -> None:
    """Main WebSocket listener with auto-reconnect.

    TODO: Implement actual Lark WebSocket connection.
    Steps:
    1. Get WebSocket endpoint via REST API
    2. Connect with tenant token auth
    3. Parse incoming event frames
    4. Send acknowledgments
    5. Handle heartbeats
    6. Auto-reconnect on disconnect
    """
    while self._running:
      try:
        # PLACEHOLDER: Replace with actual Lark WebSocket implementation
        log.warning("LarkEventStream._listen_loop not yet implemented")
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        break
      except Exception as e:
        log.error("EventStream error: %s — reconnecting in 5s", e)
        await asyncio.sleep(5)
