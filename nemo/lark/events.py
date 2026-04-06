"""Lark WebSocket event subscription (长连接).

Uses the lark-oapi SDK to connect via WebSocket persistent connection.
Receives both events (im.message.receive_v1) and card action callbacks
(card.action.trigger) through the same connection.

Requires: pip install lark-oapi
Requires: App configured with eventMode=4 and callbackMode=4 in console.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import lark_oapi as lark
from lark_oapi.event.callback.model.p2_card_action_trigger import (
  P2CardActionTrigger, P2CardActionTriggerResponse,
)

log = logging.getLogger(__name__)


def _model_to_dict(obj: Any) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
  """Recursively convert a lark-oapi SDK model object to a plain dict.

  SDK objects like EventMessage, EventSender etc. have __dict__ with
  their fields but don't support .get(). This converts the entire
  tree to plain dicts so parse_event() can process them.
  """
  if obj is None:
    return obj
  if isinstance(obj, (str, int, float, bool)):
    return obj
  if isinstance(obj, dict):
    return {k: _model_to_dict(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_model_to_dict(item) for item in obj]
  # SDK model object — convert via __dict__
  if hasattr(obj, '__dict__'):
    return {k: _model_to_dict(v) for k, v in obj.__dict__.items()
            if not k.startswith('_')}
  return obj


@dataclass
class LarkEvent:
  """Parsed Lark event from WebSocket."""
  event_type: str = ""
  chat_id: str = ""
  chat_type: str = ""       # "p2p" or "group"
  sender_id: str = ""
  message_id: str = ""
  msg_type: str = ""         # "text", "image", "file"
  text: str = ""
  mentions: list[dict[str, str]] = field(default_factory=list)
  image_key: str = ""
  file_key: str = ""
  file_name: str = ""
  parent_id: str = ""
  create_time: str = ""
  # Card action fields
  action_value: dict[str, Any] = field(default_factory=dict)
  action_tag: str = ""
  operator_id: str = ""
  raw: dict[str, Any] = field(default_factory=dict)


def _parse_message_event(payload: dict[str, Any]) -> LarkEvent:
  """Parse an im.message.receive_v1 event."""
  event = payload.get("event", {})
  msg = event.get("message", {})
  sender = event.get("sender", {}).get("sender_id", {})

  text = ""
  image_key = ""
  file_key = ""
  file_name = ""
  content_str = msg.get("content", "{}")
  try:
    content = json.loads(content_str)
    text = content.get("text", "")
    image_key = content.get("image_key", "")
    file_key = content.get("file_key", "")
    file_name = content.get("file_name", "")
  except (json.JSONDecodeError, TypeError):
    pass

  mentions_raw = msg.get("mentions") or []
  mentions = [{"key": m.get("key", ""), "id": m.get("id", {}).get("open_id", "")}
              for m in mentions_raw]

  return LarkEvent(
    event_type="im.message.receive_v1",
    chat_id=msg.get("chat_id", ""),
    chat_type=msg.get("chat_type", ""),
    sender_id=sender.get("open_id", ""),
    message_id=msg.get("message_id", ""),
    msg_type=msg.get("message_type", ""),
    text=text,
    mentions=mentions,
    image_key=image_key,
    file_key=file_key,
    file_name=file_name,
    parent_id=msg.get("parent_id", "") or msg.get("root_id", ""),
    create_time=msg.get("create_time", ""),
    raw=payload,
  )


def _parse_card_action(payload: dict[str, Any]) -> LarkEvent:
  """Parse a card.action.trigger callback."""
  event = payload.get("event", {})
  operator = event.get("operator", {})
  action = event.get("action", {})
  context = event.get("context", {})

  return LarkEvent(
    event_type="card.action.trigger",
    chat_id=context.get("open_chat_id", ""),
    message_id=context.get("open_message_id", ""),
    sender_id=operator.get("open_id", ""),
    operator_id=operator.get("open_id", ""),
    action_value=action.get("value", {}),
    action_tag=action.get("tag", ""),
    raw=payload,
  )


def _parse_reaction_event(payload: dict[str, Any]) -> LarkEvent:
  """Parse an im.message.reaction.created_v1 event."""
  event = payload.get("event", {})
  return LarkEvent(
    event_type="im.message.reaction.created_v1",
    message_id=event.get("message_id", ""),
    sender_id=event.get("user_id", {}).get("open_id", ""),
    raw=payload,
  )


def parse_event(payload: dict[str, Any]) -> LarkEvent:
  """Parse a raw event payload into a LarkEvent."""
  header = payload.get("header", {})
  event_type = header.get("event_type", "")

  if event_type == "im.message.receive_v1":
    return _parse_message_event(payload)
  elif event_type == "card.action.trigger":
    return _parse_card_action(payload)
  elif event_type == "im.message.reaction.created_v1":
    return _parse_reaction_event(payload)
  else:
    return LarkEvent(event_type=event_type, raw=payload)


class LarkEventStream:
  """Async event stream from Lark WebSocket 长连接.

  Uses lark-oapi SDK's ws.Client to connect. Events and card action
  callbacks arrive through the same WebSocket connection.

  Usage:
    stream = LarkEventStream(app_id, app_secret)
    await stream.start()
    async for event in stream:
      handle(event)
    await stream.stop()
  """

  def __init__(self, app_id: str, app_secret: str):
    self._app_id = app_id
    self._app_secret = app_secret
    self._queue: asyncio.Queue[LarkEvent] = asyncio.Queue()
    self._running = False
    self._ws_client: Any = None
    self._ws_thread: threading.Thread | None = None

  async def start(self) -> None:
    """Start the WebSocket connection in a background thread."""
    loop = asyncio.get_event_loop()
    self._running = True

    def _on_message(data: Any) -> None:
      # Convert entire SDK model tree to plain dicts
      event_payload: dict[str, Any] = {
        "header": {"event_type": "im.message.receive_v1"},
        "event": _model_to_dict(data.event) if hasattr(data, 'event') else {},
      }
      log.debug("WS raw event: %s", json.dumps(event_payload, ensure_ascii=False, default=str)[:500])
      parsed = parse_event(event_payload)
      log.debug("WS parsed: chat=%s sender=%s text=%r", parsed.chat_id, parsed.sender_id, parsed.text[:80] if parsed.text else "")
      loop.call_soon_threadsafe(self._queue.put_nowait, parsed)

    def _on_card_action(data: P2CardActionTrigger) -> P2CardActionTriggerResponse:
      # Convert entire SDK model tree to plain dicts
      event_payload: dict[str, Any] = {
        "header": {"event_type": "card.action.trigger"},
        "event": _model_to_dict(data.event) if hasattr(data, 'event') else {},
      }
      parsed = parse_event(event_payload)
      loop.call_soon_threadsafe(self._queue.put_nowait, parsed)
      return P2CardActionTriggerResponse()

    handler: Any = lark.EventDispatcherHandler.builder("", "") \
      .register_p2_im_message_receive_v1(_on_message) \
      .register_p2_card_action_trigger(_on_card_action) \
      .build()

    self._ws_client = lark.ws.Client(
      self._app_id, self._app_secret,
      event_handler=handler,
      log_level=lark.LogLevel.INFO,
      domain=lark.LARK_DOMAIN,
    )

    self._ws_thread = threading.Thread(
      target=self._ws_client.start,
      daemon=True,
      name="lark-ws",
    )
    self._ws_thread.start()
    log.info("LarkEventStream started")

  async def stop(self) -> None:
    """Stop the WebSocket connection."""
    self._running = False
    # Put a sentinel to unblock any waiting consumers
    await self._queue.put(LarkEvent(event_type="_stop"))
    log.info("LarkEventStream stopped")

  def __aiter__(self) -> LarkEventStream:
    return self

  async def __anext__(self) -> LarkEvent:
    if not self._running and self._queue.empty():
      raise StopAsyncIteration
    event = await self._queue.get()
    if event.event_type == "_stop":
      raise StopAsyncIteration
    return event

  async def next_event(self, timeout: float = 300) -> LarkEvent | None:
    """Wait for the next event with timeout. Returns None on timeout."""
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

  # Aliases for backward compatibility with scaffold code
  async def connect(self) -> None:
    return await self.start()

  async def close(self) -> None:
    return await self.stop()

  async def next_message(self, timeout: float = 300) -> LarkEvent | None:
    return await self.next_event(timeout=timeout)
