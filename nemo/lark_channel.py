"""Concrete Lark-backed Channel implementation."""

from __future__ import annotations

import os
from typing import Any

from .channel import Channel, IncomingMessage
from .config import load_credentials, load_relay_config
from .lark import api as lark_api
from .lark import auth as lark_auth
from .lark.events import LarkEvent, LarkEventStream
from .relay_events import RelayEventStream


def _to_incoming(event: LarkEvent) -> IncomingMessage:
  return IncomingMessage(
    event_type=event.event_type,
    chat_id=event.chat_id,
    chat_type=event.chat_type,
    sender_id=event.sender_id,
    message_id=event.message_id,
    msg_type=event.msg_type,
    text=event.text,
    mentions=list(event.mentions),
    image_key=event.image_key,
    file_key=event.file_key,
    file_name=event.file_name,
    parent_id=event.parent_id,
    create_time=event.create_time,
    action_value=dict(event.action_value),
    action_tag=event.action_tag,
    operator_id=event.operator_id,
    raw=dict(event.raw),
  )


class LarkChannel(Channel):
  """Channel implementation backed by Lark IM APIs and event streams."""

  def __init__(self, chat_id: str):
    self.chat_id = chat_id
    self.credentials = load_credentials() or {}
    self.token = ""
    self._events: LarkEventStream | RelayEventStream | None = None

  async def start(self) -> None:
    self.token = lark_auth.get_token(
      self.credentials["app_id"], self.credentials["app_secret"])
    relay_url, relay_api_key = load_relay_config()
    if relay_url:
      self._events = RelayEventStream(relay_url, relay_api_key, self.chat_id)
    else:
      self._events = LarkEventStream(
        self.credentials["app_id"], self.credentials["app_secret"])
    await self._events.connect()

  async def stop(self) -> None:
    if self._events is not None:
      await self._events.close()

  async def receive(self, timeout: float = 300) -> IncomingMessage | None:
    if self._events is None:
      raise RuntimeError("Channel not started")
    event = await self._events.next_message(timeout=timeout)
    if event is None:
      return None
    return _to_incoming(event)

  def push_back(self, message: IncomingMessage) -> None:
    if self._events is None:
      raise RuntimeError("Channel not started")
    event = LarkEvent(
      event_type=message.event_type,
      chat_id=message.chat_id,
      chat_type=message.chat_type,
      sender_id=message.sender_id,
      message_id=message.message_id,
      msg_type=message.msg_type,
      text=message.text,
      mentions=list(message.mentions),
      image_key=message.image_key,
      file_key=message.file_key,
      file_name=message.file_name,
      parent_id=message.parent_id,
      create_time=message.create_time,
      action_value=dict(message.action_value),
      action_tag=message.action_tag,
      operator_id=message.operator_id,
      raw=dict(message.raw),
    )
    self._events.push_back(event)

  async def send_card(self, chat_id: str, card: dict[str, Any]) -> str:
    return lark_api.send_card(self.token, chat_id, card)

  async def update_card(self, message_id: str, card: dict[str, Any]) -> None:
    lark_api.update_card(self.token, message_id, card)

  async def send_text(self, chat_id: str, text: str) -> str:
    return lark_api.send_text(self.token, chat_id, text)

  async def download_image(self, message_id: str, image_key: str) -> str:
    return lark_api.download_image(self.token, message_id, image_key)

  async def download_file(
    self, message_id: str, file_key: str, file_name: str = "",
  ) -> str:
    return lark_api.download_file(self.token, message_id, file_key, file_name)

  async def add_reaction(self, message_id: str, emoji_type: str) -> Any:
    return lark_api.add_reaction(self.token, message_id, emoji_type)

  async def remove_reaction(self, message_id: str, reaction_id: str) -> None:
    lark_api.remove_reaction(self.token, message_id, reaction_id)

  async def get_bot_id(self) -> str:
    bot_info = lark_api.get_bot_info(self.token)
    return bot_info.get("open_id", "")

  async def get_chat_members(self, chat_id: str) -> list[dict[str, Any]]:
    return lark_api.get_chat_members(self.token, chat_id)

  async def lookup_open_id_by_email(self, email: str) -> str:
    return lark_api.lookup_open_id_by_email(self.token, email) or ""

  async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
    return lark_api.get_chat_info(self.token, chat_id)

  async def refresh_token(self) -> str:
    self.token = lark_auth.get_token(
      self.credentials["app_id"], self.credentials["app_secret"])
    return self.token

  async def delete_message(self, message_id: str) -> None:
    lark_api.delete_message(self.token, message_id)

  async def resolve_operator_and_bot(self, email: str = "") -> tuple[str, str]:
    operator_open_id = ""
    if email:
      operator_open_id = lark_api.lookup_open_id_by_email(self.token, email) or ""
    bot_info = lark_api.get_bot_info(self.token)
    return operator_open_id, bot_info.get("open_id", "")

  async def ensure_workspace_claimed(self, project_dir: str, model: str) -> None:
    from .workspace import ensure_workspace_tag, evict_existing, claim_group
    ensure_workspace_tag(self.token, self.chat_id, project_dir)
    evict_existing(self.token, self.chat_id)
    claim_group(self.token, self.chat_id, model=model)

  async def release_workspace(self) -> None:
    from .workspace import release_group
    release_group(self.token, self.chat_id)

  async def update_status(self, model: str, state: str) -> None:
    from . import status_tab
    status_tab.update_status(self.token, self.chat_id, model, state)

  async def send_heartbeat(self, model: str) -> None:
    relay_url, _ = load_relay_config()
    if not relay_url:
      return
    from . import relay as relay_client
    from .workspace import get_machine_name
    relay_client.send_heartbeat(
      self.chat_id, pid=os.getpid(), model=model,
      machine=get_machine_name())

  async def dissolve_chat(self) -> None:
    lark_api.dissolve_chat(self.token, self.chat_id)

  @property
  def permission_active(self) -> bool:
    if self._events is None:
      return False
    return bool(self._events.permission_active)

  @permission_active.setter
  def permission_active(self, active: bool) -> None:
    if self._events is None:
      return
    self._events.permission_active = active
