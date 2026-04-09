"""Abstract Channel interface — how Nemo talks to the outside world.

A Channel handles all I/O with the user: receiving messages/actions,
sending cards and text, downloading media, and requesting permissions.

Lark is the first implementation. The core orchestration depends only
on this interface, not on Lark specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .types import JsonObject


@dataclass
class IncomingMessage:
  """A message or action received from the channel."""
  event_type: str = ""       # "message", "card_action", "reaction"
  chat_id: str = ""
  chat_type: str = ""        # "p2p" or "group"
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
  action_value: JsonObject = field(default_factory=dict)
  action_tag: str = ""
  operator_id: str = ""
  raw: JsonObject = field(default_factory=dict)


class Channel(ABC):
  """Abstract channel for user I/O."""

  @abstractmethod
  async def receive(self, timeout: float = 300) -> IncomingMessage | None:
    """Wait for the next incoming message/action. Returns None on timeout."""
    ...

  @abstractmethod
  def push_back(self, message: IncomingMessage) -> None:
    """Re-queue a message so it can be consumed again."""
    ...

  @abstractmethod
  async def send_card(self, chat_id: str, card: JsonObject) -> str:
    """Send an interactive card. Returns message_id."""
    ...

  @abstractmethod
  async def update_card(self, message_id: str, card: JsonObject) -> None:
    """Update (PATCH) an existing card message."""
    ...

  @abstractmethod
  async def send_text(self, chat_id: str, text: str) -> str:
    """Send a plain text message. Returns message_id."""
    ...

  @abstractmethod
  async def download_image(
    self, message_id: str, image_key: str,
  ) -> str:
    """Download an image. Returns local file path."""
    ...

  @abstractmethod
  async def download_file(
    self, message_id: str, file_key: str, file_name: str = "",
  ) -> str:
    """Download a file. Returns local file path."""
    ...

  @abstractmethod
  async def add_reaction(self, message_id: str, emoji_type: str) -> str:
    """Add an emoji reaction to a message. Returns reaction_id."""
    ...

  @abstractmethod
  async def start(self) -> None:
    """Start the channel (connect, authenticate, etc.)."""
    ...

  @abstractmethod
  async def stop(self) -> None:
    """Stop the channel and release resources."""
    ...

  @abstractmethod
  async def get_bot_id(self) -> str:
    """Get the bot's user ID."""
    ...

  @abstractmethod
  async def get_chat_members(self, chat_id: str) -> list[JsonObject]:
    """Get members of a chat."""
    ...

  @property
  @abstractmethod
  def permission_active(self) -> bool:
    """Whether the permission bridge is currently consuming channel events."""
    ...

  @permission_active.setter
  @abstractmethod
  def permission_active(self, active: bool) -> None:
    """Set whether the permission bridge is currently consuming channel events."""
    ...
