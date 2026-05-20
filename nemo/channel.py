"""Abstract Channel interface — how Nemo talks to the outside world.

A Channel handles all I/O with the user: receiving messages/actions,
sending cards and text, downloading media, and requesting permissions.

Lark is the first implementation. The core orchestration depends only
on this interface, not on Lark specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from .types import JsonObject


@dataclass
class PendingQuestion:
  """An in-flight AskUserQuestion call rendered inline in the working
  turn card. The action strings on the buttons follow
  ``askq:{nonce}:{qidx}:{oidx}`` so ``nemo.permissions._parse_askq_action``
  still parses them.

  ``answers`` maps the question index to the selected value: a label
  string for single-select questions, a list of labels for multi-select.
  ``multi_done`` tracks which multi-select questions have been
  finalised by the user clicking Submit.
  """

  questions: list[JsonObject] = field(default_factory=list)
  answers: dict[int, object] = field(default_factory=dict)
  nonce: str = ""
  multi_done: set[int] = field(default_factory=set)


@dataclass
class AnsweredQuestion:
  """A previously-answered question, rendered as a compact summary line
  near the top of the working / done turn card (``❓ <header> → ✅
  <answer>``). Persists in the turn card for the lifetime of the turn so
  the user can always see what they picked."""

  header: str = ""
  question: str = ""
  # str for single-select, list[str] for multi-select.
  answer: object = ""


def _noop_redraw() -> None:
  pass


@dataclass
class TurnCardCtx:
  """Per-turn state shared between the agent loop and tool handlers
  (e.g. the AskUserQuestion handler) that need to repaint the working
  turn card without owning it.

  The agent wires ``redraw`` to a thread-safe wrapper around
  ``_update_working`` and resets the rest of the fields at the start of
  every turn. Handlers mutate ``pending_question`` / ``answered_questions``
  and call ``redraw()`` to push their changes into the card.

  The default-constructed value is intentionally inert: ``redraw`` is a
  no-op and there is no card, so calling these on a Channel that hasn't
  been wired into a turn does nothing harmful — handlers can fall back
  to sending their own card if needed.
  """

  redraw: Callable[[], None] = field(default=_noop_redraw)
  pending_question: PendingQuestion | None = None
  answered_questions: list[AnsweredQuestion] = field(default_factory=list)
  # Non-empty once the agent has actually sent the working card. Handlers
  # use this to decide whether the embedded path is usable or whether
  # they need to fall back to a standalone card.
  turn_card_id: str = ""


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
  # Thread identifier in threaded chats (e.g. Lark topic groups). Empty
  # outside of topic/thread contexts.
  thread_id: str = ""
  create_time: str = ""
  # Card action fields
  action_value: JsonObject = field(default_factory=dict)
  action_tag: str = ""
  operator_id: str = ""
  raw: JsonObject = field(default_factory=dict)
  # True for messages synthesised by Nemo itself (e.g. /session recall
  # injection) rather than received from the channel. The main loop
  # bypasses the human-sender authorization + need_mention filters for
  # these so they reach the SDK as a normal turn.
  is_internal: bool = False


class Channel(ABC):
  """Abstract channel for user I/O."""

  # Per-turn turn-card state. The agent assigns a fresh TurnCardCtx at
  # the start of each turn; tool handlers (AskUserQuestion) read it to
  # request a card redraw. Annotated only — concrete subclasses are
  # responsible for creating their own instance in ``__init__`` so the
  # mutable state is not shared across channels.
  turn_ctx: TurnCardCtx

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
  async def update_card(self, message_id: str, card: JsonObject) -> str:
    """Update (PATCH) an existing card message. Returns the effective
    message_id — normally the input, but implementations may return a
    different id if they had to send a replacement card (e.g. the
    original message is no longer editable). Callers tracking the card
    id should update their stored id to the returned value."""
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

  async def update_workspace_tag(self, project_dir: str) -> None:
    """Update the workspace tag for the current chat. No-op by default."""
    pass
