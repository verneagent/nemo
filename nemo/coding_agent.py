"""Abstract Agent interface — how Nemo runs coding tasks.

An Agent handles turn execution: running prompts, streaming events,
interrupting running turns, and resetting conversation state.

Claude Agent SDK is the first implementation. The core orchestration
depends only on this interface, not on Claude SDK specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .turn import TurnEvent
from .types import JsonObject


@dataclass(frozen=True)
class EndpointConfig:
  """Custom endpoint override shared across providers.

  The CLI exposes ``--base-url`` / ``--api-key`` as provider-agnostic flags;
  each adapter translates these into its own vendor-specific env vars
  (``ANTHROPIC_BASE_URL`` for Claude, ``OPENAI_BASE_URL`` for Codex, etc.).

  Empty strings mean "leave the adapter's default behavior alone" — i.e.
  let Claude SDK / Codex CLI / OpenCode resolve credentials from their
  usual sources (login session, ``~/.codex/config.toml``, ``opencode.json``).
  """
  base_url: str = ""
  api_key: str = ""


class CodingAgent(ABC):
  """Abstract coding agent that executes turns."""

  @abstractmethod
  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    """Execute one agent turn with the given prompt.

    Calls on_event for each TurnEvent (tool use, text output, done, error).
    Returns when the turn completes.
    """
    ...

  @abstractmethod
  async def interrupt(self) -> None:
    """Cancel the currently running turn."""
    ...

  @abstractmethod
  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    """Initialize the agent (connect to SDK, etc.)."""
    ...

  @abstractmethod
  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    """Reset or reconnect the agent with updated runtime settings."""
    ...

  @abstractmethod
  async def stop(self) -> None:
    """Shut down the agent and release resources."""
    ...

  def set_effort(self, effort: str) -> None:
    """Set reasoning effort for subsequent turns.

    `effort` is one of "", "low", "medium", "high", "max". Empty string
    clears the setting. Adapters whose backend tops out below `max`
    should clamp it down (e.g. Codex/OpenCode → `high`). Default is no-op.

    Effort is stored in the adapter; the host typically calls `reset()`
    afterwards so the new value flows into the next turn.
    """
    del effort

  def trailing_note(self, sdk_session_id: str) -> str:
    """Return an optional markdown note to append to the final turn response.

    Concrete adapters use this to surface provider-specific health signals
    (e.g. "session context is getting large, consider /clear"). Default
    returns empty string — no note appended.
    """
    del sdk_session_id
    return ""
