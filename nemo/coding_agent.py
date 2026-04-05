"""Abstract Agent interface — how Nemo runs coding tasks.

An Agent handles turn execution: running prompts, streaming events,
interrupting running turns, and resetting conversation state.

Claude Agent SDK is the first implementation. The core orchestration
depends only on this interface, not on Claude SDK specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .turn import TurnEvent


class CodingAgent(ABC):
  """Abstract coding agent that executes turns."""

  @abstractmethod
  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], Any],
  ) -> None:
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
  async def reset(self) -> None:
    """Reset conversation state (new conversation)."""
    ...

  @abstractmethod
  async def start(self, project_dir: str, model: str) -> None:
    """Initialize the agent (connect to SDK, etc.)."""
    ...

  @abstractmethod
  async def stop(self) -> None:
    """Shut down the agent and release resources."""
    ...
