"""Shared lightweight types for Nemo."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, Self

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | JsonObject | JsonArray
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]


class TurnClient(Protocol):
  """Minimal client surface required by turn.run_turn()."""

  async def query(self, prompt: str) -> None:
    ...

  def receive_messages(self) -> AsyncIterator[object]:
    ...

  async def stop_task(self, task_id: str) -> None:
    ...


class ClaudeSDKClientLike(TurnClient, Protocol):
  """Claude SDK client surface used by SDKThread."""

  async def __aenter__(self) -> Self:
    ...

  async def __aexit__(
    self,
    exc_type: object,
    exc: object,
    tb: object,
  ) -> None:
    ...

  async def interrupt(self) -> None:
    ...
