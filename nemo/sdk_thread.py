"""Dedicated thread for SDK operations — isolates anyio from nemo's asyncio.

The Claude Agent SDK uses anyio internally. When running in nemo's complex
asyncio environment (signal handlers, WebSocket streams, heartbeat loops),
the anyio transport intermittently hangs. Running the SDK in its own thread
with a dedicated event loop avoids this entirely.

Proven: asyncio.run(sdk_test()) works 100% of the time in isolation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Callable

from .turn import TurnEvent, run_turn
from .types import ClaudeSDKClientLike, JsonObject

log = logging.getLogger(__name__)

MAX_CONNECT_ATTEMPTS = 5


class SDKThread:
  """Run SDK client in a dedicated thread with its own event loop."""

  def __init__(self):
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread: threading.Thread | None = None
    self._client: ClaudeSDKClientLike | None = None
    self._ready = threading.Event()
    self._cancelled = threading.Event()

  def start(self) -> None:
    """Start the SDK thread. Blocks until the event loop is ready."""
    self._thread = threading.Thread(target=self._run, daemon=True, name="sdk-loop")
    self._thread.start()
    self._ready.wait(timeout=10)
    if self._loop is None:
      raise RuntimeError("SDK thread failed to start")
    log.info("SDK thread started")

  def _run(self) -> None:
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    self._ready.set()
    self._loop.run_forever()

  def _schedule(self, coro: Coroutine[object, object, object]) -> Future[object]:
    """Schedule a coroutine on the SDK thread, return concurrent.futures.Future."""
    if self._loop is None:
      raise RuntimeError("SDK thread not started")
    return asyncio.run_coroutine_threadsafe(coro, self._loop)

  async def run_on_sdk_loop(self, coro: Coroutine[object, object, object]) -> object:
    """Schedule a coroutine on the SDK thread, awaitable from the main loop."""
    future = self._schedule(coro)
    return await asyncio.wrap_future(future)

  async def create_client(self, options: object) -> None:
    """Create and connect SDK client on the SDK thread."""
    from claude_agent_sdk import ClaudeSDKClient

    async def _create():
      for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
        c = ClaudeSDKClient(options=options)
        try:
          await asyncio.wait_for(c.__aenter__(), timeout=30)
          transport = getattr(c, '_transport', None)
          proc = getattr(transport, '_process', None) if transport else None
          if proc and proc.returncode is not None:
            raise RuntimeError(f"CLI exited during connect: rc={proc.returncode}")
          log.info("SDK client connected (attempt %d/%d)", attempt, MAX_CONNECT_ATTEMPTS)
          self._client = c
          return
        except Exception as e:
          log.warning("SDK connect attempt %d/%d failed: %s", attempt, MAX_CONNECT_ATTEMPTS, e)
          try:
            await c.__aexit__(None, None, None)
          except Exception:
            pass
          if attempt == MAX_CONNECT_ATTEMPTS:
            raise RuntimeError(f"SDK connect failed after {MAX_CONNECT_ATTEMPTS} attempts") from e
          await asyncio.sleep(2)

    await self.run_on_sdk_loop(_create())

  async def close_client(self) -> None:
    """Close the SDK client and ensure its CLI subprocess is dead."""
    if self._client is None:
      return

    async def _close():
      client = self._client
      self._client = None

      # Get the CLI subprocess PID before closing
      cli_pid = None
      transport = getattr(client, '_transport', None)
      proc = getattr(transport, '_process', None) if transport else None
      if proc:
        cli_pid = getattr(proc, 'pid', None)

      try:
        await asyncio.wait_for(client.__aexit__(None, None, None), timeout=5)
      except Exception:
        pass

      # Ensure CLI subprocess is dead
      if cli_pid:
        import os
        import signal as _signal
        try:
          os.kill(cli_pid, 0)  # Check if alive
          log.warning("CLI subprocess %d still alive after close, killing", cli_pid)
          os.kill(cli_pid, _signal.SIGKILL)
        except OSError:
          pass  # Already dead — good

    await self.run_on_sdk_loop(_close())

  async def reconnect(self, options: object) -> None:
    """Close and recreate the client."""
    await self.close_client()
    await self.create_client(options)

  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
    stale_tasks: set[str] | None = None,
  ) -> tuple[float, JsonObject]:
    """Run a single SDK turn on the SDK thread.

    on_event is called from the SDK thread — it must be thread-safe.
    Lark API calls (send_card, update_card) are synchronous HTTP and safe.
    """
    if self._client is None:
      raise RuntimeError("SDK client not connected")

    async def _turn():
      return await run_turn(self._client, prompt, on_event, stale_tasks=stale_tasks)

    return await self.run_on_sdk_loop(_turn())

  def cancel(self) -> None:
    """Signal the SDK thread to abort reconnect loops."""
    self._cancelled.set()

  async def run_turn_with_reconnect(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
    stale_tasks: set[str] | None = None,
    options: object = None,
    max_attempts: int = 3,
  ) -> tuple[float, JsonObject]:
    """Run turn with automatic reconnect on timeout or disconnection.

    On timeout or client disconnection, reconnects and retries. On the
    last attempt, raises immediately without wasting a reconnect.
    Checks _cancelled flag between attempts so interrupt can abort.
    """
    self._cancelled.clear()
    for attempt in range(max_attempts):
      if self._cancelled.is_set():
        raise asyncio.CancelledError("SDK turn cancelled")
      try:
        return await self.run_turn(prompt, on_event, stale_tasks=stale_tasks)
      except (TimeoutError, RuntimeError) as exc:
        is_disconnected = isinstance(exc, RuntimeError) and "not connected" in str(exc)
        if isinstance(exc, TimeoutError) or is_disconnected:
          label = "disconnected" if is_disconnected else "hung"
          log.warning("SDK turn %s (attempt %d/%d)", label, attempt + 1, max_attempts)
          if attempt == max_attempts - 1 or options is None:
            raise
          if self._cancelled.is_set():
            raise asyncio.CancelledError("SDK turn cancelled")
          log.info("Reconnecting...")
          await self.reconnect(options)
        else:
          raise  # Other RuntimeErrors should not be retried
    raise RuntimeError(f"SDK turn failed after {max_attempts} attempts")

  async def interrupt(self) -> None:
    """Interrupt the current SDK turn."""
    if self._client is None:
      return

    async def _interrupt():
      try:
        await self._client.interrupt()
      except Exception as e:
        log.warning("interrupt() error: %s", e)

    await self.run_on_sdk_loop(_interrupt())

  def stop(self) -> None:
    """Stop the SDK thread."""
    if self._loop:
      self._loop.call_soon_threadsafe(self._loop.stop)
    if self._thread:
      self._thread.join(timeout=1)
    log.info("SDK thread stopped")
