"""Dedicated thread for SDK operations — isolates anyio from nemo's asyncio.

The Claude Agent SDK uses anyio internally. When running in nemo's complex
asyncio environment (signal handlers, WebSocket streams, heartbeat loops),
the anyio transport intermittently hangs. Running the SDK in its own thread
with a dedicated event loop avoids this entirely.

Proven: asyncio.run(sdk_test()) works 100% of the time in isolation.

Lifecycle ownership
-------------------
ClaudeSDKClient is opened/closed via anyio task groups whose cancel scopes
are *task-bound*: ``__aenter__`` and ``__aexit__`` MUST execute on the same
asyncio.Task or anyio raises "Attempted to exit cancel scope in a different
task than it was entered in" — leaving the task group in a broken state with
orphan tasks that spin on the event loop forever (observable as 30–40% CPU
on idle nemos that have ever experienced a watchdog-forced reconnect).

Previously each operation was scheduled via ``run_coroutine_threadsafe``,
which created a fresh task per call — so ``create_client`` opened the client
in Task A and ``close_client`` tried to close it in Task B → broken state.

Now a single long-lived "lifecycle owner" task runs on the SDK loop and
processes ``open``/``close`` commands serially. Both __aenter__ and
__aexit__ run on the same task, no orphan tasks, no busy spin.
``run_turn`` and ``interrupt`` still use short-lived per-call tasks because
they only operate on the existing message stream and don't touch cancel
scopes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal as _signal
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Callable

from claude_agent_sdk import CLIConnectionError as _CLIConnectionError

from .turn import TurnEvent, run_turn
from .types import ClaudeSDKClientLike, JsonObject

log = logging.getLogger(__name__)

MAX_CONNECT_ATTEMPTS = 5

# Lifecycle command kinds dispatched to the owner task
_CMD_OPEN = "open"
_CMD_CLOSE = "close"
_CMD_STOP = "stop"


class SDKThread:
  """Run SDK client in a dedicated thread with its own event loop."""

  def __init__(self):
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread: threading.Thread | None = None
    self._client: ClaudeSDKClientLike | None = None
    self._ready = threading.Event()
    self._cancelled = threading.Event()
    # Lifecycle command queue (asyncio.Queue created on the SDK loop).
    self._lifecycle_q: asyncio.Queue | None = None
    self._lifecycle_started = threading.Event()

  def start(self) -> None:
    """Start the SDK thread. Blocks until the event loop is ready."""
    self._thread = threading.Thread(target=self._run, daemon=True, name="sdk-loop")
    self._thread.start()
    self._ready.wait(timeout=10)
    if self._loop is None:
      raise RuntimeError("SDK thread failed to start")
    if not self._lifecycle_started.wait(timeout=10):
      raise RuntimeError("SDK lifecycle owner failed to start")
    log.info("SDK thread started")

  def _run(self) -> None:
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    # Bootstrap the lifecycle owner from inside the loop so the queue and
    # owner task live on this loop.
    self._loop.call_soon(self._bootstrap_lifecycle)
    self._ready.set()
    self._loop.run_forever()

  def _bootstrap_lifecycle(self) -> None:
    """Runs once on the SDK loop to spin up the lifecycle owner."""
    self._lifecycle_q = asyncio.Queue()
    assert self._loop is not None
    self._loop.create_task(self._lifecycle_owner(), name="sdk-lifecycle-owner")
    self._lifecycle_started.set()

  async def _lifecycle_owner(self) -> None:
    """Long-lived task that owns the ClaudeSDKClient lifecycle.

    All open/close calls funnel through this single task so anyio cancel
    scopes opened by __aenter__ are always exited from the same task.
    """
    assert self._lifecycle_q is not None
    while True:
      kind, payload, future = await self._lifecycle_q.get()
      try:
        if kind == _CMD_OPEN:
          await self._do_open(payload)
          _set_future_result(future, None)
        elif kind == _CMD_CLOSE:
          await self._do_close()
          _set_future_result(future, None)
        elif kind == _CMD_STOP:
          await self._do_close()
          _set_future_result(future, None)
          return
        else:
          _set_future_exception(
            future, RuntimeError(f"unknown lifecycle cmd: {kind}"))
      except BaseException as exc:  # noqa: BLE001 — surface to caller
        _set_future_exception(future, exc)

  async def _do_open(self, options: object) -> None:
    """Connect a new SDK client. Must run on the lifecycle owner task."""
    from claude_agent_sdk import ClaudeSDKClient

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
        except Exception as close_err:
          log.warning("Failed to close client after connect failure: %s", close_err)
        if attempt == MAX_CONNECT_ATTEMPTS:
          raise RuntimeError(f"SDK connect failed after {MAX_CONNECT_ATTEMPTS} attempts") from e
        await asyncio.sleep(2)

  async def _do_close(self) -> None:
    """Close the active SDK client. Must run on the lifecycle owner task."""
    if self._client is None:
      return

    client = self._client
    self._client = None

    cli_pid = None
    transport = getattr(client, '_transport', None)
    proc = getattr(transport, '_process', None) if transport else None
    if proc:
      cli_pid = getattr(proc, 'pid', None)

    try:
      await asyncio.wait_for(client.__aexit__(None, None, None), timeout=5)
    except Exception as e:
      log.warning("SDK client __aexit__ failed: %s", e)

    if cli_pid:
      try:
        os.kill(cli_pid, 0)  # Check if alive
        log.warning("CLI subprocess %d still alive after close, killing", cli_pid)
        os.kill(cli_pid, _signal.SIGKILL)
      except OSError:
        pass  # Already dead — good

  async def _submit_lifecycle(self, kind: str, payload: object = None) -> object:
    """Post a lifecycle command and await its completion from any loop."""
    if self._loop is None or self._lifecycle_q is None:
      raise RuntimeError("SDK thread not started")
    fut: Future = Future()
    self._loop.call_soon_threadsafe(
      self._lifecycle_q.put_nowait, (kind, payload, fut))
    return await asyncio.wrap_future(fut)

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
    """Create and connect SDK client on the lifecycle owner task."""
    await self._submit_lifecycle(_CMD_OPEN, options)

  async def close_client(self) -> None:
    """Close the SDK client on the lifecycle owner task."""
    if self._client is None:
      return
    await self._submit_lifecycle(_CMD_CLOSE)

  async def reconnect(self, options: object) -> None:
    """Close and recreate the client (both on the owner task)."""
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
    options_factory: Callable[[], object] | None = None,
    max_attempts: int = 3,
  ) -> tuple[float, JsonObject]:
    """Run turn with automatic reconnect on timeout or disconnection.

    On timeout or client disconnection, reconnects and retries. On the
    last attempt, raises immediately without wasting a reconnect.
    Checks _cancelled flag between attempts so interrupt can abort.

    options_factory: optional callable invoked before each reconnect to
    produce fresh options. Used by ClaudeCodingAgent to inject
    `resume=<latest_session_id>` so a mid-turn watchdog reconnect
    preserves conversation context. Falls back to the static `options`
    parameter when not provided.
    """
    from .turn import TransientAPIError  # avoid import cycle
    self._cancelled.clear()
    for attempt in range(max_attempts):
      if self._cancelled.is_set():
        raise asyncio.CancelledError("SDK turn cancelled")
      try:
        return await self.run_turn(prompt, on_event, stale_tasks=stale_tasks)
      except (TimeoutError, TransientAPIError, RuntimeError, _CLIConnectionError) as exc:
        exc_msg = str(exc).lower()
        is_transient_api = isinstance(exc, TransientAPIError)
        is_disconnected = (
          isinstance(exc, _CLIConnectionError)
          or (isinstance(exc, RuntimeError)
              and not is_transient_api
              and "not connected" in exc_msg)
        )
        if isinstance(exc, TimeoutError) or is_disconnected or is_transient_api:
          if is_transient_api:
            label = "transient-api-error"
          elif is_disconnected:
            label = "disconnected"
          else:
            label = "hung"
          log.warning("SDK turn %s (attempt %d/%d)", label, attempt + 1, max_attempts)
          fresh_options = options_factory() if options_factory is not None else options
          if attempt == max_attempts - 1 or fresh_options is None:
            raise
          if self._cancelled.is_set():
            raise asyncio.CancelledError("SDK turn cancelled")
          log.info("Reconnecting...")
          await self.reconnect(fresh_options)
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
    if self._loop and self._lifecycle_q is not None:
      # Ask the owner to clean up the active client (if any) and exit.
      # Bounded wait so a hung client cannot block shutdown indefinitely.
      try:
        fut: Future = Future()
        self._loop.call_soon_threadsafe(
          self._lifecycle_q.put_nowait, (_CMD_STOP, None, fut))
        try:
          fut.result(timeout=5)
        except Exception as e:
          log.warning("Lifecycle owner stop did not complete cleanly: %s", e)
      except RuntimeError as e:
        # Loop may already be stopped
        log.debug("Could not enqueue stop command: %s", e)
    if self._loop:
      try:
        self._loop.call_soon_threadsafe(self._loop.stop)
      except RuntimeError:
        pass
    if self._thread:
      self._thread.join(timeout=1)
    log.info("SDK thread stopped")


def _set_future_result(future: Future, result: object) -> None:
  if not future.done():
    future.set_result(result)


def _set_future_exception(future: Future, exc: BaseException) -> None:
  if not future.done():
    future.set_exception(exc)
