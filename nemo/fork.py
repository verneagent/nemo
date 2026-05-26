"""`/fork` — read-only, multi-turn, tool-enabled sub-conversations.

A fork branches the current conversation into a Lark sub-thread. Each fork is
its own read-only `CodingAgent` (own SDK subprocess, sandboxed so it cannot
modify the project — see `ClaudeCodingAgent.fork`) plus a fixed thread anchor.
Messages that land in a fork's thread are routed here and run as concurrent
turns: different forks and the main conversation each have their own SDK
client, so their turns overlap freely; turns *within one fork* are serialized
by a per-fork lock.

This module owns all fork lifecycle + card rendering so `agent.py` stays
orchestration-only: it just asks `ForkManager` to open / route / close.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from . import cards
from .channel import Channel
from .coding_agent import CodingAgent
from .turn import AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent, TurnEvent
from .types import JsonObject

log = logging.getLogger(__name__)

# Cap concurrent forks: each open fork holds a live SDK subprocess. Past this,
# /fork declines until one is closed (/fork close) or the session is cleared.
_MAX_FORKS = 3


@dataclass
class ForkSession:
  """One live fork: its read-only agent, Lark sub-thread, and turn lock."""
  agent: CodingAgent
  root_msg_id: str          # the thread's root card (reply anchor)
  thread_id: str            # routing key — every message in the thread has it
  chat_id: str              # parent chat (for the stop button's value)
  prompt0: str              # the opening /fork prompt (for the root card)
  lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  # Set by ForkManager.interrupt so the in-flight turn renders "stopped"
  # instead of "done"; reset at the start of each turn.
  interrupt_requested: bool = False
  # The active turn's renderer (for interrupt to PATCH the card + read state);
  # None between turns.
  renderer: "_ForkRenderer | None" = None


class _ForkRenderer:
  """Renders one fork turn's cards into the fork's sub-thread.

  Uses the SAME ``build_turn_card`` as a main turn (full thinking timeline,
  current tool, elapsed title) but with a FORK-SCOPED stop button — its action
  is ``fork_stop:<thread_id>`` so a click interrupts this fork's turn, not the
  main one. ``on_event`` runs on the fork's SDK thread, so channel calls are
  marshalled onto the main loop via run_coroutine_threadsafe (mirrors the main
  turn's ``_await_channel``).
  """

  def __init__(self, channel: Channel, sess: ForkSession,
               loop: asyncio.AbstractEventLoop):
    self._ch = channel
    self._sess = sess
    self._loop = loop
    self._steps: list[cards.ThinkingStep] = []
    self._card_id = ""
    self._start = time.time()
    self._current_tool = ""

  @property
  def card_id(self) -> str:
    return self._card_id

  @property
  def start(self) -> float:
    return self._start

  def _elapsed(self) -> int:
    return int(time.time() - self._start)

  def _await(self, coro: Awaitable[object]) -> object:
    return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

  def _progress_card(self, phase: str = "working") -> JsonObject:
    return cards.build_turn_card(
      phase, steps=self._steps, current_tool=self._current_tool,
      elapsed=self._elapsed(), chat_id=self._sess.chat_id,
      stop_action=f"fork_stop:{self._sess.thread_id}")

  def stopped_card(self) -> JsonObject:
    thinking = [s for s in self._steps if s.kind != "answer"]
    return cards.build_turn_card("stopped", steps=thinking, elapsed=self._elapsed())

  def _ensure_card(self) -> None:
    if self._card_id:
      return
    try:
      mid, _ = self._await(
        self._ch.send_card_in_thread(self._sess.root_msg_id, self._progress_card()))
      self._card_id = mid
    except Exception as e:
      log.warning("fork: failed to create working card: %s", e)

  def _update_working(self) -> None:
    if not self._card_id:
      return
    try:
      self._card_id = self._await(
        self._ch.update_card(self._card_id, self._progress_card()))
    except Exception as e:
      log.debug("fork: working card update failed: %s", e)

  async def render_stopping(self) -> None:
    """PATCH the working card to the 'stopping' state. Called from the main
    loop (ForkManager.interrupt), so awaits the channel directly."""
    if not self._card_id:
      return
    try:
      await self._ch.update_card(self._card_id, self._progress_card("stopping"))
    except Exception as e:
      log.debug("fork: stopping-card patch failed: %s", e)

  def on_event(self, ev: TurnEvent) -> None:
    if isinstance(ev, ProgressEvent):
      self._steps.append(cards.ThinkingStep(ev.kind, ev.summary))
      if ev.kind == "tool":
        self._current_tool = ev.summary
      if ev.first:
        self._ensure_card()
      self._update_working()
    elif isinstance(ev, AnswerEvent):
      self._steps.append(cards.ThinkingStep("answer", ev.text))
    elif isinstance(ev, DoneEvent):
      thinking = [s for s in self._steps if s.kind != "answer"]
      if self._sess.interrupt_requested:
        # Interrupted mid-turn — show "stopped" (matches the main-turn stop UX)
        # rather than a "done" card for a turn the user cut short.
        card = self.stopped_card()
      else:
        answers = [s.content for s in self._steps if s.kind == "answer"]
        final = answers[-1] if answers else ""
        card = cards.build_turn_card(
          "done", body=final, steps=thinking, elapsed=self._elapsed(),
          usage=ev.usage, session_id=ev.session_id)
      try:
        if self._card_id:
          self._await(self._ch.update_card(self._card_id, card))
        else:
          # Pure text answer, no tools → no working card was created.
          self._await(
            self._ch.send_card_in_thread(self._sess.root_msg_id, card))
      except Exception as e:
        log.warning("fork: failed to render final card: %s", e)
    elif isinstance(ev, ErrorEvent):
      self._steps.append(cards.ThinkingStep("reasoning", f"⚠️ {ev.message}"))
      self._update_working()


class ForkManager:
  """Owns all live forks for one chat and routes turns to them."""

  def __init__(
    self,
    channel: Channel,
    chat_id: str,
    notify: Callable[[str], Awaitable[None]],
    *,
    max_forks: int = _MAX_FORKS,
  ):
    self._ch = channel
    self._chat_id = chat_id
    self._notify = notify  # async: post a status line to the MAIN chat
    self._max = max_forks
    self._loop = asyncio.get_running_loop()
    self._forks: dict[str, ForkSession] = {}  # thread_id -> session
    self._tasks: set[asyncio.Task[None]] = set()

  def get(self, thread_id: str) -> ForkSession | None:
    return self._forks.get(thread_id) if thread_id else None

  def count(self) -> int:
    return len(self._forks)

  async def open(
    self,
    *,
    main_agent: CodingAgent,
    anchor_msg_id: str,
    parent_sid: str,
    project_dir: str,
    model: str,
    prompt: str,
  ) -> None:
    """Open a fork: validate, spin up the read-only agent, open the Lark
    sub-thread, register it, and run the first turn. Self-contained — posts
    its own decline/status messages so the caller can fire-and-forget."""
    if not main_agent.supports_fork():
      await self._notify(
        "`/fork` isn't supported by the current agent. It needs Claude or "
        "Codex (session forking + a read-only sandbox) — switch with "
        "`/agent claude` or `/agent codex`.")
      return
    if not self._ch.supports_threads():
      await self._notify(
        "`/fork` needs a group chat that supports sub-threads — this chat "
        "doesn't.")
      return
    if len(self._forks) >= self._max:
      await self._notify(
        f"Too many open forks (max {self._max}). Close one with "
        f"`/fork close` inside its thread.")
      return

    fork_agent: CodingAgent | None = None
    try:
      fork_agent = await main_agent.fork(parent_sid, project_dir, model)
      if fork_agent is None:
        await self._notify("`/fork` is unsupported by the current agent.")
        return
      root_card = cards.build_markdown_card(
        f"Read-only branch — shares the current context, has tools, but "
        f"**cannot modify project files** (sandboxed). Keep replying in this "
        f"thread to continue; `/fork close` to end.\n\n> {prompt[:300]}",
        title="🍴 Fork", color="indigo")
      root_id, thread_id = await self._ch.send_card_in_thread(
        anchor_msg_id, root_card)
    except Exception as e:
      log.warning("fork: open failed: %s", e)
      if fork_agent is not None:
        await fork_agent.stop()
      await self._notify(f"⚠️ Couldn't open the fork: {e}")
      return

    if not thread_id:
      # No thread id back → can't route follow-ups; don't leak the subprocess.
      await fork_agent.stop()
      await self._notify(
        "⚠️ Lark didn't return a thread id; fork not started.")
      return

    sess = ForkSession(
      agent=fork_agent, root_msg_id=root_id, thread_id=thread_id,
      chat_id=self._chat_id, prompt0=prompt)
    self._forks[thread_id] = sess
    # Full ids (not truncated): correlating the Lark-assigned thread_id across
    # the open → follow-up → close round-trip needs the exact value.
    log.info("fork opened: thread=%s root=%s", thread_id, root_id)
    self._spawn_turn(sess, prompt)

  def route(self, thread_id: str, prompt: str) -> bool:
    """Route a follow-up message (already known to be in this fork's thread)
    to a new fork turn. Returns False if the thread is not a live fork."""
    sess = self._forks.get(thread_id)
    if sess is None:
      return False
    log.info("fork route: thread=%s", thread_id)
    self._spawn_turn(sess, prompt)
    return True

  async def interrupt(self, thread_id: str) -> bool:
    """Interrupt this fork's in-flight turn (the fork-scoped Stop button).
    PATCHes its card to 'stopping' and interrupts ONLY this fork's agent —
    the main turn and other forks are untouched. No-op if the thread isn't a
    live fork or has no turn running."""
    sess = self._forks.get(thread_id)
    if sess is None:
      return False
    log.info("fork interrupt: thread=%s", thread_id)
    sess.interrupt_requested = True
    if sess.renderer is not None:
      await sess.renderer.render_stopping()
    try:
      await sess.agent.interrupt()
    except Exception as e:
      log.warning("fork interrupt failed: %s", e)
    return True

  def _spawn_turn(self, sess: ForkSession, prompt: str) -> None:
    t = asyncio.create_task(self._run_turn(sess, prompt))
    self._tasks.add(t)
    t.add_done_callback(self._tasks.discard)

  async def _run_turn(self, sess: ForkSession, prompt: str) -> None:
    # Per-fork lock serializes turns *within* one fork (single SDK client);
    # different forks + the main turn still run concurrently.
    async with sess.lock:
      log.info("fork turn start: thread=%s", sess.thread_id)
      sess.interrupt_requested = False
      renderer = _ForkRenderer(self._ch, sess, self._loop)
      sess.renderer = renderer
      try:
        await sess.agent.run_turn(prompt, renderer.on_event)
      except Exception as e:
        if sess.interrupt_requested:
          # Interrupt can surface as an exception (e.g. the agent aborts its
          # turn) rather than a DoneEvent — render "stopped", not "error".
          log.info("fork turn interrupted: thread=%s", sess.thread_id)
          try:
            if renderer.card_id:
              await self._ch.update_card(renderer.card_id, renderer.stopped_card())
          except Exception as e2:
            log.debug("fork: stopped-card render failed: %s", e2)
        else:
          log.warning("fork turn failed: %s", e)
          card = cards.build_turn_card(
            "error", body=f"⚠️ fork turn failed: {e}",
            elapsed=int(time.time() - renderer.start))
          try:
            if renderer.card_id:
              await self._ch.update_card(renderer.card_id, card)
            else:
              await self._ch.send_card_in_thread(sess.root_msg_id, card)
          except Exception as e2:
            log.warning("fork: failed to render error card: %s", e2)
      finally:
        sess.renderer = None

  async def close(self, thread_id: str) -> bool:
    """Close a fork: stop its agent (frees the subprocess + scratch dir) and
    post a closing note in the thread. Returns False if not a live fork."""
    sess = self._forks.pop(thread_id, None)
    if sess is None:
      return False
    try:
      await sess.agent.stop()
    except Exception as e:
      log.warning("fork: agent stop on close failed: %s", e)
    try:
      await self._ch.send_card_in_thread(
        sess.root_msg_id,
        cards.build_markdown_card(
          "Fork closed — this branch is done.", title="🍴 Fork closed",
          color="grey"))
    except Exception as e:
      log.debug("fork: close note failed: %s", e)
    log.info("fork closed: thread=%s", thread_id)
    return True

  async def shutdown(self) -> None:
    """Stop every live fork (daemon shutdown). Fire-and-forget safe."""
    for sess in list(self._forks.values()):
      try:
        await sess.agent.stop()
      except Exception as e:
        log.warning("fork: shutdown stop failed: %s", e)
    self._forks.clear()
