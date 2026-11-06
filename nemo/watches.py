"""Daemon-owned watch processes — the Monitor tool's execution engine.

WHY THIS EXISTS (2026-11-06, oc_9540cd47…)

The Monitor feature used to be a child process of the Claude CLI, and every
bug it had was the same bug in a different costume: the daemon had to
RECONSTRUCT which watch a report belonged to and whether it was still alive,
because the CLI owns the process. Identity was inferred from the only live
watch, then from a byte cursor over the CLI's private transcript JSONL;
lifecycle was inferred from task-start / task-notification messages that the
CLI may or may not surface between turns. Each fix added another
reconciliation rule, and each new rule could disagree with the next CLI
behaviour (mid-turn injection, a completion pushed back as a real turn, a
reconnect that kills every CLI child).

Here the daemon spawns the command itself. Identity is the spawn — the report
is read from THIS process's stdout. Lifecycle is the process — it is live
until it exits. Nothing is reconstructed, so nothing can drift.

The model still DECIDES what to watch (the ``Monitor`` tool call carries the
command and the description); the Claude adapter denies that call and hands
its input here instead, so the model's interface is unchanged while the
execution is ours.

Contract with the host (all callbacks run on the main loop except
``on_finished``'s caller, which is still the main loop):

- ``on_start(watch)``  — post the 👀 card. Raises if the card cannot be sent;
                          arming is then refused (a watch nobody can see is
                          not a watch).
- ``on_report(watch, text)`` — append one line of output to that card.
- ``on_end(watch)``    — freeze the card with ``watch.status``.
- ``on_finished(watch)`` — the process is gone: wake the model once so it can
                          answer the user (or explicitly say nothing is new).

Stopping: the model (or a user relaying through it) touches
``<stop_dir>/<watch_id>.stop`` — a plain file operation any Bash call can do,
so no new binary has to be installed or reached from the CLI's environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

MAX_LIVE_WATCHES = 3
STOP_POLL_SECONDS = 1.0
KILL_GRACE_SECONDS = 3.0
MAX_QUEUED_REPORTS = 100
TAIL_LINES = 40
_MAX_LINE_CHARS = 2000
# Long enough that a single chatty line cannot make StreamReader.readline()
# raise; a line longer than this is truncated anyway.
_STREAM_LIMIT = 1024 * 1024
_MAX_REMEMBERED_ENDED = 20


@dataclass
class Watch:
  """One running watch: the process, its card identity, and its status."""

  watch_id: str
  description: str
  command: str
  timeout_ms: int
  chat_id: str
  started_at: float
  stop_file: Path | None = None
  proc: asyncio.subprocess.Process | None = None
  task: asyncio.Task[None] | None = None
  # "" while live; one of completed / failed / stopped / timeout when over.
  status: str = ""
  reports: int = 0
  tail: deque[str] = field(default_factory=lambda: deque(maxlen=TAIL_LINES))

  @property
  def live(self) -> bool:
    return not self.status


class WatchManager:
  """Owns the watch processes. Daemon-scoped, main-loop owned."""

  def __init__(
    self,
    *,
    on_start: Callable[[Watch], Awaitable[None]],
    on_report: Callable[[Watch, str], Awaitable[None]],
    on_end: Callable[[Watch], Awaitable[None]],
    on_finished: Callable[[Watch], None],
    spawn_cwd: str,
    stop_dir: Path,
    max_live: int = MAX_LIVE_WATCHES,
  ) -> None:
    self.on_start = on_start
    self.on_report = on_report
    self.on_end = on_end
    self.on_finished = on_finished
    self.spawn_cwd = spawn_cwd
    self.stop_dir = stop_dir
    self.max_live = max_live
    self._watches: dict[str, Watch] = {}
    self._reap_tasks: set[asyncio.Task[None]] = set()
    try:
      stop_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
      # Arming still works; only the file-based stop path is unavailable,
      # which is a degradation the deny reason must not promise.
      log.warning("Watch stop dir %s unavailable: %s", stop_dir, exc)

  # ---- public surface -----------------------------------------------------

  def live_ids(self) -> list[str]:
    return [w.watch_id for w in self._watches.values() if w.live]

  def get(self, watch_id: str) -> Watch | None:
    return self._watches.get(watch_id)

  async def arm(
    self, *, chat_id: str, description: str, command: str, timeout_ms: int,
  ) -> tuple[bool, str]:
    """Spawn a watch. Returns ``(ok, text-for-the-model)``.

    The text is the deny reason the model reads, so it must say what actually
    happened — a silent success would leave the model believing it still owns
    the CLI task it was about to start.
    """
    command = command.strip()
    if not command:
      return False, "Monitor 的 command 是空的，没有可执行的内容。"
    live = self.live_ids()
    if len(live) >= self.max_live:
      return False, (
        f"同时最多 {self.max_live} 个后台监控，现在已经有 {len(live)} 个在跑"
        f"（{', '.join(live)}）。先停掉不需要的："
        f"`touch {self.stop_dir}/<watch_id>.stop`，或者把多个观察合并成一条 "
        "Monitor 命令，再挂新的。"
      )
    watch_id = f"w{uuid.uuid4().hex[:8]}"
    watch = Watch(
      watch_id=watch_id,
      description=description.strip() or "后台监控",
      command=command,
      timeout_ms=max(0, int(timeout_ms or 0)),
      chat_id=chat_id,
      started_at=time.time(),
      stop_file=self.stop_dir / f"{watch_id}.stop",
    )
    self._watches[watch.watch_id] = watch
    try:
      await self.on_start(watch)
    except Exception as exc:
      log.exception("Watch %s: marker card failed", watch.watch_id)
      watch.status = "failed"
      self._prune_ended()
      return False, (
        "监控没有挂上：卡片发送失败（"
        f"{exc}）。不要用同样的参数重试，先告诉用户这次没挂成功。"
      )
    watch.task = asyncio.ensure_future(self._supervise(watch))
    log.info(
      "Watch armed id=%s chat=%s timeout_ms=%s cmd=%r",
      watch.watch_id, chat_id, watch.timeout_ms, command[:120])
    return True, (
      f"已由 Nemo 直接接管执行：watch {watch.watch_id}（后台进程独立于 CLI，"
      "/clear 或重连都不会中断它）。每次输出的每一行都会自动追加到它的"
      "👀 卡片上，你不用再转述。它结束时你会收到一条自动通知，届时再决定"
      f"是否跟进。要停止它：`touch {watch.stop_file}`。"
    )

  async def stop(self, watch_id: str) -> bool:
    watch = self._watches.get(watch_id)
    if watch is None or not watch.live:
      return False
    watch.status = "stopped"
    log.info("Watch %s stopped by request", watch_id)
    self._terminate(watch)
    return True

  async def close(self) -> None:
    """Terminate everything (daemon shutdown). Watches do not outlive us.

    Termination, not cancellation: a cancelled supervise task raises at its
    first await in ``finally``, so the card would never be closed and
    ``on_finished`` would never run. Killing the process ends the reader
    naturally, which is the same path a normal exit takes.
    """
    live = [w for w in self._watches.values() if w.live]
    for watch in live:
      watch.status = "stopped"
      self._terminate(watch)
    tasks = [w.task for w in live if w.task is not None]
    if not tasks:
      return
    _, pending = await asyncio.wait(tasks, timeout=KILL_GRACE_SECONDS + 2)
    for task in pending:
      task.cancel()
    if pending:
      await asyncio.gather(*pending, return_exceptions=True)

  # ---- process supervision ------------------------------------------------

  async def _supervise(self, watch: Watch) -> None:
    try:
      watch.proc = await asyncio.create_subprocess_shell(
        watch.command,
        cwd=self.spawn_cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        limit=_STREAM_LIMIT,
      )
    except Exception:
      log.exception("Watch %s failed to spawn", watch.watch_id)
      watch.status = "failed"
      await self._finish(watch)
      return
    log.info(
      "Watch %s process started pid=%s", watch.watch_id, watch.proc.pid)
    guard = asyncio.ensure_future(self._guard(watch))
    try:
      await self._pump(watch)
    except asyncio.CancelledError:
      watch.status = watch.status or "stopped"
      self._terminate(watch)
      raise
    except Exception:
      log.exception("Watch %s pump failed", watch.watch_id)
      watch.status = watch.status or "failed"
      self._terminate(watch)
    finally:
      guard.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await guard
      if watch.live:
        rc = watch.proc.returncode if watch.proc is not None else None
        watch.status = "completed" if rc == 0 else "failed"
      await self._finish(watch)

  async def _pump(self, watch: Watch) -> None:
    """Stream stdout lines to the card, one report per line."""
    proc = watch.proc
    if proc is None or proc.stdout is None:
      return
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    dropped = 0

    async def report_loop() -> None:
      while True:
        item = await queue.get()
        if item is None:
          return
        watch.reports += 1
        watch.tail.append(item)
        try:
          await self.on_report(watch, item)
        except Exception:
          # The host already surfaces a failed card write in the chat; keep
          # reading so a slow/failed edit never wedges the child on a full
          # pipe (that looks exactly like a hung watch).
          log.exception("Watch %s: report delivery failed", watch.watch_id)

    reporter = asyncio.ensure_future(report_loop())
    try:
      while True:
        try:
          raw = await proc.stdout.readline()
        except ValueError:
          # A single line exceeded the reader's limit; take what is there.
          raw = await proc.stdout.read(65536)
        if not raw:
          break
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.strip():
          continue
        if len(line) > _MAX_LINE_CHARS:
          line = line[:_MAX_LINE_CHARS] + " …"
        if queue.qsize() >= MAX_QUEUED_REPORTS:
          dropped += 1
          continue
        if dropped:
          queue.put_nowait(f"（输出过快，省略 {dropped} 行）")
          dropped = 0
        queue.put_nowait(line.rstrip())
      await proc.wait()
    finally:
      if dropped:
        queue.put_nowait(f"（输出过快，省略 {dropped} 行）")
      queue.put_nowait(None)
      await reporter

  async def _guard(self, watch: Watch) -> None:
    """Enforce the stop file and the ``timeout_ms`` lifetime cap."""
    deadline = (
      time.monotonic() + watch.timeout_ms / 1000.0
      if watch.timeout_ms > 0 else None)
    while watch.live:
      if watch.stop_file is not None and watch.stop_file.exists():
        watch.status = "stopped"
        log.info("Watch %s stop file seen", watch.watch_id)
        self._terminate(watch)
        return
      if deadline is not None and time.monotonic() >= deadline:
        watch.status = "timeout"
        log.info(
          "Watch %s exceeded its %ss cap", watch.watch_id,
          watch.timeout_ms / 1000.0)
        self._terminate(watch)
        return
      await asyncio.sleep(STOP_POLL_SECONDS)

  def _terminate(self, watch: Watch) -> None:
    proc = watch.proc
    if proc is None or proc.returncode is not None:
      return
    try:
      os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
      return
    except OSError as exc:
      log.warning("Watch %s: SIGTERM failed: %s", watch.watch_id, exc)

    async def hard_kill() -> None:
      await asyncio.sleep(KILL_GRACE_SECONDS)
      if proc.returncode is not None:
        return
      try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        log.warning("Watch %s ignored SIGTERM, killed", watch.watch_id)
      except ProcessLookupError:
        pass
      except OSError as exc:
        log.warning("Watch %s: SIGKILL failed: %s", watch.watch_id, exc)

    task = asyncio.ensure_future(hard_kill())
    self._reap_tasks.add(task)
    task.add_done_callback(self._reap_tasks.discard)

  async def _finish(self, watch: Watch) -> None:
    if watch.stop_file is not None:
      with contextlib.suppress(OSError):
        watch.stop_file.unlink()
    log.info(
      "Watch ended id=%s status=%s reports=%d",
      watch.watch_id, watch.status or "?", watch.reports)
    try:
      await self.on_end(watch)
    except Exception:
      log.exception("Watch %s: card close failed", watch.watch_id)
    try:
      self.on_finished(watch)
    except Exception:
      log.exception("Watch %s: wake-up failed", watch.watch_id)
    self._prune_ended()

  def _prune_ended(self) -> None:
    ended = [w for w in self._watches.values() if not w.live]
    excess = len(ended) - _MAX_REMEMBERED_ENDED
    if excess <= 0:
      return
    ended.sort(key=lambda w: w.started_at)
    for watch in ended[:excess]:
      self._watches.pop(watch.watch_id, None)
