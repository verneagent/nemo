"""User-triggered shell command execution for Nemo.

This is intentionally outside ``CodingAgent`` adapters: shell shortcuts are
channel-side operator actions.  ``!cmd`` records the completed result for the
next agent turn; ``!!cmd`` is shown to the user only.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from . import cards
from .types import JsonObject

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300.0
CARD_UPDATE_INTERVAL = 3.0
DISPLAY_TAIL_CHARS = 16_000
CONTEXT_TAIL_CHARS = 32_000
MAX_PENDING_CONTEXTS = 8


@dataclass(frozen=True)
class ShellShortcut:
  command: str
  inject_context: bool


def parse_shell_shortcut(text: str) -> ShellShortcut | None:
  """Parse a whole-message shell shortcut.

  ``!cmd`` executes and injects the final result into the next agent turn.
  ``!!cmd`` executes without context injection.
  """
  stripped = text.strip()
  if not stripped.startswith("!"):
    return None
  inject = not stripped.startswith("!!")
  command = stripped[1:] if inject else stripped[2:]
  return ShellShortcut(command=command.strip(), inject_context=inject)


@dataclass
class ShellResult:
  job_id: str
  command: str
  cwd: str
  status: str
  exit_code: int | None
  duration: float
  stdout: str
  stderr: str
  timed_out: bool = False


class ShellChannel(Protocol):
  async def send_card(self, chat_id: str, card: JsonObject) -> str:
    ...

  async def update_card(self, message_id: str, card: JsonObject) -> str:
    ...


def _trim_middle(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  omitted = len(text) - limit
  return f"... {omitted} chars omitted ...\n{text[-limit:]}"


def format_context(result: ShellResult) -> str:
  """Render a completed shell result for the next LLM turn."""
  status_bits = [result.status]
  if result.exit_code is not None:
    status_bits.append(f"exit {result.exit_code}")
  if result.timed_out:
    status_bits.append("timed out")
  output_parts: list[str] = []
  if result.stdout:
    output_parts.append("stdout:\n" + _trim_middle(result.stdout, CONTEXT_TAIL_CHARS))
  if result.stderr:
    output_parts.append("stderr:\n" + _trim_middle(result.stderr, CONTEXT_TAIL_CHARS))
  output = "\n\n".join(output_parts) or "(no output)"
  return (
    "[Nemo shell context]\n"
    "The user ran this shell command outside the coding agent.\n\n"
    f"Command:\n{result.command}\n\n"
    f"Working directory:\n{result.cwd}\n\n"
    f"Status:\n{' · '.join(status_bits)} · {result.duration:.1f}s\n\n"
    f"Output:\n{output}"
  )


@dataclass
class ShellJob:
  job_id: str
  command: str
  cwd: str
  chat_id: str
  inject_context: bool
  card_id: str = ""
  status: str = "running"
  exit_code: int | None = None
  start_time: float = field(default_factory=time.time)
  end_time: float = 0.0
  stdout_chunks: deque[str] = field(default_factory=deque)
  stderr_chunks: deque[str] = field(default_factory=deque)
  stdout_omitted: int = 0
  stderr_omitted: int = 0
  proc: asyncio.subprocess.Process | None = None
  task: asyncio.Task[None] | None = None
  abort_requested: bool = False

  def append_output(self, stream: str, text: str) -> None:
    chunks = self.stderr_chunks if stream == "stderr" else self.stdout_chunks
    chunks.append(text)
    total = sum(len(chunk) for chunk in chunks)
    while total > DISPLAY_TAIL_CHARS * 2 and chunks:
      dropped = chunks.popleft()
      total -= len(dropped)
      if stream == "stderr":
        self.stderr_omitted += len(dropped)
      else:
        self.stdout_omitted += len(dropped)

  def _joined_stream(self, stream: str) -> str:
    chunks = self.stderr_chunks if stream == "stderr" else self.stdout_chunks
    omitted = self.stderr_omitted if stream == "stderr" else self.stdout_omitted
    text = "".join(chunks)
    if omitted:
      return f"... {omitted} chars omitted ...\n{text}"
    return text

  @property
  def stdout(self) -> str:
    return self._joined_stream("stdout")

  @property
  def stderr(self) -> str:
    return self._joined_stream("stderr")

  @property
  def duration(self) -> float:
    return (self.end_time or time.time()) - self.start_time


class ShellJobManager:
  """Run shell commands without blocking the Nemo loop."""

  def __init__(
    self,
    channel: ShellChannel,
    *,
    chat_id: str,
    project_dir: str,
    on_context: Callable[[str], None],
    timeout: float | None = DEFAULT_TIMEOUT,
  ) -> None:
    self._channel = channel
    self._chat_id = chat_id
    self._project_dir = project_dir
    self._on_context = on_context
    self._timeout = timeout
    self._jobs: dict[str, ShellJob] = {}

  def set_project_dir(self, project_dir: str) -> None:
    self._project_dir = project_dir

  async def start(self, shortcut: ShellShortcut) -> str:
    if not shortcut.command:
      return "Usage: `!<shell command>` to inject output, `!!<shell command>` to run without injection."
    job_id = uuid.uuid4().hex[:8]
    job = ShellJob(
      job_id=job_id,
      command=shortcut.command,
      cwd=self._project_dir,
      chat_id=self._chat_id,
      inject_context=shortcut.inject_context,
    )
    card = cards.build_shell_card(
      "running",
      job_id=job.job_id,
      command=job.command,
      cwd=job.cwd,
      elapsed=0,
      inject_context=job.inject_context,
      chat_id=self._chat_id,
    )
    job.card_id = await self._channel.send_card(self._chat_id, card)
    self._jobs[job.job_id] = job
    log.info(
      "Shell job %s started inject=%s command=%r",
      job.job_id,
      job.inject_context,
      job.command,
    )
    task = asyncio.create_task(self._run(job))
    job.task = task
    task.add_done_callback(lambda done: self._task_done(job.job_id, done))
    return ""

  async def abort(self, job_id: str) -> bool:
    job = self._jobs.get(job_id) if job_id else self._latest_running_job()
    if job is None or job.status != "running":
      return False
    job.abort_requested = True
    proc = job.proc
    if proc is not None and proc.returncode is None:
      await self._terminate_process_group(proc)
    return True

  def _latest_running_job(self) -> ShellJob | None:
    running = [job for job in self._jobs.values() if job.status == "running"]
    if not running:
      return None
    return max(running, key=lambda job: job.start_time)

  async def close(self) -> None:
    jobs = [job for job in self._jobs.values() if job.status == "running"]
    await asyncio.gather(*(self.abort(job.job_id) for job in jobs), return_exceptions=True)

  def _task_done(self, job_id: str, task: asyncio.Task[None]) -> None:
    with contextlib.suppress(asyncio.CancelledError):
      exc = task.exception()
      if exc is not None:
        # Keep the exception visible through the event loop's exception handler.
        asyncio.get_running_loop().call_exception_handler({
          "message": f"Shell job {job_id} failed",
          "exception": exc,
        })

  async def _run(self, job: ShellJob) -> None:
    update_task: asyncio.Task[None] | None = None
    timed_out = False
    try:
      proc = await asyncio.create_subprocess_shell(
        job.command,
        cwd=job.cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
      )
      job.proc = proc
      if job.abort_requested:
        await self._terminate_process_group(proc)
      update_task = asyncio.create_task(self._periodic_update(job))
      readers = [
        asyncio.create_task(self._read_stream(job, "stdout", proc.stdout)),
        asyncio.create_task(self._read_stream(job, "stderr", proc.stderr)),
      ]
      if job.inject_context and self._timeout is not None:
        try:
          await asyncio.wait_for(proc.wait(), timeout=self._timeout)
        except TimeoutError:
          timed_out = True
          job.status = "timeout"
          await self._terminate_process_group(proc)
      else:
        await proc.wait()
      await asyncio.gather(*readers, return_exceptions=True)
      if job.abort_requested:
        job.status = "aborted"
      elif timed_out:
        job.status = "timeout"
      elif proc.returncode == 0:
        job.status = "done"
      else:
        job.status = "failed"
      job.exit_code = proc.returncode
    except Exception as exc:
      job.status = "error"
      job.append_output("stderr", f"{type(exc).__name__}: {exc}")
    finally:
      job.end_time = time.time()
      if update_task is not None:
        update_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
          await update_task
      await self._safe_update_card(job)
      if job.inject_context:
        result = ShellResult(
          job_id=job.job_id,
          command=job.command,
          cwd=job.cwd,
          status=job.status,
          exit_code=job.exit_code,
          duration=job.duration,
          stdout=job.stdout,
          stderr=job.stderr,
          timed_out=timed_out,
        )
        self._on_context(format_context(result))
      log.info(
        "Shell job %s completed status=%s exit=%s duration=%.1fs",
        job.job_id,
        job.status,
        job.exit_code,
        job.duration,
      )

  async def _read_stream(
    self,
    job: ShellJob,
    stream_name: str,
    stream: asyncio.StreamReader | None,
  ) -> None:
    if stream is None:
      return
    while True:
      chunk = await stream.read(4096)
      if not chunk:
        return
      job.append_output(stream_name, chunk.decode(errors="replace"))

  async def _periodic_update(self, job: ShellJob) -> None:
    while job.status == "running":
      await asyncio.sleep(CARD_UPDATE_INTERVAL)
      await self._safe_update_card(job)

  async def _safe_update_card(self, job: ShellJob) -> None:
    try:
      await self._update_card(job)
    except Exception as exc:
      log.warning("Failed to update shell job %s card: %s", job.job_id, exc)

  async def _update_card(self, job: ShellJob) -> None:
    if not job.card_id:
      return
    card = cards.build_shell_card(
      job.status,
      job_id=job.job_id,
      command=job.command,
      cwd=job.cwd,
      elapsed=int(job.duration),
      inject_context=job.inject_context,
      chat_id=self._chat_id,
      exit_code=job.exit_code,
      stdout=job.stdout,
      stderr=job.stderr,
    )
    job.card_id = await self._channel.update_card(job.card_id, card)

  async def _terminate_process_group(
    self,
    proc: asyncio.subprocess.Process,
  ) -> None:
    if proc.returncode is not None:
      return
    with contextlib.suppress(ProcessLookupError):
      os.killpg(proc.pid, signal.SIGTERM)
    try:
      await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=2)
