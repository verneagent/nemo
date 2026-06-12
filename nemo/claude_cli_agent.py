"""CodingAgent adapter that drives the *unmodified interactive* `claude` TUI
under a pseudo-terminal (pty).

Why this exists — billing surface. The Claude CLI tags its API requests with a
self-reported surface in the User-Agent header:

  * interactive TUI    → ``claude-cli/<ver> (external, cli)``    (subscription)
  * headless / SDK     → ``claude-cli/<ver> (external, sdk-cli)`` (Agent credits)

Every SDK-based adapter (``ClaudeCodingAgent``) drives the headless stream-json
path, so it bills against the Agent SDK credit pool. Spawning the real
interactive TUI under a pty makes the *same* binary report ``(external, cli)``
— verified by ``scripts/cli_billing_probe.py``. This adapter is the
feasibility prototype for billing turns at the subscription rate.

How it works — the TUI is built for humans, not machines, so this is
screen-scraping, not a clean API:

  * Spawn ``claude`` on a pty (stdlib ``pty``), fixed winsize, ``TERM`` set.
  * A reader thread continuously feeds pty bytes into a ``pyte`` terminal
    emulator, which maintains a stable screen buffer + scrollback under a lock.
  * ``run_turn`` writes the prompt, then polls the emulated screen: it waits for
    work to start (the "esc to interrupt" spinner appears), then for it to end
    (spinner gone + empty input box + screen stable), then scrapes the answer.
  * The answer is the last ``⏺`` block that is NOT a tool call — tool calls
    render as ``⏺ Name(args)`` and are always followed by a ``⎿`` result line;
    prose answers are not.

KNOWN LIMITATIONS (this is a prototype — see the experiment write-up):
  * No structured usage/cost. ``DoneEvent`` reports empty usage / 0 cost; the
    TUI does not expose per-turn token counts on a machine channel.
  * Fragile to TUI layout/version changes — any reflow of the markers
    (``⏺`` / ``⎿`` / ``❯`` / "esc to interrupt") breaks scraping.
  * The interactive TUI does NOT flush its transcript jsonl per-turn (verified),
    so ``resume`` across daemon restarts is unsupported — ``reset`` respawns a
    fresh session and loses context.
  * Tool observability is best-effort (scraped ``⏺`` lines), far weaker than the
    SDK adapter's structured ProgressEvents.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import termios
import threading
import time
from typing import Callable

import pyte

from .channel import Channel
from .coding_agent import CodingAgent, EndpointConfig
from .db import Database
from .turn import AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent, TurnEvent
from .types import JsonObject

log = logging.getLogger(__name__)

# Fixed terminal geometry. Tall so a typical turn's output stays in the live
# pyte screen (plus scrollback) without us having to page history.
_ROWS = 200
_COLS = 160

# Markers in the claude TUI render (observed on claude-cli 2.1.175).
_USER_ECHO = "❯"          # user prompt echo AND the (empty) input box
_ASSISTANT = "⏺"          # assistant text block OR tool invocation
_TOOL_RESULT = "⎿"        # tool result (indented under a ⏺ tool call)
_THINKING = "✻"           # post-hoc "Cogitated for Ns" summary line
# Present while a turn is running; absent when idle. The single most reliable
# "is the agent working" signal the TUI gives us.
_WORKING_HINT = "esc to interrupt"
# A ⏺ line that is a tool call: a single CapitalCase identifier then "(".
_TOOL_CALL_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*\(")


class _PtyTui:
  """Owns the pty + claude subprocess + a pyte emulator fed by a reader thread.

  Thread-safety: only the reader thread feeds the pyte stream; ``snapshot`` and
  ``feed`` both take ``_lock`` so a snapshot never races a feed.
  """

  def __init__(self, argv: list[str], cwd: str, env: dict[str, str]):
    self._argv = argv
    self._cwd = cwd
    self._env = env
    self._master = -1
    self._proc: subprocess.Popen[bytes] | None = None
    self._screen = pyte.HistoryScreen(_COLS, _ROWS, history=10000, ratio=0.5)
    self._stream = pyte.ByteStream(self._screen)
    self._lock = threading.Lock()
    self._reader: threading.Thread | None = None
    self._alive = False

  def spawn(self) -> None:
    if not os.path.exists(self._argv[0]):
      raise RuntimeError(f"`claude` not found: {self._argv[0]}")
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ,
                struct.pack("HHHH", _ROWS, _COLS, 0, 0))
    # subprocess.Popen with setsid is the proven-reliable spawn here; a manual
    # os.fork()+TIOCSCTTY variant rendered output fine but silently dropped
    # keyboard input (the input box never received the typed prompt).
    self._proc = subprocess.Popen(
      self._argv, stdin=slave, stdout=slave, stderr=slave,
      cwd=self._cwd, env=self._env, preexec_fn=os.setsid, close_fds=True,
    )
    os.close(slave)
    self._master = master
    self._alive = True
    self._reader = threading.Thread(target=self._read_loop, daemon=True)
    self._reader.start()

  def _read_loop(self) -> None:
    while self._alive:
      try:
        r, _, _ = select.select([self._master], [], [], 0.2)
      except (OSError, ValueError):
        break
      if self._master not in r:
        continue
      try:
        data = os.read(self._master, 65536)
      except OSError:
        break
      if not data:
        break
      with self._lock:
        self._stream.feed(data)
    self._alive = False

  def write(self, data: bytes) -> None:
    if self._master >= 0:
      try:
        os.write(self._master, data)
      except OSError as e:
        log.warning("pty write failed: %s", e)

  def _row_text(self, row: object) -> str:
    """Render one pyte history row (a col→Char mapping) to a plain string."""
    chars: list[str] = []
    for x in range(_COLS):
      try:
        cell = row[x]  # type: ignore[index]
        chars.append(cell.data)
      except (KeyError, AttributeError, TypeError):
        chars.append(" ")
    return "".join(chars).rstrip()

  def snapshot(self) -> list[str]:
    """Full logical screen = scrollback history + visible display, as lines."""
    with self._lock:
      lines: list[str] = []
      top = getattr(self._screen.history, "top", None)
      if top:
        for row in list(top):
          lines.append(self._row_text(row))
      lines.extend(line.rstrip() for line in self._screen.display)
    return lines

  def is_working(self) -> bool:
    return any(_WORKING_HINT in line for line in self.snapshot())

  def submit(self, text: str) -> None:
    """Type a prompt then submit it. Sending the text and the CR as one write
    can be dropped by the TUI; a brief gap between them is reliable."""
    self.write(text.encode())
    time.sleep(0.3)
    self.write(b"\r")

  def close(self) -> None:
    self._alive = False
    if self._proc is not None:
      try:
        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
      except (ProcessLookupError, OSError):
        pass
      try:
        self._proc.wait(timeout=5)
      except (subprocess.TimeoutExpired, OSError):
        pass
      self._proc = None
    if self._master >= 0:
      try:
        os.close(self._master)
      except OSError:
        pass
      self._master = -1


def _is_prompt_echo(stripped: str) -> bool:
  """True for a user prompt echo line ``❯ <text>`` (not the empty input box
  ``❯`` and not the box border)."""
  if not stripped.startswith(_USER_ECHO):
    return False
  return bool(stripped[len(_USER_ECHO):].strip())


def _region_after_echo(lines: list[str], prompt: str) -> list[str]:
  """The screen lines that belong to THIS turn: from the last echo of its
  prompt (``❯ <prompt>``) up to the NEXT prompt echo (or end). Anchor not
  found ⇒ all lines. Bounding at the next echo keeps a mid-scrollback turn
  from bleeding into a later turn's output."""
  anchor = -1
  needle = prompt.strip()[:48]
  for i, line in enumerate(lines):
    s = line.strip()
    if _is_prompt_echo(s) and needle and needle in s:
      anchor = i
  if anchor < 0:
    return lines
  end = len(lines)
  for j in range(anchor + 1, len(lines)):
    if _is_prompt_echo(lines[j].strip()):
      end = j
      break
  return lines[anchor + 1:end]


def _extract_answer(lines: list[str], prompt: str) -> str:
  """Scrape the final assistant prose answer from the screen lines.

  Anchors on the echo of THIS turn's prompt (``❯ <prompt>``), then returns the
  last ``⏺`` block in the region that is NOT a tool call. A tool call is a ``⏺``
  block whose text matches ``Name(...)`` or that is immediately followed by a
  ``⎿`` result line; prose answers are neither.
  """
  region = _region_after_echo(lines, prompt)

  # Collect ⏺ blocks: each block is the ⏺ line plus following indented
  # continuation lines, until the next marker / input-box border.
  blocks: list[tuple[str, bool]] = []  # (text, is_tool)
  i = 0
  n = len(region)
  while i < n:
    s = region[i].strip()
    if s.startswith(_ASSISTANT):
      body = s[len(_ASSISTANT):].strip()
      is_tool = bool(_TOOL_CALL_RE.match(body))
      cont: list[str] = [body] if body else []
      j = i + 1
      while j < n:
        nxt = region[j].strip()
        if (not nxt or nxt.startswith(_ASSISTANT) or nxt.startswith(_THINKING)
            or nxt.startswith("─") or nxt.startswith(_USER_ECHO)):
          break
        if nxt.startswith(_TOOL_RESULT):
          is_tool = True  # ⏺ block followed by ⎿ result ⇒ it was a tool call
          break
        cont.append(nxt)
        j += 1
      blocks.append(("\n".join(cont).strip(), is_tool))
      i = j
    else:
      i += 1

  for text, is_tool in reversed(blocks):
    if not is_tool and text:
      return text
  return ""


def _latest_tool_summary(lines: list[str], seen: set[str]) -> str | None:
  """Return a newly-appeared ``⏺ Name(args)`` tool line not yet in ``seen``."""
  for line in lines:
    s = line.strip()
    if s.startswith(_ASSISTANT):
      body = s[len(_ASSISTANT):].strip()
      if _TOOL_CALL_RE.match(body) and body not in seen:
        seen.add(body)
        return body
  return None


class ClaudeCliCodingAgent(CodingAgent):
  """Drives the interactive ``claude`` TUI over a pty (subscription billing).

  See the module docstring for the rationale and known limitations.
  """

  # Tunables (seconds).
  _BOOT_WAIT = 6.0          # let the TUI splash/onboarding render before turn 1
  _START_TIMEOUT = 20.0     # max wait for a turn to BEGIN working after submit
  _TURN_TIMEOUT = 600.0     # hard ceiling on a single turn
  _SETTLE = 2.5             # screen stable this long + idle ⇒ turn done
  _POLL = 0.3               # screen poll cadence

  def __init__(
    self,
    credentials: dict[str, str],
    chat_id: str,
    db: Database,
    channel: Channel,
    permission_mode: str = "bypassPermissions",
    system_prompt: str = "",
    endpoint: EndpointConfig | None = None,
  ):
    self._credentials = credentials
    self._chat_id = chat_id
    self._db = db
    self._channel = channel
    self._permission_mode = permission_mode
    self._system_prompt = system_prompt
    self._endpoint = endpoint or EndpointConfig()
    self._project_dir = ""
    self._model = ""
    self._effort = ""
    self._tui: _PtyTui | None = None
    self._booted = False

  def set_effort(self, effort: str) -> None:
    # The interactive TUI has no per-turn effort flag we drive here; record it
    # for parity but it is a no-op in this prototype.
    self._effort = effort

  def set_endpoint(self, endpoint: EndpointConfig) -> None:
    self._endpoint = endpoint

  def _build_argv(self) -> list[str]:
    claude = shutil.which("claude") or "claude"
    argv = [claude]
    # Permission handling: the daemon runs unattended, so bypass interactive
    # permission panels. bypassPermissions → --dangerously-skip-permissions;
    # anything else maps to the closest non-interactive stance we can take.
    if self._permission_mode == "bypassPermissions":
      argv.append("--dangerously-skip-permissions")
    else:
      argv += ["--permission-mode", self._permission_mode or "acceptEdits"]
    if self._model:
      argv += ["--model", self._model]
    return argv

  def _build_env(self) -> dict[str, str]:
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    # Do NOT set CLAUDE_CODE_ENTRYPOINT — letting the CLI choose its own keeps
    # the interactive ``(external, cli)`` surface (the whole point).
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    if self._endpoint.base_url:
      env["ANTHROPIC_BASE_URL"] = self._endpoint.base_url
    if self._endpoint.api_key:
      env["ANTHROPIC_API_KEY"] = self._endpoint.api_key
    env["NEMO_CHAT_ID"] = self._chat_id
    return env

  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    del resume  # interactive TUI holds context in-process; no resume support
    self._project_dir = project_dir
    self._model = model
    await self._spawn()

  async def _spawn(self) -> None:
    tui = _PtyTui(self._build_argv(), self._project_dir, self._build_env())
    await asyncio.to_thread(tui.spawn)
    self._tui = tui
    self._booted = False

  def _ensure_booted(self) -> None:
    """Sync — runs on the turn worker thread (see run_turn)."""
    if self._booted or self._tui is None:
      return
    # Let the splash / onboarding settle, then nudge past any trust/theme
    # dialog so the input box is ready for the first prompt.
    time.sleep(self._BOOT_WAIT)
    self._tui.write(b"\r")
    time.sleep(0.6)
    self._booted = True

  def _wait_idle(self, tui: _PtyTui, timeout: float) -> bool:
    """Block until the TUI is not working and its screen has been stable for a
    beat — i.e. ready to accept a new prompt. Prevents submitting into a busy
    TUI, which would queue prompts and desync answers from turns. Sync — runs
    on the turn worker thread."""
    last = ""
    stable_since = time.monotonic()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      time.sleep(self._POLL)
      lines = tui.snapshot()
      working = any(_WORKING_HINT in ln for ln in lines)
      disp = "\n".join(lines)
      now = time.monotonic()
      if disp != last:
        last = disp
        stable_since = now
      if not working and (now - stable_since) >= 1.0:
        return True
    return False

  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    """Run one turn on a worker thread.

    Critical: the host's ``on_event`` callback marshals card sends back to the
    main loop with a *blocking* ``run_coroutine_threadsafe(...).result()`` — it
    is written to be invoked from a worker thread (the SDK adapter calls it from
    ``SDKThread``). If we called ``on_event`` directly from this coroutine (i.e.
    on the main loop) the DoneEvent's blocking marshal would deadlock the loop.
    So the whole turn — pty polling and every ``on_event`` call — runs in a
    thread, exactly like the SDK path. The main loop stays free to receive
    /esc and call ``interrupt()`` concurrently.
    """
    if self._tui is None:
      await self._spawn()
    assert self._tui is not None
    return await asyncio.to_thread(self._run_turn_sync, self._tui, prompt, on_event)

  def _run_turn_sync(
    self,
    tui: _PtyTui,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    self._ensure_booted()

    # Never submit into a busy TUI — wait for the previous turn to fully drain
    # so prompts can't queue and desync.
    log.info("claude-cli run_turn: waiting for idle TUI (prompt=%d chars)", len(prompt))
    self._wait_idle(tui, timeout=self._TURN_TIMEOUT)

    # Submit the prompt (text + CR sent separately; see _PtyTui.submit). The
    # TUI echoes "❯ <prompt>" once it accepts the input.
    log.info("claude-cli run_turn: submitting prompt")
    tui.submit(prompt)

    start = time.monotonic()
    began = False
    seen_tools: set[str] = set()
    last_display = ""
    last_change = start
    progress_started = False
    timed_out = False
    needle = prompt.strip()[:48]

    def _echo_present(lines: list[str]) -> bool:
      return any(ln.strip().startswith(_USER_ECHO) and needle in ln
                 for ln in lines)

    while True:
      time.sleep(self._POLL)
      now = time.monotonic()
      lines = tui.snapshot()
      working = any(_WORKING_HINT in ln for ln in lines)

      # Surface tool calls as ProgressEvents as they appear. Scope to THIS
      # turn's region so stale ⏺ tool lines from prior turns (still in
      # scrollback) aren't re-emitted.
      tool = _latest_tool_summary(_region_after_echo(lines, prompt), seen_tools)
      if tool is not None:
        on_event(ProgressEvent(
          kind="tool", summary=tool, first=not progress_started))
        progress_started = True

      if working:
        began = True

      display = "\n".join(lines[-_ROWS:])
      if display != last_display:
        last_display = display
        last_change = now

      if not began:
        # Waiting for the turn to start. A trivial turn can finish before the
        # spinner is ever caught; treat "our prompt echoed + content present +
        # stable while idle" as an implicit start.
        if _echo_present(lines) and any(
            ln.strip().startswith(_ASSISTANT) for ln in lines):
          began = True
        elif now - start > self._START_TIMEOUT:
          timed_out = not _echo_present(lines)
          began = True  # give the done-check a chance; if nothing, answer="".
        continue

      # Turn has begun: done when no longer working AND screen is stable.
      if not working and (now - last_change) >= self._SETTLE:
        break
      if now - start > self._TURN_TIMEOUT:
        timed_out = True
        break

    lines = tui.snapshot()
    answer = _extract_answer(lines, prompt)
    log.info("claude-cli run_turn: done (began=%s timed_out=%s answer=%d chars, %.0fs)",
             began, timed_out, len(answer), time.monotonic() - start)

    if timed_out and not answer:
      on_event(ErrorEvent(message="claude-cli turn timed out (no answer scraped)"))
      on_event(DoneEvent(cost=0.0, usage={}))
      return 0.0, {}

    if answer:
      on_event(AnswerEvent(text=answer))
    elif progress_started:
      # Tools ran but no prose tail was scraped — surface a minimal note rather
      # than a silent empty card.
      on_event(AnswerEvent(text="(done — no text answer captured from the TUI)"))

    on_event(DoneEvent(cost=0.0, usage={}))
    return 0.0, {}

  async def interrupt(self) -> None:
    # ESC interrupts the current turn without killing the session (matches the
    # CLI's Escape semantics — context preserved).
    if self._tui is not None:
      self._tui.write(b"\x1b")

  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    del resume  # no resume; respawn is a fresh session (context is lost)
    await self.stop()
    self._project_dir = project_dir
    self._model = model
    await self._spawn()

  async def stop(self) -> None:
    if self._tui is not None:
      tui = self._tui
      self._tui = None
      await asyncio.to_thread(tui.close)
