"""CodingAgent adapter that drives the *unmodified interactive* ``claude`` TUI
under a pseudo-terminal (pty), billing turns at the subscription rate.

Why this exists — billing surface. The Claude CLI tags its API requests with a
self-reported surface in the User-Agent header:

  * interactive TUI    → ``claude-cli/<ver> (external, cli)``    (subscription)
  * headless / SDK     → ``claude-cli/<ver> (external, sdk-cli)`` (Agent credits)

Every SDK-based adapter (``ClaudeCodingAgent``) drives the headless stream-json
path, so it bills against the metered Agent-SDK credit pool. Spawning the real
interactive TUI under a pty makes the *same* binary report ``(external, cli)``
(verified by ``scripts/cli_billing_probe.py``) — i.e. it bills against the
Claude subscription.

Two channels:
  * CONTROL/output via the screen — a reader thread feeds pty bytes into a
    ``pyte`` terminal emulator; we type prompts in, watch the rendered screen to
    know when a turn starts/finishes and to scrape the answer, and send ESC to
    interrupt. Each turn runs on a worker thread (see ``run_turn``).
  * DATA via the session JSONL — the CLI persists each turn's transcript to
    ``~/.claude/projects/<slug>/<session_id>.jsonl`` (the file ``--resume``
    uses). ``_SessionLog`` tails it for real per-turn token usage and the
    resumable session id.

Env caveat that gates persistence: if nemo is launched from *inside* a Claude
Code session, the inherited ``CLAUDE_CODE_CHILD_SESSION`` / ``CLAUDECODE``
markers make the spawned CLI act as a nested child and skip persisting its own
transcript. ``_build_env`` strips all ``CLAUDE_CODE_*`` / ``CLAUDECODE`` /
``AI_AGENT`` vars so the spawned CLI is a normal top-level session that persists
per-turn (and still reports the ``(external, cli)`` surface).

Design choices that make this reliable rather than a toy:
  * Worker-thread turns — the host's ``on_event`` marshals card sends to the
    main loop with a *blocking* call, so it must be invoked off the main loop.
  * Readiness detection — wait for the TUI footer instead of a blind sleep.
  * Idle-gating — never submit into a busy TUI (would queue prompts and desync).
  * Completion = "esc to interrupt" gone + screen stable.
  * Process-death detection — surface an error instead of hanging if the TUI
    exits mid-turn.

Remaining limitations (see CLAUDE_CLI_EXPERIMENT.md):
  * No USD cost — the transcript records token counts, not a per-turn cost.
  * Answer scraping is coupled to the TUI's markers (``⏺`` / ``⎿`` / ``❯`` /
    "esc to interrupt"); a major TUI reflow could break it. (Usage/resume come
    from the structured JSONL and are layout-independent.)
  * Running the official binary on the user's account, automating their own
    terminal, is a ToS gray area with account-suspension risk — opt-in only.
"""

from __future__ import annotations

import asyncio
import fcntl
import glob
import json
import logging
import os
import pty
import re
import select
import shlex
import shutil
import signal
import struct
import subprocess
import tempfile
import termios
import threading
import time
from typing import Callable

import pyte

from .channel import Channel
from .coding_agent import CodingAgent, EndpointConfig
from .db import Database
from .sessions import claude_project_slug
from .turn import (
  AnswerEvent, CompactNoticeEvent, CompactStartedEvent, DoneEvent, ErrorEvent,
  ProgressEvent, TaskDoneEvent, TaskStartedEvent, TurnEvent, canonical_usage,
)
from .types import JsonObject

log = logging.getLogger(__name__)

# Fixed terminal geometry. Tall + wide so a turn's output and the input-box
# footer stay in the live pyte buffer (plus scrollback) for scraping.
_ROWS = 120
_COLS = 160

# Markers in the claude TUI render (observed on claude-cli 2.1.175).
_USER_ECHO = "❯"          # user prompt echo AND the (empty) input box
_ASSISTANT = "⏺"          # assistant text block OR tool invocation
_TOOL_RESULT = "⎿"        # tool result (indented under a ⏺ tool call)
_THINKING = "✻"           # post-hoc "Cogitated for Ns" summary line
# Present while a turn is running; absent when idle. The single most reliable
# "is the agent working" signal the TUI gives us.
_WORKING_HINT = "esc to interrupt"
# Footer shown once the TUI is booted and ready for input (both permission
# modes render "shift+tab to cycle"); used for readiness detection.
_READY_HINT = "shift+tab to cycle"
# A ⏺ line that is a tool call: a single CapitalCase identifier then "(".
_TOOL_CALL_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*\(")

# Valid values for the interactive CLI's --effort flag.
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "max"})


# ---------------------------------------------------------------------------
# pty terminal session
# ---------------------------------------------------------------------------

class _PtyTui:
  """Owns the pty + claude subprocess + a pyte emulator fed by a reader thread.

  Thread-safety: only the reader thread feeds the pyte stream; ``snapshot`` and
  the feed both take ``_lock`` so a snapshot never races a feed.
  """

  def __init__(self, argv: list[str], cwd: str, env: dict[str, str]):
    self._argv = argv
    self._cwd = cwd
    self._env = env
    self._master = -1
    self._proc: subprocess.Popen[bytes] | None = None
    self._screen = pyte.HistoryScreen(_COLS, _ROWS, history=4000, ratio=0.5)
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

  def alive(self) -> bool:
    return self._alive

  def write(self, data: bytes) -> None:
    if self._master >= 0:
      try:
        os.write(self._master, data)
      except OSError as e:
        log.warning("pty write failed: %s", e)

  def submit(self, text: str) -> None:
    """Type a prompt then submit it. Sending the text and the CR as one write
    can be dropped by the TUI; a brief gap between them is reliable."""
    self.write(text.encode())
    time.sleep(0.3)
    self.write(b"\r")

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


# ---------------------------------------------------------------------------
# screen scraping
# ---------------------------------------------------------------------------

def _is_prompt_echo(stripped: str) -> bool:
  """True for a user prompt echo line ``❯ <text>`` (not the empty input box
  ``❯`` and not the box border)."""
  if not stripped.startswith(_USER_ECHO):
    return False
  return bool(stripped[len(_USER_ECHO):].strip())


def _region_after_echo(lines: list[str], prompt: str) -> list[str]:
  """The screen lines that belong to THIS turn: from the last echo of its
  prompt (``❯ <prompt>``) up to the NEXT prompt echo (or end). Bounding at the
  next echo keeps a mid-scrollback turn from bleeding into a later turn."""
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


def _iter_assistant_blocks(region: list[str]) -> list[tuple[str, bool]]:
  """Parse a turn region into ``(text, is_tool)`` blocks.

  A ``⏺`` block runs until the next marker. It is a tool call if its head
  matches ``Name(...)`` or it is followed by a ``⎿`` result line; otherwise it
  is prose. Tool blocks keep the ``⏺`` head text (the invocation) as their text.
  """
  blocks: list[tuple[str, bool]] = []
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
          is_tool = True
          break
        cont.append(nxt)
        j += 1
      blocks.append(("\n".join(cont).strip(), is_tool))
      i = j
    else:
      i += 1
  return blocks


def _extract_answer(lines: list[str], prompt: str) -> str:
  """The final assistant prose answer: the last ``⏺`` block in this turn's
  region that is NOT a tool call."""
  for text, is_tool in reversed(_iter_assistant_blocks(_region_after_echo(lines, prompt))):
    if not is_tool and text:
      return text
  return ""


def _new_tool_summaries(lines: list[str], prompt: str, seen: set[str]) -> list[str]:
  """Tool invocations (``⏺ Name(args)``) in this turn's region not yet seen.

  Scoped to the current turn so stale ``⏺`` lines from earlier turns (still in
  scrollback) aren't re-emitted as progress."""
  out: list[str] = []
  for text, is_tool in _iter_assistant_blocks(_region_after_echo(lines, prompt)):
    head = text.splitlines()[0] if text else ""
    if is_tool and head and head not in seen:
      seen.add(head)
      out.append(head)
  return out


# ---------------------------------------------------------------------------
# session JSONL (token usage + resumable session id)
# ---------------------------------------------------------------------------
#
# The interactive CLI persists each turn's transcript to
# ``<config>/projects/<cwd-slug>/<session_id>.jsonl`` — the SAME file ``--resume``
# uses. (NB: it only does so when NOT inherited as a nested Claude-Code child
# session; ``_build_env`` strips the parent's CLAUDE_CODE_* markers so a freshly
# spawned CLI persists like any normal top-level session.) We tail it after each
# turn for real per-turn token usage, and capture the session id so a daemon
# restart can resume the conversation with ``claude --resume <id>``.


def _config_dir() -> str:
  return os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")


class _SessionLog:
  """Locates and tails the interactive session's JSONL for usage + session id.

  Binds lazily: the file only appears after the first prompt. The baseline of
  pre-existing files is snapshotted at construction (before spawn) so the new
  file this session creates is identifiable; a resumed session binds directly to
  ``<resume_id>.jsonl`` (seeking to its end so only new turns are read).
  """

  def __init__(self, project_dir: str, resume_id: str = ""):
    self._dir = os.path.join(
      _config_dir(), "projects", claude_project_slug(project_dir))
    self._baseline: set[str] = set(self._list())
    self._resume_id = resume_id
    self._path = ""
    self._pos = 0
    self._buf = ""
    self._session_id = ""

  def _list(self) -> list[str]:
    return glob.glob(os.path.join(self._dir, "*.jsonl"))

  def _bind(self) -> bool:
    if self._path:
      return True
    if self._resume_id:
      cand = os.path.join(self._dir, f"{self._resume_id}.jsonl")
      if os.path.exists(cand):
        self._path, self._session_id = cand, self._resume_id
        self._pos = os.path.getsize(cand)
        return True
      return False
    new = [f for f in self._list() if f not in self._baseline]
    if new:
      self._path = max(new, key=os.path.getmtime)
      self._session_id = os.path.splitext(os.path.basename(self._path))[0]
      self._pos = 0
      return True
    return False

  @property
  def session_id(self) -> str:
    if not self._path:
      self._bind()
    return self._session_id

  def read_new(self) -> list[JsonObject]:
    """JSON rows appended since the last read; advances the offset. Binds
    lazily, tolerates partial trailing lines."""
    if not self._path and not self._bind():
      return []
    if not os.path.exists(self._path):
      return []
    try:
      with open(self._path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(self._pos)
        chunk = f.read()
        self._pos = f.tell()
    except OSError as e:
      log.warning("session log read failed: %s", e)
      return []
    self._buf += chunk
    *complete, self._buf = self._buf.split("\n")
    rows: list[JsonObject] = []
    for line in complete:
      line = line.strip()
      if not line:
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
    return rows


def _sum_turn_usage(rows: list[JsonObject]) -> JsonObject:
  """Sum per-message token usage across a turn's assistant rows → canonical
  usage. Empty when no usage seen (card omits the token line)."""
  inp = cr = cc = out = 0
  seen = False
  for row in rows:
    if row.get("type") != "assistant":
      continue
    msg = row.get("message")
    if not isinstance(msg, dict):
      continue
    usage = msg.get("usage")
    if not isinstance(usage, dict):
      continue
    def _i(key: str) -> int:
      v = usage.get(key)
      return max(0, int(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
    inp += _i("input_tokens")
    cr += _i("cache_read_input_tokens")
    cc += _i("cache_creation_input_tokens")
    out += _i("output_tokens")
    seen = True
  if not seen:
    return {}
  return canonical_usage(
    input_tokens=inp, cache_read=cr, cache_creation=cc, output_tokens=out)


# ---------------------------------------------------------------------------
# structured event mapping (jsonl rows + hook events → TurnEvents)
# ---------------------------------------------------------------------------
#
# The transcript jsonl is the primary structured channel: assistant text /
# thinking / tool_use, per-message usage, and ``system`` rows (turn_duration =
# turn end, api_error, compact_boundary). Hooks (Phase 2, via --settings) add
# the realtime/control signals the jsonl lacks: PreCompact (BEFORE compaction)
# and Stop (authoritative completion), plus SubagentStop / Notification. Both
# feed the SAME TurnEvent stream the SDK adapter produces, so the host renders
# claude-cli turns with the same fidelity.


def _new_turn_state() -> dict[str, object]:
  return {
    "progress_started": False,
    "answer_seen": False,
    "turn_done": False,        # set by jsonl turn_duration OR hook Stop
    "error": "",               # set by api_error
    "usage": {"input_tokens": 0, "cache_read": 0,
              "cache_creation": 0, "output_tokens": 0},
  }


def _accumulate_usage(state: dict[str, object], usage: object) -> None:
  if not isinstance(usage, dict):
    return
  acc = state["usage"]
  assert isinstance(acc, dict)
  for src, dst in (
    ("input_tokens", "input_tokens"),
    ("cache_read_input_tokens", "cache_read"),
    ("cache_creation_input_tokens", "cache_creation"),
    ("output_tokens", "output_tokens"),
  ):
    v = usage.get(src)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      acc[dst] = int(acc.get(dst, 0)) + max(0, int(v))


def _emit_jsonl_events(
  rows: list[JsonObject],
  on_event: Callable[[TurnEvent], None],
  state: dict[str, object],
) -> None:
  """Map transcript jsonl rows to ordered TurnEvents + accumulate usage."""
  from .cards import tool_use_summary

  for row in rows:
    rtype = row.get("type")
    if rtype == "assistant":
      msg = row.get("message")
      if not isinstance(msg, dict):
        continue
      content = msg.get("content")
      if isinstance(content, list):
        for b in content:
          if not isinstance(b, dict):
            continue
          bt = b.get("type")
          if bt == "thinking":
            think = b.get("thinking") or ""
            if think:
              on_event(ProgressEvent(kind="thinking", summary=think,
                                     first=not state["progress_started"]))
              state["progress_started"] = True
          elif bt == "text":
            text = b.get("text") or ""
            if text:
              on_event(AnswerEvent(text=text))
              state["answer_seen"] = True
          elif bt == "tool_use":
            name = b.get("name") or ""
            on_event(ProgressEvent(
              kind="tool", summary=tool_use_summary(name, b.get("input") or {}),
              first=not state["progress_started"]))
            state["progress_started"] = True
            # The Task/Agent tool spawns a sub-agent; surface its start.
            if name in ("Task", "Agent"):
              tid = str(b.get("id") or "")
              log.info("claude-cli subagent start: tool=%s id=%s", name, tid[:12])
              on_event(TaskStartedEvent(task_id=tid))
      _accumulate_usage(state, msg.get("usage"))
    elif rtype == "system":
      sub = row.get("subtype")
      if sub == "turn_duration":
        state["turn_done"] = True
      elif sub == "api_error":
        err = row.get("error")
        text = ""
        code = ""
        if isinstance(err, dict):
          text = str(err.get("formatted") or err.get("message") or "")
          conn = err.get("connection")
          if isinstance(conn, dict):
            code = str(conn.get("code") or "")
        # Forensic breadcrumb: error classification + auto-reconnect are not yet
        # implemented for claude-cli (see CLAUDE_CLI_EXPERIMENT.md), so log the
        # raw error so a wedged/failed turn is diagnosable from the daemon log.
        log.warning("claude-cli api_error: code=%s msg=%s", code or "?", text[:200])
        on_event(ErrorEvent(message=text or "claude-cli: API error"))
        state["error"] = text or "API error"
      elif sub == "compact_boundary":
        meta = row.get("compact_metadata")
        meta = meta if isinstance(meta, dict) else {}
        log.info("claude-cli compaction (post): trigger=%s pre=%s post=%s dur=%sms",
                 meta.get("trigger"), meta.get("pre_tokens"),
                 meta.get("post_tokens"), meta.get("duration_ms"))
        on_event(CompactNoticeEvent(
          trigger=str(meta.get("trigger") or ""),
          pre_tokens=int(meta.get("pre_tokens") or 0),
          post_tokens=int(meta.get("post_tokens") or 0),
          duration_ms=int(meta.get("duration_ms") or 0)))
      elif sub in ("microcompact_boundary", "microcompact"):
        log.info("claude-cli microcompaction (suppressed)")


def _emit_hook_events(
  rows: list[JsonObject],
  on_event: Callable[[TurnEvent], None],
  state: dict[str, object],
) -> None:
  """Map hook NDJSON lines (Phase 2) to TurnEvents. Keyed by hook_event_name."""
  for row in rows:
    name = row.get("hook_event_name") or ""
    if name == "Stop":
      state["turn_done"] = True
    elif name == "PreCompact":
      log.info("claude-cli PreCompact hook (realtime compaction): trigger=%s",
               row.get("trigger"))
      on_event(CompactStartedEvent(trigger=str(row.get("trigger") or "")))
    elif name == "SubagentStop":
      log.info("claude-cli SubagentStop hook")
      on_event(TaskDoneEvent(task_id="", status="done"))
    elif name == "Notification":
      note = str(row.get("message") or "").strip()
      # Notifications include permission/idle prompts — e.g. an AskUserQuestion
      # picker (disabled, but log defensively) or a blocking prompt. Worth a
      # breadcrumb since these can correlate with a stalled turn.
      log.info("claude-cli Notification hook: %s", note[:200])
      if note:
        on_event(ProgressEvent(kind="reasoning", summary=f"ℹ️ {note}"))


def _tail_ndjson(path: str, pos: int, buf: str) -> tuple[list[JsonObject], int, str]:
  """Read complete JSON lines appended to ``path`` since byte offset ``pos``.
  Returns (rows, new_pos, new_buf). Tolerates partial trailing lines."""
  if not path or not os.path.exists(path):
    return [], pos, buf
  try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
      f.seek(pos)
      chunk = f.read()
      pos = f.tell()
  except OSError as e:
    log.warning("ndjson tail failed (%s): %s", path, e)
    return [], pos, buf
  buf += chunk
  *complete, buf = buf.split("\n")
  rows: list[JsonObject] = []
  for line in complete:
    line = line.strip()
    if not line:
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return rows, pos, buf


class _HookStream:
  """Per-session hook IPC. Writes a settings JSON whose hooks append their
  stdin payload to an NDJSON file, injected into the CLI via ``--settings``
  (a trusted source — project .claude/settings.json hooks are trust-gated and
  won't fire). The adapter tails the NDJSON for realtime control signals.

  Isolation: the settings file carries ONLY nemo's hooks and is passed
  explicitly; it does not touch the user's global/project settings.
  """

  # Only the signals the jsonl can't give (or gives only post-hoc): realtime
  # compaction, authoritative completion, sub-agent end, idle/permission.
  _EVENTS = ("PreCompact", "Stop", "SubagentStop", "Notification")

  def __init__(self, dirpath: str):
    self._events_path = os.path.join(dirpath, "nemo_hooks.ndjson")
    self._settings_path = os.path.join(dirpath, "nemo_settings.json")
    self._pos = 0
    self._buf = ""

  @property
  def settings_path(self) -> str:
    return self._settings_path

  def write_settings(self) -> None:
    open(self._events_path, "w").close()
    q = shlex.quote(self._events_path)
    # Append the hook's stdin JSON (one object) + a newline, atomically enough
    # for the low-frequency control events we register.
    cmd = f"cat >> {q}; printf '\\n' >> {q}"
    hooks = {ev: [{"hooks": [{"type": "command", "command": cmd}]}]
             for ev in self._EVENTS}
    with open(self._settings_path, "w", encoding="utf-8") as f:
      json.dump({"hooks": hooks}, f)

  def read_new(self) -> list[JsonObject]:
    rows, self._pos, self._buf = _tail_ndjson(
      self._events_path, self._pos, self._buf)
    return rows


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

class ClaudeCliCodingAgent(CodingAgent):
  """Drives the interactive ``claude`` TUI over a pty (subscription billing).

  See the module docstring for the rationale, design, and limitations.
  """

  # Tunables (seconds).
  _BOOT_TIMEOUT = 30.0      # max wait for the TUI to become ready for input
  _START_TIMEOUT = 25.0     # max wait for a turn to BEGIN after submit
  _TURN_TIMEOUT = 1800.0    # hard ceiling on a single turn
  _SETTLE = 2.0             # screen idle + stable this long ⇒ turn done (fallback)
  _POLL = 0.3               # poll cadence
  _JSONL_GRACE = 6.0        # after completion, drain trailing structured rows

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
    self._log: _SessionLog | None = None
    self._session_id = ""
    self._booted = False
    # Phase 2 hook IPC: a per-session settings file injected via --settings and
    # the NDJSON file its hooks append to. Populated in _spawn.
    self._settings_path = ""
    self._hooks: _HookStream | None = None
    self._hookdir = ""

  def set_effort(self, effort: str) -> None:
    # The interactive CLI accepts a session-wide --effort flag; store the
    # validated level (host calls reset() after, which respawns with it).
    self._effort = effort if effort in _EFFORT_LEVELS else ""

  def set_endpoint(self, endpoint: EndpointConfig) -> None:
    self._endpoint = endpoint

  def _build_argv(self, resume: str = "") -> list[str]:
    claude = shutil.which("claude") or "claude"
    argv = [claude]
    # Unattended daemon: bypass interactive permission panels.
    if self._permission_mode == "bypassPermissions":
      argv.append("--dangerously-skip-permissions")
    else:
      argv += ["--permission-mode", self._permission_mode or "acceptEdits"]
    # Disable the AskUserQuestion tool. In the interactive TUI it renders an
    # on-screen arrow-key picker that waits for *terminal* keyboard input — but
    # the user is on Lark, not the terminal, so the question is invisible and the
    # session wedges on the picker. With the tool disabled the model just asks
    # the question in plain text; the user answers as the next message — natural
    # for a chat channel, and no screen-picker bridging needed. (The SDK adapter
    # instead bridges it to a Lark card via can_use_tool, which the pty path has
    # no equivalent for.)
    argv += ["--disallowed-tools", "AskUserQuestion"]
    if self._model:
      argv += ["--model", self._model]
    if self._effort:
      # The interactive CLI takes a session-wide --effort flag (low/medium/
      # high/max), so we set it at spawn; /effort changes flow through reset().
      argv += ["--effort", self._effort]
    if self._settings_path:
      # Trusted hook injection (Phase 2). Project .claude/settings.json hooks are
      # trust-gated and won't fire, but --settings is an explicit trusted source.
      argv += ["--settings", self._settings_path]
    if resume:
      # Resume the prior conversation's transcript (daemon restart / model
      # switch). The session jsonl is persisted per-turn, so context is intact.
      argv += ["--resume", resume]
    return argv

  def _build_env(self) -> dict[str, str]:
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    # Strip the parent Claude Code session's identity markers. If nemo itself is
    # ever launched from *inside* a Claude Code session, these are inherited and
    # the spawned ``claude`` treats itself as a nested CHILD session — which
    # makes it NOT persist its own transcript jsonl (so --resume has nothing to
    # resume). Removing them makes the spawned CLI a normal top-level
    # interactive session that persists like any other. Also keeps the
    # ``(external, cli)`` surface (the CLI re-derives its own entrypoint=cli).
    for key in list(env):
      if key.startswith("CLAUDE_CODE") or key in ("CLAUDECODE", "AI_AGENT"):
        env.pop(key, None)
    if self._endpoint.base_url:
      env["ANTHROPIC_BASE_URL"] = self._endpoint.base_url
    if self._endpoint.api_key:
      env["ANTHROPIC_API_KEY"] = self._endpoint.api_key
    env["NEMO_CHAT_ID"] = self._chat_id
    return env

  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    self._project_dir = project_dir
    self._model = model
    self._session_id = resume
    await self._spawn(resume)

  async def _spawn(self, resume: str = "") -> None:
    def _do() -> tuple[_PtyTui, _SessionLog]:
      # Snapshot existing transcripts BEFORE spawn so the new one is identifiable.
      sess = _SessionLog(self._project_dir, resume_id=resume)
      # Per-session hook IPC: write an isolated settings file (injected via
      # --settings in _build_argv) whose hooks append realtime control signals
      # to an NDJSON file we tail.
      self._hookdir = tempfile.mkdtemp(prefix="nemo_clicli_")
      hooks = _HookStream(self._hookdir)
      hooks.write_settings()
      self._hooks = hooks
      self._settings_path = hooks.settings_path
      tui = _PtyTui(self._build_argv(resume), self._project_dir, self._build_env())
      tui.spawn()
      return tui, sess
    self._tui, self._log = await asyncio.to_thread(_do)
    self._booted = False
    if resume:
      self._session_id = self._log.session_id or resume

  def _wait_ready(self, tui: _PtyTui) -> None:
    """Block until the TUI footer shows it's ready for input, then nudge past
    any first-run trust/theme dialog. Replaces a blind fixed sleep."""
    deadline = time.monotonic() + self._BOOT_TIMEOUT
    while time.monotonic() < deadline:
      if not tui.alive():
        return
      if any(_READY_HINT in ln for ln in tui.snapshot()):
        break
      time.sleep(self._POLL)
    # A first run in an untrusted dir can park on a trust/theme prompt; a CR
    # accepts the default. Harmless once already trusted.
    tui.write(b"\r")
    time.sleep(0.5)
    self._booted = True

  def _wait_idle(self, tui: _PtyTui, timeout: float) -> bool:
    """Block until the TUI is not working and its screen has been stable for a
    beat — ready for a new prompt. Prevents submitting into a busy TUI (which
    queues prompts and desyncs answers from turns)."""
    last = ""
    stable_since = time.monotonic()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not tui.alive():
        return False
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

    Critical: the host's ``on_event`` marshals card sends to the main loop with
    a *blocking* ``run_coroutine_threadsafe(...).result()`` — it is written to
    be invoked from a worker thread (the SDK adapter calls it from ``SDKThread``).
    Calling it from this coroutine (the main loop) would deadlock. So the whole
    turn runs in a thread, like the SDK path; the main loop stays free to take
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
    if not tui.alive():
      log.warning("claude-cli: TUI process not running at turn start "
                  "(crashed/exited) — turn cannot run")
      on_event(ErrorEvent(message="claude-cli: TUI process is not running"))
      on_event(DoneEvent(cost=0.0, usage={}))
      return 0.0, {}

    if not self._booted:
      self._wait_ready(tui)

    # Never submit into a busy TUI — wait for any prior turn to fully drain.
    log.info("claude-cli run_turn: waiting for idle TUI (prompt=%d chars)", len(prompt))
    self._wait_idle(tui, timeout=self._TURN_TIMEOUT)

    # Drain pre-turn rows from both structured channels so we only emit THIS
    # turn's events.
    if self._log is not None:
      self._log.read_new()
    if self._hooks is not None:
      self._hooks.read_new()

    log.info("claude-cli run_turn: submitting prompt")
    tui.submit(prompt)

    state = _new_turn_state()
    start = time.monotonic()
    began = False
    last_display = ""
    last_change = start
    needle = prompt.strip()[:48]
    timed_out = False
    used_screen_completion = False

    def _echo_present(lines: list[str]) -> bool:
      return any(_is_prompt_echo(ln.strip()) and needle in ln for ln in lines)

    def _drain() -> None:
      # Structured channels are the event source (layout-independent).
      if self._log is not None:
        _emit_jsonl_events(self._log.read_new(), on_event, state)
      if self._hooks is not None:
        _emit_hook_events(self._hooks.read_new(), on_event, state)

    while True:
      time.sleep(self._POLL)
      now = time.monotonic()
      if not tui.alive():
        log.warning("claude-cli: TUI exited mid-turn (%.0fs in) — surfacing error",
                    now - start)
        on_event(ErrorEvent(message="claude-cli: TUI exited mid-turn"))
        on_event(DoneEvent(cost=0.0, usage={}))
        return 0.0, {}
      _drain()
      lines = tui.snapshot()
      working = any(_WORKING_HINT in ln for ln in lines)

      if working or state["progress_started"] or state["answer_seen"]:
        began = True
      display = "\n".join(lines[-_ROWS:])
      if display != last_display:
        last_display = display
        last_change = now

      # Authoritative completion: jsonl turn_duration OR hook Stop.
      if state["turn_done"]:
        break
      if not began:
        if _echo_present(lines) and any(
            ln.strip().startswith(_ASSISTANT) for ln in lines):
          began = True
        elif now - start > self._START_TIMEOUT:
          timed_out = not _echo_present(lines)
          began = True
        continue
      # Fallback completion: screen idle + stable (marker-drift canary below).
      if not working and (now - last_change) >= self._SETTLE:
        used_screen_completion = True
        break
      if now - start > self._TURN_TIMEOUT:
        timed_out = True
        break

    # Grace: the final text/usage rows can lag the completion signal by a beat.
    grace_end = time.monotonic() + self._JSONL_GRACE
    while time.monotonic() < grace_end:
      _drain()
      if state["answer_seen"]:
        time.sleep(0.4)
        _drain()
        break
      time.sleep(0.3)

    if self._log is not None and self._log.session_id:
      self._session_id = self._log.session_id

    acc = state["usage"]
    assert isinstance(acc, dict)
    usage = canonical_usage(
      input_tokens=int(acc["input_tokens"]), cache_read=int(acc["cache_read"]),
      cache_creation=int(acc["cache_creation"]), output_tokens=int(acc["output_tokens"]),
    ) if any(acc.values()) else {}

    if used_screen_completion:
      # Marker-drift canary: completion fell back to screen scraping. With hooks
      # the Stop hook should always fire, so a fallback there signals real drift.
      if self._hooks is not None:
        log.warning("claude-cli: completed via SCREEN fallback despite hooks "
                    "(no Stop/turn_duration) — possible TUI/hook drift")
      else:
        log.info("claude-cli: completed via screen idle (no turn_duration row)")

    log.info("claude-cli run_turn: done (began=%s timed_out=%s answer=%s "
             "out_tokens=%s session=%s screen_fallback=%s %.0fs)", began,
             timed_out, state["answer_seen"], acc["output_tokens"],
             (self._session_id[:8] or "?"), used_screen_completion,
             time.monotonic() - start)

    # Fallback answer: structured channel produced no text — scrape the screen
    # so the turn still yields a reply.
    if not state["answer_seen"]:
      scraped = _extract_answer(tui.snapshot(), prompt)
      if scraped:
        on_event(AnswerEvent(text=scraped))
      elif timed_out and not state["error"]:
        on_event(ErrorEvent(message="claude-cli turn timed out (no answer)"))

    on_event(DoneEvent(cost=0.0, usage=usage, session_id=self._session_id))
    return 0.0, usage

  async def interrupt(self) -> None:
    # ESC interrupts the current turn without killing the session (matches the
    # CLI's Escape semantics — context preserved).
    if self._tui is not None:
      self._tui.write(b"\x1b")

  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    # Respawn, resuming the persisted transcript so a model switch / reconnect /
    # daemon restart keeps the conversation (falls back to a fresh session if the
    # resume id can't be materialised — see start()/_spawn).
    await self.stop()
    self._project_dir = project_dir
    self._model = model
    resume_id = resume or self._session_id
    self._session_id = resume_id
    await self._spawn(resume_id)

  async def stop(self) -> None:
    self._log = None
    self._hooks = None
    self._settings_path = ""
    hookdir = self._hookdir
    self._hookdir = ""
    if self._tui is not None:
      tui = self._tui
      self._tui = None
      await asyncio.to_thread(tui.close)
    if hookdir:
      shutil.rmtree(hookdir, ignore_errors=True)
