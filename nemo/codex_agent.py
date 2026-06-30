"""Concrete CodingAgent implementation backed by the Codex CLI JSON stream."""

from __future__ import annotations

import asyncio
import datetime
import glob
import json
import logging
import os
import time
import uuid
from pathlib import Path
import shutil
from asyncio.subprocess import Process
from typing import Callable

from .channel import Channel
from .coding_agent import CodingAgent, EndpointConfig
from .db import Database
from .turn import (
  AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent, TurnEvent, canonical_usage,
)
from .types import JsonObject

log = logging.getLogger(__name__)

# Sidecar lives inside the nemo package so it's shipped via package-data in
# the wheel. Keep the source files (package.json, run_turn.mjs,
# package-lock.json); node_modules/ is installed on first use, not packaged.
_SIDE_CAR_DIR = Path(__file__).resolve().parent / "codex_sidecar"
_SIDE_CAR_SCRIPT = _SIDE_CAR_DIR / "run_turn.mjs"
_SIDE_CAR_PACKAGE = _SIDE_CAR_DIR / "package.json"
_SIDE_CAR_NODE_MODULES = _SIDE_CAR_DIR / "node_modules"

# Valid values for Codex SDK's ThreadOptions.modelReasoningEffort.
_CODEX_EFFORT_LEVELS = frozenset({"low", "medium", "high"})
# Shared knob accepts "max" (Claude's top tier). Codex tops out at "high",
# so we clamp instead of rejecting.
_CLAUDE_TO_CODEX_EFFORT = {"max": "high"}

# Codex persists each thread as a rollout jsonl under here, keyed by thread id.
_CODEX_SESSIONS = Path(os.path.expanduser("~/.codex/sessions"))
_CODEX_CONTEXT_WARNING_TOKENS = 180_000
_CODEX_STDERR_TAIL_LINES = 20
_CODEX_BACKEND_RETRY_ATTEMPTS = 3
_CODEX_BACKEND_RETRY_DELAYS = (2.0, 5.0)
_CODEX_COMPACT_FAILURE_MARKERS = (
  "remote compact",
  "responses/compact",
  "pre-sampling compact",
  "failed to run pre-sampling compact",
  "error running remote compact task",
)
_CODEX_BACKEND_FAILURE_MARKERS = (
  "chatgpt.com/backend-api/codex/responses",
  "failed to connect to websocket",
  "tls handshake eof",
  "error sending request for url",
)

# Prepended to a read-only fork's first turn. The Codex sandbox physically
# blocks writes (sandboxMode=read-only), but telling the model up front saves
# it from burning a turn discovering it can't write.
_CODEX_FORK_DIRECTIVE = (
  "FORK MODE — read-only branch. You are a forked side-conversation that "
  "branched from the main session's context; it is ephemeral and nothing you "
  "do is written back to the main conversation. The project at {project} is "
  "STRICTLY READ-ONLY: you may read files, grep, and run read-only commands, "
  "but you CANNOT modify any file (an OS sandbox blocks all writes — write "
  "attempts will fail). Investigate and answer; do not attempt edits."
)


def _uuid7() -> str:
  """Generate a UUIDv7 (time-ordered) — matches Codex's rollout id format so a
  seeded copy lands in today's date dir and is resolvable by thread id."""
  ms = int(time.time() * 1000)
  b = bytearray(ms.to_bytes(6, "big") + os.urandom(10))
  b[6] = (b[6] & 0x0F) | 0x70  # version 7
  b[8] = (b[8] & 0x3F) | 0x80  # variant
  return str(uuid.UUID(bytes=bytes(b)))


def _find_codex_rollout(thread_id: str) -> str:
  """Locate the rollout jsonl for a Codex thread id (scans the dated dirs)."""
  if not thread_id:
    return ""
  hits = glob.glob(str(_CODEX_SESSIONS / "**" / f"*{thread_id}*.jsonl"),
                   recursive=True)
  return hits[0] if hits else ""


def _find_total_token_usage(obj: object) -> JsonObject | None:
  """Return the nested ``total_token_usage`` dict from a rollout record."""
  if isinstance(obj, dict):
    found = obj.get("total_token_usage")
    if isinstance(found, dict):
      return found
    for value in obj.values():
      hit = _find_total_token_usage(value)
      if hit is not None:
        return hit
  elif isinstance(obj, list):
    for value in obj:
      hit = _find_total_token_usage(value)
      if hit is not None:
        return hit
  return None


def _read_codex_cumulative(thread_id: str) -> dict[str, int]:
  """Seed the per-turn differencing baseline after a resume/restart.

  Codex's ``turn.completed.usage`` is SESSION-CUMULATIVE (it maps to the
  rollout's ``total_token_usage``), so per-turn counts come from differencing
  successive totals on the live ``CodexCodingAgent`` instance. When a fresh
  daemon resumes an existing thread that in-memory baseline is gone — recover
  it from the last ``token_count`` record in the rollout so the first
  post-resume turn isn't reported as the entire session so far. Returns the
  cumulative {input_tokens, cached_input_tokens, output_tokens} or {} when the
  rollout has no token usage yet (fresh thread).
  """
  path = _find_codex_rollout(thread_id)
  if not path:
    return {}
  baseline: dict[str, int] = {}
  try:
    for line in Path(path).read_text().splitlines():
      if "total_token_usage" not in line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        continue
      total = _find_total_token_usage(record)
      if total is None:
        continue
      baseline = {
        "input_tokens": int(total.get("input_tokens", 0) or 0),
        "cached_input_tokens": int(total.get("cached_input_tokens", 0) or 0),
        "output_tokens": int(total.get("output_tokens", 0) or 0),
      }  # keep the LAST record's totals
  except OSError as e:
    log.warning("codex usage baseline read failed (%s) — baseline reset", e)
    return {}
  return baseline


def _seed_fork_rollout(parent_thread_id: str) -> tuple[str, str]:
  """Copy a parent rollout to a new thread id so a fork can resume it without
  mutating the parent (Codex resume APPENDS to the resumed file, so the fork
  MUST own a private copy). Returns (new_thread_id, copied_path), or ("", "")
  if the parent rollout isn't on disk (then the fork starts fresh).

  This is the manual equivalent of Claude's fork_session=True: rewrite the
  copy's session_meta.id to the new id and place it where Codex resolves it.
  """
  src = _find_codex_rollout(parent_thread_id)
  if not src:
    return "", ""
  try:
    lines = Path(src).read_text().splitlines()
    if not lines:
      return "", ""
    newid = _uuid7()
    meta = json.loads(lines[0])
    if isinstance(meta.get("payload"), dict):
      meta["payload"]["id"] = newid
    lines[0] = json.dumps(meta)
    now = datetime.datetime.now()
    ddir = _CODEX_SESSIONS / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}"
    ddir.mkdir(parents=True, exist_ok=True)
    dst = ddir / f"rollout-{now:%Y-%m-%dT%H-%M-%S}-{newid}.jsonl"
    dst.write_text("\n".join(lines) + "\n")
    return newid, str(dst)
  except (OSError, ValueError, json.JSONDecodeError) as e:
    log.warning("codex fork rollout seed failed (%s) — fork starts fresh", e)
    return "", ""


def _short(text: str, limit: int) -> str:
  """Truncate ``text`` to ``limit`` chars with an ellipsis when oversized."""
  if len(text) <= limit:
    return text
  return text[: limit - 3] + "..."


def _codex_cumulative_total(usage: dict[str, int]) -> int:
  """Return Codex's best known cumulative context footprint."""
  return max(0, usage.get("input_tokens", 0)) + max(0, usage.get("output_tokens", 0))


def _is_codex_compact_failure(message: str) -> bool:
  lower = message.lower()
  return any(marker in lower for marker in _CODEX_COMPACT_FAILURE_MARKERS)


def _codex_recovery_message(message: str) -> str:
  if not _is_codex_compact_failure(message):
    return message
  return (
    "Codex session compaction failed. The current Codex thread is large enough "
    "that Codex must compact it before the next response, but the remote "
    "compact request failed. Nemo keeps the session memory intact; retrying "
    "may succeed if this was a transient backend/network failure.\n\n"
    "If it keeps failing, run `/clear` only when you explicitly want to start "
    "a fresh session, then use `/session recall` to bring back a summary of "
    "the previous thread.\n\n"
    f"Original error: {message}"
  )


def _is_codex_backend_failure(message: str) -> bool:
  lower = message.lower()
  return any(marker in lower for marker in _CODEX_BACKEND_FAILURE_MARKERS)


class _CodexTurnFailure(RuntimeError):
  """Internal marker carrying retry metadata for one sidecar attempt."""

  def __init__(self, message: str, *, retryable: bool, progressed: bool):
    super().__init__(message)
    self.retryable = retryable
    self.progressed = progressed


class CodexCodingAgent(CodingAgent):
  """CodingAgent adapter for the local Codex SDK sidecar runtime."""

  def __init__(
    self,
    credentials: dict[str, str],
    chat_id: str,
    db: Database,
    channel: Channel,
    permission_mode: str = "bypassPermissions",
    system_prompt: str = "",
    endpoint: EndpointConfig | None = None,
    read_only: bool = False,
  ):
    del credentials, chat_id, db, channel
    self._permission_mode = permission_mode
    self._system_prompt = system_prompt
    self._endpoint = endpoint or EndpointConfig()
    self._project_dir = ""
    self._model = ""
    self._session_id = ""
    self._effort = ""
    self._proc: Process | None = None
    self._interrupted = False
    # Codex reports usage as a SESSION-CUMULATIVE total each turn, so we keep
    # the previous cumulative here and emit the per-turn delta (seeded from the
    # rollout on resume — see _read_codex_cumulative).
    self._cum_usage: dict[str, int] = {}
    # Read-only fork mode: runs the sidecar with sandboxMode=read-only so it
    # cannot modify the project; resumes a private copy of the parent rollout
    # (see fork()). _fork_rollout_path is that throwaway copy, removed on stop.
    self._read_only = read_only
    self._fork_note_pending = read_only
    self._fork_rollout_path = ""

  def set_effort(self, effort: str) -> None:
    mapped = _CLAUDE_TO_CODEX_EFFORT.get(effort, effort)
    self._effort = mapped if mapped in _CODEX_EFFORT_LEVELS else ""

  def set_endpoint(self, endpoint: EndpointConfig) -> None:
    self._endpoint = endpoint

  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    self._ensure_runtime()
    self._project_dir = project_dir
    self._model = model
    self._session_id = resume
    # Resuming an existing thread: recover the cumulative baseline so the first
    # turn's per-turn delta is correct. A fresh thread has no rollout yet → {}.
    self._cum_usage = _read_codex_cumulative(resume) if resume else {}

  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    last_error = ""
    visible_progress = False
    for attempt in range(1, _CODEX_BACKEND_RETRY_ATTEMPTS + 1):
      if attempt > 1:
        await self._emit_event(
          on_event,
          ProgressEvent(
            kind="reasoning",
            summary=(
              "Codex backend connection dropped before the turn started; "
              f"retrying ({attempt}/{_CODEX_BACKEND_RETRY_ATTEMPTS})..."
            ),
            first=not visible_progress,
          ),
        )
        visible_progress = True
      try:
        return await self._run_sidecar_once(prompt, on_event)
      except _CodexTurnFailure as exc:
        last_error = str(exc)
        if exc.progressed:
          visible_progress = True
        can_retry = (
          exc.retryable
          and not exc.progressed
          and not self._interrupted
          and attempt < _CODEX_BACKEND_RETRY_ATTEMPTS
        )
        if not can_retry:
          await self._emit_event(on_event, ErrorEvent(message=last_error))
          raise RuntimeError(last_error) from exc
        delay = _CODEX_BACKEND_RETRY_DELAYS[
          min(attempt - 1, len(_CODEX_BACKEND_RETRY_DELAYS) - 1)]
        log.warning(
          "codex backend failed before progress (attempt %d/%d); retrying in %.1fs: %s",
          attempt, _CODEX_BACKEND_RETRY_ATTEMPTS, delay, _short(last_error, 300))
        await asyncio.sleep(delay)
    # Unreachable in normal control flow, but keeps the return type honest.
    await self._emit_event(on_event, ErrorEvent(message=last_error))
    raise RuntimeError(last_error)

  async def _run_sidecar_once(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    self._ensure_runtime()
    self._interrupted = False

    args = self._build_command()
    log.info("codex sdk start model=%s resume=%s", self._model, bool(self._session_id))
    proc = await asyncio.create_subprocess_exec(
      *args,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      cwd=self._project_dir or None,
      env=self._build_env(),
      # A single JSON event from the sidecar (large reasoning block or
      # agent_message) can exceed asyncio's 64 KB default line buffer and
      # make readline() raise "Separator is found, but chunk is longer
      # than limit". Match the Claude SDK cap at 16 MB.
      limit=16 * 1024 * 1024,
    )
    self._proc = proc

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_tail: list[str] = []
    stderr_task = asyncio.create_task(self._log_stderr(proc.stderr, stderr_tail))
    proc.stdin.write(self._prepare_prompt(prompt).encode())
    await proc.stdin.drain()
    proc.stdin.close()

    usage: JsonObject = {}
    progress_started = False
    failure: str | None = None

    try:
      while True:
        raw = await proc.stdout.readline()
        if not raw:
          break
        line = raw.decode(errors="replace").strip()
        if not line:
          continue
        event = self._parse_event(line)
        if event is None:
          continue

        event_type = str(event.get("type", ""))
        if event_type == "thread.started":
          self._session_id = str(event.get("thread_id", "") or "")
          continue
        if event_type == "turn.completed":
          usage = self._per_turn_usage(self._coerce_json_object(event.get("usage")))
          continue
        if event_type == "turn.failed":
          error = self._coerce_json_object(event.get("error"))
          failure = _codex_recovery_message(str(error.get("message", "Codex turn failed")))
          break
        if event_type != "item.completed":
          continue

        item = self._coerce_json_object(event.get("item"))
        item_type = str(item.get("type", ""))
        if item_type == "agent_message":
          text = str(item.get("text", "") or "")
          if text:
            await self._emit_event(on_event, AnswerEvent(text=text))
        else:
          summary = self._item_summary(item)
          if not summary:
            continue
          kind = "reasoning" if item_type == "reasoning" else "tool"
          is_first = not progress_started
          progress_started = True
          await self._emit_event(on_event, ProgressEvent(kind=kind, summary=summary, first=is_first))

      rc = await proc.wait()
      try:
        await asyncio.wait_for(asyncio.shield(stderr_task), timeout=0.2)
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
      if failure is None and rc != 0 and not self._interrupted:
        failure = f"Codex sidecar exited with status {rc}"
        if stderr_tail:
          stderr_text = "\n".join(stderr_tail)
          recovered = _codex_recovery_message(stderr_text)
          failure = recovered if recovered != stderr_text else stderr_text
      if failure is not None:
        raise _CodexTurnFailure(
          failure,
          retryable=_is_codex_backend_failure(failure),
          progressed=progress_started,
        )

      await self._emit_event(
        on_event,
        DoneEvent(cost=0.0, usage=usage, session_id=self._session_id),
      )
      return 0.0, usage
    finally:
      stderr_task.cancel()
      try:
        await stderr_task
      except asyncio.CancelledError:
        pass  # cleanup on task cancel
      self._proc = None

  async def interrupt(self) -> None:
    proc = self._proc
    if proc is None or proc.returncode is not None:
      return
    self._interrupted = True
    proc.terminate()
    try:
      await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
      proc.kill()
      await proc.wait()

  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    await self.interrupt()
    self._project_dir = project_dir
    self._model = model
    self._session_id = resume
    self._cum_usage = _read_codex_cumulative(resume) if resume else {}

  def supports_fork(self) -> bool:
    return True

  async def fork(
    self, parent_session_id: str, project_dir: str, model: str,
  ) -> "CodingAgent | None":
    """Spawn a started, read-only Codex fork — see CodingAgent.fork.

    Branches the parent thread by copying its rollout to a private id (Codex
    resume APPENDS, so a shared id would clobber the main session) and resuming
    that copy; runs sandboxMode=read-only so it cannot modify the project.
    Inherits this agent's endpoint / effort / system prompt. If the parent
    rollout isn't on disk, the fork starts fresh (no branched context).
    """
    fork = CodexCodingAgent(
      {}, "", None, None,
      permission_mode="bypassPermissions",
      system_prompt=self._system_prompt,
      endpoint=self._endpoint,
      read_only=True,
    )
    fork.set_effort(self._effort)
    new_id, copied = _seed_fork_rollout(parent_session_id)
    fork._fork_rollout_path = copied
    await fork.start(project_dir, model, resume=new_id)
    return fork

  async def stop(self) -> None:
    await self.interrupt()
    if self._fork_rollout_path:
      try:
        os.remove(self._fork_rollout_path)
      except OSError:
        pass
      self._fork_rollout_path = ""

  def trailing_note(self, sdk_session_id: str) -> str:
    del sdk_session_id
    total = _codex_cumulative_total(self._cum_usage)
    if total < _CODEX_CONTEXT_WARNING_TOKENS:
      return ""
    return (
      "\n\n<font color='grey'>⚠️ Codex context is getting large "
      f"(~{total:,} tokens). Consider `/clear` followed by `/session recall` "
      "before another long turn to avoid remote compact failures.</font>"
    )

  def _per_turn_usage(self, cumulative: JsonObject) -> JsonObject:
    """Difference Codex's session-cumulative usage into a per-turn canonical dict.

    ``turn.completed.usage`` is the running total since the thread started, so
    the per-turn cost is (this total − previous total). Codex folds cached
    tokens INTO input_tokens (not a sibling), so the new-uncached input is
    ``input_tokens − cached_input_tokens``; Codex has no cache-creation concept.
    """
    def _i(key: str) -> int:
      value = cumulative.get(key)
      if isinstance(value, bool):
        return 0
      if isinstance(value, (int, float)):
        return max(0, int(value))
      return 0

    current = {
      "input_tokens": _i("input_tokens"),
      "cached_input_tokens": _i("cached_input_tokens"),
      "output_tokens": _i("output_tokens"),
    }
    prev = self._cum_usage
    delta_in = max(0, current["input_tokens"] - prev.get("input_tokens", 0))
    delta_cached = max(0, current["cached_input_tokens"] - prev.get("cached_input_tokens", 0))
    delta_out = max(0, current["output_tokens"] - prev.get("output_tokens", 0))
    self._cum_usage = current
    return canonical_usage(
      input_tokens=max(0, delta_in - delta_cached),  # codex input includes cached
      cache_read=delta_cached,
      cache_creation=0,  # codex doesn't report cache-creation tokens
      output_tokens=delta_out,
    )

  def _prepare_prompt(self, prompt: str) -> str:
    # A read-only fork resumes a copied rollout (so _session_id is already
    # set) — the system-prompt-injection branch below wouldn't fire. Prepend
    # the fork directive once, on the fork's first turn.
    if self._read_only and self._fork_note_pending:
      self._fork_note_pending = False
      return f"{_CODEX_FORK_DIRECTIVE.format(project=self._project_dir)}\n\n{prompt}"
    # Codex SDK has no system-prompt field, so inject custom instructions
    # by prepending to the first turn of a thread. Subsequent turns inherit
    # them via the persisted thread context.
    if self._system_prompt and not self._session_id:
      return (
        "<system_instructions>\n"
        f"{self._system_prompt}\n"
        "</system_instructions>\n\n"
        f"{prompt}"
      )
    return prompt

  def _build_command(self) -> list[str]:
    if self._permission_mode != "bypassPermissions":
      raise RuntimeError("Codex provider currently supports only bypassPermissions mode")

    args = ["node", str(_SIDE_CAR_SCRIPT)]
    args.extend(["--cwd", self._project_dir])
    if self._model:
      args.extend(["--model", self._model])
    if self._effort:
      args.extend(["--effort", self._effort])
    if self._read_only:
      # Native Codex read-only sandbox: blocks all project writes while reads
      # and read-only commands still work. No scratch-cwd trick needed (resume
      # is keyed by rollout id, not cwd), so cwd stays the project.
      args.extend(["--sandbox", "read-only"])
    if self._session_id:
      args.extend(["--resume", self._session_id])
    return args

  def _build_env(self) -> dict[str, str]:
    env = {
      "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
      "HOME": os.environ.get("HOME", ""),
      "USER": os.environ.get("USER", ""),
    }
    # OPENAI_BASE_URL / OPENAI_API_BASE: point the codex CLI at any
    # OpenAI-compatible endpoint without touching ~/.codex/config.toml.
    passthrough = (
      "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
      "CODEX_API_KEY",
      # nemo-vision reads ~/.nemo/vision.json from disk but resolves its
      # apiKey from $BAILIAN_API_KEY; the codex sidecar env is a whitelist,
      # so forward the key (base/model come from the file).
      "BAILIAN_API_KEY",
      "http_proxy", "https_proxy", "all_proxy",
    )
    for key in passthrough:
      val = os.environ.get(key)
      if val:
        env[key] = val

    # Explicit --base-url / --api-key flags overlay on top of shell env.
    if self._endpoint.base_url:
      env["OPENAI_BASE_URL"] = self._endpoint.base_url
    if self._endpoint.api_key:
      env["OPENAI_API_KEY"] = self._endpoint.api_key
    return env

  async def _log_stderr(
    self,
    stream: asyncio.StreamReader,
    tail: list[str],
  ) -> None:
    while True:
      raw = await stream.readline()
      if not raw:
        return
      line = raw.decode(errors="replace").rstrip()
      tail.append(line)
      del tail[:-_CODEX_STDERR_TAIL_LINES]
      log.info("[codex-stderr] %s", line)

  async def _emit_event(
    self,
    on_event: Callable[[TurnEvent], None],
    event: TurnEvent,
  ) -> None:
    await asyncio.to_thread(on_event, event)

  def _ensure_runtime(self) -> None:
    if not shutil.which("node"):
      raise RuntimeError("node not found in PATH")
    if not shutil.which("codex"):
      raise RuntimeError("codex CLI not found in PATH")
    if not _SIDE_CAR_SCRIPT.is_file() or not _SIDE_CAR_PACKAGE.is_file():
      raise RuntimeError(
        f"codex sidecar source missing at {_SIDE_CAR_DIR} "
        "(wheel packaging issue — reinstall captain-nemo)"
      )
    if not _SIDE_CAR_NODE_MODULES.is_dir():
      self._install_sidecar_deps()
      return
    if self._sidecar_deps_stale():
      log.info("codex sidecar deps stale — reinstalling for new captain-nemo")
      shutil.rmtree(_SIDE_CAR_NODE_MODULES, ignore_errors=True)
      self._install_sidecar_deps()

  def _sidecar_deps_stale(self) -> bool:
    """True if the installed @openai/codex-sdk version doesn't match the
    pin in package.json. captain-nemo upgrades that bump the codex pin
    (e.g. for a new model the old codex CLI doesn't recognize) need to
    reinstall the sidecar's node_modules — npm install on its own won't
    cross a 0.x minor boundary because the caret semantics in 0.y.z
    constrain to >=0.y.z <0.(y+1).0 in older configs and we now pin
    exact versions anyway. Compare installed vs declared and force a
    fresh install on mismatch."""
    expected = self._declared_sdk_version()
    if not expected:
      return False  # no pin to compare against — keep what's there
    installed = self._installed_sdk_version()
    return installed != expected

  def _declared_sdk_version(self) -> str:
    try:
      data = json.loads(_SIDE_CAR_PACKAGE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
      log.warning("codex sidecar package.json unreadable: %s", exc)
      return ""
    deps = data.get("dependencies", {}) if isinstance(data, dict) else {}
    raw = deps.get("@openai/codex-sdk", "") if isinstance(deps, dict) else ""
    # Accept "0.128.0", "^0.128.0", "~0.128.0" — strip semver prefix.
    return str(raw).lstrip("^~>=< ").strip() if isinstance(raw, str) else ""

  def _installed_sdk_version(self) -> str:
    pkg = _SIDE_CAR_NODE_MODULES / "@openai" / "codex-sdk" / "package.json"
    try:
      data = json.loads(pkg.read_text())
    except (OSError, json.JSONDecodeError):
      return ""
    return str(data.get("version", "")) if isinstance(data, dict) else ""

  def _install_sidecar_deps(self) -> None:
    """Run `npm install` in the sidecar dir on first use.

    Node deps are NOT packaged in the wheel (too heavy, platform-specific);
    they're fetched from the registry the first time a codex provider
    daemon starts on a new install.
    """
    npm = shutil.which("npm")
    if not npm:
      raise RuntimeError("npm not found in PATH — required to install codex sidecar deps")
    log.info("installing codex sidecar node_modules in %s", _SIDE_CAR_DIR)
    import subprocess
    result = subprocess.run(
      [npm, "install", "--no-audit", "--no-fund", "--silent"],
      cwd=str(_SIDE_CAR_DIR),
      capture_output=True,
      text=True,
    )
    if result.returncode != 0:
      raise RuntimeError(
        f"npm install failed in {_SIDE_CAR_DIR}: {result.stderr.strip() or result.stdout.strip()}"
      )
    log.info("codex sidecar node_modules installed")

  def _parse_event(self, line: str) -> JsonObject | None:
    try:
      parsed = json.loads(line)
    except json.JSONDecodeError:
      log.warning("codex sidecar json parse failed: %s", line[:200])
      return None
    if isinstance(parsed, dict):
      return self._coerce_json_object(parsed)
    return None

  def _coerce_json_object(self, value: object) -> JsonObject:
    if isinstance(value, dict):
      return value
    return {}

  def _item_summary(self, item: JsonObject) -> str:
    """Short, one-line label for progress card — mirrors cards.tool_use_summary.

    Codex emits free-form shell / Python / file paths; we truncate each to
    the same ~60-char budget Claude's Bash line uses so the Working card
    preview doesn't overflow with multi-hundred-char heredocs.
    """
    item_type = str(item.get("type", ""))
    if item_type == "reasoning":
      return _short(str(item.get("text", "") or ""), 200)
    if item_type == "command_execution":
      command = str(item.get("command", "") or "")
      if not command:
        return "command"
      # Flatten multi-line commands (heredocs, etc.) for the preview line.
      flat = " ".join(command.split())
      return f"$ {_short(flat, 60)}"
    if item_type == "file_change":
      changes = item.get("changes")
      if isinstance(changes, list):
        paths = []
        for change in changes[:3]:
          if isinstance(change, dict):
            path = str(change.get("path", "") or "")
            kind = str(change.get("kind", "") or "")
            # Show just the basename; full paths are noisy in a one-liner.
            base = os.path.basename(path) if path else ""
            if base:
              paths.append(f"{kind}:{base}" if kind else base)
        return _short(", ".join(paths), 80)
      return "file change"
    if item_type == "mcp_tool_call":
      server = str(item.get("server", "") or "")
      tool = str(item.get("tool", "") or "")
      return _short(": ".join(part for part in (server, tool) if part), 60)
    if item_type == "web_search":
      query = str(item.get("query", "") or "")
      return _short(f"web: {query}", 60) if query else "web search"
    if item_type == "todo_list":
      return "todo list updated"
    if item_type == "error":
      return _short(str(item.get("message", "") or "error"), 200)
    return ""
