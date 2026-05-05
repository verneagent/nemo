"""Concrete CodingAgent implementation backed by the Codex CLI JSON stream."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
from asyncio.subprocess import Process
from typing import Callable

from .channel import Channel
from .coding_agent import CodingAgent, EndpointConfig
from .db import Database
from .turn import AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent, TurnEvent
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


def _short(text: str, limit: int) -> str:
  """Truncate ``text`` to ``limit`` chars with an ellipsis when oversized."""
  if len(text) <= limit:
    return text
  return text[: limit - 3] + "..."


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

  def set_effort(self, effort: str) -> None:
    mapped = _CLAUDE_TO_CODEX_EFFORT.get(effort, effort)
    self._effort = mapped if mapped in _CODEX_EFFORT_LEVELS else ""

  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    self._ensure_runtime()
    self._project_dir = project_dir
    self._model = model
    self._session_id = resume

  async def run_turn(
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

    stderr_task = asyncio.create_task(self._log_stderr(proc.stderr))
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
          usage = self._coerce_json_object(event.get("usage"))
          continue
        if event_type == "turn.failed":
          error = self._coerce_json_object(event.get("error"))
          failure = str(error.get("message", "Codex turn failed"))
          await self._emit_event(on_event, ErrorEvent(message=failure))
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
      if failure is None and rc != 0 and not self._interrupted:
        failure = f"Codex sidecar exited with status {rc}"
        await self._emit_event(on_event, ErrorEvent(message=failure))
      if failure is not None:
        raise RuntimeError(failure)

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

  async def stop(self) -> None:
    await self.interrupt()

  def _prepare_prompt(self, prompt: str) -> str:
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

  async def _log_stderr(self, stream: asyncio.StreamReader) -> None:
    while True:
      raw = await stream.readline()
      if not raw:
        return
      log.info("[codex-stderr] %s", raw.decode(errors="replace").rstrip())

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
