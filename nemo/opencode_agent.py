"""Concrete CodingAgent implementation backed by an OpenCode SDK sidecar."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
from asyncio.subprocess import Process
import subprocess
from typing import Callable

from .channel import Channel
from .coding_agent import CodingAgent, EndpointConfig
from .db import Database
from .turn import AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent, TurnEvent
from .types import JsonObject

log = logging.getLogger(__name__)

_SIDE_CAR_DIR = Path(__file__).resolve().parent / "opencode_sidecar"
_SIDE_CAR_SCRIPT = _SIDE_CAR_DIR / "run_turn.mjs"
_SIDE_CAR_PACKAGE = _SIDE_CAR_DIR / "package.json"
_SIDE_CAR_NODE_MODULES = _SIDE_CAR_DIR / "node_modules"
_LIST_MODELS_SCRIPT = _SIDE_CAR_DIR / "list_models.mjs"

_EFFORT_PREFIX: dict[str, str] = {
  "low": (
    "Reason tersely. Prefer the smallest sufficient amount of deliberation "
    "before acting."
  ),
  "high": (
    "Reason more carefully than usual. Double-check assumptions, edge cases, "
    "and tool choices before acting."
  ),
}


def _build_agent_prompt(system_prompt: str) -> str:
  prompt = (
    "You are running inside Nemo, a Lark-connected coding agent daemon. "
    "Users interact with you through Lark mobile app. "
    "Process one message at a time. Return your response as text, "
    "the agent process sends it to Lark for you.\n\n"
    "Keep responses concise (mobile reading). Use 2-space indentation in code blocks.\n\n"
    "The Nemo host process handles these slash commands (not you):\n"
    "- /guest add <name> [coowner] — add a group member as guest or coowner\n"
    "- /guest remove <name> — remove a guest\n"
    "- /guest list — list all guests\n"
    "- /norm add <name> <text> — add a group norm\n"
    "- /norm remove <name> — remove a norm\n"
    "- /mention on|off — toggle @mention requirement\n"
    "- /name <name> — rename the current group\n"
    "- /model <name> — switch model\n"
    "When users ask to do these things in natural language, tell them "
    "the exact slash command to use. Do NOT try to execute them yourself.\n\n"
    "Chat history is stored in a SQLite DB. To find past messages:\n"
    "  sqlite3 \"$NEMO_DB\" \"SELECT direction, text, datetime(sent_at, 'unixepoch', 'localtime') FROM messages ORDER BY id DESC LIMIT 20;\"\n\n"
    "To send an image or file to the user:\n"
    "  nemo-send image /path/to/screenshot.png\n"
    "  nemo-send file /path/to/document.pdf"
  )
  if system_prompt:
    return f"{prompt}\n\n{system_prompt}"
  return prompt


class OpenCodeCodingAgent(CodingAgent):
  """CodingAgent adapter for the local OpenCode SDK runtime."""

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
    del credentials, db, channel
    self._chat_id = chat_id
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
    # OpenCode prefix tops out at "high"; clamp Claude's "max" down.
    if effort == "max":
      effort = "high"
    self._effort = effort if effort in ("", "low", "medium", "high") else ""

  def set_endpoint(self, endpoint: EndpointConfig) -> None:
    self._endpoint = endpoint

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
    log.info("opencode sdk start model=%s resume=%s", self._model, bool(self._session_id))
    proc = await asyncio.create_subprocess_exec(
      *args,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      cwd=self._project_dir or None,
      env=self._build_env(),
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
    cost = 0.0
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
        if event_type == "session.started":
          self._session_id = str(event.get("session_id", "") or "")
          continue
        if event_type == "turn.completed":
          usage = self._coerce_json_object(event.get("usage"))
          event_cost = event.get("cost")
          if isinstance(event_cost, int | float):
            cost = float(event_cost)
          continue
        if event_type == "turn.failed":
          error = self._coerce_json_object(event.get("error"))
          failure = str(error.get("message", "OpenCode turn failed"))
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
          await self._emit_event(
            on_event,
            ProgressEvent(kind=kind, summary=summary, first=is_first),
          )

      rc = await proc.wait()
      if failure is None and rc != 0 and not self._interrupted:
        failure = f"OpenCode sidecar exited with status {rc}"
        await self._emit_event(on_event, ErrorEvent(message=failure))
      if failure is not None:
        raise RuntimeError(failure)

      await self._emit_event(
        on_event,
        DoneEvent(cost=cost, usage=usage, session_id=self._session_id),
      )
      return cost, usage
    finally:
      stderr_task.cancel()
      try:
        await stderr_task
      except asyncio.CancelledError:
        pass
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
    parts: list[str] = []
    keyword = _EFFORT_PREFIX.get(self._effort, "")
    if keyword:
      parts.append(keyword)
    parts.append(prompt)
    return "\n\n".join(part for part in parts if part)

  def _build_command(self) -> list[str]:
    if self._permission_mode != "bypassPermissions":
      raise RuntimeError("OpenCode provider currently supports only bypassPermissions mode")

    args = ["node", str(_SIDE_CAR_SCRIPT)]
    args.extend(["--cwd", self._project_dir])
    if self._model:
      args.extend(["--model", self._model])
    if self._session_id:
      args.extend(["--resume", self._session_id])
    return args

  def _build_env(self) -> dict[str, str]:
    from .db import _db_path

    env = os.environ.copy()
    env["NEMO_CHAT_ID"] = self._chat_id
    env["NEMO_DB"] = _db_path(self._project_dir)
    env["NEMO_OPENCODE_SYSTEM_PROMPT"] = _build_agent_prompt(self._system_prompt)

    # OpenCode is multi-provider — a single base_url is ambiguous. Dispatch
    # by the model's `provider/` prefix:
    #   anthropic/* → only ANTHROPIC_*
    #   openai/*    → only OPENAI_*
    #   anything else (no slash, "default", or third-party prefix) → set
    #   both, defensively, since the configured opencode provider plugin
    #   could read from either.
    if self._endpoint.base_url or self._endpoint.api_key:
      prefix = self._model.split("/", 1)[0].lower() if "/" in self._model else ""
      if prefix == "anthropic":
        targets = ("anthropic",)
      elif prefix == "openai":
        targets = ("openai",)
      else:
        targets = ("anthropic", "openai")
      if self._endpoint.base_url:
        if "anthropic" in targets:
          env["ANTHROPIC_BASE_URL"] = self._endpoint.base_url
        if "openai" in targets:
          env["OPENAI_BASE_URL"] = self._endpoint.base_url
      if self._endpoint.api_key:
        if "anthropic" in targets:
          env["ANTHROPIC_API_KEY"] = self._endpoint.api_key
        if "openai" in targets:
          env["OPENAI_API_KEY"] = self._endpoint.api_key
    return env

  async def _log_stderr(self, stream: asyncio.StreamReader) -> None:
    while True:
      raw = await stream.readline()
      if not raw:
        return
      log.info("[opencode-stderr] %s", raw.decode(errors="replace").rstrip())

  async def _emit_event(
    self,
    on_event: Callable[[TurnEvent], None],
    event: TurnEvent,
  ) -> None:
    await asyncio.to_thread(on_event, event)

  def _ensure_runtime(self) -> None:
    if not shutil.which("node"):
      raise RuntimeError("node not found in PATH")
    if not shutil.which("opencode"):
      raise RuntimeError("opencode CLI not found in PATH")
    if not _SIDE_CAR_SCRIPT.is_file() or not _SIDE_CAR_PACKAGE.is_file():
      raise RuntimeError(
        f"opencode sidecar source missing at {_SIDE_CAR_DIR} "
        "(wheel packaging issue — reinstall captain-nemo)"
      )
    if not _SIDE_CAR_NODE_MODULES.is_dir():
      self._install_sidecar_deps()

  def _install_sidecar_deps(self) -> None:
    npm = shutil.which("npm")
    if not npm:
      raise RuntimeError("npm not found in PATH — required to install opencode sidecar deps")
    log.info("installing opencode sidecar node_modules in %s", _SIDE_CAR_DIR)
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
    log.info("opencode sidecar node_modules installed")

  def _parse_event(self, line: str) -> JsonObject | None:
    try:
      parsed = json.loads(line)
    except json.JSONDecodeError:
      log.warning("opencode sidecar json parse failed: %s", line[:200])
      return None
    if isinstance(parsed, dict):
      return self._coerce_json_object(parsed)
    return None

  def _coerce_json_object(self, value: object) -> JsonObject:
    if isinstance(value, dict):
      return value
    return {}

  def _item_summary(self, item: JsonObject) -> str:
    item_type = str(item.get("type", ""))
    if item_type == "reasoning":
      text = str(item.get("text", "") or "")
      return text[:200]
    if item_type == "tool_call":
      tool = str(item.get("tool", "") or "")
      title = str(item.get("title", "") or "")
      if title:
        return f"{tool}: {title}" if tool else title
      return tool
    return ""


def query_opencode_model_catalog_data(project_dir: str) -> tuple[tuple[str, ...], str]:
  """Query configured OpenCode models via the SDK sidecar.

  Returns (models, note). On failure we fall back to an empty model set with
  a note explaining how to inspect the live config manually.
  """
  base_note = (
    "Dynamic models come from the local OpenCode config. Use full "
    "`provider/model` names."
  )
  if not project_dir:
    return (), base_note
  node = shutil.which("node")
  opencode = shutil.which("opencode")
  if not node or not opencode or not _LIST_MODELS_SCRIPT.is_file():
    note = (
      f"{base_note} Live query unavailable here; run `opencode models` on the "
      "host to inspect configured values."
    )
    return (), note

  try:
    result = subprocess.run(
      [node, str(_LIST_MODELS_SCRIPT), "--cwd", project_dir],
      capture_output=True,
      text=True,
      check=True,
      timeout=20,
    )
  except (subprocess.SubprocessError, OSError) as exc:
    note = (
      f"{base_note} Live query failed ({exc}); run `opencode models` on the "
      "host to inspect configured values."
    )
    return (), note

  try:
    parsed = json.loads(result.stdout)
  except json.JSONDecodeError:
    note = (
      f"{base_note} Live query returned malformed output; run `opencode models` "
      "on the host to inspect configured values."
    )
    return (), note
  if not isinstance(parsed, dict):
    return (), base_note

  raw_models = parsed.get("models")
  models: list[str] = []
  if isinstance(raw_models, list):
    for model in raw_models:
      if isinstance(model, str) and model:
        models.append(model)
  default_model = parsed.get("default_model")
  note = base_note
  if isinstance(default_model, str) and default_model:
    note += f" Config default: `{default_model}`."
  return tuple(sorted(dict.fromkeys(models))), note
