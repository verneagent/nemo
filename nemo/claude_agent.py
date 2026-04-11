"""Concrete CodingAgent implementation backed by Claude Agent SDK."""

from __future__ import annotations

import logging
import os
from typing import Callable, cast

from .channel import Channel
from .coding_agent import CodingAgent
from .db import Database
from .permissions import build_permission_handler
from .sdk_thread import SDKThread
from .turn import TurnEvent
from .types import JsonObject

log = logging.getLogger(__name__)


class ClaudeCodingAgent(CodingAgent):
  """CodingAgent adapter for the Claude Agent SDK."""

  def __init__(
    self,
    credentials: dict[str, str],
    chat_id: str,
    db: Database,
    channel: Channel,
    permission_mode: str = "bypassPermissions",
  ):
    self._credentials = credentials
    self._chat_id = chat_id
    self._db = db
    self._channel = channel
    self._permission_mode = permission_mode
    self._sdk = SDKThread()
    self._sdk_started = False
    self._options: object = None

  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    if not self._sdk_started:
      self._sdk.start()
      self._sdk_started = True
    self._options = self._build_options(project_dir, model, resume=resume)
    await self._sdk.create_client(self._options)

  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
    stale_tasks: set[str] | None = None,
  ) -> tuple[float, JsonObject]:
    return await self._sdk.run_turn_with_reconnect(
      prompt, on_event, stale_tasks=stale_tasks, options=self._options)

  async def interrupt(self) -> None:
    self._sdk.cancel()
    await self._sdk.interrupt()

  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    self._options = self._build_options(project_dir, model, resume=resume)
    await self._sdk.reconnect(self._options)

  async def stop(self) -> None:
    await self._sdk.close_client()
    if self._sdk_started:
      self._sdk.stop()
      self._sdk_started = False

  def _build_options(self, project_dir: str, model: str, resume: str = "") -> object:
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk.types import PermissionMode

    agent_prompt = (
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

    from .db import _db_path
    db_path = _db_path(project_dir)

    env = {
      "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
      "HOME": os.environ.get("HOME", ""),
      "USER": os.environ.get("USER", ""),
      "NEMO_CHAT_ID": self._chat_id,
      "NEMO_DB": db_path,
      "CLAUDE_ENABLE_STREAM_WATCHDOG": "1",
      "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "90000",
    }
    for key in ("http_proxy", "https_proxy", "all_proxy"):
      val = os.environ.get(key)
      if val:
        env[key] = val

    perm_handler = None
    if self._permission_mode != "bypassPermissions":
      perm_handler = build_permission_handler(
        self._credentials, self._chat_id, self._db, self._channel)

    def _stderr_handler(line: str) -> None:
      log.info("[sdk-stderr] %s", line.rstrip())

    opts: dict[str, object] = dict(
      allowed_tools=["Agent", "Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
      setting_sources=["user", "project"],
      permission_mode=cast(PermissionMode, self._permission_mode),
      system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": agent_prompt,
      },
      cwd=project_dir,
      model=model,
      env=env,
      stderr=_stderr_handler,
      hooks={},
    )
    if perm_handler is not None:
      opts["can_use_tool"] = perm_handler
    if resume:
      opts["resume"] = resume

    return ClaudeAgentOptions(**opts)
