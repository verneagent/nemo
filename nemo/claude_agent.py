"""Concrete CodingAgent implementation backed by Claude Agent SDK."""

from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable, cast

from .channel import Channel
from .coding_agent import CodingAgent
from .db import Database
from .permissions import build_ask_user_question_handler, build_permission_handler
from .sdk_thread import SDKThread
from .turn import TurnEvent
from .types import JsonObject

log = logging.getLogger(__name__)


# Claude Agent SDK triggers extended thinking via keywords embedded in the
# user prompt. Map nemo's shared effort levels to the strongest keyword that
# still reliably maps to a distinct thinking budget inside Claude Code.
_EFFORT_TO_KEYWORD: dict[str, str] = {
  "low": "think",
  "medium": "think hard",
  "high": "ultrathink",
}


# Session jsonl size thresholds for the /clear reminder. The Claude CLI
# appends every user/assistant/tool message to ~/.claude/projects/<slug>/
# <session_id>.jsonl. When this file grows past ~30MB the resumed context
# starts pressuring the CLI: we've seen it wedge in silent retry loops
# (SystemMessage every ~50s, no real progress) around 100MB. We nudge the
# user to /clear well before that cliff. Sizes are in bytes.
_SESSION_SIZE_NUDGE = 30 * 1024 * 1024
_SESSION_SIZE_STRONG = 60 * 1024 * 1024


def _session_jsonl_path(project_dir: str, sdk_session_id: str) -> str:
  """Compute the absolute path to a Claude session's jsonl transcript."""
  config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
  slug = project_dir.replace("/", "-")
  return os.path.join(config_dir, "projects", slug, f"{sdk_session_id}.jsonl")


def _format_size_warning(size_bytes: int) -> str:
  """Return a markdown note for an oversized session, or '' if under threshold."""
  if size_bytes < _SESSION_SIZE_NUDGE:
    return ""
  mb = size_bytes / 1024 / 1024
  marker = "⚠️⚠️" if size_bytes >= _SESSION_SIZE_STRONG else "⚠️"
  return (
    f"\n\n---\n{marker} 当前会话上下文已 {mb:.0f} MB，"
    "建议发送 `/clear` 重置，避免响应变慢或卡死。"
  )


class ClaudeCodingAgent(CodingAgent):
  """CodingAgent adapter for the Claude Agent SDK."""

  def __init__(
    self,
    credentials: dict[str, str],
    chat_id: str,
    db: Database,
    channel: Channel,
    permission_mode: str = "bypassPermissions",
    system_prompt: str = "",
  ):
    self._credentials = credentials
    self._chat_id = chat_id
    self._db = db
    self._channel = channel
    self._permission_mode = permission_mode
    self._system_prompt = system_prompt
    self._sdk = SDKThread()
    self._sdk_started = False
    self._options: object = None
    self._effort = ""
    self._project_dir = ""
    # SDK #788 workaround — stale task ids carry across turns inside this
    # adapter. Kept here (not in the abstract CodingAgent interface) because
    # only the Claude SDK exhibits the bug.
    self._stale_tasks: set[str] = set()

  def set_effort(self, effort: str) -> None:
    self._effort = effort if effort in _EFFORT_TO_KEYWORD else ""

  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    if not self._sdk_started:
      self._sdk.start()
      self._sdk_started = True
    # Fresh session — any task ids we thought were stale belong to no
    # session now.
    self._stale_tasks.clear()
    self._project_dir = project_dir
    self._options = self._build_options(project_dir, model, resume=resume)
    await self._sdk.create_client(self._options)

  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    return await self._sdk.run_turn_with_reconnect(
      self._prefix_prompt(prompt), on_event,
      stale_tasks=self._stale_tasks, options=self._options)

  def _prefix_prompt(self, prompt: str) -> str:
    keyword = _EFFORT_TO_KEYWORD.get(self._effort, "")
    if not keyword:
      return prompt
    return f"{keyword}\n\n{prompt}"

  async def interrupt(self) -> None:
    self._sdk.cancel()
    await self._sdk.interrupt()

  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    # Reconnecting spawns a new CLI / SDK session. Any task ids previously
    # marked stale belong to the old session and will never be delivered
    # on the new one — keeping them would force a pointless drain turn.
    self._stale_tasks.clear()
    self._project_dir = project_dir
    self._options = self._build_options(project_dir, model, resume=resume)
    await self._sdk.reconnect(self._options)

  def trailing_note(self, sdk_session_id: str) -> str:
    if not sdk_session_id or not self._project_dir:
      return ""
    path = _session_jsonl_path(self._project_dir, sdk_session_id)
    try:
      size = os.path.getsize(path)
    except OSError:
      return ""
    return _format_size_warning(size)

  async def stop(self) -> None:
    await self._sdk.close_client()
    if self._sdk_started:
      self._sdk.stop()
      self._sdk_started = False

  def _build_agent_prompt(self) -> str:
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
    if self._system_prompt:
      agent_prompt = f"{agent_prompt}\n\n{self._system_prompt}"
    return agent_prompt

  def _build_options(self, project_dir: str, model: str, resume: str = "") -> object:
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk.types import PermissionMode

    agent_prompt = self._build_agent_prompt()

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

    from claude_agent_sdk import PermissionResultAllow

    # AskUserQuestion is a built-in tool with shouldDefer=true,
    # requiresUserInteraction=true. Without a can_use_tool hook in headless
    # SDK mode it executes with empty `answers` and the model gets nothing —
    # which is the "skill silently exits without asking" failure mode.
    # We always attach an askq handler so AskUserQuestion can render to Lark
    # regardless of permission_mode. In bypass mode the dispatcher allows
    # everything else; in non-bypass mode it delegates to the regular
    # permission handler.
    askq_handler = build_ask_user_question_handler(
      self._credentials, self._chat_id, self._channel)
    perm_handler = None
    if self._permission_mode != "bypassPermissions":
      perm_handler = build_permission_handler(
        self._credentials, self._chat_id, self._db, self._channel)

    async def _can_use_tool(
      tool_name: str,
      tool_input: JsonObject,
      context: object,
    ) -> object:
      if tool_name == "AskUserQuestion":
        return await cast(Awaitable[object], askq_handler(tool_name, tool_input, context))
      if perm_handler is not None:
        return await cast(Awaitable[object], perm_handler(tool_name, tool_input, context))
      return PermissionResultAllow()

    def _stderr_handler(line: str) -> None:
      log.info("[sdk-stderr] %s", line.rstrip())

    opts: dict[str, object] = dict(
      allowed_tools=[
        "Agent", "Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "AskUserQuestion",
      ],
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
      can_use_tool=_can_use_tool,
      # Single stream_json messages can exceed the SDK's 1 MB default when
      # a tool result (large file read, heavy bash output) lands in one chunk.
      # Raise to 16 MB so the reader doesn't kill the turn with a decode error.
      max_buffer_size=16 * 1024 * 1024,
    )
    if resume:
      opts["resume"] = resume

    return ClaudeAgentOptions(**opts)
