"""Concrete CodingAgent implementation backed by Claude Agent SDK."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, cast

from .channel import Channel
from .coding_agent import CodingAgent, EndpointConfig
from .db import Database
from .permissions import build_ask_user_question_handler, build_permission_handler
from .sdk_thread import SDKThread
from .sessions import claude_project_slug
from .turn import CompactStartedEvent, TurnEvent
from .types import JsonObject

# Env vars Claude Code / claude-agent-sdk honor for endpoint overrides
# and model routing. We passthrough whichever the host shell already exports
# so users can configure entirely via env without touching nemo flags;
# explicit --base-url/--api-key flags overlay on top.
_CLAUDE_ENDPOINT_ENV_KEYS = (
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_MODEL",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
  "CLAUDE_CODE_SUBAGENT_MODEL",
)

log = logging.getLogger(__name__)


# claude-agent-sdk exposes a native `effort` literal on ClaudeAgentOptions.
# Pass it through directly; the SDK forwards to the Messages API's effort
# parameter. The host triggers a reconnect (with resume) when the user
# changes effort so the new value lands on the next turn.
_CLAUDE_EFFORT_LEVELS = frozenset({"low", "medium", "high", "max"})


# Session jsonl size thresholds for the /clear reminder. The Claude CLI
# appends every user/assistant/tool message to ~/.claude/projects/<slug>/
# <session_id>.jsonl. When this file grows past ~30MB the resumed context
# starts pressuring the CLI: we've seen it wedge in silent retry loops
# (SystemMessage every ~50s, no real progress) around 100MB. We nudge the
# user to /clear well before that cliff. Sizes are in bytes.
_SESSION_SIZE_NUDGE = 30 * 1024 * 1024
_SESSION_SIZE_STRONG = 60 * 1024 * 1024


# Appended after the main system prompt for `/btw` side questions.
_BTW_DIRECTIVE = (
  "SIDE QUESTION MODE: The user asked a quick out-of-band question with "
  "`/btw`. Answer ONLY from the conversation context you already have. "
  "You have NO tools — do not attempt to read files, run commands, or "
  "search. Give one short, direct answer (a few sentences at most). Do "
  "not resume or comment on the main task; this exchange is ephemeral "
  "and will not be part of the conversation."
)

# Appended after the main system prompt for `/fork` read-only branches.
# The OS sandbox physically blocks writes to the project (cwd is a private
# scratch dir, the project lives outside the writable workspace), but the
# model still needs to be told where the project is and that it is read-only,
# otherwise it operates in the empty scratch dir.
_FORK_DIRECTIVE = (
  "FORK MODE — read-only branch. You are a forked side-conversation that "
  "branched from the main session's context; it is ephemeral and nothing you "
  "do is written back to the main conversation. The project under review is "
  "at {project} — treat it as STRICTLY READ-ONLY: you may read files, grep, "
  "and run read-only commands or tests, but you CANNOT modify any file there "
  "(an OS sandbox blocks all writes outside your scratch dir, so write "
  "attempts will fail with 'operation not permitted'). Use absolute paths "
  "under {project} to read project files. Your working directory is a private "
  "scratch space ({scratch}) — put any temporary scripts or output there."
)

# Tools a read-only fork may use: everything investigative, minus the file
# mutators. Bash stays (sandbox makes project writes impossible) so the fork
# can run tests / git / grep. Write/Edit/NotebookEdit are removed because they
# bypass the bash sandbox; Agent/Skill are removed as escape hatches (a
# subagent/skill could write outside the sandbox); AskUserQuestion is removed
# because rendering it into the fork sub-thread is not wired for v1.
_FORK_ALLOWED_TOOLS = [
  "Bash", "BashOutput", "KillShell",
  "Read", "Grep", "Glob", "WebSearch", "WebFetch", "TodoWrite",
]
_FORK_DISALLOWED_TOOLS = [
  "Write", "Edit", "NotebookEdit", "Agent", "Skill", "AskUserQuestion",
  "mcp__*",
]

# Hard ceiling so a wedged side query can't hang the chat indefinitely.
_BTW_TIMEOUT = 120

# `/session recall` digest: a throwaway read-only session reads a past
# transcript and returns a compact summary (see ``digest_transcript``).
# Transcripts can be multi-MB, so it gets a longer ceiling and more turns
# than `/btw` (it reads the file in chunks via the Read tool).
_DIGEST_TIMEOUT = 240
_DIGEST_MAX_TURNS = 24

# System prompt for the digest session — keep it lean (no nemo-host preamble:
# this session never talks to the user, it only reads a file and returns text).
_DIGEST_DIRECTIVE = (
  "TRANSCRIPT DIGEST MODE: You are a throwaway, read-only session whose only "
  "job is to read a past coding session's transcript file and summarise it for "
  "another agent. You are NOT resuming that session and must not act on, "
  "re-run, or continue anything it describes. Only Read/Grep/Glob are "
  "available. Produce the requested summary, then stop."
)

# The digest task prompt — filled with the transcript path + on-disk format
# hint. Asks for a compact, fixed-section briefing so the recall prompt that
# carries it into the main agent's context stays small and predictable.
_DIGEST_TASK = (
  "A past coding session in this project was recorded as a JSONL transcript:\n\n"
  "  {path}\n\n"
  "On-disk format: {fmt}\n\n"
  "The file may be large. Prefer the TAIL — the most recent turns are usually "
  "the most relevant; for big files use Read with an offset/limit rather than "
  "loading the whole thing. Skim tool calls; don't quote them verbatim.\n\n"
  "Reply with ONLY this markdown (no preamble), each section a few terse "
  "bullets, and omit any section that would be empty:\n\n"
  "### Working on\n### Done\n### Open / not done\n### Key files\n### Watch out\n\n"
  "Keep the whole summary under ~400 words."
)

# Why not 1: the side query exposes NO tools, but an agentic model
# (notably non-Claude models resumed into a tool-heavy coding session)
# will still *attempt* tool calls before answering in prose. Each blocked
# attempt burns a turn; with max_turns=1 the turn ends as `error_max_turns`
# before any text block is produced and `/btw` comes back blank (observed
# on deepseek-v4-pro). A few turns let the model recover and answer after
# its tool calls are denied. Read-only is still guaranteed by the empty
# allow-list — extra turns can't execute anything; well-behaved models
# answer on turn 1 regardless, so this adds no latency to the common path.
_BTW_MAX_TURNS = 6


# The Claude CLI's project-dir → transcript-dir slug. ``_seed_fork_transcript``
# copies a parent transcript into a fork's slug dir, and the CLI only finds it
# if the slug matches byte-for-byte; see ``claude_project_slug`` for the rule.
_project_slug = claude_project_slug


def _session_jsonl_path(project_dir: str, sdk_session_id: str) -> str:
  """Compute the absolute path to a Claude session's jsonl transcript."""
  config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
  return os.path.join(
    config_dir, "projects", _project_slug(project_dir), f"{sdk_session_id}.jsonl")


def _short_error(exc: BaseException) -> str:
  """One-line summary suitable for a log breadcrumb."""
  return str(exc).split("\n")[0][:200]


def _stderr_tail(lines: "deque[str] | list[str]", limit: int = 20) -> str:
  """Last few CLI stderr lines, flattened to one log-friendly breadcrumb."""
  tail = [ln.strip() for ln in list(lines)[-limit:] if ln.strip()]
  return " ⏎ ".join(tail) if tail else "(empty)"


_EXIT_1_PATTERN = re.compile(r"\bexit code 1\b", re.IGNORECASE)
_NO_SESSION_PATTERN = re.compile(r"\bno session\b", re.IGNORECASE)


def _is_resume_unrecoverable(exc: BaseException) -> bool:
  """True if the SDK connect failure looks like a stale resume target.

  The bundled claude CLI exits 1 when it can't materialise a session
  from its local jsonl. The SDK reports that as a chained
  ProcessError("Command failed with exit code 1") wrapped by SDKThread's
  RuntimeError. We walk the cause chain looking for the exit code so
  we don't false-positive on real init bugs (network, CLI missing,
  etc.). When we can't decide, return False — better to surface a
  genuine bug than to silently lose conversation context.
  """
  cur: BaseException | None = exc
  while cur is not None:
    name = type(cur).__name__
    msg = str(cur)
    if name == "ProcessError":
      code = getattr(cur, "exit_code", None)
      if code == 1:
        return True
    # Some SDK versions surface the same failure as a plain
    # RuntimeError with text only — match exit code 1 (with word
    # boundary so "exit code 137" doesn't trigger) or "no session".
    if _EXIT_1_PATTERN.search(msg) or _NO_SESSION_PATTERN.search(msg):
      return True
    cur = cur.__cause__ or cur.__context__
  return False


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


@dataclass
class _OneShotResult:
  """Outcome of a one-shot ``query()`` (see ``_run_oneshot_query``).

  ``text`` is the concatenated assistant text (possibly empty — the caller
  decides how to render an empty answer). ``error`` is a short breadcrumb
  when the worker thread itself failed (so the caller can surface or
  swallow it); it stays "" on the normal path.
  """
  text: str = ""
  timed_out: bool = False
  n_assistant: int = 0
  result_subtype: str | None = None
  result_is_error: bool | None = None
  error: str = ""


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
    endpoint: EndpointConfig | None = None,
    read_only: bool = False,
  ):
    self._credentials = credentials
    self._chat_id = chat_id
    self._db = db
    self._channel = channel
    self._permission_mode = permission_mode
    self._system_prompt = system_prompt
    self._endpoint = endpoint or EndpointConfig()
    # Read-only fork mode (see fork()/_build_options): the project is
    # exposed read-only via an OS sandbox while cwd points at a private
    # scratch dir. _fork_parent_id is the session this fork branched from;
    # it drives fork_session=True on (re)connect until the fork's own
    # session id materialises.
    self._read_only = read_only
    self._scratch_dir = ""
    self._fork_parent_id = ""
    self._sdk = SDKThread()
    self._sdk_started = False
    self._options: object = None
    self._effort = ""
    self._project_dir = ""
    self._model = ""
    # Latest sdk_session_id observed via DoneEvent. Used by the options
    # factory so a mid-turn watchdog reconnect (sdk_thread.run_turn_with_reconnect)
    # resumes the live session instead of starting a fresh one and dropping
    # conversation context.
    self._latest_session_id: str = ""
    # SDK #788 workaround — stale task ids carry across turns inside this
    # adapter. Kept here (not in the abstract CodingAgent interface) because
    # only the Claude SDK exhibits the bug.
    self._stale_tasks: set[str] = set()
    # Live per-turn on_event sink for hook callbacks. Hooks are session-
    # scoped (registered once on ClaudeAgentOptions) but events are
    # per-turn, so the PreCompact hook reads this attribute to find the
    # current turn's callback. Set at the top of run_turn(), cleared on
    # exit. None means no turn is active — we drop the hook event.
    self._current_on_event: Callable[[TurnEvent], None] | None = None
    # Shared steer counter (see steer()). One element so the SDK loop can
    # bump it in place; _single_turn absorbs one extra ResultMessage per
    # count. Lives on the agent (not per-turn) so steer() and the running
    # turn reference the same object.
    self._pending_steers: list[int] = [0]

  def set_effort(self, effort: str) -> None:
    self._effort = effort if effort in _CLAUDE_EFFORT_LEVELS else ""

  def set_endpoint(self, endpoint: EndpointConfig) -> None:
    self._endpoint = endpoint

  async def start(
    self, project_dir: str, model: str, resume: str = "",
    fork_session: bool = False,
  ) -> None:
    if not self._sdk_started:
      self._sdk.start()
      self._sdk_started = True
    # Fresh session — any task ids we thought were stale belong to no
    # session now.
    self._stale_tasks.clear()
    self._project_dir = project_dir
    self._model = model
    self._latest_session_id = resume
    # fork_session=True means "branch from `resume` on first connect". Record
    # the parent id so _build_options re-forks from it on any mid-first-turn
    # reconnect, and stops forking once the fork's own session id arrives.
    self._fork_parent_id = resume if (fork_session and resume) else ""
    if self._read_only and not self._scratch_dir:
      import tempfile
      self._scratch_dir = tempfile.mkdtemp(prefix="nemo_fork_")
    # A read-only fork runs with cwd=scratch (so the project stays read-only),
    # but the Claude CLI keys session transcripts by a cwd-derived slug. The
    # parent transcript lives under the PROJECT slug; resuming/forking it from
    # the scratch cwd would look under the SCRATCH slug and miss it (→ silent
    # "fresh session", losing all branched context). Copy the parent jsonl
    # into the scratch slug so resume+fork_session find it.
    if self._read_only and self._fork_parent_id:
      self._seed_fork_transcript(project_dir, self._fork_parent_id)
    self._options = self._build_options(project_dir, model, resume=resume)
    try:
      await self._sdk.create_client(self._options)
    except Exception as exc:
      # Resume-id can become unusable between daemon runs: the bundled
      # claude CLI hard-exits 1 if the local session jsonl is missing
      # (process killed mid-turn last time, jsonl never flushed; the
      # session was created on a different machine; the user's
      # `~/.claude/projects/...` got pruned; or some endpoint that
      # speaks Anthropic protocol returned a session id that the
      # SDK's local resume path doesn't materialise into a jsonl).
      # The exit-1 propagates as a ProcessError nested inside the
      # SDKThread retry RuntimeError — same shape as the codex
      # sidecar's "no rollout found" failure, just on the Claude side.
      # Drop the resume id and try once more so the daemon doesn't
      # die. The next DoneEvent will overwrite the stale id in nemo's
      # DB on its way out.
      if not resume or not _is_resume_unrecoverable(exc):
        raise
      log.warning(
        "Claude SDK resume %s unusable (%s) — starting fresh session",
        resume[:8], _short_error(exc),
      )
      self._latest_session_id = ""
      self._options = self._build_options(project_dir, model, resume="")
      await self._sdk.create_client(self._options)

  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    from .turn import DoneEvent

    def _wrapped(ev: TurnEvent) -> None:
      if isinstance(ev, DoneEvent) and ev.session_id:
        self._latest_session_id = ev.session_id
      on_event(ev)

    def _options_factory() -> object:
      return self._build_options(
        self._project_dir, self._model, resume=self._latest_session_id)

    def _is_paused() -> bool:
      # True while an AskUserQuestion / permission prompt is awaiting the
      # user. The SDK is then blocked inside can_use_tool and emits nothing;
      # without this the turn watchdog would treat the wait as a hang and
      # force-reconnect, discarding the in-flight question.
      return bool(getattr(self._channel, "permission_active", False))

    self._current_on_event = _wrapped
    try:
      return await self._sdk.run_turn_with_reconnect(
        prompt, _wrapped,
        stale_tasks=self._stale_tasks,
        options=self._options,
        options_factory=_options_factory,
        is_paused=_is_paused,
        pending_steers=self._pending_steers)
    finally:
      self._current_on_event = None

  async def _on_pre_compact(
    self,
    hook_input: object,
    tool_use_id: object,
    context: object,
  ) -> dict[str, object]:
    """SDK PreCompact hook — surface a CompactStartedEvent.

    Fires from the SDK's control channel before the CLI begins compaction.
    Returns an empty dict so the SDK's default-allow path runs (returning
    a dict with ``decision: "block"`` would actually block compaction —
    not what we want).
    """
    del tool_use_id, context
    sink = self._current_on_event
    if sink is None:
      log.warning("PreCompact hook fired outside a live turn — dropping")
      return {}
    try:
      trigger = ""
      if isinstance(hook_input, dict):
        trigger = str(hook_input.get("trigger") or "")
      sink(CompactStartedEvent(trigger=trigger))
    except Exception as exc:  # never block compaction on our notice failure
      log.warning("PreCompact hook handler error: %s", exc)
    return {}

  async def interrupt(self) -> None:
    self._sdk.cancel()
    await self._sdk.interrupt()

  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    # Reconnecting spawns a new CLI / SDK session. Any task ids previously
    # marked stale belong to the old session and will never be delivered
    # on the new one — keeping them would force a pointless drain turn.
    self._stale_tasks.clear()
    self._project_dir = project_dir
    self._model = model
    self._latest_session_id = resume
    self._options = self._build_options(project_dir, model, resume=resume)
    try:
      await self._sdk.reconnect(self._options)
    except Exception as exc:
      if not resume or not _is_resume_unrecoverable(exc):
        raise
      log.warning(
        "Claude SDK resume %s unusable (%s) — starting fresh session",
        resume[:8], _short_error(exc),
      )
      self._latest_session_id = ""
      self._options = self._build_options(project_dir, model, resume="")
      await self._sdk.reconnect(self._options)

  async def side_question(self, question: str, sdk_session_id: str) -> str:
    """Read-only ephemeral side question — see CodingAgent.side_question.

    Implementation: a one-shot ``query()``. When a live session exists
    we ``resume`` it for full context + parent prompt-cache reuse, with
    ``fork_session=True`` so every message it generates lands in a
    throwaway forked transcript — the real session is never touched, so
    the answer can never re-enter conversation history. Before the first
    turn completes there is no session id yet; we still answer, just
    from a fresh ephemeral session (no prior context to draw on, but
    /btw must work as the very first message too — Claude Code's does).
    Read-only is enforced solely by the empty tool allow-list (no tool
    can execute regardless of how many turns run); ``max_turns`` is a few
    (see ``_BTW_MAX_TURNS``) rather than 1 so an agentic model that burns
    turns attempting blocked tool calls still reaches a prose answer
    instead of ending as ``error_max_turns`` with no text.

    Isolation: ``claude_agent_sdk.query()`` drives an internal anyio
    task group whose cancel scopes MUST be entered and exited on the
    same task/loop. Running it on the host's main loop and letting the
    async generator be finalised (timeout-cancel or GC) crashed the
    daemon with "Attempted to exit cancel scope in a different task"
    (anyio), taking the whole chat down. So — exactly like ``SDKThread``
    does for the main client — we run the entire query lifecycle on a
    dedicated worker thread with its own ``asyncio.run`` loop, and
    explicitly ``aclose()`` the generator in-task there. The host only
    ``await``s the worker via ``to_thread``; nothing from the SDK's
    cancel scopes can ever cross back into the main loop, and a
    concurrent main turn is neither blocked nor interrupted.
    """
    if not self._project_dir:
      return ""

    from claude_agent_sdk import ClaudeAgentOptions

    # Reuse the main turn's system prompt verbatim so a forked query
    # hits the parent's cached prefix; the btw directive is a short
    # suffix appended after it.
    agent_prompt = (
      f"{self._build_agent_prompt()}\n\n{_BTW_DIRECTIVE}"
    )
    # Capture the forked CLI's stderr. Without a callback the SDK discards
    # it and a non-zero exit surfaces only as the opaque "Command failed
    # with exit code N (Check stderr output for details)" — leaving no way
    # to tell WHY (`⚠️ btw failed: …`). Bounded so a chatty/wedged CLI
    # can't grow it without limit.
    stderr_lines: deque[str] = deque(maxlen=200)
    opts_kwargs: dict[str, object] = dict(
      stderr=stderr_lines.append,
      allowed_tools=[],
      # Belt-and-suspenders: even with an empty allow-list, name every
      # mutating/IO tool so a preset system prompt can't coax one in.
      disallowed_tools=[
        "Agent", "Skill", "Read", "Write", "Edit", "NotebookEdit",
        "Bash", "BashOutput", "KillShell",
        "Glob", "Grep", "WebSearch", "WebFetch", "TodoWrite",
        "AskUserQuestion", "mcp__*",
      ],
      max_turns=_BTW_MAX_TURNS,
      permission_mode="bypassPermissions",
      cwd=self._project_dir,
      model=self._model,
      env=self._build_env(self._project_dir, self._model),
      system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": agent_prompt,
      },
      max_buffer_size=16 * 1024 * 1024,
    )
    if sdk_session_id:
      # Fork only makes sense with a session to fork; without resume it
      # would be meaningless. With a session: read context + reuse the
      # parent prompt cache, but keep the answer out of real history.
      opts_kwargs["resume"] = sdk_session_id
      opts_kwargs["fork_session"] = True
    opts = ClaudeAgentOptions(**opts_kwargs)
    try:
      res = await self._run_oneshot_query(
        prompt=question, opts=opts, timeout=_BTW_TIMEOUT,
        stderr_lines=stderr_lines, label="btw",
      )
    except Exception as exc:
      log.warning("side_question dispatch failed: %s", _short_error(exc))
      return f"⚠️ btw failed: {_short_error(exc)}"
    if res.error:
      return f"⚠️ btw failed: {res.error}"
    if not res.text:
      # The empty/timeout fallbacks used to return silently, leaving no
      # trace in the daemon log of WHY `/btw` came back blank. Record the
      # shape of the forked turn so the next blank answer is diagnosable
      # (timeout vs. a result with no assistant text vs. an error result).
      log.warning(
        "side_question empty: timed_out=%s assistant_msgs=%d "
        "result_subtype=%s result_is_error=%s model=%s stderr=%s",
        res.timed_out, res.n_assistant, res.result_subtype,
        res.result_is_error, self._model, _stderr_tail(stderr_lines),
      )
      if res.timed_out:
        return "⚠️ btw timed out before an answer came back."
      return "⚠️ btw returned no answer."
    return res.text

  async def _run_oneshot_query(
    self,
    *,
    prompt: str,
    opts: object,
    timeout: int,
    stderr_lines: "deque[str]",
    label: str,
  ) -> _OneShotResult:
    """Drive a one-shot ``claude_agent_sdk.query()`` to completion and
    return the collected assistant text — the anyio-safe execution core
    shared by ``side_question`` and ``digest_transcript``.

    ``query()`` runs an internal anyio task group whose cancel scopes MUST
    be entered and exited on the same task/loop; running it on the host
    loop and letting the generator be finalised cross-task crashes the
    daemon with "Attempted to exit cancel scope in a different task". So
    the entire lifecycle (drive + explicit ``aclose()``) lives on a
    dedicated worker thread with its own ``asyncio.run`` loop; the host
    only ``await``s via ``to_thread`` and nothing from the SDK's scopes
    crosses back. See ``side_question``'s docstring for the full rationale.

    Never raises for an in-query failure: a worker-thread exception is
    captured into ``_OneShotResult.error`` (with the CLI stderr tail
    logged) so callers choose whether to surface or swallow it.
    """
    from claude_agent_sdk import (
      AssistantMessage,
      ResultMessage,
      TextBlock,
      query,
    )

    async def _amain() -> _OneShotResult:
      # Enforce the timeout INSIDE this loop with a monotonic deadline
      # instead of cancelling from outside: cancelling the SDK generator
      # across tasks is precisely what corrupts the anyio scope. We close
      # the generator explicitly, in this task, in the finally.
      loop = asyncio.get_running_loop()
      deadline = loop.time() + timeout
      out = _OneShotResult()
      parts: list[str] = []
      gen = query(prompt=prompt, options=opts)
      try:
        async for msg in gen:
          if isinstance(msg, AssistantMessage):
            out.n_assistant += 1
            for block in msg.content:
              if isinstance(block, TextBlock) and block.text:
                parts.append(block.text)
          elif isinstance(msg, ResultMessage):
            out.result_subtype = getattr(msg, "subtype", None)
            out.result_is_error = getattr(msg, "is_error", None)
            # Deliberately NO `break` here. Breaking out of the async-for
            # leaves the SDK generator suspended at its yield; the
            # subsequent aclose() then finalises it from a different task
            # than the one that entered its anyio task group, raising
            # "Attempted to exit cancel scope in a different task than it
            # was entered in" — the crash class this whole isolation
            # exists to prevent. Letting the loop run on to
            # StopAsyncIteration makes the SDK close its own task group on
            # THIS worker task; the one-shot ``query()`` stops right after
            # its final ResultMessage, and the deadline below still bounds it.
          if loop.time() >= deadline:
            out.timed_out = True
            break
      finally:
        # On the normal path the generator is already exhausted, so this
        # is a harmless no-op; only the timeout-break leaves it suspended,
        # where closing it here (in-task) is the safe place to do so. Any
        # late anyio trip is swallowed so it can't escape the worker loop.
        aclose = getattr(gen, "aclose", None)
        if aclose is not None:
          try:
            await aclose()
          except (Exception, asyncio.CancelledError) as exc:
            log.warning("%s generator close: %s", label, _short_error(exc))
      out.text = "".join(parts).strip()
      return out

    def _run() -> _OneShotResult:
      # Fresh loop, own thread — the SDK's anyio scopes live and die
      # entirely here, never touching the host loop.
      try:
        return asyncio.run(_amain())
      except BaseException as exc:  # nothing may propagate to the host
        # The SDK's ProcessError says only "Command failed with exit code
        # N (Check stderr output for details)" — the CLI's real stderr is
        # the only thing that explains it, so log the captured tail.
        tail = _stderr_tail(stderr_lines)
        log.warning("%s worker failed: %s; cli stderr: %s",
                    label, _short_error(exc), tail)
        return _OneShotResult(error=_short_error(exc))

    return await asyncio.to_thread(_run)

  async def digest_transcript(self, transcript_path: str, fmt_hint: str) -> str:
    """Summarise a past transcript in a fresh, context-free session.

    See ``CodingAgent.digest_transcript``. Implementation mirrors
    ``side_question``'s one-shot ``query()`` but with NO ``resume`` /
    ``fork_session`` (a blank session — the transcript itself is the only
    context) and a read-only Read/Grep/Glob tool set so it can open the
    JSONL at ``transcript_path``. Returns "" on any failure / empty result
    so the caller falls back to the inline "read the file yourself" recall.
    """
    if not self._project_dir or not self._model:
      return ""

    from claude_agent_sdk import ClaudeAgentOptions

    stderr_lines: deque[str] = deque(maxlen=200)
    opts = ClaudeAgentOptions(
      stderr=stderr_lines.append,
      # Read is what opens the transcript; Grep/Glob help seek inside a
      # large one. Everything mutating/agentic is denied so a blank
      # session can't be coaxed into doing more than reading the file.
      allowed_tools=["Read", "Grep", "Glob"],
      disallowed_tools=[
        "Write", "Edit", "NotebookEdit", "Bash", "BashOutput", "KillShell",
        "Agent", "Skill", "AskUserQuestion", "WebSearch", "WebFetch",
        "TodoWrite", "mcp__*",
      ],
      max_turns=_DIGEST_MAX_TURNS,
      permission_mode="bypassPermissions",
      cwd=self._project_dir,
      model=self._model,
      env=self._build_env(self._project_dir, self._model),
      system_prompt=_DIGEST_DIRECTIVE,
      max_buffer_size=16 * 1024 * 1024,
    )
    task = _DIGEST_TASK.format(
      path=transcript_path, fmt=fmt_hint or "One JSON event per line.")
    try:
      res = await self._run_oneshot_query(
        prompt=task, opts=opts, timeout=_DIGEST_TIMEOUT,
        stderr_lines=stderr_lines, label="recall-digest",
      )
    except Exception as exc:
      log.warning("digest_transcript dispatch failed: %s", _short_error(exc))
      return ""
    if res.error:
      log.warning("digest_transcript worker error: %s", res.error)
      return ""
    if not res.text:
      log.warning(
        "digest_transcript empty: timed_out=%s assistant_msgs=%d "
        "result_subtype=%s model=%s stderr=%s",
        res.timed_out, res.n_assistant, res.result_subtype,
        self._model, _stderr_tail(stderr_lines),
      )
      return ""
    return res.text

  def trailing_note(self, sdk_session_id: str) -> str:
    if not sdk_session_id or not self._project_dir:
      return ""
    path = _session_jsonl_path(self._project_dir, sdk_session_id)
    try:
      size = os.path.getsize(path)
    except OSError:
      return ""
    return _format_size_warning(size)

  def _seed_fork_transcript(self, parent_project_dir: str, parent_sid: str) -> None:
    """Copy the parent session jsonl into this fork's scratch-cwd slug.

    Claude stores transcripts at ``<config>/projects/<cwd-slug>/<sid>.jsonl``.
    Resume/fork look them up by the *current* cwd's slug, so a fork running in
    a scratch cwd can only branch the parent if the parent transcript also
    exists under the scratch slug. Best-effort: if the source isn't there yet
    (parent never flushed), resume falls back to a fresh session, same as the
    main agent's start() path.
    """
    import shutil
    src = _session_jsonl_path(parent_project_dir, parent_sid)
    dst = _session_jsonl_path(self._scratch_dir, parent_sid)
    try:
      if os.path.exists(src) and src != dst:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    except OSError as e:
      log.warning("fork transcript seed failed (%s) — fork starts fresh", e)

  def supports_steering(self) -> bool:
    return True

  async def steer(self, text: str) -> bool:
    """Fold a follow-up into the running turn — see CodingAgent.steer.

    Gated on a live turn: ``_current_on_event`` is set only while run_turn
    is in flight. If no turn is running a query() would start a brand-new
    turn out of band (not steering), so we decline and let the host queue.
    """
    if self._current_on_event is None:
      return False
    text = text.strip()
    if not text:
      return False
    return await self._sdk.steer(text, self._pending_steers)

  def supports_fork(self) -> bool:
    return True

  async def fork(
    self, parent_session_id: str, project_dir: str, model: str,
  ) -> "CodingAgent | None":
    """Spawn a started, read-only Claude fork — see CodingAgent.fork.

    Branches from ``parent_session_id`` (fork_session=True) so the fork sees
    the current context but writes to a throwaway transcript; runs sandboxed
    over a scratch cwd so it cannot modify ``project_dir``. Inherits this
    agent's credentials / endpoint / effort / system prompt so the fork hits
    the same model routing (and the parent's cached prompt prefix).
    """
    fork = ClaudeCodingAgent(
      self._credentials, self._chat_id, self._db, self._channel,
      permission_mode="bypassPermissions",
      system_prompt=self._system_prompt,
      endpoint=self._endpoint,
      read_only=True,
    )
    fork.set_effort(self._effort)
    await fork.start(
      project_dir, model, resume=parent_session_id,
      fork_session=bool(parent_session_id),
    )
    return fork

  def _reply_anchor_path(self) -> str:
    """Path of the fork's reply-anchor handoff file (in the scratch dir).
    Shared by _build_env (exports it to nemo-send) and bind_reply_anchor
    (writes it). Empty when this agent has no scratch dir (not a fork)."""
    if not self._scratch_dir:
      return ""
    return os.path.join(self._scratch_dir, ".nemo_reply_thread")

  async def bind_reply_anchor(self, anchor_msg_id: str) -> None:
    """Record the fork's sub-thread anchor so `nemo-send image/file` replies
    into the thread (see CodingAgent.bind_reply_anchor). Writes the anchor to
    the scratch-dir file that _build_env exported as NEMO_REPLY_THREAD_FILE.
    No-op without a scratch dir (i.e. not a read-only fork)."""
    path = self._reply_anchor_path()
    if not path:
      return
    try:
      with open(path, "w", encoding="utf-8") as f:
        f.write(anchor_msg_id)
    except OSError as e:
      log.warning("fork: failed to write reply anchor %s: %s", path, e)

  async def stop(self) -> None:
    await self._sdk.close_client()
    if self._sdk_started:
      self._sdk.stop()
      self._sdk_started = False
    if self._scratch_dir:
      import shutil
      shutil.rmtree(self._scratch_dir, ignore_errors=True)
      # Also drop the fork's per-cwd transcript dir (seeded parent copy +
      # the fork's own throwaway jsonl) so ~/.claude/projects doesn't grow a
      # dead slug per fork.
      slug_dir = os.path.dirname(_session_jsonl_path(self._scratch_dir, "x"))
      shutil.rmtree(slug_dir, ignore_errors=True)
      self._scratch_dir = ""

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
    # Standing nemo-vision note for text-only models (deepseek/kimi). Computed
    # from the live model so it refreshes on each reconnect / /model switch;
    # empty for vision models or when no helper is configured.
    from .agent_factory import model_media_vision
    from . import vision_cli
    vision_note = vision_cli.standing_hint(
      model_media_vision("claude", self._model).image)
    if vision_note:
      agent_prompt = f"{agent_prompt}\n\n{vision_note}"
    if self._system_prompt:
      agent_prompt = f"{agent_prompt}\n\n{self._system_prompt}"
    return agent_prompt

  def _build_env(self, project_dir: str, model: str) -> dict[str, str]:
    """Build the SDK subprocess env. Shared by main turns and side
    questions so a forked side-question subprocess sees the same
    endpoint/model routing (→ parent prompt-cache reuse)."""
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
    # Read-only fork: media from `nemo-send` must land in the fork's Lark
    # sub-thread, not the main chat. The thread doesn't exist yet (it's created
    # after this subprocess starts), so we point nemo-send at a file in the
    # scratch dir; bind_reply_anchor fills it with the thread anchor once Lark
    # assigns the thread. stop()'s rmtree of the scratch dir reaps it.
    if self._read_only and self._scratch_dir:
      env["NEMO_REPLY_THREAD_FILE"] = self._reply_anchor_path()
    for key in ("http_proxy", "https_proxy", "all_proxy"):
      val = os.environ.get(key)
      if val:
        env[key] = val

    # Endpoint passthrough: shell-exported ANTHROPIC_*/CLAUDE_CODE_* vars
    # need to reach the SDK subprocess, since this env dict starts blank
    # rather than inheriting os.environ.
    for key in _CLAUDE_ENDPOINT_ENV_KEYS:
      val = os.environ.get(key)
      if val:
        env[key] = val

    # Explicit --base-url / --api-key flags overlay on top of shell env.
    if self._endpoint.base_url:
      env["ANTHROPIC_BASE_URL"] = self._endpoint.base_url
    if self._endpoint.api_key:
      env["ANTHROPIC_AUTH_TOKEN"] = self._endpoint.api_key

    # When pointing at a third-party Anthropic-compatible endpoint, the
    # remote almost certainly does not understand the canonical Claude
    # slugs ("claude-opus-4-7", etc.) — neither for the primary request
    # nor for Claude Code's internal subagent / tier routing. Fan the
    # user-supplied --model out to every routing knob so all internal
    # paths use the same third-party slug. setdefault preserves any
    # explicit overrides the user already set in their shell env.
    if self._endpoint.base_url and model:
      env.setdefault("ANTHROPIC_MODEL", model)
      env.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", model)
      env.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", model)
      env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", model)
      env.setdefault("CLAUDE_CODE_SUBAGENT_MODEL", model)
    return env

  def _build_fork_options(
    self, project_dir: str, model: str, resume: str = "",
  ) -> object:
    """Options for a read-only fork (see _FORK_DIRECTIVE).

    cwd is a private scratch dir, NOT the project — the OS bash sandbox makes
    cwd writable but blocks writes everywhere else, so keeping the project
    outside the workspace (and out of add_dirs, which would re-add it as
    writable) makes it physically read-only while reads/Read/Grep/Glob still
    work on absolute project paths. File-mutating tools are disallowed
    outright since they bypass the bash sandbox. Runs bypassPermissions: with
    writes impossible there is nothing dangerous to gate, so no permission
    cards are needed in the fork sub-thread.
    """
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from claude_agent_sdk.types import PermissionMode

    agent_prompt = (
      f"{self._build_agent_prompt()}\n\n"
      + _FORK_DIRECTIVE.format(project=project_dir, scratch=self._scratch_dir)
    )

    def _stderr_handler(line: str) -> None:
      log.info("[fork-stderr] %s", line.rstrip())

    opts: dict[str, object] = dict(
      allowed_tools=list(_FORK_ALLOWED_TOOLS),
      disallowed_tools=list(_FORK_DISALLOWED_TOOLS),
      sandbox={"enabled": True, "autoAllowBashIfSandboxed": True},
      permission_mode=cast(PermissionMode, "bypassPermissions"),
      # Don't auto-load the project's hooks/settings into the fork — its cwd
      # is scratch anyway, and a fork should stay lean and side-effect-free.
      setting_sources=["user"],
      system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": agent_prompt,
      },
      cwd=self._scratch_dir,
      model=model,
      env=self._build_env(project_dir, model),
      stderr=_stderr_handler,
      hooks={
        "PreCompact": [HookMatcher(hooks=[self._on_pre_compact])],
      },
      max_buffer_size=16 * 1024 * 1024,
    )
    if resume:
      opts["resume"] = resume
      # Branch from the parent transcript on first connect (and on any
      # reconnect that still resumes the parent); once the fork's own
      # session id materialises, resume != parent and we stop re-forking.
      if self._fork_parent_id and resume == self._fork_parent_id:
        opts["fork_session"] = True
    if self._effort:
      opts["effort"] = self._effort
    return ClaudeAgentOptions(**opts)

  def _build_options(self, project_dir: str, model: str, resume: str = "") -> object:
    if self._read_only:
      return self._build_fork_options(project_dir, model, resume=resume)

    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from claude_agent_sdk.types import PermissionMode

    agent_prompt = self._build_agent_prompt()

    env = self._build_env(project_dir, model)

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
        "Agent", "Skill", "Read", "Write", "Edit", "NotebookEdit",
        "Bash", "BashOutput", "KillShell",
        "Glob", "Grep", "WebSearch", "WebFetch", "TodoWrite",
        "AskUserQuestion",
        "mcp__*",
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
      # PreCompact fires before the CLI starts summarising context. Without
      # it, the user sees a silent 10-60s working-card stall mid-turn; with
      # it we surface CompactStartedEvent and the host can render a banner.
      # The matching completion arrives on the message stream as
      # SystemMessage(subtype="compact_boundary"); see turn.py.
      hooks={
        "PreCompact": [HookMatcher(hooks=[self._on_pre_compact])],
      },
      can_use_tool=_can_use_tool,
      # Single stream_json messages can exceed the SDK's 1 MB default when
      # a tool result (large file read, heavy bash output) lands in one chunk.
      # Raise to 16 MB so the reader doesn't kill the turn with a decode error.
      max_buffer_size=16 * 1024 * 1024,
    )
    if resume:
      opts["resume"] = resume
    if self._effort:
      opts["effort"] = self._effort

    return ClaudeAgentOptions(**opts)
