"""Abstract Agent interface — how Nemo runs coding tasks.

An Agent handles turn execution: running prompts, streaming events,
interrupting running turns, and resetting conversation state.

Claude Agent SDK is the first implementation. The core orchestration
depends only on this interface, not on Claude SDK specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .turn import TurnEvent
from .types import JsonObject


@dataclass(frozen=True)
class EndpointConfig:
  """Custom endpoint override shared across providers.

  The CLI exposes ``--base-url`` / ``--api-key`` as provider-agnostic flags;
  each adapter translates these into its own vendor-specific env vars
  (``ANTHROPIC_BASE_URL`` for Claude, ``OPENAI_BASE_URL`` for Codex, etc.).

  Empty strings mean "leave the adapter's default behavior alone" — i.e.
  let Claude SDK / Codex CLI / OpenCode resolve credentials from their
  usual sources (login session, ``~/.codex/config.toml``, ``opencode.json``).
  """
  base_url: str = ""
  api_key: str = ""


class CodingAgent(ABC):
  """Abstract coding agent that executes turns."""

  @abstractmethod
  async def run_turn(
    self,
    prompt: str,
    on_event: Callable[[TurnEvent], None],
  ) -> tuple[float, JsonObject]:
    """Execute one agent turn with the given prompt.

    Calls on_event for each TurnEvent (tool use, text output, done, error).
    Returns when the turn completes.
    """
    ...

  @abstractmethod
  async def interrupt(self) -> None:
    """Cancel the currently running turn."""
    ...

  @abstractmethod
  async def start(self, project_dir: str, model: str, resume: str = "") -> None:
    """Initialize the agent (connect to SDK, etc.)."""
    ...

  @abstractmethod
  async def reset(self, project_dir: str, model: str, resume: str = "") -> None:
    """Reset or reconnect the agent with updated runtime settings."""
    ...

  @abstractmethod
  async def stop(self) -> None:
    """Shut down the agent and release resources."""
    ...

  def set_effort(self, effort: str) -> None:
    """Set reasoning effort for subsequent turns.

    `effort` is one of "", "low", "medium", "high", "max". Empty string
    clears the setting. Adapters whose backend tops out below `max`
    should clamp it down (e.g. Codex/OpenCode → `high`). Default is no-op.

    Effort is stored in the adapter; the host typically calls `reset()`
    afterwards so the new value flows into the next turn.
    """
    del effort

  def set_endpoint(self, endpoint: EndpointConfig) -> None:
    """Replace the active endpoint config (base_url + api_key).

    Triggered by ``/model`` switching to/from a preset that points at a
    different endpoint than the current daemon was started with. The
    host calls ``reset()`` after this so the new env vars propagate
    into the SDK subprocess on its next reconnect.
    """
    del endpoint

  async def side_question(self, question: str, sdk_session_id: str) -> str:
    """Answer a read-only, ephemeral side question ("by the way…").

    Semantics mirror Claude Code's ``/btw``: the answer is produced
    against the *current* conversation context but MUST NOT mutate it —
    no tools, a single response, never written back to the session
    transcript — and it MUST run independently of any turn that may be
    executing concurrently (the caller does not interrupt the turn).

    `sdk_session_id` is the live session to read context from. Returns
    the answer text, or "" if this adapter does not support side
    questions (the caller surfaces a "not supported" note). Default is
    unsupported, so Codex / OpenCode inherit the no-op.
    """
    del question, sdk_session_id
    return ""

  async def forward_native_command(self, command: str) -> str:
    """Forward a CLI-native slash command (e.g. ``/compact``, ``/usage``) to the
    underlying agent and return its rendered result as text.

    Only the claude-cli (pty TUI) adapter implements this — it types the command
    into the live TUI and scrapes the result. SDK/headless adapters have no such
    surface, so the default is a no-op returning "" (the host then reports the
    command as unsupported for that agent).
    """
    del command
    return ""

  async def digest_transcript(self, transcript_path: str, fmt_hint: str) -> str:
    """Summarise a past session transcript in a FRESH, context-free session.

    Used by ``/session recall``: instead of handing the *live* agent the
    raw JSONL path and letting it read chunks into its own working context
    (verbose, non-deterministic, pollutes the conversation), a throwaway
    read-only session reads the transcript and returns a compact structured
    summary. Only that summary enters the main agent's context; the caller
    still hands it the path so it can Read specific slices on demand.

    ``transcript_path`` is the absolute path to the JSONL. ``fmt_hint``
    describes the on-disk event shape (it depends on which agent *wrote*
    the transcript, not on this agent). Returns the summary text, or ""
    when this adapter can't run a blank side session (the caller then
    falls back to the inline "read the file yourself" recall). Default is
    unsupported, so Codex / OpenCode inherit the no-op.
    """
    del transcript_path, fmt_hint
    return ""

  def supports_fork(self) -> bool:
    """True if this adapter can spawn a read-only forked sub-session.

    Forking branches the *current* conversation context into an
    independent, multi-turn, tool-enabled sub-session that is physically
    unable to modify the project (see ``fork``). Claude (SDK
    ``fork_session`` + bash sandbox) and Codex (rollout-copy +
    ``sandboxMode=read-only``) support it; OpenCode inherits ``False``.
    """
    return False

  async def fork(
    self, parent_session_id: str, project_dir: str, model: str,
  ) -> "CodingAgent | None":
    """Spawn a read-only forked sub-agent for a `/fork` sub-thread.

    The returned agent is started and ready to ``run_turn``: it branches
    from ``parent_session_id`` (so it sees the current context, ephemeral
    — nothing written back), exposes tools but is sandboxed so it CANNOT
    modify ``project_dir``. Returns ``None`` when unsupported (the caller
    surfaces a "Claude-only" note). Caller owns the lifecycle and must
    ``stop()`` the fork to release its subprocess + scratch dir.
    """
    del parent_session_id, project_dir, model
    return None

  def trailing_note(self, sdk_session_id: str) -> str:
    """Return an optional markdown note to append to the final turn response.

    Concrete adapters use this to surface provider-specific health signals
    (e.g. "session context is getting large, consider /clear"). Default
    returns empty string — no note appended.
    """
    del sdk_session_id
    return ""

  async def bind_reply_anchor(self, anchor_msg_id: str) -> None:
    """Bind a Lark message id as the anchor for out-of-band media sends.

    A `/fork` runs in a Lark sub-thread, so its `nemo-send image/file` must
    reply into that thread, not the main chat. The thread is created AFTER the
    fork's SDK subprocess starts (its env is already frozen), so the anchor is
    handed over here once Lark assigns the thread. Only the read-only fork
    adapter (Claude) wires this; everything else inherits the no-op (the main
    conversation posts media to the chat root, and Codex/OpenCode forks don't
    route nemo-send into a thread)."""
    del anchor_msg_id
