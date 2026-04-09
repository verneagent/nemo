"""Provider-based CodingAgent factory."""

from __future__ import annotations

from typing import Literal

from .channel import Channel
from .claude_agent import ClaudeCodingAgent
from .coding_agent import CodingAgent
from .codex_agent import CodexCodingAgent
from .db import Database

type AgentProvider = Literal["claude", "codex"]

DEFAULT_PROVIDER: AgentProvider = "claude"

_DEFAULT_MODEL_BY_PROVIDER: dict[AgentProvider, str] = {
  "claude": "claude-opus-4-6",
  "codex": "gpt-5-codex",
}


def default_model_for_provider(provider: AgentProvider) -> str:
  return _DEFAULT_MODEL_BY_PROVIDER[provider]


def is_model_compatible(provider: AgentProvider, model: str) -> bool:
  normalized = model.strip().lower()
  if not normalized:
    return False
  if provider == "claude":
    return not normalized.startswith(("gpt-", "o1", "o3", "o4", "codex"))
  if provider == "codex":
    return not normalized.startswith("claude-")
  return False


def build_coding_agent(
  provider: AgentProvider,
  credentials: dict[str, str],
  chat_id: str,
  db: Database,
  channel: Channel,
  *,
  permission_mode: str = "bypassPermissions",
) -> CodingAgent:
  if provider == "claude":
    return ClaudeCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
    )
  if provider == "codex":
    return CodexCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
    )
  raise ValueError(f"Unsupported provider: {provider}")
