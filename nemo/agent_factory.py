"""Provider-based CodingAgent factory."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ModelCatalog:
  """Model catalog for a provider.

  - ``visible``: full slugs shown in the picker.
  - ``hidden``: full slugs accepted but not shown (legacy / experimental).
  - ``aliases``: short name → canonical full slug (e.g. ``opus`` → ``claude-opus-4-6``).
  """
  visible: tuple[str, ...] = ()
  hidden: tuple[str, ...] = ()
  aliases: dict[str, str] = field(default_factory=dict)

  def all_names(self) -> tuple[str, ...]:
    return self.visible + self.hidden + tuple(self.aliases.keys())


# Claude aliases mirror the Claude CLI's /model picker.
# Codex slugs come from github.com/openai/codex `models-manager/models.json`.
# Older bundled `codex` binaries may reject newer slugs at turn time.
_CATALOG_BY_PROVIDER: dict[AgentProvider, ModelCatalog] = {
  "claude": ModelCatalog(
    visible=(
      "claude-opus-4-6",
      "claude-sonnet-4-6",
      "claude-haiku-4-5",
      "opusplan",
    ),
    hidden=(),
    aliases={
      "opus": "claude-opus-4-6",
      "sonnet": "claude-sonnet-4-6",
      "haiku": "claude-haiku-4-5",
    },
  ),
  "codex": ModelCatalog(
    visible=(
      "gpt-5.4",
      "gpt-5.3-codex",
      "gpt-5.2-codex",
      "gpt-5.2",
      "gpt-5.1-codex-max",
      "gpt-5.1-codex-mini",
    ),
    hidden=(
      "gpt-5.1-codex",
      "gpt-5.1",
      "gpt-5-codex",
      "gpt-5-codex-mini",
      "gpt-5",
      "gpt-oss-120b",
      "gpt-oss-20b",
    ),
    aliases={},
  ),
}


def default_model_for_provider(provider: AgentProvider) -> str:
  return _DEFAULT_MODEL_BY_PROVIDER[provider]


def model_catalog_for_provider(provider: AgentProvider) -> ModelCatalog:
  return _CATALOG_BY_PROVIDER.get(provider, ModelCatalog())


def is_model_compatible(provider: AgentProvider, model: str) -> bool:
  return model.strip().lower() in model_catalog_for_provider(provider).all_names()


def build_coding_agent(
  provider: AgentProvider,
  credentials: dict[str, str],
  chat_id: str,
  db: Database,
  channel: Channel,
  *,
  permission_mode: str = "bypassPermissions",
  system_prompt: str = "",
) -> CodingAgent:
  if provider == "claude":
    return ClaudeCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
    )
  if provider == "codex":
    return CodexCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
    )
  raise ValueError(f"Unsupported provider: {provider}")
