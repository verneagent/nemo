"""Provider-based CodingAgent factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .channel import Channel
from .claude_agent import ClaudeCodingAgent
from .coding_agent import CodingAgent, EndpointConfig
from .codex_agent import CodexCodingAgent
from .db import Database
from .opencode_agent import OpenCodeCodingAgent

type AgentProvider = Literal["claude", "codex", "opencode"]

DEFAULT_PROVIDER: AgentProvider = "claude"

__all__ = [
  "AgentProvider",
  "DEFAULT_PROVIDER",
  "EndpointConfig",
  "ModelCatalog",
  "build_coding_agent",
  "default_model_for_provider",
  "is_model_compatible",
  "model_catalog_for_provider",
]

_DEFAULT_MODEL_BY_PROVIDER: dict[AgentProvider, str] = {
  "claude": "claude-opus-4-7",
  # gpt-5.5 is the current top-priority codex model — works for both
  # ChatGPT subscribers and API users. The codex-specialized slugs
  # (-codex variants) are API-only and return HTTP 400 for ChatGPT
  # accounts, so they make a poor default.
  "codex": "gpt-5.5",
  # OpenCode resolves the configured default model on its side.
  "opencode": "default",
}


@dataclass(frozen=True)
class ModelCatalog:
  """Model catalog for a provider.

  - ``visible``: full slugs shown in the picker.
  - ``api_only``: full slugs that require API auth (ChatGPT subscribers
    can't use these — e.g. codex-specialized variants). Still accepted
    by the picker; rendered in a separate help section.
  - ``hidden``: full slugs accepted but not shown (legacy / experimental).
  - ``aliases``: short name → canonical full slug (e.g. ``opus`` → ``claude-opus-4-7``).
  """
  visible: tuple[str, ...] = ()
  api_only: tuple[str, ...] = ()
  hidden: tuple[str, ...] = ()
  aliases: dict[str, str] = field(default_factory=dict)
  note: str = ""

  def all_names(self) -> tuple[str, ...]:
    return (
      self.visible + self.api_only + self.hidden
      + tuple(self.aliases.keys())
    )


# Claude aliases mirror the Claude CLI's /model picker.
# Codex slugs come from github.com/openai/codex `models-manager/models.json`.
# Older bundled `codex` binaries may reject newer slugs at turn time.
_CATALOG_BY_PROVIDER: dict[AgentProvider, ModelCatalog] = {
  "claude": ModelCatalog(
    visible=(
      "claude-opus-4-7",
      "claude-sonnet-4-6",
      "claude-haiku-4-5",
      "opusplan",
    ),
    hidden=(
      "claude-opus-4-6",
    ),
    aliases={
      "opus": "claude-opus-4-7",
      "sonnet": "claude-sonnet-4-6",
      "haiku": "claude-haiku-4-5",
    },
  ),
  "codex": ModelCatalog(
    # Slugs from the codex CLI's `~/.codex/models_cache.json`
    # (priority-ordered, all marked supported_in_api). gpt-5.5 is the
    # current default-tier coding model. Older 5.0 / 5.1 / -codex
    # variants OpenAI dropped from the live list have been removed —
    # passing them at /model would just 400.
    visible=(
      "gpt-5.5",
      "gpt-5.4",
      "gpt-5.4-mini",
      "gpt-5.2",
    ),
    # API-only codex-specialized slug — works on API auth, returns 400
    # ("not supported when using Codex with a ChatGPT account") for
    # ChatGPT subscribers.
    api_only=(
      "gpt-5.3-codex",
    ),
    hidden=(
      # gpt-5.3-codex-spark is ChatGPT-only (supported_in_api: false in
      # the cache), kept here so /model accepts it for users who have it.
      "gpt-5.3-codex-spark",
      "gpt-oss-120b",
      "gpt-oss-20b",
    ),
    aliases={},
  ),
  "opencode": ModelCatalog(
    note=(
      "Dynamic models come from the local OpenCode config. Use full "
      "`provider/model` names from `opencode models`, or `default` to keep "
      "OpenCode's configured default."
    ),
  ),
}


def default_model_for_provider(provider: AgentProvider) -> str:
  return _DEFAULT_MODEL_BY_PROVIDER[provider]


def model_catalog_for_provider(
  provider: AgentProvider,
  project_dir: str = "",
) -> ModelCatalog:
  if provider == "opencode":
    from .opencode_agent import query_opencode_model_catalog_data
    models, note = query_opencode_model_catalog_data(project_dir)
    return ModelCatalog(visible=models, note=note)
  return _CATALOG_BY_PROVIDER.get(provider, ModelCatalog())


def is_model_compatible(
  provider: AgentProvider,
  model: str,
  project_dir: str = "",
) -> bool:
  if provider == "opencode":
    candidate = model.strip().lower()
    if candidate == "default":
      return True
    catalog = model_catalog_for_provider(provider, project_dir)
    if catalog.visible:
      return candidate in catalog.all_names()
    return "/" in candidate
  return model.strip().lower() in model_catalog_for_provider(
    provider,
    project_dir,
  ).all_names()


def build_coding_agent(
  provider: AgentProvider,
  credentials: dict[str, str],
  chat_id: str,
  db: Database,
  channel: Channel,
  *,
  permission_mode: str = "bypassPermissions",
  system_prompt: str = "",
  endpoint: EndpointConfig | None = None,
) -> CodingAgent:
  endpoint = endpoint or EndpointConfig()
  if provider == "claude":
    return ClaudeCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
      endpoint=endpoint,
    )
  if provider == "codex":
    return CodexCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
      endpoint=endpoint,
    )
  if provider == "opencode":
    return OpenCodeCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
      endpoint=endpoint,
    )
  raise ValueError(f"Unsupported provider: {provider}")
