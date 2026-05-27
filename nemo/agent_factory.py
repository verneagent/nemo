"""Coding-agent factory.

"Agent" here means the coding-agent runtime / harness — Claude Agent SDK,
Codex CLI, or OpenCode — selected by the daemon's ``--agent`` flag and
runtime ``/agent`` command. (This is intentionally distinct from "model
provider" in the models.json schema — see nemo/presets.py; config lives in
~/.nemo/models.json — where ``provider`` groups models that share the same
upstream gateway like DeepSeek or Kimi.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .channel import Channel
from .claude_agent import ClaudeCodingAgent
from .coding_agent import CodingAgent, EndpointConfig
from .codex_agent import CodexCodingAgent
from .db import Database
from .opencode_agent import OpenCodeCodingAgent

type AgentKind = Literal["claude", "codex", "opencode"]

DEFAULT_AGENT: AgentKind = "claude"

__all__ = [
  "AgentKind",
  "DEFAULT_AGENT",
  "EndpointConfig",
  "ModelCatalog",
  "build_coding_agent",
  "default_model_for_agent",
  "is_model_compatible",
  "model_catalog_for_agent",
  "MediaVision",
  "model_media_vision",
]

_DEFAULT_MODEL_BY_AGENT: dict[AgentKind, str] = {
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
  """Model catalog for an agent kind.

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
_CATALOG_BY_AGENT: dict[AgentKind, ModelCatalog] = {
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


def default_model_for_agent(agent: AgentKind) -> str:
  return _DEFAULT_MODEL_BY_AGENT[agent]


def _preset_names_for_agent(agent: AgentKind) -> tuple[str, ...]:
  """Names of model presets whose endpoints are populated for ``agent``."""
  from .presets import load_presets
  return tuple(
    name for name, p in load_presets().items() if p.supports(agent)
  )


def model_catalog_for_agent(
  agent: AgentKind,
  project_dir: str = "",
) -> ModelCatalog:
  if agent == "opencode":
    from .opencode_agent import query_opencode_model_catalog_data
    models, note = query_opencode_model_catalog_data(project_dir)
    presets = _preset_names_for_agent(agent)
    visible = tuple(dict.fromkeys((*models, *presets)))
    return ModelCatalog(visible=visible, note=note)
  base = _CATALOG_BY_AGENT.get(agent, ModelCatalog())
  presets = _preset_names_for_agent(agent)
  if not presets:
    return base
  # Drop any preset name that already lives in the static catalog so
  # /model listing doesn't duplicate.
  existing = set(base.all_names())
  extra = tuple(p for p in presets if p not in existing)
  return ModelCatalog(
    visible=base.visible + extra,
    api_only=base.api_only,
    hidden=base.hidden,
    aliases=base.aliases,
    note=base.note,
  )


def is_model_compatible(
  agent: AgentKind,
  model: str,
  project_dir: str = "",
) -> bool:
  candidate = model.strip().lower()
  # Preset names expand to agent-specific endpoints, so they're
  # compatible iff the preset itself supports this agent kind — short-
  # circuit before falling back to the static catalog.
  from .presets import resolve_preset
  preset = resolve_preset(candidate)
  if preset is not None:
    return preset.supports(agent)
  if agent == "opencode":
    if candidate == "default":
      return True
    catalog = model_catalog_for_agent(agent, project_dir)
    if candidate in catalog.all_names():
      return True
    # OpenCode resolves the actual provider at the SDK layer, and the
    # live catalog isn't always reachable (pytest, fresh installs).
    # Accept any ``provider/model`` slug as a compatibility fallback.
    return "/" in candidate
  return candidate in model_catalog_for_agent(
    agent,
    project_dir,
  ).all_names()


@dataclass(frozen=True)
class MediaVision:
  """Which media inputs a model can natively see.

  ``image`` and ``video`` are independent: Claude/Codex see images (via Read /
  view_image) but not video; a text-only model (deepseek/kimi) sees neither; a
  multimodal preset may see both. A False axis means the corresponding
  ``[image:]`` / ``[video:]`` marker is routed to the nemo-vision shell tool.
  """
  image: bool = False
  video: bool = False


def model_media_vision(agent: AgentKind, model: str) -> MediaVision:
  """Native media support for ``model`` under ``agent``.

  Presets declare it in models.json's ``vision`` block (default text-only).
  A non-preset is the agent's own built-in model: Claude (Read) and Codex
  (view_image) ingest images natively, none ingest video, and OpenCode's
  default is typically a frontier model — so assume image-yes / video-no.
  """
  from .presets import load_presets, resolve_preset
  preset = resolve_preset(model)
  if preset is None:
    # After a preset /model switch the live model is the resolved remote id
    # (e.g. "deepseek-v4-pro[1m]"), not the preset name — resolve_preset only
    # matches names, so map the remote id back to its preset here. Without
    # this a text-only preset reads as a vision model and gets no nemo-vision
    # hint.
    for candidate in load_presets().values():
      if candidate.remote_for(agent) == model:
        preset = candidate
        break
  if preset is not None:
    return MediaVision(image=preset.sees_image, video=preset.sees_video)
  return MediaVision(image=True, video=False)


def build_coding_agent(
  agent: AgentKind,
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
  if agent == "claude":
    return ClaudeCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
      endpoint=endpoint,
    )
  if agent == "codex":
    return CodexCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
      endpoint=endpoint,
    )
  if agent == "opencode":
    return OpenCodeCodingAgent(
      credentials, chat_id, db, channel,
      permission_mode=permission_mode,
      system_prompt=system_prompt,
      endpoint=endpoint,
    )
  raise ValueError(f"Unsupported agent: {agent}")
