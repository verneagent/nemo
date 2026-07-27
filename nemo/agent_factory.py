"""Coding-agent factory.

"Agent" here means the coding-agent runtime / harness — Claude Agent SDK,
Codex CLI, or OpenCode — selected by the daemon's ``--agent`` flag and
runtime ``/agent`` command. (This is intentionally distinct from "model
provider" in the models.json schema — see nemo/presets.py; config lives in
~/.nemo/models.json — where ``provider`` groups models that share the same
upstream gateway like DeepSeek or Kimi.)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Literal, get_args

from .channel import Channel
from .claude_agent import ClaudeCodingAgent
from .claude_cli_agent import ClaudeCliCodingAgent
from .coding_agent import CodingAgent, EndpointConfig
from .codex_agent import CodexCodingAgent
from .db import Database
from .opencode_agent import OpenCodeCodingAgent

type AgentKind = Literal["claude", "codex", "opencode", "claude-cli"]

AGENT_KINDS: frozenset[str] = frozenset(get_args(AgentKind.__value__))

DEFAULT_AGENT: AgentKind = "claude"

__all__ = [
  "AgentKind",
  "AGENT_KINDS",
  "DEFAULT_AGENT",
  "EndpointConfig",
  "ModelCatalog",
  "build_coding_agent",
  "default_model_for_agent",
  "is_model_compatible",
  "model_catalog_for_agent",
  "query_codex_model_catalog",
  "resolve_boot_model",
  "MediaVision",
  "model_media_vision",
]

_DEFAULT_MODEL_BY_AGENT: dict[AgentKind, str] = {
  "claude": "claude-opus-5",
  # claude-cli drives the interactive TUI; same model slugs as claude.
  "claude-cli": "claude-opus-5",
  # Kept as the startup default. The interactive /model catalog is read
  # dynamically from `codex debug models`.
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
  - ``aliases``: short name → canonical full slug (e.g. ``opus`` → ``claude-opus-5``).
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
_CATALOG_BY_AGENT: dict[AgentKind, ModelCatalog] = {
  "claude": ModelCatalog(
    visible=(
      "claude-fable-5",
      "claude-opus-5",
      "claude-sonnet-5",
      "claude-haiku-4-5",
      "opusplan",
    ),
    hidden=(
      "claude-opus-4-8",
      "claude-opus-4-7",
      "claude-opus-4-6",
      "claude-sonnet-4-6",
      "claude-sonnet-4-5",
    ),
    aliases={
      "fable": "claude-fable-5",
      "opus": "claude-opus-5",
      "sonnet": "claude-sonnet-5",
      "haiku": "claude-haiku-4-5",
    },
  ),
  "opencode": ModelCatalog(
    note=(
      "Dynamic models come from the local OpenCode config. Use full "
      "`provider/model` names from `opencode models`, or `default` to keep "
      "OpenCode's configured default."
    ),
  ),
}

# claude-cli (pty-driven interactive TUI) accepts the same model slugs as the
# SDK-backed claude agent — it's the same binary, just driven differently.
_CATALOG_BY_AGENT["claude-cli"] = _CATALOG_BY_AGENT["claude"]


def default_model_for_agent(agent: AgentKind) -> str:
  return _DEFAULT_MODEL_BY_AGENT[agent]


def _preset_names_for_agent(agent: AgentKind) -> tuple[str, ...]:
  """Names of model presets whose endpoints are populated for ``agent``."""
  from .presets import load_presets
  return tuple(
    name for name, p in load_presets().items() if p.supports(agent)
  )


def _codex_model_sort_key(item: object, index: int) -> tuple[int, int]:
  if not isinstance(item, dict):
    return (1_000_000, index)
  priority = item.get("priority")
  if isinstance(priority, int):
    return (priority, index)
  return (1_000_000, index)


def query_codex_model_catalog() -> ModelCatalog:
  """Read Codex's live model catalog via `codex debug models`."""
  try:
    result = subprocess.run(
      ["codex", "debug", "models"],
      capture_output=True,
      text=True,
      timeout=10,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    return ModelCatalog(note=f"Codex model catalog unavailable: {exc}")
  if result.returncode != 0:
    detail = (result.stderr or result.stdout).strip()
    suffix = f": {detail}" if detail else ""
    return ModelCatalog(note=f"Codex model catalog unavailable{suffix}")
  try:
    parsed = json.loads(result.stdout)
  except json.JSONDecodeError as exc:
    return ModelCatalog(note=f"Codex model catalog unreadable: {exc}")
  if not isinstance(parsed, dict):
    return ModelCatalog(note="Codex model catalog unreadable: expected object")
  raw_models = parsed.get("models")
  if not isinstance(raw_models, list):
    return ModelCatalog(note="Codex model catalog unreadable: missing models")

  visible: list[str] = []
  hidden: list[str] = []
  for _, item in sorted(
    enumerate(raw_models),
    key=lambda pair: _codex_model_sort_key(pair[1], pair[0]),
  ):
    if not isinstance(item, dict):
      continue
    slug = item.get("slug")
    if not isinstance(slug, str) or not slug:
      continue
    if item.get("visibility") == "list":
      visible.append(slug)
  return ModelCatalog(
    visible=tuple(dict.fromkeys(visible)),
    hidden=tuple(dict.fromkeys(hidden)),
    note="Dynamic models from `codex debug models`.",
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
  if agent == "codex":
    base = query_codex_model_catalog()
    presets = _preset_names_for_agent(agent)
    if not presets:
      return base
    existing = set(base.all_names())
    extra = tuple(p for p in presets if p not in existing)
    return ModelCatalog(
      visible=base.visible + extra,
      api_only=base.api_only,
      hidden=base.hidden,
      aliases=base.aliases,
      note=base.note,
    )
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


def _catalog_is_authoritative(agent: AgentKind, project_dir: str = "") -> bool:
  """Whether ``agent``'s model list can be trusted to be complete.

  claude / claude-cli ship a static compiled-in catalog, so it always is.
  codex and opencode read theirs from an external CLI (`codex debug
  models`, the opencode config); when that call fails the catalog
  degrades to just the models.json presets, which says nothing about
  whether a native slug still exists.
  """
  if agent in ("claude", "claude-cli"):
    return True
  if agent == "codex":
    return bool(query_codex_model_catalog().visible)
  if agent == "opencode":
    from .opencode_agent import query_opencode_model_catalog_data
    models, _ = query_opencode_model_catalog_data(project_dir)
    return bool(models)
  return False


def resolve_boot_model(
  agent: AgentKind,
  model: str,
  project_dir: str = "",
) -> tuple[str, str]:
  """Resolve the model a daemon should actually boot with.

  Returns ``(model, notice)``. A model retired upstream (Claude/Codex
  rotate slugs) otherwise boots fine and only fails on the first turn,
  which reads as "restart succeeded, then the bot went silent" — so fall
  back to the agent default and hand back a notice for the start card.
  Only downgrades when the catalog is authoritative; an unreadable
  catalog is not evidence that the model is gone.
  """
  if is_model_compatible(agent, model, project_dir):
    return model, ""
  if not _catalog_is_authoritative(agent, project_dir):
    return model, ""
  fallback = default_model_for_agent(agent)
  return fallback, (
    f"Model `{model}` is unavailable for {agent} — fell back to `{fallback}`"
  )


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
  if agent == "claude-cli":
    return ClaudeCliCodingAgent(
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
