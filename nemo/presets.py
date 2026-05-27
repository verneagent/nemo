"""Model preset registry.

A *model preset* is a logical model name (e.g. ``deepseek-v4-pro``) that
expands at startup or on ``/model`` switch into:

  - the endpoint URL for the active coding agent's wire protocol
    (Anthropic-format for ``--agent claude``, OpenAI-format for
    ``--agent codex``),
  - the API key, resolved from an environment variable so secrets stay
    out of argv / config dumps,
  - the actual model id sent to the remote (which may differ between
    protocols — e.g. DeepSeek advertises ``deepseek-v4-pro`` on its
    OpenAI endpoint but ``deepseek-v4-pro[1m]`` on its Anthropic
    endpoint).

The on-disk schema is provider-grouped JSON so multiple models that
share a base URL / API key don't have to repeat themselves::

    {
      "providers": {
        "deepseek": {
          "anthropic": {
            "baseURL": "https://api.deepseek.com/anthropic",
            "apiKey": "{env:DEEPSEEK_API_KEY}"
          },
          "openai": {
            "baseURL": "https://api.deepseek.com",
            "apiKey": "{env:DEEPSEEK_API_KEY}"
          },
          "models": {
            "deepseek-v4-pro": {
              "anthropic": { "remote": "deepseek-v4-pro[1m]" }
            },
            "deepseek-v4-flash": {}
          }
        }
      }
    }

A model is **callable under a given agent kind** iff its parent provider
declares the matching protocol block. A model's per-protocol ``remote``
override falls back to the model id itself when omitted.

An optional ``vision`` block declares the native media input the model
accepts — ``{ "image": bool, "video": bool }`` — so incoming media that the
model can't see gets routed to the ``nemo-vision`` shell tool instead. It may
sit at the provider level (a default for its models) and/or per-model (a
field-by-field override). Omitted entirely → text-only (image+video False),
which is the right default for coding presets like deepseek/kimi::

        "qwen": {
          "openai": { "baseURL": "…", "apiKey": "{env:DASHSCOPE_API_KEY}" },
          "vision": { "image": true, "video": false },
          "models": {
            "qwen-vl-max": { "vision": { "image": true, "video": true } },
            "qwen-text":   { "vision": { "image": false } }
          }
        }

Sources, in order of precedence (later overrides earlier):
  1. ``nemo/models.json`` — a default ``builtin_path`` hook that the package
     does NOT ship (the file is absent → reads as ``{}``). Provider/model
     config (endpoints, API-key env, remote slugs) is *deployment* config, not
     package code, so the published package carries no presets and no
     opinionated endpoints. (Drop a file here only if a fork wants to ship
     defaults.)
  2. ``~/.nemo/models.json`` — where presets actually live. Same schema,
     merged at the provider level (a user-defined provider entry fully
     replaces the base's entry for that name; entries it doesn't touch pass
     through). Each deployment defines its own here; without it there are no
     third-party presets (plain Anthropic model ids still work — they need no
     preset).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .coding_agent import EndpointConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal flat record — one per (provider, model) pair after flattening.
# Public API (resolve_preset / endpoint_for / remote_for) is unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
  """A logical model name + how to reach it.

  Stays a flat per-model record after flattening the provider-grouped
  on-disk schema. Callers (agent_factory, agent.py, commands.py) treat
  this as the unit of /model dispatch.
  """
  name: str
  # `{env:VAR}` apiKeys keep secrets out of argv / ps / config dumps and are
  # preferred for shared/remote providers. `api_key_env` is the VAR name.
  api_key_env: str = ""
  # A literal apiKey written straight in models.json. Allowed because
  # ~/.nemo/models.json is user-private and never committed (the package
  # ships none) — convenient for local/self-hosted endpoints (e.g. an MLX
  # server). Takes precedence over `api_key_env` in endpoint_for. Empty
  # `api_key_env` + empty literal means no key (the agent's default auth).
  api_key_literal: str = ""
  # Anthropic-protocol endpoint. Empty → not callable via --agent claude
  # (or via opencode against an anthropic/* model).
  anthropic_url: str = ""
  anthropic_remote: str = ""
  # OpenAI-protocol endpoint. Empty → not callable via --agent codex
  # (or via opencode against an openai/* model).
  openai_url: str = ""
  openai_remote: str = ""
  # Native media input the model accepts, from the `vision` block. Drives
  # whether incoming [image:]/[video:] markers need a nemo-vision hint.
  # Third-party presets default to text-only (False/False) — declare the
  # block to opt a VL model out of the hint.
  sees_image: bool = False
  sees_video: bool = False

  def supports(self, agent: str) -> bool:
    if agent == "claude":
      return bool(self.anthropic_url)
    if agent == "codex":
      return bool(self.openai_url)
    if agent == "opencode":
      # OpenCode resolves the actual provider from the model slug
      # prefix at the SDK layer; either url being set is enough for us
      # to plumb env vars through.
      return bool(self.anthropic_url or self.openai_url)
    return False

  def remote_for(self, agent: str) -> str:
    """The model id to send downstream when the active agent is ``agent``."""
    if agent == "claude":
      return self.anthropic_remote or self.name
    if agent == "codex":
      return self.openai_remote or self.name
    return self.name

  def base_url_for(self, agent: str) -> str:
    if agent == "claude":
      return self.anthropic_url
    if agent == "codex":
      return self.openai_url
    if agent == "opencode":
      return self.anthropic_url or self.openai_url
    return ""

  def endpoint_for(self, agent: str) -> EndpointConfig:
    """Materialise the per-turn endpoint config for ``agent``.

    Returns an empty ``EndpointConfig`` (no overrides) when this preset
    doesn't apply to ``agent`` — caller should pre-check ``supports``.
    """
    base = self.base_url_for(agent)
    # A literal key wins; otherwise resolve the env var at use-time.
    api_key = self.api_key_literal or (
      os.environ.get(self.api_key_env, "") if self.api_key_env else "")
    return EndpointConfig(base_url=base, api_key=api_key)


# ---------------------------------------------------------------------------
# JSON loading + flattening
# ---------------------------------------------------------------------------


_ENV_PATTERN = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")


def _parse_api_key(raw: object, *, where: str) -> tuple[str, str]:
  """Parse an apiKey into ``(env_var_name, literal)`` — exactly one is set.

  ``{env:VARNAME}`` → (``VARNAME``, "") — resolved from the environment at
  use-time (preferred for shared/remote secrets). Any other non-empty string
  → ("", literal): a literal key, allowed because ~/.nemo/models.json is
  user-private and uncommitted (handy for local/self-hosted endpoints).
  Missing/blank/non-string → ("", "").
  """
  if raw is None or raw == "":
    return "", ""
  if not isinstance(raw, str):
    log.warning("%s: apiKey must be a string, got %s — ignoring",
                where, type(raw).__name__)
    return "", ""
  raw = raw.strip()
  m = _ENV_PATTERN.match(raw)
  if m:
    return m.group(1), ""
  return "", raw


def _protocol_block(provider_data: dict, key: str) -> tuple[str, str, str]:
  """Pull (baseURL, api_key_env, api_key_literal) out of one protocol section."""
  raw = provider_data.get(key)
  if not isinstance(raw, dict):
    return "", "", ""
  base = raw.get("baseURL", "")
  if not isinstance(base, str):
    log.warning("provider.%s.baseURL must be a string, got %s",
                key, type(base).__name__)
    base = ""
  api_key_env, api_key_literal = _parse_api_key(
    raw.get("apiKey"), where=f"provider.{key}.apiKey")
  return base, api_key_env, api_key_literal


def _model_remote(model_data: object, protocol: str) -> str:
  """Pull the per-protocol ``remote`` override from one model entry."""
  if not isinstance(model_data, dict):
    return ""
  sub = model_data.get(protocol)
  if not isinstance(sub, dict):
    return ""
  remote = sub.get("remote", "")
  if not isinstance(remote, str):
    log.warning("model.%s.remote must be a string, got %s",
                protocol, type(remote).__name__)
    return ""
  return remote


def _parse_vision(
  raw: object, default_image: bool, default_video: bool,
) -> tuple[bool, bool]:
  """Parse a ``vision`` block into ``(sees_image, sees_video)``.

  Missing block → inherit the passed defaults (provider default for a model,
  ``False`` for a provider). Each field falls back to its default when absent
  or non-boolean, so a model can override just ``image`` and keep the
  provider's ``video``.
  """
  if not isinstance(raw, dict):
    return default_image, default_video
  image = raw.get("image", default_image)
  video = raw.get("video", default_video)
  if not isinstance(image, bool):
    log.warning("vision.image must be a bool, got %s", type(image).__name__)
    image = default_image
  if not isinstance(video, bool):
    log.warning("vision.video must be a bool, got %s", type(video).__name__)
    video = default_video
  return image, video


def _flatten_providers(providers: dict) -> dict[str, Preset]:
  """Expand provider-grouped JSON into a flat {model_name: Preset} table."""
  out: dict[str, Preset] = {}
  for provider_name, pdata in providers.items():
    if not isinstance(pdata, dict):
      log.warning("provider %r: expected object, got %s — skipping",
                  provider_name, type(pdata).__name__)
      continue
    anthropic_url, anthropic_key_env, anthropic_key_lit = _protocol_block(pdata, "anthropic")
    openai_url, openai_key_env, openai_key_lit = _protocol_block(pdata, "openai")
    if not anthropic_url and not openai_url:
      # Provider declares models but no protocol blocks — flattening
      # would produce Presets with empty URLs that supports() rejects
      # for every agent. They'd vanish from /model without explanation,
      # so warn loudly and skip the whole provider.
      log.warning(
        "provider %r: no anthropic/openai block — models %s are unreachable, skipping",
        provider_name, list(pdata.get("models", {}).keys()),
      )
      continue
    # We expect each provider to use a single env var across protocols
    # (deepseek does — same DEEPSEEK_API_KEY on both endpoints). If they
    # differ, prefer the protocol the model actually exposes; if the
    # model exposes both, take anthropic's. This is a rare-enough edge
    # case that a hard error would be more annoying than useful.
    api_key_env = anthropic_key_env or openai_key_env
    api_key_literal = anthropic_key_lit or openai_key_lit
    # Provider-level media defaults (text-only unless declared); a model's
    # own `vision` block overrides field-by-field.
    prov_image, prov_video = _parse_vision(pdata.get("vision"), False, False)

    models = pdata.get("models", {})
    if not isinstance(models, dict):
      log.warning("provider %r: models must be an object, got %s — skipping",
                  provider_name, type(models).__name__)
      continue
    for model_name, mdata in models.items():
      if model_name in out:
        log.warning("model %r appears under multiple providers; later entry wins",
                    model_name)
      sees_image, sees_video = _parse_vision(
        mdata.get("vision") if isinstance(mdata, dict) else None,
        prov_image, prov_video)
      out[model_name] = Preset(
        name=str(model_name),
        api_key_env=api_key_env,
        api_key_literal=api_key_literal,
        anthropic_url=anthropic_url,
        anthropic_remote=_model_remote(mdata, "anthropic") or model_name,
        openai_url=openai_url,
        openai_remote=_model_remote(mdata, "openai") or model_name,
        sees_image=sees_image,
        sees_video=sees_video,
      )
  return out


def _read_json(path: str | os.PathLike) -> dict:
  try:
    with open(path, encoding="utf-8") as f:
      raw = json.load(f)
  except FileNotFoundError:
    return {}
  except (OSError, json.JSONDecodeError) as exc:
    log.warning("failed to read %s: %s", path, exc)
    return {}
  if not isinstance(raw, dict):
    log.warning("%s: expected an object at the top level", path)
    return {}
  providers = raw.get("providers", {})
  if not isinstance(providers, dict):
    log.warning("%s: 'providers' must be an object", path)
    return {}
  return providers


_BUILTIN_PATH = Path(__file__).resolve().parent / "models.json"
_USER_OVERRIDE_PATH = os.path.expanduser("~/.nemo/models.json")


def _merge_providers(base: dict, override: dict) -> dict:
  """User-provided provider entries fully replace package entries by name.

  Provider-level (not model-level) replacement keeps the schema's
  shared-endpoint-config story coherent: if you override deepseek you
  own its endpoint *and* its model list, no surprise interleaving.
  """
  merged = dict(base)
  merged.update(override)
  return merged


def load_presets(
  *,
  builtin_path: str | os.PathLike = _BUILTIN_PATH,
  user_path: str = _USER_OVERRIDE_PATH,
) -> dict[str, Preset]:
  """Return the merged {model_name: Preset} table.

  Reads ``builtin_path`` (defaults to ``nemo/models.json``, which the package
  doesn't ship → reads as ``{}``) plus ``~/.nemo/models.json`` (where presets
  actually live) and flattens the provider-grouped schema into per-model
  ``Preset`` records.
  """
  base_providers = _read_json(builtin_path)
  user_providers = _read_json(user_path)
  merged = _merge_providers(base_providers, user_providers)
  return _flatten_providers(merged)


def resolve_preset(
  name: str,
  *,
  builtin_path: str | os.PathLike = _BUILTIN_PATH,
  user_path: str = _USER_OVERRIDE_PATH,
) -> Preset | None:
  """Resolve a preset by name, CASE-INSENSITIVELY.

  The match ignores case but the returned ``Preset`` keeps its original-case
  name/remote, so the exact model id still reaches the upstream server. Needed
  because callers normalise case differently (``is_model_compatible``
  lowercases the model name) while a model's key in models.json can be
  mixed-case (e.g. ``Qwen3-Coder-Next-MLX-8bit``) — a case-sensitive lookup
  silently missed those and reported the model "not available" for the agent.
  """
  presets = load_presets(builtin_path=builtin_path, user_path=user_path)
  exact = presets.get(name)
  if exact is not None:
    return exact
  folded = name.casefold()
  for key, preset in presets.items():
    if key.casefold() == folded:
      return preset
  return None


def preset_name_for_endpoint(
  remote_model: str,
  base_url: str,
  agent: str,
  *,
  builtin_path: str | os.PathLike = _BUILTIN_PATH,
  user_path: str = _USER_OVERRIDE_PATH,
) -> str | None:
  """Inverse of preset resolution: map a *live* (remote model id, endpoint
  base URL) back to the preset NAME that produced them for ``agent``.

  Resolution is one-way at startup / on ``/model``: ``deepseek-v4-pro`` →
  remote ``deepseek-v4-pro[1m]`` + an endpoint URL, and only the remote id
  survives in the running daemon's ``model`` variable. But ``--model``
  accepts preset *names*, so a restart must hand the name back or the new
  process re-sends the remote id to the default endpoint and every turn
  fails "model not found". This recovers the name. Any preset matching the
  same (url, remote) round-trips to identical routing, so ties are
  harmless; ``None`` means no preset matches (a plain model on the default
  endpoint — pass it through unchanged).
  """
  if not base_url:
    return None
  presets = load_presets(builtin_path=builtin_path, user_path=user_path)
  for preset in presets.values():
    if (preset.base_url_for(agent) == base_url
        and preset.remote_for(agent) == remote_model):
      return preset.name
  return None
