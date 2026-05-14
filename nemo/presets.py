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

Sources, in order of precedence (later overrides earlier):
  1. ``nemo/models.json`` shipped inside the package (the previous
     ``BUILTIN_PRESETS`` dict, externalised so adding a model doesn't
     require a release).
  2. ``~/.nemo/models.json`` user overrides — same schema, merged at
     the provider level (a user-defined provider entry fully replaces
     the package's entry for that name; entries the user doesn't touch
     pass through unchanged).
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
  # Reading the key from env (vs accepting on the command line or in
  # the JSON file) avoids leaking secrets into argv / ps / config
  # dumps. Empty string means the model has no associated key — the
  # active agent's default auth (e.g. an OAuth token from ChatGPT
  # login for codex) takes over.
  api_key_env: str = ""
  # Anthropic-protocol endpoint. Empty → not callable via --agent claude
  # (or via opencode against an anthropic/* model).
  anthropic_url: str = ""
  anthropic_remote: str = ""
  # OpenAI-protocol endpoint. Empty → not callable via --agent codex
  # (or via opencode against an openai/* model).
  openai_url: str = ""
  openai_remote: str = ""

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
    api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
    return EndpointConfig(base_url=base, api_key=api_key)


# ---------------------------------------------------------------------------
# JSON loading + flattening
# ---------------------------------------------------------------------------


_ENV_PATTERN = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")


def _parse_api_key_env(raw: object, *, where: str) -> str:
  """Extract the env var name from ``{env:VARNAME}``. Empty for missing/blank.

  Plain-string keys are rejected so secrets can't accidentally land in
  the JSON file (and from there in dotfile backups, git, etc).
  """
  if raw is None or raw == "":
    return ""
  if not isinstance(raw, str):
    log.warning("%s: apiKey must be a string, got %s — ignoring",
                where, type(raw).__name__)
    return ""
  m = _ENV_PATTERN.match(raw.strip())
  if not m:
    log.warning("%s: apiKey must use {env:VARNAME} syntax, got %r — ignoring",
                where, raw)
    return ""
  return m.group(1)


def _protocol_block(provider_data: dict, key: str) -> tuple[str, str]:
  """Pull (baseURL, api_key_env) out of one provider's protocol section."""
  raw = provider_data.get(key)
  if not isinstance(raw, dict):
    return "", ""
  base = raw.get("baseURL", "")
  if not isinstance(base, str):
    log.warning("provider.%s.baseURL must be a string, got %s",
                key, type(base).__name__)
    base = ""
  api_key_env = _parse_api_key_env(raw.get("apiKey"), where=f"provider.{key}.apiKey")
  return base, api_key_env


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


def _flatten_providers(providers: dict) -> dict[str, Preset]:
  """Expand provider-grouped JSON into a flat {model_name: Preset} table."""
  out: dict[str, Preset] = {}
  for provider_name, pdata in providers.items():
    if not isinstance(pdata, dict):
      log.warning("provider %r: expected object, got %s — skipping",
                  provider_name, type(pdata).__name__)
      continue
    anthropic_url, anthropic_key_env = _protocol_block(pdata, "anthropic")
    openai_url, openai_key_env = _protocol_block(pdata, "openai")
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

    models = pdata.get("models", {})
    if not isinstance(models, dict):
      log.warning("provider %r: models must be an object, got %s — skipping",
                  provider_name, type(models).__name__)
      continue
    for model_name, mdata in models.items():
      if model_name in out:
        log.warning("model %r appears under multiple providers; later entry wins",
                    model_name)
      out[model_name] = Preset(
        name=str(model_name),
        api_key_env=api_key_env,
        anthropic_url=anthropic_url,
        anthropic_remote=_model_remote(mdata, "anthropic") or model_name,
        openai_url=openai_url,
        openai_remote=_model_remote(mdata, "openai") or model_name,
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

  Reads ``nemo/models.json`` (package builtin) plus
  ``~/.nemo/models.json`` (user overrides) and flattens the
  provider-grouped schema into per-model ``Preset`` records.
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
  return load_presets(builtin_path=builtin_path, user_path=user_path).get(name)
