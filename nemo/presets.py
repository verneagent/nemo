"""Model preset registry.

A preset is a logical model name (e.g. ``deepseek-v4-pro``) that
expands at startup or on ``/model`` switch into:

  - the endpoint URL for the active provider's wire protocol
    (Anthropic-format for ``--provider claude``, OpenAI-format for
    ``--provider codex``),
  - the API key, read from ``$<api_key_env>`` so secrets stay out of
    argv,
  - the actual model id sent to the remote (which may differ between
    protocols — e.g. DeepSeek advertises ``deepseek-v4-pro`` on its
    OpenAI endpoint but ``deepseek-v4-pro[1m]`` on its Anthropic
    endpoint).

Presets unify what used to be three separate flags
(``--base-url`` / ``--api-key-env`` / ``--model``) into a single
``--model deepseek-v4-pro``.

Sources, in order of precedence (later overrides earlier):
  1. ``BUILTIN_PRESETS`` shipped in this module
  2. ``~/.nemo/models.json`` user overrides

The user file shape mirrors the builtin dict — each key is the preset
name, the value is a JSON object with the same fields as ``Preset``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from typing import ClassVar

from .coding_agent import EndpointConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
  """A logical model name + how to reach it."""
  name: str
  # Reading the key from env (vs accepting on the command line) avoids
  # leaking secrets into argv / ps. Empty string means the preset has
  # no associated key — the active provider's default auth (e.g. an
  # OAuth token from ChatGPT login for codex) takes over.
  api_key_env: str = ""
  # Default remote model id. Falls back to ``name`` if blank. The two
  # protocol-specific overrides below only kick in when set.
  remote_name: str = ""
  # Anthropic-protocol endpoint. Empty → not callable via --provider claude
  # (or via opencode against an anthropic/* model).
  anthropic_url: str = ""
  anthropic_remote: str = ""
  # OpenAI-protocol endpoint. Empty → not callable via --provider codex
  # (or via opencode against an openai/* model).
  openai_url: str = ""
  openai_remote: str = ""

  # Class-level set of valid keys, used by `from_dict` to reject typos.
  _ALLOWED_KEYS: ClassVar[frozenset[str]] = frozenset()

  def supports(self, provider: str) -> bool:
    if provider == "claude":
      return bool(self.anthropic_url)
    if provider == "codex":
      return bool(self.openai_url)
    if provider == "opencode":
      # OpenCode resolves the actual provider from the model slug
      # prefix at the SDK layer; either url being set is enough for us
      # to plumb env vars through.
      return bool(self.anthropic_url or self.openai_url)
    return False

  def remote_for(self, provider: str) -> str:
    """The model id to send downstream when the active provider is ``provider``."""
    if provider == "claude":
      return self.anthropic_remote or self.remote_name or self.name
    if provider == "codex":
      return self.openai_remote or self.remote_name or self.name
    return self.remote_name or self.name

  def base_url_for(self, provider: str) -> str:
    if provider == "claude":
      return self.anthropic_url
    if provider == "codex":
      return self.openai_url
    if provider == "opencode":
      return self.anthropic_url or self.openai_url
    return ""

  def endpoint_for(self, provider: str) -> EndpointConfig:
    """Materialise the per-turn endpoint config for ``provider``.

    Returns an empty ``EndpointConfig`` (no overrides) when this preset
    doesn't apply to ``provider`` — caller should pre-check ``supports``.
    """
    base = self.base_url_for(provider)
    api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
    return EndpointConfig(base_url=base, api_key=api_key)


# Compute _ALLOWED_KEYS once Preset is fully defined.
Preset._ALLOWED_KEYS = frozenset(f.name for f in fields(Preset) if not f.name.startswith("_"))


def _preset_from_dict(name: str, data: object) -> Preset | None:
  if not isinstance(data, dict):
    log.warning("preset %r: expected object, got %s", name, type(data).__name__)
    return None
  unknown = set(data.keys()) - Preset._ALLOWED_KEYS - {"name"}
  if unknown:
    log.warning("preset %r: ignoring unknown fields %s", name, sorted(unknown))
  payload: dict[str, object] = {"name": name}
  for f in fields(Preset):
    if f.name.startswith("_") or f.name == "name":
      continue
    raw = data.get(f.name, "")
    if not isinstance(raw, str):
      log.warning("preset %r: field %s must be a string, got %s",
                  name, f.name, type(raw).__name__)
      raw = ""
    payload[f.name] = raw
  try:
    return Preset(**payload)  # type: ignore[arg-type]
  except TypeError as exc:
    log.warning("preset %r: malformed (%s)", name, exc)
    return None


# ---------------------------------------------------------------------------
# Builtin presets
# ---------------------------------------------------------------------------
# Source: https://api-docs.deepseek.com/zh-cn/quick_start/pricing
# (verified 2026-05-07).  The Anthropic endpoint advertises a 1M-context
# variant via the ``[1m]`` suffix; the OpenAI endpoint only documents
# the bare slug.

BUILTIN_PRESETS: dict[str, Preset] = {
  "deepseek-v4-pro": Preset(
    name="deepseek-v4-pro",
    api_key_env="DEEPSEEK_API_KEY",
    anthropic_url="https://api.deepseek.com/anthropic",
    anthropic_remote="deepseek-v4-pro[1m]",
    openai_url="https://api.deepseek.com",
    openai_remote="deepseek-v4-pro",
  ),
  "deepseek-v4-flash": Preset(
    name="deepseek-v4-flash",
    api_key_env="DEEPSEEK_API_KEY",
    anthropic_url="https://api.deepseek.com/anthropic",
    anthropic_remote="deepseek-v4-flash",
    openai_url="https://api.deepseek.com",
    openai_remote="deepseek-v4-flash",
  ),
}


_USER_OVERRIDE_PATH = os.path.expanduser("~/.nemo/models.json")


def _load_user_overrides(path: str = _USER_OVERRIDE_PATH) -> dict[str, Preset]:
  if not os.path.isfile(path):
    return {}
  try:
    with open(path, encoding="utf-8") as f:
      raw = json.load(f)
  except (OSError, json.JSONDecodeError) as exc:
    log.warning("failed to read %s: %s", path, exc)
    return {}
  if not isinstance(raw, dict):
    log.warning("%s: expected an object at the top level", path)
    return {}
  out: dict[str, Preset] = {}
  for name, data in raw.items():
    p = _preset_from_dict(str(name), data)
    if p is not None:
      out[p.name] = p
  return out


def load_presets(*, path: str = _USER_OVERRIDE_PATH) -> dict[str, Preset]:
  """Return the merged builtin + user override preset table."""
  merged = dict(BUILTIN_PRESETS)
  merged.update(_load_user_overrides(path))
  return merged


def resolve_preset(name: str, *, path: str = _USER_OVERRIDE_PATH) -> Preset | None:
  return load_presets(path=path).get(name)
