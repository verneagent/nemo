"""Tests for nemo.presets — preset registry + per-provider expansion."""

from __future__ import annotations

import json
import os
from unittest import mock

from nemo.presets import (
  BUILTIN_PRESETS, Preset, _preset_from_dict,
  load_presets, resolve_preset,
)


# ---------------------------------------------------------------------------
# Preset.supports / .remote_for / .endpoint_for
# ---------------------------------------------------------------------------

def test_supports_only_when_protocol_url_is_set():
  p = Preset(name="x", anthropic_url="https://a.example", openai_url="")
  assert p.supports("claude") is True
  assert p.supports("codex") is False
  # opencode is OK if either protocol is set — the SDK picks based on
  # the model slug prefix, not the daemon's --provider flag.
  assert p.supports("opencode") is True

  p_neither = Preset(name="x")
  assert p_neither.supports("claude") is False
  assert p_neither.supports("codex") is False
  assert p_neither.supports("opencode") is False


def test_remote_for_falls_back_through_chain():
  # Most specific to least specific: protocol-specific override →
  # generic remote_name → preset name itself.
  p = Preset(
    name="alias",
    remote_name="generic",
    anthropic_remote="anthropic-only",
    openai_url="https://x", openai_remote="",
  )
  assert p.remote_for("claude") == "anthropic-only"
  assert p.remote_for("codex") == "generic"  # falls through to remote_name

  p2 = Preset(name="alias")  # no overrides anywhere
  assert p2.remote_for("claude") == "alias"
  assert p2.remote_for("codex") == "alias"


def test_endpoint_for_reads_api_key_from_env():
  p = Preset(
    name="ds",
    api_key_env="MY_KEY",
    anthropic_url="https://api.deepseek.com/anthropic",
    openai_url="https://api.deepseek.com",
  )
  with mock.patch.dict(os.environ, {"MY_KEY": "sk-abc"}):
    ep = p.endpoint_for("claude")
    assert ep.base_url == "https://api.deepseek.com/anthropic"
    assert ep.api_key == "sk-abc"
    ep_codex = p.endpoint_for("codex")
    assert ep_codex.base_url == "https://api.deepseek.com"
    assert ep_codex.api_key == "sk-abc"


def test_endpoint_for_with_unset_env_returns_empty_key():
  p = Preset(name="x", api_key_env="DEFINITELY_UNSET_XYZ", anthropic_url="https://a.example")
  os.environ.pop("DEFINITELY_UNSET_XYZ", None)
  ep = p.endpoint_for("claude")
  assert ep.base_url == "https://a.example"
  assert ep.api_key == ""  # caller (CLI) is responsible for failing fast


# ---------------------------------------------------------------------------
# _preset_from_dict
# ---------------------------------------------------------------------------

def test_from_dict_happy_path():
  p = _preset_from_dict("foo", {
    "api_key_env": "K",
    "remote_name": "foo-1",
    "anthropic_url": "https://a",
    "openai_url": "https://o",
  })
  assert p is not None
  assert p.name == "foo"
  assert p.api_key_env == "K"
  assert p.remote_name == "foo-1"


def test_from_dict_rejects_non_object():
  assert _preset_from_dict("foo", "not-a-dict") is None
  assert _preset_from_dict("foo", None) is None


def test_from_dict_ignores_unknown_fields(caplog):
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    p = _preset_from_dict("foo", {
      "anthropic_url": "https://a",
      "weird_field": "ignored",
      "another": 42,
    })
  assert p is not None
  assert any("unknown fields" in r.getMessage() for r in caplog.records)


def test_from_dict_coerces_bad_field_types():
  # Non-string field gets logged + reset to "" rather than crashing.
  p = _preset_from_dict("foo", {
    "anthropic_url": 123,  # not a string
    "openai_url": "https://ok",
  })
  assert p is not None
  assert p.anthropic_url == ""
  assert p.openai_url == "https://ok"


# ---------------------------------------------------------------------------
# load_presets / resolve_preset / user overrides
# ---------------------------------------------------------------------------

def test_builtin_includes_deepseek_v4_pro():
  assert "deepseek-v4-pro" in BUILTIN_PRESETS
  p = BUILTIN_PRESETS["deepseek-v4-pro"]
  # DeepSeek's Anthropic endpoint advertises [1m]; OpenAI side does not.
  assert p.anthropic_remote == "deepseek-v4-pro[1m]"
  assert p.openai_remote == "deepseek-v4-pro"
  assert p.api_key_env == "DEEPSEEK_API_KEY"


def test_user_overrides_extend_and_replace_builtins(tmp_path):
  override_path = tmp_path / "models.json"
  override_path.write_text(json.dumps({
    # New preset.
    "router-claude": {
      "api_key_env": "OPENROUTER_KEY",
      "anthropic_url": "https://openrouter.ai/anthropic",
      "remote_name": "anthropic/claude-sonnet-4-6",
    },
    # Overrides the builtin deepseek-v4-pro with a different api_key_env.
    "deepseek-v4-pro": {
      "api_key_env": "MY_DEEPSEEK_KEY",
      "anthropic_url": "https://api.deepseek.com/anthropic",
      "anthropic_remote": "deepseek-v4-pro[1m]",
      "openai_url": "https://api.deepseek.com",
      "openai_remote": "deepseek-v4-pro",
    },
  }))
  presets = load_presets(path=str(override_path))
  assert "router-claude" in presets
  # Override: api_key_env replaced, builtin's value gone.
  assert presets["deepseek-v4-pro"].api_key_env == "MY_DEEPSEEK_KEY"


def test_resolve_preset_returns_none_for_unknown(tmp_path):
  assert resolve_preset("definitely-not-a-real-model", path=str(tmp_path / "missing.json")) is None


def test_load_presets_handles_malformed_user_file(tmp_path, caplog):
  bad = tmp_path / "models.json"
  bad.write_text("{not valid json")
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    out = load_presets(path=str(bad))
  # Falls back to builtins only; log line surfaces the error.
  assert "deepseek-v4-pro" in out
  assert "router-claude" not in out  # nothing user-defined survives
  assert any("failed to read" in r.getMessage() for r in caplog.records)


def test_load_presets_user_file_top_level_must_be_object(tmp_path, caplog):
  bad = tmp_path / "models.json"
  bad.write_text(json.dumps(["not", "an", "object"]))
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    out = load_presets(path=str(bad))
  assert out == BUILTIN_PRESETS  # untouched
  assert any("expected an object" in r.getMessage() for r in caplog.records)
