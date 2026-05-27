"""Tests for nemo.presets — provider-grouped JSON registry."""

from __future__ import annotations

import json
import os
from unittest import mock

from nemo.presets import (
  Preset, _flatten_providers, _parse_api_key,
  load_presets, preset_name_for_endpoint, resolve_preset,
)


# ---------------------------------------------------------------------------
# Preset.supports / .remote_for / .endpoint_for — internal flat record
# ---------------------------------------------------------------------------

def test_supports_only_when_protocol_url_is_set():
  p = Preset(name="x", anthropic_url="https://a.example", openai_url="")
  assert p.supports("claude") is True
  assert p.supports("codex") is False
  # opencode is OK if either protocol is set — the SDK picks based on
  # the model slug prefix, not the daemon's --agent flag.
  assert p.supports("opencode") is True

  p_neither = Preset(name="x")
  assert p_neither.supports("claude") is False
  assert p_neither.supports("codex") is False
  assert p_neither.supports("opencode") is False


def test_remote_for_falls_back_to_model_name():
  # Per-protocol override → preset name itself.
  p = Preset(
    name="alias",
    anthropic_url="https://a", anthropic_remote="anthropic-only",
    openai_url="https://x",  # no openai_remote override
  )
  assert p.remote_for("claude") == "anthropic-only"
  assert p.remote_for("codex") == "alias"  # falls through to name

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
# {env:VARNAME} apiKey parsing — secrets-in-config defence
# ---------------------------------------------------------------------------

def test_parse_api_key_env_syntax():
  # {env:VAR} → (env_name, "")
  assert _parse_api_key("{env:DEEPSEEK_API_KEY}", where="t") == ("DEEPSEEK_API_KEY", "")
  # Whitespace around the syntax is fine — JSON-paste-from-docs path.
  assert _parse_api_key("  {env:FOO}  ", where="t") == ("FOO", "")


def test_parse_api_key_accepts_literal():
  # Literal keys are now allowed (~/.nemo/models.json is user-private) → ("", literal).
  assert _parse_api_key("blacktree@", where="t") == ("", "blacktree@")
  assert _parse_api_key("  sk-local  ", where="t") == ("", "sk-local")


def test_parse_api_key_blank_or_missing_is_empty():
  assert _parse_api_key("", where="t") == ("", "")
  assert _parse_api_key(None, where="t") == ("", "")


def test_endpoint_for_uses_literal_key():
  # A literal apiKey is used directly (no env lookup); it wins over api_key_env.
  p = Preset(name="x", api_key_literal="blacktree@",
             anthropic_url="http://127.0.0.1:8000")
  assert p.endpoint_for("claude").api_key == "blacktree@"


def test_literal_key_flows_from_models_json(tmp_path):
  # End-to-end: a self-hosted provider with a literal key + no /v1 on the
  # anthropic base resolves to the exact endpoint the SDK will hit.
  user = tmp_path / "models.json"
  user.write_text(json.dumps({
    "providers": {
      "omlx": {
        "anthropic": {"baseURL": "http://127.0.0.1:8000", "apiKey": "blacktree@"},
        "models": {"Qwen3-Coder-Next-MLX-8bit": {}},
      },
    },
  }))
  p = resolve_preset("Qwen3-Coder-Next-MLX-8bit",
                     builtin_path="/nonexistent", user_path=str(user))
  assert p is not None and p.supports("claude")
  ep = p.endpoint_for("claude")
  assert ep.base_url == "http://127.0.0.1:8000"
  assert ep.api_key == "blacktree@"


def test_vision_block_provider_default_and_model_override(tmp_path):
  # Provider-level vision is the default; a model's own block overrides it
  # field-by-field; an undeclared provider/model is text-only.
  user = tmp_path / "models.json"
  user.write_text(json.dumps({
    "providers": {
      "qwen": {
        "openai": {"baseURL": "https://q", "apiKey": "k"},
        "vision": {"image": True, "video": False},
        "models": {
          "qwen-text": {},                                  # inherits provider
          "qwen-vl-max": {"vision": {"video": True}},        # override video only
          "qwen-blind": {"vision": {"image": False}},        # override image only
        },
      },
      "deepseek": {
        "anthropic": {"baseURL": "https://d", "apiKey": "k"},
        "models": {"deepseek-v4-pro": {}},                   # no vision → text-only
      },
    },
  }))
  presets = load_presets(builtin_path="/nonexistent", user_path=str(user))
  assert (presets["qwen-text"].sees_image, presets["qwen-text"].sees_video) == (True, False)
  assert (presets["qwen-vl-max"].sees_image, presets["qwen-vl-max"].sees_video) == (True, True)
  assert (presets["qwen-blind"].sees_image, presets["qwen-blind"].sees_video) == (False, False)
  assert (presets["deepseek-v4-pro"].sees_image, presets["deepseek-v4-pro"].sees_video) == (False, False)


# ---------------------------------------------------------------------------
# Flattening: provider-grouped JSON → flat {name: Preset}
# ---------------------------------------------------------------------------

def test_flatten_kimi_anthropic_only():
  presets = _flatten_providers({
    "kimi": {
      "anthropic": {
        "baseURL": "https://api.kimi.com/coding",
        "apiKey": "{env:KIMI_API_KEY}",
      },
      "models": {"kimi-for-coding": {}},
    },
  })
  assert "kimi-for-coding" in presets
  p = presets["kimi-for-coding"]
  # OpenAI endpoint missing → /agent codex must not advertise it.
  assert p.supports("claude") is True
  assert p.supports("codex") is False
  assert p.anthropic_url == "https://api.kimi.com/coding"
  assert p.anthropic_remote == "kimi-for-coding"  # falls back to model name
  assert p.api_key_env == "KIMI_API_KEY"


def test_flatten_deepseek_dual_protocol_with_remote_override():
  presets = _flatten_providers({
    "deepseek": {
      "anthropic": {
        "baseURL": "https://api.deepseek.com/anthropic",
        "apiKey": "{env:DEEPSEEK_API_KEY}",
      },
      "openai": {
        "baseURL": "https://api.deepseek.com",
        "apiKey": "{env:DEEPSEEK_API_KEY}",
      },
      "models": {
        # Anthropic endpoint advertises [1m] context variant; OpenAI doesn't.
        "deepseek-v4-pro": {"anthropic": {"remote": "deepseek-v4-pro[1m]"}},
        # No override → both protocols send the bare model name.
        "deepseek-v4-flash": {},
      },
    },
  })
  pro = presets["deepseek-v4-pro"]
  assert pro.remote_for("claude") == "deepseek-v4-pro[1m]"
  assert pro.remote_for("codex") == "deepseek-v4-pro"
  flash = presets["deepseek-v4-flash"]
  assert flash.remote_for("claude") == "deepseek-v4-flash"
  assert flash.remote_for("codex") == "deepseek-v4-flash"


def test_flatten_skips_provider_with_non_object_value(caplog):
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    out = _flatten_providers({"weird": "not-an-object"})
  assert out == {}
  assert any("expected object" in r.getMessage() for r in caplog.records)


def test_flatten_warns_and_skips_orphan_provider(caplog):
  # User wrote ``models: { ... }`` but forgot the anthropic/openai
  # block. Without a guard, flattening would produce Presets with
  # empty URLs that vanish from /model — silently uncallable. Warn so
  # the misconfig surfaces, and don't pretend the model exists.
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    out = _flatten_providers({
      "broken": {
        "models": {"my-cool-model": {}},
      },
    })
  assert "my-cool-model" not in out
  assert any(
    "no anthropic/openai block" in r.getMessage() and "broken" in r.getMessage()
    for r in caplog.records
  )


# ---------------------------------------------------------------------------
# load_presets / resolve_preset — file-based plumbing + user overrides
# ---------------------------------------------------------------------------

def _write_base_models(tmp_path, name: str = "base.json") -> str:
  """Write a deepseek+kimi catalog and return its path.

  The package builtin (nemo/models.json) now ships EMPTY — provider/model
  config is deployment config in ~/.nemo/models.json, not package code — so
  merge tests supply their own base via ``builtin_path`` instead of relying
  on the package shipping presets."""
  p = tmp_path / name
  p.write_text(json.dumps({
    "providers": {
      "deepseek": {
        "anthropic": {"baseURL": "https://api.deepseek.com/anthropic",
                      "apiKey": "{env:DEEPSEEK_API_KEY}"},
        "openai": {"baseURL": "https://api.deepseek.com",
                   "apiKey": "{env:DEEPSEEK_API_KEY}"},
        "models": {
          "deepseek-v4-pro": {"anthropic": {"remote": "deepseek-v4-pro[1m]"}},
          "deepseek-v4-flash": {},
        },
      },
      "kimi": {
        "anthropic": {"baseURL": "https://api.kimi.com/coding",
                      "apiKey": "{env:KIMI_API_KEY}"},
        "models": {"kimi-for-coding": {}},
      },
    },
  }))
  return str(p)


def test_package_builtin_ships_no_presets():
  # Provider/model config is deployment config (~/.nemo/models.json), not
  # package code — the package ships NO nemo/models.json at all, so with no
  # user override there are zero presets (the missing builtin reads as {}).
  assert load_presets(user_path="/nonexistent/path") == {}


def test_load_presets_resolves_from_base_catalog(tmp_path):
  base = _write_base_models(tmp_path)
  os.environ.pop("DEEPSEEK_API_KEY", None)  # not required to *load* the catalog
  presets = load_presets(builtin_path=base, user_path="/nonexistent/path")
  assert "kimi-for-coding" in presets
  assert "deepseek-v4-pro" in presets
  assert "deepseek-v4-flash" in presets
  # Verify the [1m] override survived JSON round-trip.
  assert presets["deepseek-v4-pro"].anthropic_remote == "deepseek-v4-pro[1m]"
  assert presets["deepseek-v4-pro"].openai_remote == "deepseek-v4-pro"


def test_user_override_adds_new_provider(tmp_path):
  base = _write_base_models(tmp_path, "base.json")
  user = tmp_path / "user.json"
  user.write_text(json.dumps({
    "providers": {
      "router": {
        "anthropic": {
          "baseURL": "https://openrouter.ai/anthropic",
          "apiKey": "{env:OPENROUTER_KEY}",
        },
        "models": {"anthropic/claude-sonnet-4-6": {}},
      },
    },
  }))
  presets = load_presets(builtin_path=base, user_path=str(user))
  # Base providers still there.
  assert "kimi-for-coding" in presets
  # New entry from user file.
  assert "anthropic/claude-sonnet-4-6" in presets
  assert presets["anthropic/claude-sonnet-4-6"].api_key_env == "OPENROUTER_KEY"


def test_user_override_replaces_existing_provider(tmp_path):
  # Provider-level replacement: redefining "deepseek" in the user file
  # wipes the base's model list for that provider so the user fully
  # owns it. Provider entries the user doesn't touch pass through.
  base = _write_base_models(tmp_path, "base.json")
  user = tmp_path / "user.json"
  user.write_text(json.dumps({
    "providers": {
      "deepseek": {
        "anthropic": {
          "baseURL": "https://api.deepseek.com/anthropic",
          "apiKey": "{env:MY_OWN_KEY}",
        },
        "models": {"deepseek-v4-pro": {}},  # only one model now, no [1m]
      },
    },
  }))
  presets = load_presets(builtin_path=base, user_path=str(user))
  # kimi untouched.
  assert "kimi-for-coding" in presets
  # deepseek-v4-flash no longer present (user provider entry replaced base).
  assert "deepseek-v4-flash" not in presets
  # User's deepseek-v4-pro replaces the builtin (different api_key_env, no [1m]).
  pro = presets["deepseek-v4-pro"]
  assert pro.api_key_env == "MY_OWN_KEY"
  assert pro.anthropic_remote == "deepseek-v4-pro"  # no override now


def test_resolve_preset_returns_none_for_unknown(tmp_path):
  assert resolve_preset(
    "definitely-not-a-real-model",
    user_path=str(tmp_path / "missing.json"),
  ) is None


def test_resolve_preset_is_case_insensitive(tmp_path):
  # Regression: a mixed-case model key (e.g. Qwen3-Coder-Next-MLX-8bit) was
  # silently missed because is_model_compatible lowercases the name before
  # lookup, so /model reported it "not available for claude". The match must
  # ignore case while the returned preset keeps its original-case name/remote
  # (so the upstream server still gets the exact id).
  user = tmp_path / "models.json"
  user.write_text(json.dumps({
    "providers": {
      "omlx": {
        "anthropic": {"baseURL": "http://localhost:8000/v1",
                      "apiKey": "{env:OMLX_KEY}"},
        "models": {"Qwen3-Coder-Next-MLX-8bit": {}},
      },
    },
  }))
  kw = {"builtin_path": "/nonexistent/builtin", "user_path": str(user)}
  exact = resolve_preset("Qwen3-Coder-Next-MLX-8bit", **kw)
  lower = resolve_preset("qwen3-coder-next-mlx-8bit", **kw)  # is_model_compatible's form
  assert exact is not None and lower is not None
  # Same preset, original-case name/remote preserved (server gets exact id).
  assert exact.name == lower.name == "Qwen3-Coder-Next-MLX-8bit"
  assert lower.anthropic_remote == "Qwen3-Coder-Next-MLX-8bit"
  assert lower.supports("claude") is True


# ---------------------------------------------------------------------------
# preset_name_for_endpoint — inverse mapping used by /restart and /upgrade
# ---------------------------------------------------------------------------

def test_preset_name_for_endpoint_round_trips_resolved_remote(tmp_path):
  # The bug this guards: /restart relaunched with the *remote id*
  # (deepseek-v4-pro[1m]) instead of the preset NAME (deepseek-v4-pro),
  # so the new daemon couldn't resolve the endpoint and every turn (and
  # the forked /btw CLI) failed "model not found". Forward resolution is
  # deepseek-v4-pro → remote deepseek-v4-pro[1m] @ the anthropic URL; the
  # inverse must recover the name from exactly those two live values.
  base = _write_base_models(tmp_path, "base.json")
  name = preset_name_for_endpoint(
    "deepseek-v4-pro[1m]", "https://api.deepseek.com/anthropic", "claude",
    builtin_path=base, user_path="/nonexistent/path",
  )
  assert name == "deepseek-v4-pro"
  # codex advertises the bare name on the OpenAI endpoint — must also map.
  assert preset_name_for_endpoint(
    "deepseek-v4-pro", "https://api.deepseek.com", "codex",
    builtin_path=base, user_path="/nonexistent/path",
  ) == "deepseek-v4-pro"


def test_preset_name_for_endpoint_none_when_no_endpoint_or_no_match():
  # Default endpoint (no preset active) → nothing to reverse-resolve;
  # the caller passes the plain model id through unchanged.
  assert preset_name_for_endpoint(
    "claude-opus-4-7", "", "claude", user_path="/nonexistent/path") is None
  # Right URL but a remote id no preset emits → no match.
  assert preset_name_for_endpoint(
    "deepseek-v4-pro", "https://api.deepseek.com/anthropic", "claude",
    user_path="/nonexistent/path") is None


def test_load_presets_handles_malformed_user_file(tmp_path, caplog):
  base = _write_base_models(tmp_path, "base.json")
  bad = tmp_path / "user.json"
  bad.write_text("{not valid json")
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    out = load_presets(builtin_path=base, user_path=str(bad))
  # Falls back to the base catalog; log line surfaces the error.
  assert "deepseek-v4-pro" in out
  assert "kimi-for-coding" in out
  assert any("failed to read" in r.getMessage() for r in caplog.records)


def test_load_presets_user_file_top_level_must_be_object(tmp_path, caplog):
  base = _write_base_models(tmp_path, "base.json")
  bad = tmp_path / "user.json"
  bad.write_text(json.dumps(["not", "an", "object"]))
  import logging
  with caplog.at_level(logging.WARNING, logger="nemo.presets"):
    out = load_presets(builtin_path=base, user_path=str(bad))
  # Base untouched.
  assert "deepseek-v4-pro" in out
  assert any("expected an object" in r.getMessage() for r in caplog.records)


def test_kimi_anthropic_only_visible_in_codex_picker(tmp_path):
  # Regression: Kimi For Coding gates its OpenAI endpoint
  # (access_terminated_error for this tier). Make sure /agent codex
  # never advertises it.
  base = _write_base_models(tmp_path, "base.json")
  p = resolve_preset("kimi-for-coding", builtin_path=base,
                     user_path="/nonexistent/path")
  assert p is not None
  assert p.supports("claude") is True
  assert p.supports("codex") is False
