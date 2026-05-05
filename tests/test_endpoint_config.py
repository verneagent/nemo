"""Tests for the cross-provider --base-url / --api-key plumbing.

Each adapter must translate a shared EndpointConfig into its own vendor
env vars. Claude additionally fans the user-supplied model out across
the ANTHROPIC_DEFAULT_*_MODEL knobs so third-party endpoints (DeepSeek,
etc.) that don't speak canonical Claude slugs route everything to the
remote model.
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

from nemo.agent_factory import build_coding_agent
from nemo.claude_agent import ClaudeCodingAgent
from nemo.coding_agent import EndpointConfig
from nemo.codex_agent import CodexCodingAgent
from nemo.opencode_agent import OpenCodeCodingAgent


class _DummyDB:
  pass


class _DummyChannel:
  pass


def _strip_anthropic_env(monkeypatch_target_env=None):
  """Clear ANTHROPIC_*/CLAUDE_CODE_* so passthrough doesn't pollute assertions."""
  return mock.patch.dict(
    os.environ,
    {
      k: ""
      for k in (
        "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
      )
    },
    clear=False,
  )


# ---------------------------------------------------------------------------
# EndpointConfig basics
# ---------------------------------------------------------------------------

def test_endpoint_config_defaults_are_empty():
  ep = EndpointConfig()
  assert ep.base_url == ""
  assert ep.api_key == ""


def test_build_coding_agent_default_endpoint_is_noop():
  # No endpoint kwarg should still construct each adapter and leave the
  # adapter's _endpoint at the empty default.
  for provider, cls in (
    ("claude", ClaudeCodingAgent),
    ("codex", CodexCodingAgent),
    ("opencode", OpenCodeCodingAgent),
  ):
    agent = build_coding_agent(provider, {}, "oc_1", _DummyDB(), _DummyChannel())
    assert isinstance(agent, cls)
    assert agent._endpoint.base_url == ""
    assert agent._endpoint.api_key == ""


def test_build_coding_agent_threads_endpoint_to_adapter():
  ep = EndpointConfig(base_url="https://x.example/api", api_key="sk-1")
  agent = build_coding_agent(
    "claude", {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
  assert agent._endpoint is ep


# ---------------------------------------------------------------------------
# Claude adapter
# ---------------------------------------------------------------------------

def test_claude_endpoint_sets_anthropic_env():
  async def _run():
    with _strip_anthropic_env():
      ep = EndpointConfig(
        base_url="https://api.deepseek.com/anthropic",
        api_key="sk-deepseek",
      )
      agent = ClaudeCodingAgent(
        {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
      opts = agent._build_options("/tmp/project", "deepseek-v4-pro")
      env = opts.env
      assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
      assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-deepseek"

  asyncio.run(_run())


def test_claude_endpoint_fans_model_out_to_routing_env():
  # When base_url is set, the user-supplied model must reach every
  # internal Claude Code routing env so subagents / preset slugs all use
  # the same third-party model name.
  async def _run():
    with _strip_anthropic_env():
      ep = EndpointConfig(
        base_url="https://api.deepseek.com/anthropic",
        api_key="sk-1",
      )
      agent = ClaudeCodingAgent(
        {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
      opts = agent._build_options("/tmp/project", "deepseek-v4-pro")
      env = opts.env
      assert env["ANTHROPIC_MODEL"] == "deepseek-v4-pro"
      assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "deepseek-v4-pro"
      assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "deepseek-v4-pro"
      assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "deepseek-v4-pro"
      assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "deepseek-v4-pro"

  asyncio.run(_run())


def test_claude_no_endpoint_no_model_fanout():
  # With no base_url, the SDK talks to Anthropic directly and the model
  # fan-out env vars must be left alone — otherwise we'd break the
  # default routing (e.g. force claude-opus-4-7 to reply on every tier).
  async def _run():
    with _strip_anthropic_env():
      agent = ClaudeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
      opts = agent._build_options("/tmp/project", "claude-opus-4-7")
      env = opts.env
      assert "ANTHROPIC_BASE_URL" not in env
      assert "ANTHROPIC_AUTH_TOKEN" not in env
      assert "ANTHROPIC_MODEL" not in env
      assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env

  asyncio.run(_run())


def test_claude_shell_env_passthrough_overlay():
  # User exported ANTHROPIC_DEFAULT_HAIKU_MODEL in their shell — that
  # should reach the SDK subprocess and survive even when --base-url
  # triggers fan-out (setdefault preserves existing values).
  async def _run():
    with mock.patch.dict(
      os.environ,
      {
        "ANTHROPIC_BASE_URL": "",  # cleared so flag wins
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
      },
      clear=False,
    ):
      ep = EndpointConfig(
        base_url="https://api.deepseek.com/anthropic", api_key="sk-1")
      agent = ClaudeCodingAgent(
        {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
      opts = agent._build_options("/tmp/project", "deepseek-v4-pro")
      env = opts.env
      # Flag still wins for base_url (we explicitly overwrite).
      assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
      # Pre-existing haiku override survives the fan-out.
      assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "deepseek-v4-flash"
      # Other tiers got fanned-out from --model.
      assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "deepseek-v4-pro"

  asyncio.run(_run())


# ---------------------------------------------------------------------------
# Codex adapter
# ---------------------------------------------------------------------------

def test_codex_endpoint_sets_openai_env():
  with mock.patch.dict(
    os.environ,
    {"OPENAI_BASE_URL": "", "OPENAI_API_KEY": "", "OPENAI_API_BASE": ""},
    clear=False,
  ):
    ep = EndpointConfig(base_url="https://router.example/v1", api_key="sk-2")
    agent = CodexCodingAgent(
      {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
    env = agent._build_env()
    assert env["OPENAI_BASE_URL"] == "https://router.example/v1"
    assert env["OPENAI_API_KEY"] == "sk-2"


def test_codex_shell_env_passthrough_no_flag():
  # Shell-exported OPENAI_BASE_URL must reach the sidecar without
  # requiring --base-url. Previously the env builder filtered it out.
  with mock.patch.dict(
    os.environ,
    {
      "OPENAI_BASE_URL": "https://shell.example/v1",
      "OPENAI_API_BASE": "https://alt.example/v1",
    },
    clear=False,
  ):
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    env = agent._build_env()
    assert env["OPENAI_BASE_URL"] == "https://shell.example/v1"
    assert env["OPENAI_API_BASE"] == "https://alt.example/v1"


def test_codex_flag_overrides_shell_env():
  with mock.patch.dict(
    os.environ,
    {"OPENAI_BASE_URL": "https://shell.example/v1", "OPENAI_API_KEY": "shell-k"},
    clear=False,
  ):
    ep = EndpointConfig(base_url="https://flag.example/v1", api_key="flag-k")
    agent = CodexCodingAgent(
      {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
    env = agent._build_env()
    assert env["OPENAI_BASE_URL"] == "https://flag.example/v1"
    assert env["OPENAI_API_KEY"] == "flag-k"


# ---------------------------------------------------------------------------
# OpenCode adapter
# ---------------------------------------------------------------------------

def test_opencode_endpoint_anthropic_prefix_only_writes_anthropic():
  with mock.patch.dict(
    os.environ,
    {"ANTHROPIC_BASE_URL": "", "ANTHROPIC_API_KEY": "",
     "OPENAI_BASE_URL": "", "OPENAI_API_KEY": ""},
    clear=False,
  ):
    ep = EndpointConfig(base_url="https://x.example/api", api_key="sk-3")
    agent = OpenCodeCodingAgent(
      {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
    agent._project_dir = "/tmp/project"
    agent._model = "anthropic/claude-sonnet-4-6"
    env = agent._build_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://x.example/api"
    assert env["ANTHROPIC_API_KEY"] == "sk-3"
    # OpenAI vars are left at their pre-existing (empty) values.
    assert env.get("OPENAI_BASE_URL", "") == ""
    assert env.get("OPENAI_API_KEY", "") == ""


def test_opencode_endpoint_openai_prefix_only_writes_openai():
  with mock.patch.dict(
    os.environ,
    {"ANTHROPIC_BASE_URL": "", "ANTHROPIC_API_KEY": "",
     "OPENAI_BASE_URL": "", "OPENAI_API_KEY": ""},
    clear=False,
  ):
    ep = EndpointConfig(base_url="https://x.example/v1", api_key="sk-4")
    agent = OpenCodeCodingAgent(
      {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
    agent._project_dir = "/tmp/project"
    agent._model = "openai/gpt-4o"
    env = agent._build_env()
    assert env["OPENAI_BASE_URL"] == "https://x.example/v1"
    assert env["OPENAI_API_KEY"] == "sk-4"
    assert env.get("ANTHROPIC_BASE_URL", "") == ""
    assert env.get("ANTHROPIC_API_KEY", "") == ""


def test_opencode_endpoint_unknown_prefix_writes_both():
  # "default" or third-party prefix → defensive: set both env families.
  with mock.patch.dict(
    os.environ,
    {"ANTHROPIC_BASE_URL": "", "ANTHROPIC_API_KEY": "",
     "OPENAI_BASE_URL": "", "OPENAI_API_KEY": ""},
    clear=False,
  ):
    ep = EndpointConfig(base_url="https://x.example/api", api_key="sk-5")
    agent = OpenCodeCodingAgent(
      {}, "oc_1", _DummyDB(), _DummyChannel(), endpoint=ep)
    agent._project_dir = "/tmp/project"
    agent._model = "default"
    env = agent._build_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://x.example/api"
    assert env["OPENAI_BASE_URL"] == "https://x.example/api"
    assert env["ANTHROPIC_API_KEY"] == "sk-5"
    assert env["OPENAI_API_KEY"] == "sk-5"


# ---------------------------------------------------------------------------
# CLI plumbing through __main__
# ---------------------------------------------------------------------------

def _fake_asyncio_run_capture(captured):
  def _runner(coro):
    captured["frame"] = coro.cr_frame
    coro.close()
    return 0
  return _runner


def test_cli_base_url_and_api_key_threaded_to_main_loop(tmp_path):
  from nemo.__main__ import main
  project = str(tmp_path)
  captured: dict[str, object] = {}
  argv = [
    "nemo", "--chat-id", "oc_1", "--project-dir", project,
    "--base-url", "https://api.deepseek.com/anthropic",
    "--api-key", "sk-deepseek",
  ]
  with mock.patch("sys.argv", argv), \
       mock.patch("nemo.__main__._ensure_provider_runtime"), \
       mock.patch("nemo.config.load_credentials",
                  return_value={"app_id": "a", "app_secret": "s", "email": ""}), \
       mock.patch("nemo.preflight.run_preflight", return_value=[]), \
       mock.patch("nemo.__main__.asyncio") as mock_asyncio:
    mock_asyncio.run.side_effect = _fake_asyncio_run_capture(captured)
    rc = main()
  assert rc == 0
  endpoint = captured["frame"].f_locals["endpoint"]
  assert endpoint.base_url == "https://api.deepseek.com/anthropic"
  assert endpoint.api_key == "sk-deepseek"


def test_cli_api_key_env_reads_from_environment(tmp_path):
  from nemo.__main__ import main
  project = str(tmp_path)
  captured: dict[str, object] = {}
  argv = [
    "nemo", "--chat-id", "oc_1", "--project-dir", project,
    "--api-key-env", "MY_TEST_KEY",
  ]
  with mock.patch.dict(os.environ, {"MY_TEST_KEY": "secret-from-env"}), \
       mock.patch("sys.argv", argv), \
       mock.patch("nemo.__main__._ensure_provider_runtime"), \
       mock.patch("nemo.config.load_credentials",
                  return_value={"app_id": "a", "app_secret": "s", "email": ""}), \
       mock.patch("nemo.preflight.run_preflight", return_value=[]), \
       mock.patch("nemo.__main__.asyncio") as mock_asyncio:
    mock_asyncio.run.side_effect = _fake_asyncio_run_capture(captured)
    rc = main()
  assert rc == 0
  endpoint = captured["frame"].f_locals["endpoint"]
  assert endpoint.api_key == "secret-from-env"


def test_cli_api_key_env_unset_is_error(tmp_path):
  from nemo.__main__ import main
  project = str(tmp_path)
  argv = [
    "nemo", "--chat-id", "oc_1", "--project-dir", project,
    "--api-key-env", "DEFINITELY_NOT_SET_XYZZY",
  ]
  with mock.patch.dict(os.environ, {}, clear=False), \
       mock.patch("sys.argv", argv), \
       mock.patch("nemo.__main__._ensure_provider_runtime"), \
       mock.patch("nemo.config.load_credentials",
                  return_value={"app_id": "a", "app_secret": "s", "email": ""}), \
       mock.patch("nemo.preflight.run_preflight", return_value=[]):
    # Make sure the env var is genuinely unset.
    os.environ.pop("DEFINITELY_NOT_SET_XYZZY", None)
    rc = main()
  assert rc == 1
