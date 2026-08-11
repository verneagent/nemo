"""Tests for the OpenCodeCodingAgent provider adapter."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest import mock

from nemo.agent_factory import build_coding_agent, default_model_for_agent, is_model_compatible
from nemo.opencode_agent import (
  OpenCodeCodingAgent,
  _SIDE_CAR_SCRIPT,
  _build_agent_prompt,
  query_opencode_model_catalog_data,
)
from nemo.presets import Preset
from nemo.turn import AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent


class _DummyDB:
  pass


class _DummyChannel:
  pass


class _FakeStream:
  def __init__(self, lines: list[bytes]):
    self._lines = list(lines)

  async def readline(self) -> bytes:
    if self._lines:
      return self._lines.pop(0)
    return b""


class _FakeStdin:
  def __init__(self):
    self.writes: list[bytes] = []
    self.closed = False

  def write(self, data: bytes) -> None:
    self.writes.append(data)

  async def drain(self) -> None:
    return None

  def close(self) -> None:
    self.closed = True


class _BlockingStream:
  """Stream that yields lines then blocks forever (cancellable).

  Simulates a sidecar that produced events but whose model follow-up round
  never arrives — the exact oc_623b stall (reasoning + skill tool, then
  silence). ``asyncio.wait_for`` cancels the pending ``readline``.
  """

  def __init__(self, lines: list[bytes]):
    self._lines = list(lines)
    self._block = asyncio.Event()

  async def readline(self) -> bytes:
    if self._lines:
      return self._lines.pop(0)
    await self._block.wait()
    return b""


class _FakeProc:
  def __init__(self, stdout_lines: list[bytes] | None = None,
               returncode: int = 0, stdout=None):
    self.stdin = _FakeStdin()
    self.stdout = stdout if stdout is not None else _FakeStream(stdout_lines or [])
    self.stderr = _FakeStream([])
    self.returncode = None
    self._final_returncode = returncode
    self.terminated = False
    self.killed = False

  async def wait(self) -> int:
    self.returncode = self._final_returncode
    return self._final_returncode

  def terminate(self) -> None:
    self.terminated = True
    self.returncode = -15

  def kill(self) -> None:
    self.killed = True
    self.returncode = -9


class _HangingWaitProc(_FakeProc):
  """Fake proc whose wait() never resolves (cancellable).

  Simulates the pathological case where the sidecar died but the asyncio child
  watcher never resolves the wait — the exact oc_623b wedge. run_turn must hit
  the hard ceiling instead of hanging the group.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._wait_event = asyncio.Event()

  async def wait(self) -> int:
    await self._wait_event.wait()  # never set
    return 0


def test_default_model_for_agent():
  assert default_model_for_agent("opencode") == "default"


def test_is_model_compatible():
  assert is_model_compatible("opencode", "default")
  assert is_model_compatible("opencode", "anthropic/claude-sonnet-4-5")
  assert not is_model_compatible("opencode", "claude-sonnet-4-5")


def test_build_coding_agent_returns_expected_class():
  db = _DummyDB()
  channel = _DummyChannel()
  credentials = {"app_id": "a", "app_secret": "s"}
  agent = build_coding_agent("opencode", credentials, "oc_1", db, channel)
  assert isinstance(agent, OpenCodeCodingAgent)


def test_opencode_normalize_usage_to_canonical():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  # Sidecar input_tokens INCLUDES the cached read, so new-uncached = 1000-200.
  assert agent._normalize_usage({
    "input_tokens": 1000, "output_tokens": 80, "cached_input_tokens": 200,
  }) == {
    "input_tokens": 800, "cache_read_input_tokens": 200,
    "cache_creation_input_tokens": 0, "output_tokens": 80, "total_tokens": 1080,
  }
  assert agent._normalize_usage({}) == {}


def test_opencode_build_command_new_session():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._project_dir = "/tmp/project"
  agent._model = "anthropic/claude-sonnet-4-5"
  cmd = agent._build_command()
  assert cmd == [
    "node", str(_SIDE_CAR_SCRIPT),
    "--cwd", "/tmp/project",
    "--model", "anthropic/claude-sonnet-4-5",
  ]


def test_opencode_build_command_resume():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._project_dir = "/tmp/project"
  agent._model = "default"
  agent._session_id = "sess-123"
  cmd = agent._build_command()
  assert cmd[-2:] == ["--resume", "sess-123"]


def test_opencode_sidecar_model_passthrough():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  for model in ("", "default", "deepseek/deepseek-v4-flash"):
    agent._model = model
    assert agent._sidecar_model() == (model, None)


def test_opencode_sidecar_model_unresolvable_passthrough():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._model = "some-garbage"
  with mock.patch("nemo.presets.resolve_preset", return_value=None):
    assert agent._sidecar_model() == ("some-garbage", None)


def test_opencode_sidecar_model_translates_single_name_preset():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._model = "oc-deepseek-v4-flash"
  preset = Preset(
    name="oc-deepseek-v4-flash",
    openai_url="https://opencode.ai/zen/go/v1",
    openai_remote="deepseek-v4-flash",
    api_key_env="OPENCODE_GO_API_KEY",
  )
  with mock.patch("nemo.presets.resolve_preset", return_value=preset), \
       mock.patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "sk-zen"}):
    model_arg, provider = agent._sidecar_model()
  assert model_arg == "nemo/deepseek-v4-flash"
  assert provider == (
    "@ai-sdk/openai-compatible",
    "https://opencode.ai/zen/go/v1",
    "sk-zen",
  )


def test_opencode_sidecar_model_anthropic_only_preset():
  # Kimi For Coding has no OpenAI endpoint → the injected provider must use
  # the Anthropic AI-SDK package against the anthropic URL.
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._model = "kimi-for-coding"
  preset = Preset(
    name="kimi-for-coding",
    anthropic_url="https://api.kimi.com/coding",
    anthropic_remote="kimi-for-coding",
    api_key_env="KIMI_CODE_API_KEY",
  )
  with mock.patch("nemo.presets.resolve_preset", return_value=preset), \
       mock.patch.dict(os.environ, {"KIMI_CODE_API_KEY": "sk-kimi"}):
    model_arg, provider = agent._sidecar_model()
  assert model_arg == "nemo/kimi-for-coding"
  assert provider == (
    "@ai-sdk/anthropic",
    "https://api.kimi.com/coding",
    "sk-kimi",
  )


def test_opencode_build_command_translates_single_name_preset():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._project_dir = "/tmp/project"
  agent._model = "oc-deepseek-v4-flash"
  preset = Preset(
    name="oc-deepseek-v4-flash",
    openai_url="https://opencode.ai/zen/go/v1",
    openai_remote="deepseek-v4-flash",
  )
  with mock.patch("nemo.presets.resolve_preset", return_value=preset):
    cmd = agent._build_command()
  assert cmd[cmd.index("--model") + 1] == "nemo/deepseek-v4-flash"


def test_opencode_build_env_injects_provider_for_single_name_preset():
  agent = OpenCodeCodingAgent(
    {}, "oc_1", _DummyDB(), _DummyChannel(),
    system_prompt="Follow the house style guide.",
  )
  agent._project_dir = "/tmp/project"
  agent._model = "oc-deepseek-v4-flash"
  preset = Preset(
    name="oc-deepseek-v4-flash",
    openai_url="https://opencode.ai/zen/go/v1",
    openai_remote="deepseek-v4-flash",
    api_key_env="OPENCODE_GO_API_KEY",
  )
  with mock.patch("nemo.presets.resolve_preset", return_value=preset), \
       mock.patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "sk-zen"}):
    env = agent._build_env()
  assert env["NEMO_OPENCODE_PROVIDER_URL"] == "https://opencode.ai/zen/go/v1"
  assert env["NEMO_OPENCODE_PROVIDER_API_KEY"] == "sk-zen"
  assert env["NEMO_OPENCODE_PROVIDER_NPM"] == "@ai-sdk/openai-compatible"
  # The injected provider carries the endpoint; the legacy blanket
  # base-url override must NOT also be set (it could redirect OpenCode's
  # native providers).
  assert "OPENAI_BASE_URL" not in env
  assert "ANTHROPIC_BASE_URL" not in env


def test_opencode_prepare_prompt_injects_effort_only():
  agent = OpenCodeCodingAgent(
    {}, "oc_1", _DummyDB(), _DummyChannel(),
    system_prompt="Follow the house style guide.",
  )
  agent.set_effort("high")
  out = agent._prepare_prompt("hello")
  assert "Reason more carefully" in out
  assert "system_instructions" not in out
  assert "Follow the house style guide." not in out


def test_opencode_prepare_prompt_medium_effort_prefix():
  # /effort medium was a silent no-op (prefix map only had low/high).
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent.set_effort("medium")
  out = agent._prepare_prompt("hello")
  assert "moderate care" in out


def test_opencode_build_env_exports_nemo_system_prompt():
  agent = OpenCodeCodingAgent(
    {}, "oc_1", _DummyDB(), _DummyChannel(),
    system_prompt="Follow the house style guide.",
  )
  agent._project_dir = "/tmp/project"
  env = agent._build_env()
  assert env["NEMO_CHAT_ID"] == "oc_1"
  assert env["NEMO_DB"].endswith("nemo.db")
  assert env["NEMO_OPENCODE_SYSTEM_PROMPT"] == _build_agent_prompt(
    "Follow the house style guide."
  )


def test_opencode_build_command_rejects_permission_mode():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel(), permission_mode="default")
  agent._project_dir = "/tmp/project"
  try:
    agent._build_command()
  except RuntimeError as exc:
    assert "bypassPermissions" in str(exc)
  else:
    raise AssertionError("expected RuntimeError")


def test_opencode_parse_event_invalid_json():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  assert agent._parse_event("not-json") is None


def test_opencode_ensure_runtime_checks_sidecar():
  agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
    with mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")):
      try:
        agent._ensure_runtime()
      except RuntimeError as exc:
        assert "sidecar" in str(exc)
      else:
        raise AssertionError("expected RuntimeError")


def test_opencode_run_turn_maps_events():
  async def _run():
    lines = [
      b'{"type":"session.started","session_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"reasoning","text":"Inspect repo"}}\n',
      b'{"type":"item.completed","item":{"type":"tool_call","tool":"bash","title":"Run tests"}}\n',
      b'{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n',
      b'{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5,"cached_input_tokens":2},"cost":0.25}\n',
    ]
    proc = _FakeProc(lines)
    agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "anthropic/claude-sonnet-4-5")

    events = []
    captured_env = {}

    async def _fake_spawn(*args, **kwargs):
      del args
      captured_env.update(kwargs.get("env", {}))
      return proc

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(Path, "is_dir", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_spawn):
      cost, usage = await agent.run_turn("fix it", events.append)

    assert isinstance(events[0], ProgressEvent)
    assert events[0].kind == "reasoning"
    assert isinstance(events[1], ProgressEvent)
    assert events[1].summary == "bash: Run tests"
    assert isinstance(events[2], AnswerEvent)
    assert events[2].text == "Done"
    assert isinstance(events[3], DoneEvent)
    assert events[3].session_id == "sess-1"
    assert "NEMO_OPENCODE_SYSTEM_PROMPT" in captured_env
    assert cost == 0.25
    # Usage is normalized to the canonical schema: sidecar input (10) includes
    # the cached read (2), so new-uncached in = 8, cache_r = 2, total = 15.
    assert usage["input_tokens"] == 8
    assert usage["cache_read_input_tokens"] == 2
    assert usage["total_tokens"] == 15

  asyncio.run(_run())


def test_opencode_run_turn_idle_timeout_on_silent_sidecar():
  # Regression for the oc_623b hang: the model reasoned, called the native
  # `skill` tool (completed), then the follow-up round never arrived. The
  # sidecar went silent; run_turn must raise TimeoutError (→ the main loop's
  # "Timed out — context preserved" card) instead of wedging the group.
  async def _run():
    lines = [
      b'{"type":"session.started","session_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"reasoning","text":"Inspect"}}\n',
      b'{"type":"item.completed","item":{"type":"tool_call","tool":"skill","title":"agent-reach","status":"completed"}}\n',
    ]
    proc = _FakeProc(stdout=_BlockingStream(lines))
    agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "anthropic/claude-sonnet-4-5")

    events = []

    async def _fake_spawn(*args, **kwargs):
      return proc

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(Path, "is_dir", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_spawn), \
         mock.patch("nemo.opencode_agent._IDLE_TIMEOUT", 0.05), \
         mock.patch("nemo.opencode_agent._TURN_TIMEOUT", 30.0):
      try:
        await agent.run_turn("fix it", events.append)
      except TimeoutError as exc:
        assert "stopped responding" in str(exc)
      else:
        raise AssertionError("expected TimeoutError")

    # The sidecar was force-stopped on timeout.
    assert proc.terminated
    # Events up to the stall were surfaced before the timeout.
    assert any(isinstance(e, ProgressEvent) and e.kind == "reasoning" for e in events)
    assert any(isinstance(e, ProgressEvent) and e.kind == "tool" for e in events)

  asyncio.run(_run())


def test_opencode_run_turn_hard_ceiling_while_tool_in_flight():
  # The idle clock is disarmed while a tool is running (a long bash produces no
  # events), but the hard ceiling must still fire.
  async def _run():
    lines = [
      b'{"type":"session.started","session_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"reasoning","text":"Inspect"}}\n',
      b'{"type":"item.completed","item":{"type":"tool_call","tool":"bash","title":"Build","status":"running"}}\n',
    ]
    proc = _FakeProc(stdout=_BlockingStream(lines))
    agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "anthropic/claude-sonnet-4-5")

    events = []

    async def _fake_spawn(*args, **kwargs):
      return proc

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(Path, "is_dir", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_spawn), \
         mock.patch("nemo.opencode_agent._IDLE_TIMEOUT", 30.0), \
         mock.patch("nemo.opencode_agent._TURN_TIMEOUT", 0.05):
      try:
        await agent.run_turn("fix it", events.append)
      except TimeoutError as exc:
        assert "ceiling" in str(exc)
      else:
        raise AssertionError("expected TimeoutError")

    assert proc.terminated

  asyncio.run(_run())


def test_opencode_run_turn_eof_without_completion_raises_recoverable():
  # Regression for the oc_623b `task`-tool recurrence: the sidecar died
  # mid-turn and the stream ended (EOF) WITHOUT any completion event. run_turn
  # must raise a recoverable TimeoutError (→ "context preserved, re-send"),
  # never emit a silent empty DoneEvent.
  async def _run():
    lines = [
      b'{"type":"session.started","session_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"reasoning","text":"Inspect"}}\n',
      b'{"type":"item.completed","item":{"type":"tool_call","tool":"task","title":"research aptos","status":"completed"}}\n',
    ]
    proc = _FakeProc(lines)  # rc=0, stream ends (EOF) after the lines
    agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "anthropic/claude-sonnet-4-5")

    events = []

    async def _fake_spawn(*args, **kwargs):
      return proc

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(Path, "is_dir", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_spawn):
      try:
        await agent.run_turn("fix it", events.append)
      except TimeoutError as exc:
        assert "without a completion event" in str(exc)
      else:
        raise AssertionError("expected TimeoutError")

    # Events before the drop were surfaced; no DoneEvent was invented.
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert not any(isinstance(e, DoneEvent) for e in events)

  asyncio.run(_run())


def test_opencode_run_turn_session_idle_relay_is_a_completion():
  # The sidecar relays session.idle on a normal idle finish. That must be
  # treated as a completion (not a silent drop), so the turn emits DoneEvent.
  async def _run():
    lines = [
      b'{"type":"session.started","session_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"agent_message","text":"Aptos news"}}\n',
      b'{"type":"session.idle","session_id":"sess-1"}\n',
    ]
    proc = _FakeProc(lines)  # rc=0
    agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "anthropic/claude-sonnet-4-5")

    events = []

    async def _fake_spawn(*args, **kwargs):
      return proc

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(Path, "is_dir", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_spawn):
      cost, usage = await agent.run_turn("Aptos 近况怎样", events.append)

    assert any(isinstance(e, AnswerEvent) and e.text == "Aptos news" for e in events)
    assert any(isinstance(e, DoneEvent) for e in events)
    assert not any(isinstance(e, ErrorEvent) for e in events)

  asyncio.run(_run())


def test_opencode_run_turn_wedged_proc_wait_hits_ceiling():
  # The sidecar died and readline hit EOF, but proc.wait() never resolves (the
  # pathological asyncio child-watcher wedge). The hard ceiling must fire
  # instead of hanging the group until an external SIGKILL.
  async def _run():
    lines = [
      b'{"type":"session.started","session_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"reasoning","text":"Inspect"}}\n',
    ]
    proc = _HangingWaitProc(stdout=_FakeStream(lines))
    agent = OpenCodeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "anthropic/claude-sonnet-4-5")

    events = []

    async def _fake_spawn(*args, **kwargs):
      return proc

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.opencode_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(Path, "is_dir", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_spawn), \
         mock.patch("nemo.opencode_agent._IDLE_TIMEOUT", 30.0), \
         mock.patch("nemo.opencode_agent._TURN_TIMEOUT", 0.05):
      try:
        await agent.run_turn("fix it", events.append)
      except TimeoutError as exc:
        assert "ceiling" in str(exc)
      else:
        raise AssertionError("expected TimeoutError")

    assert proc.terminated

  asyncio.run(_run())


def test_query_opencode_model_catalog_data():
  payload = """{"models":["anthropic/claude-sonnet-4-5","openai/gpt-5"],"default_model":"openai/gpt-5"}"""
  completed = mock.Mock(stdout=payload)
  with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
       mock.patch("nemo.opencode_agent._LIST_MODELS_SCRIPT", Path("/tmp/list_models.mjs")), \
       mock.patch.object(Path, "is_file", return_value=True), \
       mock.patch("subprocess.run", return_value=completed):
    models, note = query_opencode_model_catalog_data("/tmp/project")
  assert models == ("anthropic/claude-sonnet-4-5", "openai/gpt-5")
  assert "openai/gpt-5" in note
