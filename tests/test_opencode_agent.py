"""Tests for the OpenCodeCodingAgent provider adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

from nemo.agent_factory import build_coding_agent, default_model_for_provider, is_model_compatible
from nemo.opencode_agent import (
  OpenCodeCodingAgent,
  _SIDE_CAR_SCRIPT,
  _build_agent_prompt,
  query_opencode_model_catalog_data,
)
from nemo.turn import AnswerEvent, DoneEvent, ProgressEvent


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


class _FakeProc:
  def __init__(self, stdout_lines: list[bytes], returncode: int = 0):
    self.stdin = _FakeStdin()
    self.stdout = _FakeStream(stdout_lines)
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


def test_default_model_for_provider():
  assert default_model_for_provider("opencode") == "default"


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
    assert usage["input_tokens"] == 10
    assert usage["cached_input_tokens"] == 2

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
