"""Tests for the CodexCodingAgent provider adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

from nemo.agent_factory import build_coding_agent, default_model_for_provider, is_model_compatible
from nemo.claude_agent import ClaudeCodingAgent
from nemo.codex_agent import CodexCodingAgent, _SIDE_CAR_SCRIPT
from nemo.turn import DoneEvent, TextEvent, ToolProgressEvent, ToolStartEvent


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
  assert default_model_for_provider("claude") == "claude-opus-4-6"
  assert default_model_for_provider("codex") == "gpt-5-codex"


def test_is_model_compatible():
  assert is_model_compatible("claude", "claude-opus-4-6")
  assert not is_model_compatible("claude", "gpt-5-codex")
  assert is_model_compatible("codex", "gpt-5-codex")
  assert not is_model_compatible("codex", "claude-sonnet-4-6")


def test_build_coding_agent_returns_expected_class():
  db = _DummyDB()
  channel = _DummyChannel()
  credentials = {"app_id": "a", "app_secret": "s"}
  claude = build_coding_agent("claude", credentials, "oc_1", db, channel)
  codex = build_coding_agent("codex", credentials, "oc_1", db, channel)
  assert isinstance(claude, ClaudeCodingAgent)
  assert isinstance(codex, CodexCodingAgent)


def test_codex_build_command_new_session():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._project_dir = "/tmp/project"
  agent._model = "gpt-5-codex"
  cmd = agent._build_command()
  assert cmd == [
    "node", str(_SIDE_CAR_SCRIPT),
    "--cwd", "/tmp/project",
    "--model", "gpt-5-codex",
  ]


def test_codex_build_command_resume():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._project_dir = "/tmp/project"
  agent._model = "gpt-5-codex"
  agent._session_id = "sess-123"
  cmd = agent._build_command()
  assert cmd[-2:] == ["--resume", "sess-123"]


def test_codex_build_command_effort():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._project_dir = "/tmp/project"
  agent._model = "gpt-5-codex"
  agent.set_effort("high")
  cmd = agent._build_command()
  assert "--effort" in cmd
  assert cmd[cmd.index("--effort") + 1] == "high"


def test_codex_set_effort_rejects_invalid():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent.set_effort("bogus")
  assert agent._effort == ""
  agent.set_effort("medium")
  assert agent._effort == "medium"
  agent.set_effort("")
  assert agent._effort == ""


def test_claude_prefix_prompt_effort_keywords():
  agent = ClaudeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  assert agent._prefix_prompt("fix bug") == "fix bug"
  agent.set_effort("low")
  assert agent._prefix_prompt("fix bug") == "think\n\nfix bug"
  agent.set_effort("medium")
  assert agent._prefix_prompt("fix bug") == "think hard\n\nfix bug"
  agent.set_effort("high")
  assert agent._prefix_prompt("fix bug") == "ultrathink\n\nfix bug"
  agent.set_effort("")
  assert agent._prefix_prompt("fix bug") == "fix bug"
  agent.set_effort("bogus")
  assert agent._prefix_prompt("fix bug") == "fix bug"


def test_codex_build_command_rejects_permission_mode():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel(), permission_mode="default")
  agent._project_dir = "/tmp/project"
  try:
    agent._build_command()
  except RuntimeError as exc:
    assert "bypassPermissions" in str(exc)
  else:
    raise AssertionError("expected RuntimeError")


def test_codex_parse_event_invalid_json():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  assert agent._parse_event("not-json") is None


def test_codex_ensure_runtime_checks_sidecar():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
    with mock.patch("nemo.codex_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.codex_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")):
      try:
        agent._ensure_runtime()
      except RuntimeError as exc:
        assert "sidecar" in str(exc)
      else:
        raise AssertionError("expected RuntimeError")


def test_codex_item_summary_variants():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  assert agent._item_summary({"type": "command_execution", "command": "ls -la"}) == "$ ls -la"
  assert agent._item_summary({
    "type": "file_change",
    "changes": [{"kind": "update", "path": "nemo/agent.py"}],
  }) == "update:nemo/agent.py"
  assert agent._item_summary({
    "type": "mcp_tool_call", "server": "github", "tool": "fetch_pr",
  }) == "github: fetch_pr"


def test_codex_run_turn_maps_events():
  async def _run():
    lines = [
      b'{"type":"thread.started","thread_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"reasoning","text":"Inspect repo"}}\n',
      b'{"type":"item.completed","item":{"type":"command_execution","command":"pytest","status":"completed"}}\n',
      b'{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n',
      b'{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5,"cached_input_tokens":0}}\n',
    ]
    proc = _FakeProc(lines)
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex")

    events = []
    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.codex_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.codex_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", return_value=proc):
      cost, usage = await agent.run_turn("fix it", events.append)

    assert cost == 0.0
    assert usage["input_tokens"] == 10
    assert agent._session_id == "sess-1"
    assert isinstance(events[0], ToolStartEvent)
    assert isinstance(events[1], ToolProgressEvent)
    assert isinstance(events[2], TextEvent)
    assert events[2].text == "Done"
    assert isinstance(events[3], DoneEvent)
    assert proc.stdin.writes == [b"fix it"]
    assert proc.stdin.closed is True

  asyncio.run(_run())
