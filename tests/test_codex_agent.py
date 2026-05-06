"""Tests for the CodexCodingAgent provider adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

from nemo.agent_factory import build_coding_agent, default_model_for_provider, is_model_compatible
from nemo.claude_agent import ClaudeCodingAgent
from nemo.codex_agent import CodexCodingAgent, _SIDE_CAR_SCRIPT
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
  assert default_model_for_provider("claude") == "claude-opus-4-7"
  # Codex default must work for both ChatGPT subscribers and API users —
  # the codex-specialized (-codex) slugs are API-only.
  assert default_model_for_provider("codex") == "gpt-5.5"


def test_is_model_compatible():
  assert is_model_compatible("claude", "claude-opus-4-7")
  assert is_model_compatible("claude", "claude-opus-4-6")
  assert not is_model_compatible("claude", "gpt-5.3-codex")
  assert is_model_compatible("codex", "gpt-5.3-codex")
  assert is_model_compatible("codex", "gpt-5.5")
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


def test_codex_set_effort_clamps_max_to_high():
  # Codex SDK has no `max` tier — clamp Claude's max down to high so the
  # shared knob accepts it without rejecting the user's intent.
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent.set_effort("max")
  assert agent._effort == "high"


def test_claude_set_effort_uses_native_sdk_field():
  agent = ClaudeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  # All four levels supported by claude-agent-sdk's effort literal.
  for level in ("low", "medium", "high", "max"):
    agent.set_effort(level)
    assert agent._effort == level
  agent.set_effort("")
  assert agent._effort == ""
  agent.set_effort("bogus")
  assert agent._effort == ""
  agent.set_effort("ultrathink")  # old keyword, no longer accepted as a level
  assert agent._effort == ""


def test_claude_build_options_passes_native_effort():
  # _build_options grabs the running loop for the askq handler, so wrap
  # the assertions in an async context.
  async def _run():
    agent = ClaudeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    agent.set_effort("max")
    opts = agent._build_options("/tmp/project", "claude-opus-4-7")
    assert getattr(opts, "effort", None) == "max"

    agent.set_effort("")
    opts = agent._build_options("/tmp/project", "claude-opus-4-7")
    # Empty string clears effort — option should be unset (None).
    assert getattr(opts, "effort", None) is None

  asyncio.run(_run())


def test_codex_prepare_prompt_injects_system_prompt_on_first_turn():
  agent = CodexCodingAgent(
    {}, "oc_1", _DummyDB(), _DummyChannel(),
    system_prompt="Be extra polite.",
  )
  # No session_id yet → first turn gets the instructions prepended.
  out = agent._prepare_prompt("hello")
  assert "<system_instructions>" in out
  assert "Be extra polite." in out
  assert out.endswith("hello")
  # After a session is established, subsequent turns do NOT re-inject.
  agent._session_id = "sess-1"
  assert agent._prepare_prompt("next") == "next"


def test_codex_prepare_prompt_noop_without_system_prompt():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  assert agent._prepare_prompt("hello") == "hello"


def test_claude_system_prompt_appended_to_agent_prompt():
  agent = ClaudeCodingAgent(
    {}, "oc_1", _DummyDB(), _DummyChannel(),
    system_prompt="Follow the house style guide.",
  )
  built = agent._build_agent_prompt()
  assert "Follow the house style guide." in built
  # Default agent_prompt preamble is still present.
  assert "Nemo" in built


def test_claude_build_agent_prompt_without_user_prompt():
  agent = ClaudeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  built = agent._build_agent_prompt()
  assert "Nemo" in built
  # No trailing double newline from empty append.
  assert not built.endswith("\n\n")


def test_claude_system_prompt_empty_by_default():
  agent = ClaudeCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  assert agent._system_prompt == ""


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
  # File-change preview shows basename only (mirrors Claude's Edit/Write).
  assert agent._item_summary({
    "type": "file_change",
    "changes": [{"kind": "update", "path": "nemo/agent.py"}],
  }) == "update:agent.py"
  # Long commands are flattened + truncated to 60 chars with ellipsis.
  long_cmd = "python3 - <<'PY'\n" + "x = 1\n" * 30 + "PY"
  summary = agent._item_summary({"type": "command_execution", "command": long_cmd})
  assert summary.startswith("$ ")
  assert summary.endswith("...")
  assert len(summary) <= 62  # "$ " + 60
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
    assert isinstance(events[0], ProgressEvent)
    assert events[0].first is True
    assert isinstance(events[1], ProgressEvent)
    assert events[1].first is False
    assert isinstance(events[2], AnswerEvent)
    assert events[2].text == "Done"
    assert isinstance(events[3], DoneEvent)
    assert proc.stdin.writes == [b"fix it"]
    assert proc.stdin.closed is True

  asyncio.run(_run())


def test_codex_run_turn_raises_stdout_buffer_limit():
  # A single sidecar JSON event (large reasoning / agent_message) must not
  # be capped at asyncio's 64 KB default, which would raise
  # "Separator is found, but chunk is longer than limit".
  async def _run():
    proc = _FakeProc([
      b'{"type":"turn.completed","usage":{}}\n',
    ])
    captured: dict[str, object] = {}

    async def _fake_exec(*args: object, **kwargs: object):
      captured["kwargs"] = kwargs
      return proc

    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex")

    with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
         mock.patch("nemo.codex_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")), \
         mock.patch("nemo.codex_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")), \
         mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
      await agent.run_turn("hi", lambda _e: None)

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("limit") == 16 * 1024 * 1024

  asyncio.run(_run())
