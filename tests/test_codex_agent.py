"""Tests for the CodexCodingAgent provider adapter."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from nemo.agent_factory import (
  build_coding_agent,
  default_model_for_agent,
  is_model_compatible,
  ModelCatalog,
  model_catalog_for_agent,
  query_codex_model_catalog,
)
from nemo.claude_agent import ClaudeCodingAgent
from nemo.codex_agent import CodexCodingAgent, _SIDE_CAR_SCRIPT
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


class _FakeProc:
  def __init__(
    self,
    stdout_lines: list[bytes],
    returncode: int = 0,
    stderr_lines: list[bytes] | None = None,
  ):
    self.stdin = _FakeStdin()
    self.stdout = _FakeStream(stdout_lines)
    self.stderr = _FakeStream(stderr_lines or [])
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


def test_default_model_for_agent():
  assert default_model_for_agent("claude") == "claude-opus-5"
  # Startup default stays stable; /model gets the live catalog from Codex.
  assert default_model_for_agent("codex") == "gpt-5.5"


def test_claude_model_catalog_tracks_current_aliases():
  catalog = model_catalog_for_agent("claude")
  assert catalog.visible[:5] == (
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "opusplan",
  )
  assert catalog.aliases["opus"] == "claude-opus-5"
  assert catalog.aliases["sonnet"] == "claude-sonnet-5"
  assert "claude-opus-4-8" in catalog.hidden
  assert "claude-sonnet-4-6" in catalog.hidden


def test_is_model_compatible():
  assert is_model_compatible("claude", "claude-opus-5")
  assert is_model_compatible("claude", "claude-sonnet-5")
  assert is_model_compatible("claude", "claude-opus-4-7")
  assert is_model_compatible("claude", "claude-opus-4-6")
  assert not is_model_compatible("claude", "gpt-5.5")
  with mock.patch(
    "nemo.agent_factory.query_codex_model_catalog",
    return_value=ModelCatalog(
      visible=("gpt-5.5",),
      hidden=("gpt-5-hidden",),
    ),
  ):
    assert is_model_compatible("codex", "gpt-5-hidden")
    assert is_model_compatible("codex", "gpt-5.5")
    assert not is_model_compatible("codex", "claude-sonnet-4-6")


def test_codex_model_catalog_uses_debug_models():
  payload = {
    "models": [
      {"slug": "codex-auto-review", "visibility": "hide", "priority": 43},
      {"slug": "gpt-5.3-codex-spark", "visibility": "list", "priority": 26},
      {"slug": "gpt-5.5", "visibility": "list", "priority": 7},
    ],
  }
  completed = subprocess.CompletedProcess(
    ["codex", "debug", "models"],
    0,
    stdout=json.dumps(payload),
    stderr="",
  )
  with mock.patch("nemo.agent_factory.subprocess.run", return_value=completed) as run:
    catalog = query_codex_model_catalog()

  run.assert_called_once()
  assert catalog.visible == ("gpt-5.5", "gpt-5.3-codex-spark")
  assert catalog.hidden == ()
  assert "codex-auto-review" not in catalog.all_names()
  assert catalog.api_only == ()
  assert "codex debug models" in catalog.note


def test_codex_model_catalog_failure_has_no_static_fallback():
  completed = subprocess.CompletedProcess(
    ["codex", "debug", "models"],
    1,
    stdout="",
    stderr="boom",
  )
  with mock.patch("nemo.agent_factory.subprocess.run", return_value=completed):
    catalog = query_codex_model_catalog()

  assert catalog.visible == ()
  assert catalog.hidden == ()
  assert "unavailable" in catalog.note
  assert "boom" in catalog.note


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


def test_codex_sidecar_deps_stale_detection(tmp_path):
  """When package.json pins a different codex-sdk version than what's
  installed, _ensure_runtime must wipe node_modules and reinstall — npm
  install alone won't cross a 0.x minor pin boundary, so a stale
  sidecar would keep using the old codex CLI even after the wheel
  upgrade. The most common symptom is the OpenAI API rejecting newer
  models ("'gpt-5.5' requires a newer version of Codex")."""
  import json as _json

  side_dir = tmp_path / "codex_sidecar"
  node_modules = side_dir / "node_modules"
  pkg_path = side_dir / "package.json"
  installed_path = node_modules / "@openai" / "codex-sdk" / "package.json"
  installed_path.parent.mkdir(parents=True)
  pkg_path.write_text(_json.dumps({
    "dependencies": {"@openai/codex-sdk": "0.128.0"}
  }))
  installed_path.write_text(_json.dumps({"version": "0.118.0"}))

  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  with mock.patch("nemo.codex_agent._SIDE_CAR_DIR", side_dir), \
       mock.patch("nemo.codex_agent._SIDE_CAR_PACKAGE", pkg_path), \
       mock.patch("nemo.codex_agent._SIDE_CAR_NODE_MODULES", node_modules):
    assert agent._sidecar_deps_stale() is True

    # Match → no reinstall needed.
    installed_path.write_text(_json.dumps({"version": "0.128.0"}))
    assert agent._sidecar_deps_stale() is False

    # Caret prefix in package.json should be stripped before compare.
    pkg_path.write_text(_json.dumps({
      "dependencies": {"@openai/codex-sdk": "^0.128.0"}
    }))
    assert agent._sidecar_deps_stale() is False
    installed_path.write_text(_json.dumps({"version": "0.127.5"}))
    assert agent._sidecar_deps_stale() is True


def test_codex_ensure_runtime_reinstalls_when_stale(tmp_path):
  """_ensure_runtime: stale node_modules → rmtree + _install_sidecar_deps."""
  side_dir = tmp_path / "codex_sidecar"
  node_modules = side_dir / "node_modules"
  script_path = side_dir / "run_turn.mjs"
  pkg_path = side_dir / "package.json"
  side_dir.mkdir()
  script_path.write_text("// stub")
  pkg_path.write_text('{"dependencies": {"@openai/codex-sdk": "0.128.0"}}')
  installed = node_modules / "@openai" / "codex-sdk" / "package.json"
  installed.parent.mkdir(parents=True)
  installed.write_text('{"version": "0.118.0"}')

  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  with mock.patch("nemo.codex_agent._SIDE_CAR_DIR", side_dir), \
       mock.patch("nemo.codex_agent._SIDE_CAR_SCRIPT", script_path), \
       mock.patch("nemo.codex_agent._SIDE_CAR_PACKAGE", pkg_path), \
       mock.patch("nemo.codex_agent._SIDE_CAR_NODE_MODULES", node_modules), \
       mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), \
       mock.patch.object(CodexCodingAgent, "_install_sidecar_deps") as mock_install:
    agent._ensure_runtime()
    mock_install.assert_called_once()
    # node_modules removed before reinstall.
    assert not node_modules.exists()


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
    # Fresh thread: cumulative == this turn, so the per-turn delta equals it.
    assert usage["input_tokens"] == 10
    assert usage["total_tokens"] == 15
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


def _codex_runtime_patches():
  """The mock.patch context every run_turn test reuses (sidecar + subprocess)."""
  return (
    mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
    mock.patch("nemo.codex_agent._SIDE_CAR_SCRIPT", Path("/tmp/run_turn.mjs")),
    mock.patch("nemo.codex_agent._SIDE_CAR_PACKAGE", Path("/tmp/package.json")),
    mock.patch.object(Path, "is_file", return_value=True),
  )


def test_codex_usage_is_per_turn_not_cumulative():
  # Codex reports turn.completed.usage as a SESSION-CUMULATIVE total. The
  # adapter must difference successive totals so each card shows THAT turn's
  # cost (matching Claude), not an ever-growing running total.
  async def _run():
    proc1 = _FakeProc([
      b'{"type":"thread.started","thread_id":"sess-1"}\n',
      b'{"type":"item.completed","item":{"type":"agent_message","text":"one"}}\n',
      b'{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":30,"output_tokens":10}}\n',
    ])
    proc2 = _FakeProc([
      b'{"type":"item.completed","item":{"type":"agent_message","text":"two"}}\n',
      b'{"type":"turn.completed","usage":{"input_tokens":250,"cached_input_tokens":80,"output_tokens":25}}\n',
    ])
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex")

    w, s, p, f = _codex_runtime_patches()
    with w, s, p, f, \
         mock.patch("asyncio.create_subprocess_exec", side_effect=[proc1, proc2]):
      _, usage1 = await agent.run_turn("first", lambda e: None)
      _, usage2 = await agent.run_turn("second", lambda e: None)

    # Turn 1 (cumulative == first turn): in = 100-30, cache_r = 30, out = 10.
    assert usage1 == {
      "input_tokens": 70, "cache_read_input_tokens": 30,
      "cache_creation_input_tokens": 0, "output_tokens": 10, "total_tokens": 110,
    }
    # Turn 2 is the DELTA of the cumulative totals (150/50/15), NOT raw 250/80/25.
    assert usage2 == {
      "input_tokens": 100, "cache_read_input_tokens": 50,
      "cache_creation_input_tokens": 0, "output_tokens": 15, "total_tokens": 165,
    }

  asyncio.run(_run())


def test_codex_usage_baseline_seeded_from_rollout(tmp_path):
  # On resume the in-memory baseline is gone; recover the cumulative total from
  # the LAST token_count record in the rollout so the first turn isn't reported
  # as the whole session.
  from nemo.codex_agent import _read_codex_cumulative
  rollout = tmp_path / "rollout.jsonl"
  rollout.write_text(
    '{"payload":{"type":"token_count","info":{"total_token_usage":'
    '{"input_tokens":100,"cached_input_tokens":20,"output_tokens":5}}}}\n'
    '{"payload":{"type":"token_count","info":{"total_token_usage":'
    '{"input_tokens":250,"cached_input_tokens":60,"output_tokens":18}}}}\n'
  )
  with mock.patch("nemo.codex_agent._find_codex_rollout", return_value=str(rollout)):
    baseline = _read_codex_cumulative("sess-x")
  assert baseline == {"input_tokens": 250, "cached_input_tokens": 60, "output_tokens": 18}


def test_codex_first_turn_after_resume_diffs_seeded_baseline(tmp_path):
  async def _run():
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
      '{"payload":{"type":"token_count","info":{"total_token_usage":'
      '{"input_tokens":200,"cached_input_tokens":50,"output_tokens":20}}}}\n'
    )
    proc = _FakeProc([
      b'{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}\n',
      b'{"type":"turn.completed","usage":{"input_tokens":260,"cached_input_tokens":70,"output_tokens":28}}\n',
    ])
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    with mock.patch("nemo.codex_agent._find_codex_rollout", return_value=str(rollout)):
      await agent.start("/tmp/project", "gpt-5-codex", resume="sess-x")

    w, s, p, f = _codex_runtime_patches()
    with w, s, p, f, \
         mock.patch("asyncio.create_subprocess_exec", return_value=proc):
      _, usage = await agent.run_turn("next", lambda e: None)

    # Delta vs seeded {200,50,20}: in=(260-200)-(70-50)=40, cache_r=20, out=8.
    assert usage == {
      "input_tokens": 40, "cache_read_input_tokens": 20,
      "cache_creation_input_tokens": 0, "output_tokens": 8, "total_tokens": 68,
    }

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


def test_codex_compact_failure_gets_recovery_hint_from_turn_failed():
  async def _run():
    def _proc():
      return _FakeProc([
        b'{"type":"turn.failed","error":{"message":"Error running remote compact task: stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses/compact)"}}\n',
      ])
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex", resume="sess-x")

    events = []
    w, s, p, f = _codex_runtime_patches()
    with w, s, p, f, \
         mock.patch("asyncio.create_subprocess_exec",
                    side_effect=[_proc(), _proc(), _proc()]) as exec_mock, \
         mock.patch("asyncio.sleep", new=mock.AsyncMock()), \
         pytest.raises(RuntimeError) as raised:
      await agent.run_turn("continue", events.append)

    assert exec_mock.call_count == 3
    assert "Codex session compaction failed" in str(raised.value)
    assert "`/clear`" in str(raised.value)
    assert "`/session recall`" in str(raised.value)
    assert any(
      isinstance(e, ProgressEvent) and "retrying (2/3)" in e.summary
      for e in events
    )
    assert isinstance(events[-1], ErrorEvent)
    assert "memory intact" in events[-1].message

  asyncio.run(_run())


def test_codex_compact_failure_gets_recovery_hint_from_stderr_exit():
  async def _run():
    def _proc():
      return _FakeProc(
        [],
        returncode=1,
        stderr_lines=[
          b"2026-06-29T09:41:11Z ERROR codex_core::compact_remote: remote compaction failed compact_error=stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses/compact)\n",
        ],
      )
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex", resume="sess-x")

    events = []
    w, s, p, f = _codex_runtime_patches()
    with w, s, p, f, \
         mock.patch("asyncio.create_subprocess_exec",
                    side_effect=[_proc(), _proc(), _proc()]) as exec_mock, \
         mock.patch("asyncio.sleep", new=mock.AsyncMock()), \
         pytest.raises(RuntimeError) as raised:
      await agent.run_turn("continue", events.append)

    assert exec_mock.call_count == 3
    assert "Codex session compaction failed" in str(raised.value)
    assert "`/clear`" in str(raised.value)
    assert isinstance(events[-1], ErrorEvent)
    assert "responses/compact" in events[-1].message

  asyncio.run(_run())


def test_codex_backend_failure_retries_before_progress():
  async def _run():
    fail = _FakeProc(
      [],
      returncode=1,
      stderr_lines=[
        b"failed to connect to websocket: IO error: tls handshake eof, url: wss://chatgpt.com/backend-api/codex/responses\n",
      ],
    )
    ok = _FakeProc([
      b'{"type":"thread.started","thread_id":"sess-2"}\n',
      b'{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n',
      b'{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":0,"output_tokens":2}}\n',
    ])
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex")

    events = []
    w, s, p, f = _codex_runtime_patches()
    with w, s, p, f, \
         mock.patch("asyncio.create_subprocess_exec", side_effect=[fail, ok]) as exec_mock, \
         mock.patch("asyncio.sleep", new=mock.AsyncMock()):
      _, usage = await agent.run_turn("continue", events.append)

    assert exec_mock.call_count == 2
    assert fail.stdin.writes == [b"continue"]
    assert ok.stdin.writes == [b"continue"]
    assert any(
      isinstance(e, ProgressEvent) and "retrying (2/3)" in e.summary
      for e in events
    )
    assert any(isinstance(e, AnswerEvent) and e.text == "Done" for e in events)
    assert any(isinstance(e, DoneEvent) for e in events)
    assert usage["total_tokens"] == 12

  asyncio.run(_run())


def test_codex_backend_failure_does_not_retry_after_progress():
  async def _run():
    proc = _FakeProc(
      [
        b'{"type":"item.completed","item":{"type":"command_execution","command":"python mutate.py"}}\n',
      ],
      returncode=1,
      stderr_lines=[
        b"stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)\n",
      ],
    )
    agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
    await agent.start("/tmp/project", "gpt-5-codex")

    events = []
    w, s, p, f = _codex_runtime_patches()
    with w, s, p, f, \
         mock.patch("asyncio.create_subprocess_exec", return_value=proc) as exec_mock, \
         pytest.raises(RuntimeError) as raised:
      await agent.run_turn("continue", events.append)

    assert exec_mock.call_count == 1
    assert "chatgpt.com/backend-api/codex/responses" in str(raised.value)
    assert isinstance(events[0], ProgressEvent)
    assert isinstance(events[-1], ErrorEvent)

  asyncio.run(_run())


def test_codex_trailing_note_warns_when_context_is_large():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._cum_usage = {
    "input_tokens": 181_000,
    "cached_input_tokens": 120_000,
    "output_tokens": 2_000,
  }

  note = agent.trailing_note("sess-x")

  assert "Codex context is getting large" in note
  assert "⚠️" in note
  assert "`/clear`" in note
  assert "`/session recall`" in note


def test_codex_trailing_note_stays_quiet_below_warning_threshold():
  agent = CodexCodingAgent({}, "oc_1", _DummyDB(), _DummyChannel())
  agent._cum_usage = {
    "input_tokens": 100_000,
    "cached_input_tokens": 50_000,
    "output_tokens": 1_000,
  }

  assert agent.trailing_note("sess-x") == ""
