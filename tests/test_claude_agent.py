"""Tests for ClaudeCodingAgent helpers."""

from __future__ import annotations

import os

from nemo.claude_agent import (
  _SESSION_SIZE_NUDGE,
  _SESSION_SIZE_STRONG,
  _format_size_warning,
  _session_jsonl_path,
)


def test_session_jsonl_path_uses_slug(tmp_path, monkeypatch):
  monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
  p = _session_jsonl_path("/Users/foo/teams/irisy/mockup", "abc-123")
  assert p == str(tmp_path / "projects" / "-Users-foo-teams-irisy-mockup" / "abc-123.jsonl")


def test_session_jsonl_path_default_config_dir(monkeypatch):
  monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
  p = _session_jsonl_path("/a/b", "sid")
  assert p == os.path.expanduser("~/.claude/projects/-a-b/sid.jsonl")


def test_format_size_warning_below_threshold():
  assert _format_size_warning(0) == ""
  assert _format_size_warning(_SESSION_SIZE_NUDGE - 1) == ""


def test_format_size_warning_nudge():
  note = _format_size_warning(_SESSION_SIZE_NUDGE)
  assert "/clear" in note
  assert "⚠️" in note
  assert "⚠️⚠️" not in note


def test_format_size_warning_strong():
  note = _format_size_warning(_SESSION_SIZE_STRONG)
  assert "⚠️⚠️" in note
  assert "/clear" in note


def test_trailing_note_reports_oversized_session(tmp_path, monkeypatch):
  monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
  project_dir = "/proj/a"
  session_id = "sess-1"
  jsonl = tmp_path / "projects" / "-proj-a" / f"{session_id}.jsonl"
  jsonl.parent.mkdir(parents=True)
  jsonl.write_bytes(b"x" * (_SESSION_SIZE_NUDGE + 10))

  from nemo.claude_agent import ClaudeCodingAgent
  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = project_dir

  note = agent.trailing_note(session_id)
  assert "/clear" in note


def test_trailing_note_silent_when_small(tmp_path, monkeypatch):
  monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
  project_dir = "/proj/a"
  session_id = "sess-small"
  jsonl = tmp_path / "projects" / "-proj-a" / f"{session_id}.jsonl"
  jsonl.parent.mkdir(parents=True)
  jsonl.write_bytes(b"tiny")

  from nemo.claude_agent import ClaudeCodingAgent
  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = project_dir
  assert agent.trailing_note(session_id) == ""


def test_trailing_note_no_session_or_file(tmp_path, monkeypatch):
  monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
  from nemo.claude_agent import ClaudeCodingAgent
  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = "/proj/a"
  # No session id
  assert agent.trailing_note("") == ""
  # Session id points at non-existent file
  assert agent.trailing_note("nope") == ""
  # No project dir
  agent._project_dir = ""
  assert agent.trailing_note("anything") == ""


def test_run_turn_resumes_latest_session_after_done_event():
  """Regression for the chat-amnesia bug: when a watchdog-forced reconnect
  fires mid-turn, the new CLI must be launched with `resume=<latest session>`
  so conversation context is preserved. The fix wires an options factory
  through SDKThread that rebuilds options using the most recently seen
  sdk_session_id (captured from DoneEvent).
  """
  import asyncio
  from unittest import mock
  from nemo.claude_agent import ClaudeCodingAgent
  from nemo.turn import DoneEvent

  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = "/proj"
  agent._model = "claude-opus-4-7"
  agent._latest_session_id = "old-session"
  agent._stale_tasks = set()
  agent._options = "STATIC_OPTIONS"

  build_calls: list[dict[str, str]] = []

  def fake_build(project_dir: str, model: str, resume: str = "") -> object:
    build_calls.append({"project_dir": project_dir, "model": model, "resume": resume})
    return f"OPTIONS(resume={resume})"

  agent._build_options = fake_build  # type: ignore[method-assign]

  captured: dict[str, object] = {}

  async def fake_rwrc(prompt, on_event, stale_tasks=None, options=None,
                       options_factory=None, max_attempts=3):
    captured["options"] = options
    captured["options_factory"] = options_factory
    # Simulate the SDK reporting a session id at end of turn.
    on_event(DoneEvent(cost=0.1, usage={}, session_id="NEW_SESSION"))
    return (0.1, {})

  agent._sdk = mock.MagicMock()
  agent._sdk.run_turn_with_reconnect = fake_rwrc

  received: list[object] = []
  asyncio.run(agent.run_turn("hello", on_event=received.append))

  # User's on_event still receives the DoneEvent.
  assert any(isinstance(ev, DoneEvent) for ev in received)
  # latest_session_id was updated from the DoneEvent.
  assert agent._latest_session_id == "NEW_SESSION"
  # Static options snapshot is still passed for the first attempt.
  assert captured["options"] == "STATIC_OPTIONS"
  # The factory rebuilds options with the latest session id as resume.
  assert callable(captured["options_factory"])
  fresh = captured["options_factory"]()
  assert fresh == "OPTIONS(resume=NEW_SESSION)"
  assert build_calls[-1]["resume"] == "NEW_SESSION"


def test_run_turn_factory_uses_initial_resume_before_first_done_event():
  """Before any DoneEvent fires, the factory should fall back to the
  resume value seeded by start()/reset() — not a fresh empty session.
  """
  import asyncio
  from unittest import mock
  from nemo.claude_agent import ClaudeCodingAgent

  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = "/proj"
  agent._model = "claude-opus-4-7"
  agent._latest_session_id = "seed-from-start"
  agent._stale_tasks = set()
  agent._options = "STATIC"

  build_calls: list[str] = []

  def fake_build(project_dir: str, model: str, resume: str = "") -> object:
    build_calls.append(resume)
    return f"OPTIONS({resume})"

  agent._build_options = fake_build  # type: ignore[method-assign]

  captured: dict[str, object] = {}

  async def fake_rwrc(prompt, on_event, stale_tasks=None, options=None,
                       options_factory=None, max_attempts=3):
    captured["factory"] = options_factory
    return (0.0, {})

  agent._sdk = mock.MagicMock()
  agent._sdk.run_turn_with_reconnect = fake_rwrc

  asyncio.run(agent.run_turn("hi", on_event=lambda _e: None))
  assert captured["factory"]() == "OPTIONS(seed-from-start)"
  assert build_calls[-1] == "seed-from-start"


def test_default_trailing_note_is_empty():
  """CodingAgent default (non-Claude adapter) returns no note."""
  from nemo.coding_agent import CodingAgent

  class _Stub(CodingAgent):
    async def run_turn(self, prompt, on_event):
      return 0.0, {}
    async def interrupt(self): pass
    async def start(self, project_dir, model, resume=""): pass
    async def reset(self, project_dir, model, resume=""): pass
    async def stop(self): pass

  assert _Stub().trailing_note("some-session") == ""
