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
