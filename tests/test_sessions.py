"""Tests for nemo.sessions — scanning Claude/Codex JSONL session files."""

from __future__ import annotations

import json
import os
import time
from unittest import mock

from nemo import sessions


def _write_jsonl(path: str, events: list[dict]) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    for ev in events:
      f.write(json.dumps(ev) + "\n")


def _claude_session(home: str, project_dir: str, uuid: str,
                    events: list[dict]) -> str:
  encoded = os.path.abspath(project_dir).replace("/", "-").replace(".", "-")
  path = os.path.join(home, ".claude", "projects", encoded, f"{uuid}.jsonl")
  _write_jsonl(path, events)
  return path


def _codex_session(home: str, year: str, month: str, day: str,
                   uuid: str, events: list[dict]) -> str:
  path = os.path.join(
    home, ".codex", "sessions", year, month, day,
    f"rollout-{year}-{month}-{day}T12-00-00-{uuid}.jsonl",
  )
  _write_jsonl(path, events)
  return path


def test_list_claude_sessions_extracts_uuid_model_and_preview(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  _claude_session(home, project, "01fe69c7-5793-4ad7-9ba6-7d1aa1e01f90", [
    {"type": "file-history-snapshot", "snapshot": {}},
    {"type": "user", "message": {
      "role": "user",
      "content": "<command-name>/clear</command-name>",
    }},
    {"type": "user", "message": {
      "role": "user",
      "content": "Help me debug this bug",
    }},
    {"type": "assistant", "message": {
      "model": "claude-opus-4-7",
      "content": [{"type": "text", "text": "Looking now..."}],
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_claude_sessions(project)
  assert len(out) == 1, out
  s = out[0]
  assert s.uuid == "01fe69c7-5793-4ad7-9ba6-7d1aa1e01f90"
  assert s.agent == "claude"
  assert s.model == "claude-opus-4-7"
  # Noise tag (<command-name>) gets stripped; the real first user
  # prompt comes through as the preview.
  assert s.first_user_text == "Help me debug this bug"


def test_list_claude_sessions_handles_hidden_path_segments(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / ".prowl" / "repos" / "fived" / "stt-stuck")
  os.makedirs(project, exist_ok=True)
  _claude_session(home, project, "d65dfbbf-7c71-49cd-9761-06844d0a189f", [
    {"type": "user", "message": {
      "role": "user",
      "content": "Look at this bug",
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_claude_sessions(project)
  assert len(out) == 1
  assert out[0].uuid == "d65dfbbf-7c71-49cd-9761-06844d0a189f"
  assert out[0].first_user_text == "Look at this bug"


def test_list_claude_sessions_handles_block_content(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  _claude_session(home, project, "abc12345-0000-0000-0000-000000000000", [
    {"type": "user", "message": {
      "role": "user",
      "content": [
        {"type": "text", "text": "First part."},
        {"type": "tool_use", "id": "x"},  # ignored
        {"type": "text", "text": "Second part."},
      ],
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_claude_sessions(project)
  assert out[0].first_user_text == "First part.\nSecond part."


def test_list_codex_sessions_skips_injected_agents_md_preview(tmp_path):
  # Codex always injects the project AGENTS.md as the first "user"
  # message. The preview should land on the FIRST real user prompt,
  # not the boilerplate.
  home = str(tmp_path / "home")
  project = str(tmp_path / "p")
  os.makedirs(project, exist_ok=True)
  _codex_session(home, "2026", "04", "01",
                 "aaaaaaaa-1111-2222-3333-444444444444", [
    {"type": "session_meta", "payload": {
      "id": "aaaaaaaa-1111-2222-3333-444444444444",
      "cwd": os.path.abspath(project),
    }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text",
                   "text": "# AGENTS.md instructions for /Users/x\n\n..."}],
    }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "Actual user question here"}],
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_codex_sessions(project)
  assert out[0].first_user_text == "Actual user question here"


def test_list_codex_sessions_scopes_to_project_cwd(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "ours")
  other_project = str(tmp_path / "theirs")
  os.makedirs(project, exist_ok=True)
  os.makedirs(other_project, exist_ok=True)
  # One rollout for our cwd, one for someone else's.
  _codex_session(home, "2026", "02", "09",
                 "019c3fec-3890-78f0-988c-cdb3802197b8", [
    {"timestamp": "2026-02-09T01:02:51.591Z", "type": "session_meta",
     "payload": {
       "id": "019c3fec-3890-78f0-988c-cdb3802197b8",
       "cwd": os.path.abspath(project),
       "model": "gpt-5.5",
     }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "Run the tests please"}],
    }},
  ])
  _codex_session(home, "2026", "02", "10",
                 "019c4061-84be-7731-ac61-8db2752783ae", [
    {"timestamp": "2026-02-10T01:02:51.591Z", "type": "session_meta",
     "payload": {
       "id": "019c4061-84be-7731-ac61-8db2752783ae",
       "cwd": os.path.abspath(other_project),
       "model": "gpt-5.5",
     }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_codex_sessions(project)
  # Only our cwd's session shows up.
  assert len(out) == 1, [s.uuid for s in out]
  s = out[0]
  assert s.uuid == "019c3fec-3890-78f0-988c-cdb3802197b8"
  assert s.agent == "codex"
  assert s.model == "gpt-5.5"
  assert s.first_user_text == "Run the tests please"


def test_list_codex_sessions_extracts_model_from_turn_context(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "p")
  os.makedirs(project, exist_ok=True)
  _codex_session(home, "2026", "05", "15",
                 "019e2b0a-2519-7f81-ad04-fa2f22e450e7", [
    {"type": "session_meta", "payload": {
      "id": "019e2b0a-2519-7f81-ad04-fa2f22e450e7",
      "cwd": os.path.abspath(project),
    }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "# AGENTS.md\n..."}],
    }},
    {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "Actual prompt"}],
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_codex_sessions(project)
  assert len(out) == 1
  assert out[0].model == "gpt-5.5"
  assert out[0].first_user_text == "Actual prompt"


def test_list_sessions_merges_claude_and_codex_newest_first(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "p")
  os.makedirs(project, exist_ok=True)
  # Claude session, older.
  c_path = _claude_session(home, project, "11111111-aaaa-bbbb-cccc-dddddddddddd", [
    {"type": "user", "message": {"role": "user", "content": "old claude"}},
  ])
  os.utime(c_path, (time.time() - 100, time.time() - 100))
  # Codex session, newer.
  x_path = _codex_session(home, "2026", "03", "01",
                          "22222222-eeee-ffff-1111-222222222222", [
    {"type": "session_meta", "payload": {
      "id": "22222222-eeee-ffff-1111-222222222222",
      "cwd": os.path.abspath(project),
    }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "newer codex"}],
    }},
  ])
  os.utime(x_path, (time.time() - 10, time.time() - 10))
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_sessions(project)
  assert [s.agent for s in out] == ["codex", "claude"], out


def test_find_session_prefers_exact_over_prefix():
  from nemo.sessions import SessionInfo, find_session
  a = SessionInfo(uuid="abc123", agent="claude", path="", mtime=0,
                  first_user_text="", model="")
  b = SessionInfo(uuid="abc1234567", agent="claude", path="", mtime=0,
                  first_user_text="", model="")
  # Exact wins.
  matches = find_session("abc123", [a, b])
  assert matches == [a]
  # Prefix-only is ambiguous.
  matches = find_session("abc1", [a, b])
  assert {m.uuid for m in matches} == {"abc123", "abc1234567"}
  # Empty needle is empty result.
  assert find_session("", [a, b]) == []
  assert find_session("nope", [a, b]) == []


def test_list_claude_sessions_captures_last_three_user_prompts(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  events: list[dict] = []
  for i in range(6):
    events.append({"type": "user", "message": {
      "role": "user", "content": f"prompt {i}",
    }})
    events.append({"type": "assistant", "message": {
      "model": "claude-opus-4-7",
      "content": [{"type": "text", "text": f"reply {i}"}],
    }})
  _claude_session(home, project, "cafef00d-0000-0000-0000-000000000000", events)
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_claude_sessions(project)
  assert len(out) == 1
  s = out[0]
  # First-pass preview is the oldest user prompt.
  assert s.first_user_text == "prompt 0"
  # Tail-pass collects the last 3 in oldest-first order.
  assert s.last_user_texts == ["prompt 3", "prompt 4", "prompt 5"]


def test_list_codex_sessions_captures_last_three_user_prompts(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  events: list[dict] = [
    {"type": "session_meta", "payload": {
      "id": "beadbead-1111-2222-3333-444444444444",
      "cwd": os.path.abspath(project),
    }},
    # Injected AGENTS.md (the tail scan must skip this too).
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "# AGENTS.md\n..."}],
    }},
  ]
  for i in range(5):
    events.append({"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": f"user prompt {i}"}],
    }})
    events.append({"type": "response_item", "payload": {
      "type": "message", "role": "assistant",
      "content": [{"type": "output_text", "text": f"reply {i}"}],
    }})
  _codex_session(home, "2026", "04", "01",
                 "beadbead-1111-2222-3333-444444444444", events)
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_codex_sessions(project)
  s = out[0]
  assert s.first_user_text == "user prompt 0"
  assert s.last_user_texts == ["user prompt 2", "user prompt 3", "user prompt 4"]


def test_list_returns_empty_when_project_has_no_sessions(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "empty")
  os.makedirs(project, exist_ok=True)
  with mock.patch.dict(os.environ, {"HOME": home}):
    assert sessions.list_sessions(project) == []
