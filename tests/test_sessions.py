"""Tests for nemo.sessions — scanning Claude/Codex JSONL session files."""

from __future__ import annotations

import json
import os
import shutil
import time
from unittest import mock

import pytest

from nemo import sessions


def _write_jsonl(path: str, events: list[dict]) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    for ev in events:
      f.write(json.dumps(ev) + "\n")


def _claude_session(home: str, project_dir: str, uuid: str,
                    events: list[dict]) -> str:
  # Mirror the CLI's slug exactly (realpath + every non-alnum → "-"); on
  # macOS tmp_path is symlinked, so abspath would diverge from realpath.
  encoded = sessions.claude_project_slug(project_dir)
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


def test_list_claude_sessions_handles_spaces_in_path(tmp_path):
  # Claude CLI encodes spaces in the cwd to ``-`` (e.g. macOS
  # "Application Support"). The folder lookup must do the same or
  # /session list finds nothing for projects under such paths.
  home = str(tmp_path / "home")
  project = str(tmp_path / "Application Support" / "Muxy" / "vm-min-dur")
  os.makedirs(project, exist_ok=True)
  _claude_session(home, project, "5c0ffee0-0000-0000-0000-000000000000", [
    {"type": "user", "message": {
      "role": "user",
      "content": "Hello from a spaced path",
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    out = sessions.list_claude_sessions(project)
  assert len(out) == 1, out
  assert out[0].uuid == "5c0ffee0-0000-0000-0000-000000000000"
  assert out[0].first_user_text == "Hello from a spaced path"


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


def test_list_codex_sessions_skips_environment_context_preview(tmp_path):
  # Codex records environment context as a user-shaped message. It is host
  # metadata, not the prompt the operator needs in /session recall previews.
  home = str(tmp_path / "home")
  project = str(tmp_path / "p")
  os.makedirs(project, exist_ok=True)
  _codex_session(home, "2026", "04", "01",
                 "bbbbbbbb-1111-2222-3333-444444444444", [
    {"type": "session_meta", "payload": {
      "id": "bbbbbbbb-1111-2222-3333-444444444444",
      "cwd": os.path.abspath(project),
    }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text",
                   "text": "<environment_context>\n  <cwd>/tmp/p</cwd>"}],
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
    # Injected host context (the tail scan must skip this too).
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "<environment_context>\n..."}],
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


def test_session_detail_reads_first_and_last_three_claude_messages(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  _claude_session(home, project, "feedface-0000-0000-0000-000000000000", [
    {"timestamp": "2026-05-17T01:00:00Z", "type": "user", "message": {
      "role": "user", "content": "first user",
    }},
    {"timestamp": "2026-05-17T01:01:00Z", "type": "assistant", "message": {
      "role": "assistant", "model": "claude-opus-4-7",
      "content": [{"type": "text", "text": "first reply"}],
    }},
    {"timestamp": "2026-05-17T01:02:00Z", "type": "user", "message": {
      "role": "user", "content": "second user",
    }},
    {"timestamp": "2026-05-17T01:03:00Z", "type": "assistant", "message": {
      "role": "assistant", "model": "claude-opus-4-7",
      "content": [{"type": "text", "text": "second reply"}],
    }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.session_detail(project, "feedface")

  assert result.detail is not None
  detail = result.detail
  assert detail.session.uuid == "feedface-0000-0000-0000-000000000000"
  assert detail.message_count == 4
  assert detail.first_message is not None
  assert detail.first_message.text == "first user"
  assert detail.first_message.timestamp == "2026-05-17T01:00:00Z"
  assert [(m.role, m.text) for m in detail.last_messages] == [
    ("assistant", "first reply"),
    ("user", "second user"),
    ("assistant", "second reply"),
  ]


def test_session_detail_reads_codex_messages_and_skips_injected_context(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  _codex_session(home, "2026", "05", "17",
                 "c0dec0de-1111-2222-3333-444444444444", [
    {"timestamp": "2026-05-17T02:00:00Z", "type": "session_meta",
     "payload": {
       "id": "c0dec0de-1111-2222-3333-444444444444",
       "cwd": os.path.abspath(project),
       "model": "gpt-5.5",
     }},
    {"timestamp": "2026-05-17T02:00:01Z", "type": "response_item",
     "payload": {
       "type": "message", "role": "user",
       "content": [{"type": "input_text", "text": "# AGENTS.md\n..."}],
     }},
    {"timestamp": "2026-05-17T02:01:00Z", "type": "response_item",
     "payload": {
       "type": "message", "role": "user",
       "content": [{"type": "input_text", "text": "real question"}],
     }},
    {"timestamp": "2026-05-17T02:02:00Z", "type": "response_item",
     "payload": {
       "type": "message", "role": "assistant",
       "content": [{"type": "output_text", "text": "real answer"}],
     }},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.session_detail(project, "", current_uuid="c0dec0de")

  assert result.detail is not None
  detail = result.detail
  assert detail.message_count == 2
  assert detail.first_message is not None
  assert detail.first_message.text == "real question"
  assert [(m.role, m.text) for m in detail.last_messages] == [
    ("user", "real question"),
    ("assistant", "real answer"),
  ]


def test_session_detail_empty_current_without_session_id(tmp_path):
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  result = sessions.session_detail(project, "", current_uuid="")
  assert result.detail is None
  assert result.ambiguous == []
  assert result.not_found == ""


def test_session_detail_handles_empty_transcript_file(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  path = _claude_session(
    home, project, "eeeeeeee-0000-0000-0000-000000000000", [])
  assert os.path.exists(path)

  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.session_detail(project, "eeeeeeee")

  assert result.detail is not None
  assert result.detail.message_count == 0
  assert result.detail.first_message is None
  assert result.detail.last_messages == []


def test_list_returns_empty_when_project_has_no_sessions(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "empty")
  os.makedirs(project, exist_ok=True)
  with mock.patch.dict(os.environ, {"HOME": home}):
    assert sessions.list_sessions(project) == []


def test_remove_session_deletes_matching_file(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  path = _claude_session(home, project, "abc12345-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "delete me"}},
  ])

  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.remove_session(project, "abc12345")

  assert [s.uuid for s in result.deleted] == [
    "abc12345-0000-0000-0000-000000000000"
  ]
  assert result.failures == []
  assert not os.path.exists(path)


def test_remove_session_reports_ambiguous_prefix(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  path_a = _claude_session(home, project, "abc11111-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "one"}},
  ])
  path_b = _claude_session(home, project, "abc22222-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "two"}},
  ])

  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.remove_session(project, "abc")

  assert {s.uuid for s in result.ambiguous} == {
    "abc11111-0000-0000-0000-000000000000",
    "abc22222-0000-0000-0000-000000000000",
  }
  assert result.deleted == []
  assert os.path.exists(path_a)
  assert os.path.exists(path_b)


def test_purge_sessions_removes_older_than_target_excluding_target(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  old_path = _claude_session(home, project, "11111111-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "old"}},
  ])
  pivot_path = _claude_session(home, project, "22222222-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "pivot"}},
  ])
  new_path = _claude_session(home, project, "33333333-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "new"}},
  ])
  now = time.time()
  os.utime(old_path, (now - 300, now - 300))
  os.utime(pivot_path, (now - 200, now - 200))
  os.utime(new_path, (now - 100, now - 100))

  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.purge_sessions(project, "22222222")

  assert [s.uuid for s in result.deleted] == [
    "11111111-0000-0000-0000-000000000000"
  ]
  assert not os.path.exists(old_path)
  assert os.path.exists(pivot_path)
  assert os.path.exists(new_path)


def test_purge_sessions_without_target_keeps_current(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  current_path = _claude_session(
    home, project, "aaaaaaaa-0000-0000-0000-000000000000", [
      {"type": "user", "message": {"role": "user", "content": "current"}},
    ])
  old_path = _codex_session(home, "2026", "05", "15",
                            "bbbbbbbb-0000-0000-0000-000000000000", [
    {"type": "session_meta", "payload": {
      "id": "bbbbbbbb-0000-0000-0000-000000000000",
      "cwd": os.path.abspath(project),
    }},
    {"type": "response_item", "payload": {
      "type": "message", "role": "user",
      "content": [{"type": "input_text", "text": "old"}],
    }},
  ])

  with mock.patch.dict(os.environ, {"HOME": home}):
    result = sessions.purge_sessions(
      project, current_uuid="aaaaaaaa-0000-0000-0000-000000000000")

  assert [s.uuid for s in result.deleted] == [
    "bbbbbbbb-0000-0000-0000-000000000000"
  ]
  assert os.path.exists(current_path)
  assert not os.path.exists(old_path)


def test_session_is_active_detects_open_transcript(tmp_path):
  # An open file handle is ground truth that a daemon is using the
  # session; lsof must report it as active and report it inactive once
  # the handle is closed.
  if shutil.which("lsof") is None:
    pytest.skip("lsof not available on this host")
  path = str(tmp_path / "live.jsonl")
  with open(path, "w", encoding="utf-8") as f:
    f.write("{}\n")
    f.flush()
    assert sessions._session_is_active(path) is True
  assert sessions._session_is_active(path) is False


def test_session_is_active_false_without_lsof(tmp_path):
  # No lsof → can't probe → preserve "delete anything" behaviour.
  path = str(tmp_path / "x.jsonl")
  with open(path, "w", encoding="utf-8") as f:
    f.write("{}\n")
  with mock.patch.object(sessions.shutil, "which", return_value=None):
    assert sessions._session_is_active(path) is False


def test_remove_session_skips_active_session(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  path = _claude_session(home, project, "abc12345-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "in use"}},
  ])
  with mock.patch.dict(os.environ, {"HOME": home}), \
       mock.patch.object(sessions, "_session_is_active", return_value=True):
    result = sessions.remove_session(project, "abc12345")
  assert [s.uuid for s in result.skipped_active] == [
    "abc12345-0000-0000-0000-000000000000"
  ]
  assert result.deleted == []
  assert os.path.exists(path)  # an in-progress session is never removed


def test_purge_skips_active_sessions_of_other_daemons(tmp_path):
  # Two sessions share the workspace: the current daemon's (excluded by
  # uuid) and another daemon's still-live one. Purge must keep the live
  # one even though it isn't this daemon's "current".
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  current_path = _claude_session(
    home, project, "aaaaaaaa-0000-0000-0000-000000000000", [
      {"type": "user", "message": {"role": "user", "content": "current"}},
    ])
  live_other_path = _claude_session(
    home, project, "bbbbbbbb-0000-0000-0000-000000000000", [
      {"type": "user", "message": {"role": "user", "content": "other live"}},
    ])
  dead_path = _claude_session(
    home, project, "cccccccc-0000-0000-0000-000000000000", [
      {"type": "user", "message": {"role": "user", "content": "dead"}},
    ])

  def fake_active(path: str) -> bool:
    return path == live_other_path

  with mock.patch.dict(os.environ, {"HOME": home}), \
       mock.patch.object(sessions, "_session_is_active", side_effect=fake_active):
    result = sessions.purge_sessions(
      project, current_uuid="aaaaaaaa-0000-0000-0000-000000000000")

  assert [s.uuid for s in result.deleted] == [
    "cccccccc-0000-0000-0000-000000000000"
  ]
  assert [s.uuid for s in result.skipped_active] == [
    "bbbbbbbb-0000-0000-0000-000000000000"
  ]
  assert os.path.exists(current_path)
  assert os.path.exists(live_other_path)
  assert not os.path.exists(dead_path)


# ---------------------------------------------------------------------------
# Recall digest cache
# ---------------------------------------------------------------------------

def _session_info(path: str, uuid: str) -> sessions.SessionInfo:
  st = os.stat(path)
  return sessions.SessionInfo(
    uuid=uuid, agent="claude", path=path, mtime=st.st_mtime,
    first_user_text="hi", model="claude-opus-4-7",
  )


def test_digest_cache_round_trip(tmp_path):
  home = str(tmp_path / "home")
  transcript = str(tmp_path / "t.jsonl")
  _write_jsonl(transcript, [{"type": "user", "message": {
    "role": "user", "content": "hello"}}])
  info = _session_info(transcript, "uuid-1")
  with mock.patch.dict(os.environ, {"HOME": home}):
    assert sessions.read_cached_digest(info) == ""  # cold
    sessions.write_cached_digest(info, "### Working on\n- the thing")
    assert sessions.read_cached_digest(info) == "### Working on\n- the thing"


def test_digest_cache_invalidated_when_transcript_changes(tmp_path):
  home = str(tmp_path / "home")
  transcript = str(tmp_path / "t.jsonl")
  _write_jsonl(transcript, [{"type": "user", "message": {
    "role": "user", "content": "hello"}}])
  info = _session_info(transcript, "uuid-2")
  with mock.patch.dict(os.environ, {"HOME": home}):
    sessions.write_cached_digest(info, "old summary")
    assert sessions.read_cached_digest(info) == "old summary"
    # Append to the transcript → size changes → cache is stale.
    with open(transcript, "a", encoding="utf-8") as f:
      f.write(json.dumps({"type": "assistant", "message": {
        "model": "m", "content": [{"type": "text", "text": "more"}]}}) + "\n")
    # The cached SessionInfo still has the OLD stat, but read validates
    # against the LIVE file, so the size mismatch invalidates the cache.
    assert sessions.read_cached_digest(info) == ""


def test_digest_cache_empty_not_written(tmp_path):
  home = str(tmp_path / "home")
  transcript = str(tmp_path / "t.jsonl")
  _write_jsonl(transcript, [{"type": "user", "message": {
    "role": "user", "content": "hi"}}])
  info = _session_info(transcript, "uuid-3")
  with mock.patch.dict(os.environ, {"HOME": home}):
    sessions.write_cached_digest(info, "   \n  ")  # blank → skipped
    assert not os.path.exists(sessions._digest_cache_path("uuid-3"))


def test_delete_session_drops_cached_digest(tmp_path):
  home = str(tmp_path / "home")
  project = str(tmp_path / "project")
  os.makedirs(project, exist_ok=True)
  path = _claude_session(home, project, "cccccccc-0000-0000-0000-000000000000", [
    {"type": "user", "message": {"role": "user", "content": "x"}},
  ])
  info = _session_info(path, "cccccccc-0000-0000-0000-000000000000")
  with mock.patch.dict(os.environ, {"HOME": home}):
    sessions.write_cached_digest(info, "summary to be orphaned")
    cache = sessions._digest_cache_path("cccccccc-0000-0000-0000-000000000000")
    assert os.path.exists(cache)
    sessions.remove_session(project, "cccccccc")
    assert not os.path.exists(cache)
