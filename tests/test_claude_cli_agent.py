"""Unit tests for the claude-cli (pty-driven TUI) adapter's screen scraping.

The pty/subprocess machinery and live turns are exercised by
``scripts/cli_adapter_smoke.py`` and ``scripts/e2e_test.py --workflow`` (both
need a real account). These tests pin the *pure* scraping logic — the part most
sensitive to TUI layout — against synthetic buffers modeled on the real
claude-cli 2.1.175 render captured in development.
"""

from __future__ import annotations

import json
import os
import tempfile

from nemo.claude_cli_agent import (
  ClaudeCliCodingAgent,
  _HookStream,
  _SessionLog,
  _emit_hook_events,
  _emit_jsonl_events,
  _extract_answer,
  _iter_assistant_blocks,
  _new_tool_summaries,
  _new_turn_state,
  _region_after_echo,
  _sum_turn_usage,
)
from nemo.turn import (
  AnswerEvent, CompactNoticeEvent, CompactStartedEvent, ErrorEvent,
  ProgressEvent, TaskDoneEvent, TaskStartedEvent, TurnEvent,
)


# A realistic post-turn screen: prompt → Write tool → "done" answer.
_TOOL_TURN = [
  "❯ create a file hello.txt in the current directory with the text hi, then reply: done",
  "⏺ Write(hello.txt)",
  "  ⎿  Wrote 1 lines to hello.txt",
  "      1 hi",
  "⏺ done",
  "✻ Crunched for 5s",
  "────────────────────────",
  "❯",
  "────────────────────────",
  "  ⏵⏵ accept edits on (shift+tab to cycle) · ← for agents",
]


def test_extract_answer_prose_after_tool() -> None:
  # Write(...) is a tool (matches Name(...) and is followed by ⎿); "done" is prose.
  assert _extract_answer(_TOOL_TURN, "create a file hello.txt") == "done"


def test_extract_answer_plain_reply() -> None:
  lines = ["❯ reply with exactly the word: pong", "⏺ pong", "✻ Cogitated for 3s", "❯"]
  assert _extract_answer(lines, "reply with exactly the word: pong") == "pong"


def test_extract_answer_multiline_block() -> None:
  lines = [
    "❯ summarize",
    "⏺ First line of the answer",
    "  continues on a second line",
    "  and a third",
    "────────",
    "❯",
  ]
  out = _extract_answer(lines, "summarize")
  assert "First line of the answer" in out and "third" in out


def test_extract_answer_anchors_on_current_prompt() -> None:
  # Two turns in scrollback: extraction must return THIS prompt's answer — the
  # desync class the region anchor prevents.
  lines = [
    "❯ first question", "⏺ first answer",
    "❯ second question", "⏺ second answer", "❯",
  ]
  assert _extract_answer(lines, "second question") == "second answer"
  assert _extract_answer(lines, "first question") == "first answer"


def test_extract_answer_tool_only_turn_is_empty() -> None:
  lines = [
    "❯ run the build", "⏺ Bash(make build)", "  ⎿  build succeeded", "❯",
  ]
  assert _extract_answer(lines, "run the build") == ""


def test_region_after_echo_slices_current_turn() -> None:
  region = _region_after_echo(_TOOL_TURN, "create a file hello.txt")
  assert region[0] == "⏺ Write(hello.txt)"
  assert "❯ create a file" not in "\n".join(region)


def test_iter_assistant_blocks_classifies_tool_vs_prose() -> None:
  region = _region_after_echo(_TOOL_TURN, "create a file hello.txt")
  blocks = _iter_assistant_blocks(region)
  texts = {t: is_tool for t, is_tool in blocks}
  assert texts["Write(hello.txt)"] is True
  assert texts["done"] is False


def test_new_tool_summaries_dedupes_within_turn() -> None:
  seen: set[str] = set()
  first = _new_tool_summaries(_TOOL_TURN, "create a file hello.txt", seen)
  assert first == ["Write(hello.txt)"]
  # Same screen again → already seen → nothing new.
  assert _new_tool_summaries(_TOOL_TURN, "create a file hello.txt", seen) == []


def test_new_tool_summaries_ignores_prose_and_other_turns() -> None:
  lines = [
    "❯ old question", "⏺ Read(old.py)", "  ⎿ ...",
    "❯ new question", "⏺ pong",  # prose, not a tool
    "❯",
  ]
  # Scoped to the NEW turn: the old Read(...) tool must not be surfaced.
  assert _new_tool_summaries(lines, "new question", set()) == []


# --- session JSONL: token usage + resumable session id -----------------------

def _arow(*, usage: dict | None = None) -> dict:
  msg: dict = {"role": "assistant", "content": [{"type": "text", "text": "x"}]}
  if usage is not None:
    msg["usage"] = usage
  return {"type": "assistant", "message": msg}


def test_sum_turn_usage_sums_across_messages_to_canonical() -> None:
  rows = [
    _arow(usage={"input_tokens": 10, "output_tokens": 5,
                 "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7}),
    _arow(usage={"input_tokens": 2, "output_tokens": 8,
                 "cache_read_input_tokens": 200}),
  ]
  u = _sum_turn_usage(rows)
  assert u["input_tokens"] == 12
  assert u["output_tokens"] == 13
  assert u["cache_read_input_tokens"] == 300
  assert u["cache_creation_input_tokens"] == 7
  assert u["total_tokens"] == 12 + 13 + 300 + 7


def test_sum_turn_usage_empty_when_no_usage() -> None:
  assert _sum_turn_usage([{"type": "user", "message": {}}, _arow()]) == {}


def test_session_log_tail_and_lazy_bind() -> None:
  with tempfile.TemporaryDirectory() as d:
    log = _SessionLog.__new__(_SessionLog)
    log._dir = d  # type: ignore[attr-defined]
    log._baseline = set()  # type: ignore[attr-defined]
    log._resume_id = ""  # type: ignore[attr-defined]
    log._path = ""  # type: ignore[attr-defined]
    log._pos = 0  # type: ignore[attr-defined]
    log._buf = ""  # type: ignore[attr-defined]
    log._session_id = ""  # type: ignore[attr-defined]
    # No file yet → binds to nothing.
    assert log.read_new() == []
    # Create the "new" session file; lazy bind picks it up.
    p = os.path.join(d, "abc123.jsonl")
    with open(p, "w") as f:
      f.write(json.dumps(_arow(usage={"output_tokens": 3})) + "\n")
    rows = log.read_new()
    assert len(rows) == 1
    assert log.session_id == "abc123"        # id derived from filename
    assert log.read_new() == []              # offset advanced
    # Partial line buffered until completed.
    with open(p, "a") as f:
      f.write('{"type":"user","mes')
    assert log.read_new() == []
    with open(p, "a") as f:
      f.write('sage":{}}\n')
    assert len(log.read_new()) == 1


def test_session_log_resume_binds_to_id_at_eof() -> None:
  with tempfile.TemporaryDirectory() as d:
    # Pre-existing transcript with prior history.
    p = os.path.join(d, "resume-me.jsonl")
    with open(p, "w") as f:
      f.write(json.dumps(_arow(usage={"output_tokens": 99})) + "\n")
    log = _SessionLog.__new__(_SessionLog)
    log._dir = d  # type: ignore[attr-defined]
    log._baseline = {p}  # type: ignore[attr-defined]
    log._resume_id = "resume-me"  # type: ignore[attr-defined]
    log._path = ""  # type: ignore[attr-defined]
    log._pos = 0  # type: ignore[attr-defined]
    log._buf = ""  # type: ignore[attr-defined]
    log._session_id = ""  # type: ignore[attr-defined]
    # Resume binds to <id>.jsonl seeked to END — prior history is NOT re-read.
    assert log.read_new() == []
    assert log.session_id == "resume-me"
    with open(p, "a") as f:
      f.write(json.dumps(_arow(usage={"output_tokens": 4})) + "\n")
    rows = log.read_new()
    assert len(rows) == 1  # only the new turn, not the replayed history


# --- Phase 0/1/2: argv flags, structured event mapping, hook IPC -------------

def _agent(**kw):
  return ClaudeCliCodingAgent({}, "c", None, None, **kw)  # type: ignore[arg-type]


def test_build_argv_includes_effort_and_settings() -> None:
  a = _agent(permission_mode="bypassPermissions")
  a.set_effort("high")
  a._model = "claude-opus-4-8"
  a._settings_path = "/tmp/s.json"
  argv = a._build_argv(resume="sess-1")
  assert "--dangerously-skip-permissions" in argv
  assert argv[argv.index("--effort") + 1] == "high"
  assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
  assert argv[argv.index("--settings") + 1] == "/tmp/s.json"
  assert argv[argv.index("--resume") + 1] == "sess-1"


def test_set_effort_validates() -> None:
  a = _agent()
  a.set_effort("high")
  assert a._effort == "high"
  a.set_effort("bogus")
  assert a._effort == ""  # invalid cleared, so no --effort is passed


def _run_jsonl(rows: list[dict]) -> tuple[list[TurnEvent], dict]:
  events: list[TurnEvent] = []
  state = _new_turn_state()
  _emit_jsonl_events(rows, events.append, state)
  return events, state


def _amsg(*blocks: dict, usage: dict | None = None) -> dict:
  msg: dict = {"role": "assistant", "content": list(blocks)}
  if usage is not None:
    msg["usage"] = usage
  return {"type": "assistant", "message": msg}


def test_jsonl_text_thinking_tool_in_order() -> None:
  events, state = _run_jsonl([
    _amsg({"type": "thinking", "thinking": "let me think"}),
    _amsg({"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}),
    _amsg({"type": "text", "text": "done"}),
  ])
  kinds = [("answer", e.text) if isinstance(e, AnswerEvent)
           else (e.kind, None) for e in events
           if isinstance(e, (AnswerEvent, ProgressEvent))]
  assert kinds == [("thinking", None), ("tool", None), ("answer", "done")]
  assert state["answer_seen"] is True


def test_jsonl_turn_duration_sets_done() -> None:
  _, state = _run_jsonl([{"type": "system", "subtype": "turn_duration", "durationMs": 5}])
  assert state["turn_done"] is True


def test_jsonl_api_error_emits_and_records() -> None:
  events, state = _run_jsonl([{
    "type": "system", "subtype": "api_error",
    "error": {"message": "Connection error.", "formatted": "Unable to connect (ECONNRESET)"},
  }])
  errs = [e for e in events if isinstance(e, ErrorEvent)]
  assert errs and "ECONNRESET" in errs[0].message
  assert state["error"]


def test_jsonl_compact_boundary_emits_notice() -> None:
  events, _ = _run_jsonl([{
    "type": "system", "subtype": "compact_boundary",
    "compact_metadata": {"trigger": "auto", "pre_tokens": 100, "post_tokens": 40, "duration_ms": 1200},
  }])
  notices = [e for e in events if isinstance(e, CompactNoticeEvent)]
  assert notices and notices[0].pre_tokens == 100 and notices[0].trigger == "auto"


def test_jsonl_task_tool_emits_task_started() -> None:
  events, _ = _run_jsonl([
    _amsg({"type": "tool_use", "name": "Task", "id": "t1", "input": {"prompt": "x"}}),
  ])
  assert any(isinstance(e, TaskStartedEvent) and e.task_id == "t1" for e in events)


def test_jsonl_usage_accumulates() -> None:
  _, state = _run_jsonl([
    _amsg({"type": "text", "text": "a"}, usage={"input_tokens": 5, "output_tokens": 2}),
    _amsg({"type": "text", "text": "b"}, usage={"input_tokens": 3, "output_tokens": 4,
                                                 "cache_read_input_tokens": 10}),
  ])
  assert state["usage"] == {"input_tokens": 8, "cache_read": 10,
                            "cache_creation": 0, "output_tokens": 6}


def _run_hooks(rows: list[dict]) -> tuple[list[TurnEvent], dict]:
  events: list[TurnEvent] = []
  state = _new_turn_state()
  _emit_hook_events(rows, events.append, state)
  return events, state


def test_hook_stop_sets_done() -> None:
  _, state = _run_hooks([{"hook_event_name": "Stop", "session_id": "s"}])
  assert state["turn_done"] is True


def test_hook_precompact_emits_started() -> None:
  events, _ = _run_hooks([{"hook_event_name": "PreCompact", "trigger": "auto"}])
  assert any(isinstance(e, CompactStartedEvent) and e.trigger == "auto" for e in events)


def test_hook_subagent_stop_emits_task_done() -> None:
  events, _ = _run_hooks([{"hook_event_name": "SubagentStop"}])
  assert any(isinstance(e, TaskDoneEvent) for e in events)


def test_hookstream_writes_valid_settings_and_tails(tmp_path) -> None:
  import json
  hs = _HookStream(str(tmp_path))
  hs.write_settings()
  cfg = json.load(open(hs.settings_path))
  # Registers exactly the realtime/control signals the jsonl can't give.
  assert set(cfg["hooks"]) == {"PreCompact", "Stop", "SubagentStop", "Notification"}
  cmd = cfg["hooks"]["Stop"][0]["hooks"][0]["command"]
  assert "nemo_hooks.ndjson" in cmd
  # Tailing the events file yields appended JSON lines.
  assert hs.read_new() == []
  events_path = os.path.join(str(tmp_path), "nemo_hooks.ndjson")
  with open(events_path, "a") as f:
    f.write('{"hook_event_name":"Stop"}\n')
  rows = hs.read_new()
  assert rows and rows[0]["hook_event_name"] == "Stop"
