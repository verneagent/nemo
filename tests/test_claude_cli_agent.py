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
  AnswerEvent, CompactNoticeEvent, CompactStartedEvent, DoneEvent, ErrorEvent,
  ProgressEvent, RateLimitNoticeEvent, TaskDoneEvent, TaskStartedEvent,
  TurnEvent,
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
  """Last assistant message's usage wins — the SDK reports full-turn usage."""
  rows = [
    _arow(usage={"input_tokens": 10, "output_tokens": 5,
                 "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7}),
    _arow(usage={"input_tokens": 2, "output_tokens": 8,
                 "cache_read_input_tokens": 200}),
  ]
  u = _sum_turn_usage(rows)
  # Last message wins (full-turn usage, not incremental)
  assert u["input_tokens"] == 2
  assert u["output_tokens"] == 8
  assert u["cache_read_input_tokens"] == 200
  assert u["cache_creation_input_tokens"] == 0  # not in last message
  assert u["total_tokens"] == 2 + 8 + 200 + 0


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
  # AskUserQuestion disabled: its on-screen picker can't be answered from Lark.
  assert argv[argv.index("--disallowed-tools") + 1] == "AskUserQuestion"
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


def test_jsonl_api_error_transient_sets_reconnect_no_error_event() -> None:
  # ECONNRESET is transient → flagged for reconnect-with-resume, NOT surfaced
  # as a user-facing ErrorEvent (the retry may succeed).
  events, state = _run_jsonl([{
    "type": "system", "subtype": "api_error",
    "error": {"message": "Connection error.",
              "formatted": "Unable to connect (ECONNRESET)",
              "connection": {"code": "ECONNRESET"}},
  }])
  assert "ECONNRESET" in str(state["transient"])
  assert not state["error"]
  assert not [e for e in events if isinstance(e, ErrorEvent)]


def test_jsonl_api_error_non_retryable_surfaces() -> None:
  events, state = _run_jsonl([{
    "type": "system", "subtype": "api_error",
    "error": {"message": "402 insufficient balance"},
  }])
  assert state["error"] and not state["transient"]
  assert [e for e in events if isinstance(e, ErrorEvent)]


def test_jsonl_api_error_rate_limit_emits_notice() -> None:
  events, state = _run_jsonl([{
    "type": "system", "subtype": "api_error",
    "error": {"message": "429 Too Many Requests (rate limit)"},
  }])
  assert state["rate_limited"] is True
  assert [e for e in events if isinstance(e, RateLimitNoticeEvent)]
  assert not state["transient"]  # rate-limit is NOT auto-retried


def test_classify_api_error() -> None:
  from nemo.claude_cli_agent import _classify_api_error
  assert _classify_api_error("ECONNRESET", "Unable to connect") == "transient"
  assert _classify_api_error("", "fetch failed") == "transient"
  assert _classify_api_error("", "429 rate limit exceeded") == "rate_limit"
  assert _classify_api_error("", "overloaded") == "transient"
  assert _classify_api_error("529", "529 Overloaded") == "transient"
  assert _classify_api_error("529", "Overloaded") == "transient"
  assert _classify_api_error("", "529 Overloaded") == "transient"  # code field empty
  assert _classify_api_error("", "402 payment required") == "non_retryable"
  assert _classify_api_error("", "insufficient balance") == "non_retryable"
  assert _classify_api_error("", "weird novel error") == "unknown"


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
  """Last assistant message's usage wins — the SDK reports full-turn usage."""
  _, state = _run_jsonl([
    _amsg({"type": "text", "text": "a"}, usage={"input_tokens": 5, "output_tokens": 2}),
    _amsg({"type": "text", "text": "b"}, usage={"input_tokens": 3, "output_tokens": 4,
                                                 "cache_read_input_tokens": 10}),
  ])
  assert state["usage"] == {"input_tokens": 3, "cache_read": 10,
                            "cache_creation": 0, "output_tokens": 4}


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


# --- reconnect-with-resume orchestration (run_turn) -------------------------

def test_run_turn_reconnects_on_transient_then_succeeds() -> None:
  import asyncio
  a = _agent(permission_mode="bypassPermissions")
  a._tui = object()  # non-None sentinel so run_turn skips spawn
  calls = {"sync": 0, "reset": 0}

  def fake_sync(_tui, _prompt, on_event):
    calls["sync"] += 1
    if calls["sync"] == 1:
      return 0.0, {}, True  # transient → signal reconnect, emit nothing
    on_event(AnswerEvent("recovered"))
    on_event(DoneEvent(cost=0.0, usage={"output_tokens": 5}))
    return 0.0, {"output_tokens": 5}, False

  async def fake_reset(_pd, _m, resume=""):
    calls["reset"] += 1

  a._run_turn_sync = fake_sync  # type: ignore[assignment]
  a.reset = fake_reset          # type: ignore[assignment]
  evs: list[TurnEvent] = []
  cost, usage = asyncio.run(a.run_turn("hi", evs.append))
  assert calls["sync"] == 2 and calls["reset"] == 1
  assert usage == {"output_tokens": 5}
  assert any(isinstance(e, AnswerEvent) and e.text == "recovered" for e in evs)


def test_run_turn_gives_up_after_max_reconnects() -> None:
  import asyncio
  a = _agent(permission_mode="bypassPermissions")
  a._tui = object()
  calls = {"sync": 0, "reset": 0}

  def fake_sync(_tui, _prompt, _on_event):
    calls["sync"] += 1
    return 0.0, {}, True  # always transient

  async def fake_reset(_pd, _m, resume=""):
    calls["reset"] += 1

  a._run_turn_sync = fake_sync   # type: ignore[assignment]
  a.reset = fake_reset           # type: ignore[assignment]
  evs: list[TurnEvent] = []
  asyncio.run(a.run_turn("hi", evs.append))
  # MAX_RECONNECT=2 → 3 sync attempts, 2 resets, then an error + done emitted.
  assert calls["sync"] == a._MAX_RECONNECT + 1
  assert calls["reset"] == a._MAX_RECONNECT
  assert any(isinstance(e, ErrorEvent) for e in evs)
  assert any(isinstance(e, DoneEvent) for e in evs)


# --- forwarded slash-command result scraping (/btw popup, /usage, /compact) ---

def test_scrape_command_btw_popup() -> None:
  from nemo.claude_cli_agent import _scrape_command_result
  lines = [
    "❯ /btw what is the secret word? answer in one word",
    "  /btw what is the secret word? answer in one word",
    "    zebra",
    "  ↑/↓ to scroll · c to copy · f to fork · Esc to close",
  ]
  assert _scrape_command_result(lines, "/btw what is the secret word? answer in one word") == "zebra"


def test_scrape_command_usage_popup() -> None:
  from nemo.claude_cli_agent import _scrape_command_result
  lines = [
    "❯ /usage",
    "  Usage credits",
    "  34% of your usage was at >150k context",
    "  Esc to cancel",
  ]
  out = _scrape_command_result(lines, "/usage")
  assert "Usage credits" in out and "34%" in out
  assert "Esc to cancel" not in out


def test_scrape_command_compact_inline() -> None:
  from nemo.claude_cli_agent import _scrape_command_result
  lines = [
    "❯ /compact",
    "  ⎿  Not enough messages to compact.",
    "────────────────────────",
    "❯",
  ]
  assert _scrape_command_result(lines, "/compact") == "Not enough messages to compact."


# --- steering: inject a follow-up into the RUNNING TUI turn ----------------

class _FakeTui:
  def __init__(self, lines: list[str]):
    self._lines = lines
    self.submitted: list[str] = []
    self._alive = True

  def alive(self) -> bool:
    return self._alive

  def snapshot(self) -> list[str]:
    return list(self._lines)

  def submit(self, text: str) -> None:
    self.submitted.append(text)


def test_supports_steering_true() -> None:
  assert _agent().supports_steering() is True


def test_steer_injects_when_working() -> None:
  import asyncio
  a = _agent()
  # "esc to interrupt" present ⇒ a turn is in flight; steer folds the
  # follow-up into it and submits the STRIPPED text.
  tui = _FakeTui(["⏺ Bash(make)", "  ⎿ building", "  esc to interrupt"])
  a._tui = tui  # type: ignore[assignment]
  assert asyncio.run(a.steer("  also update the README\n")) is True
  assert tui.submitted == ["also update the README"]


def test_steer_declines_when_idle() -> None:
  import asyncio
  a = _agent()
  # No working hint ⇒ TUI idle; injecting would start a new turn (queue's
  # job), so steer must decline and let the host queue the message.
  tui = _FakeTui(["⏺ done", "────────", "❯"])
  a._tui = tui  # type: ignore[assignment]
  assert asyncio.run(a.steer("hello")) is False
  assert tui.submitted == []


def test_steer_declines_when_no_tui() -> None:
  import asyncio
  a = _agent()
  a._tui = None
  assert asyncio.run(a.steer("hello")) is False


def test_steer_declines_empty_text() -> None:
  import asyncio
  a = _agent()
  tui = _FakeTui(["  esc to interrupt"])
  a._tui = tui  # type: ignore[assignment]
  assert asyncio.run(a.steer("   ")) is False
  assert tui.submitted == []


def test_reset_clear_creates_fresh_session() -> None:
  """reset(resume="") should NOT fall back to self._session_id — /clear
  must create a brand-new session, not resume the previous one."""
  from nemo.claude_cli_agent import ClaudeCliCodingAgent
  agent = ClaudeCliCodingAgent.__new__(ClaudeCliCodingAgent)
  agent._session_id = "old-session-id"
  agent._project_dir = "/tmp"
  agent._model = "test-model"
  agent._log = None
  agent._hooks = None
  agent._settings_path = ""
  agent._hookdir = ""
  agent._tui = None
  spawned_with: list[str] = []
  async def _fake_stop():
    pass
  async def _fake_spawn(resume: str):
    spawned_with.append(resume)
  agent.stop = _fake_stop
  agent._spawn = _fake_spawn
  import asyncio
  asyncio.run(agent.reset(agent._project_dir, agent._model, resume=""))
  assert agent._session_id == ""
  assert spawned_with == [""], f"expected fresh session, got spawn({spawned_with})"
