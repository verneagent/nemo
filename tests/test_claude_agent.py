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


# ---------------------------------------------------------------------------
# Resume fallback: bundled claude exit-1 → drop resume + retry once
# ---------------------------------------------------------------------------

def test_is_resume_unrecoverable_walks_cause_chain():
  """ProcessError(exit_code=1) wrapped in SDKThread's RuntimeError must
  still be detected as a resume-class failure."""
  from nemo.claude_agent import _is_resume_unrecoverable

  class FakeProcessError(Exception):
    def __init__(self, msg, exit_code):
      super().__init__(msg)
      self.exit_code = exit_code

  inner = FakeProcessError("Command failed with exit code 1", 1)
  inner.__class__.__name__ = "ProcessError"
  outer = RuntimeError("SDK connect failed after 5 attempts")
  outer.__cause__ = inner
  assert _is_resume_unrecoverable(outer) is True

  # exit_code other than 1 → real bug, must propagate.
  inner_137 = FakeProcessError("killed", 137)
  inner_137.__class__.__name__ = "ProcessError"
  outer_137 = RuntimeError("...")
  outer_137.__cause__ = inner_137
  assert _is_resume_unrecoverable(outer_137) is False

  # No chain at all → no false positive on plain RuntimeError.
  assert _is_resume_unrecoverable(RuntimeError("network unreachable")) is False


def test_is_resume_unrecoverable_matches_text_fallback():
  """Older SDK versions surface exit-1 via plain RuntimeError text
  (no chained ProcessError). Match by message as a backup."""
  from nemo.claude_agent import _is_resume_unrecoverable

  assert _is_resume_unrecoverable(
    RuntimeError("CLI exited during connect: rc=1 — Command failed with exit code 1"),
  ) is True
  # Unrelated exit code text → False.
  assert _is_resume_unrecoverable(
    RuntimeError("Command failed with exit code 137"),
  ) is False


def _bare_claude_agent():
  """Construct a ClaudeCodingAgent without running __init__.

  __init__ pulls in claude_agent_sdk indirectly through SDKThread
  start; the resume-fallback tests don't need any of that, they just
  exercise start()'s try/except. Match the pattern used elsewhere in
  this file (test_run_turn_seeds_options_factory etc).
  """
  from nemo.claude_agent import ClaudeCodingAgent
  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._sdk_started = True
  agent._stale_tasks = set()
  agent._project_dir = ""
  agent._model = ""
  agent._latest_session_id = ""
  agent._options = None
  return agent


def test_start_falls_back_when_resume_unrecoverable():
  """Bundled claude exit-1 on resume → claude_agent rebuilds options
  with resume="" and retries once. Daemon stays alive."""
  import asyncio
  from unittest import mock

  agent = _bare_claude_agent()

  build_calls: list[str] = []
  create_calls: list[object] = []

  def fake_build(_proj, _model, resume=""):
    build_calls.append(resume)
    return f"OPTIONS(resume={resume!r})"

  attempts = {"n": 0}

  async def fake_create(opts):
    create_calls.append(opts)
    attempts["n"] += 1
    if attempts["n"] == 1:
      class _ProcessError(Exception):
        def __init__(self):
          super().__init__("Command failed with exit code 1")
          self.exit_code = 1
      _ProcessError.__name__ = "ProcessError"
      err = RuntimeError("SDK connect failed after 5 attempts")
      err.__cause__ = _ProcessError()
      raise err
    return None

  agent._sdk = mock.MagicMock()
  agent._sdk.start = lambda: None
  agent._sdk.create_client = fake_create
  agent._build_options = fake_build  # type: ignore[assignment]

  asyncio.run(agent.start("/tmp/project", "claude-opus-4-7", resume="stale-uuid"))

  # First build with the stale resume id, second build with empty
  # (fallback path).
  assert build_calls == ["stale-uuid", ""], build_calls
  # create_client called twice — first failed, second succeeded.
  assert len(create_calls) == 2
  # _latest_session_id reset to "" after fallback so the rest of the
  # adapter doesn't keep handing the bad id around.
  assert agent._latest_session_id == ""


def test_start_propagates_non_resume_failures():
  """Network errors / SDK bugs must NOT trigger the resume fallback —
  daemon should fail loudly so the operator sees the real cause."""
  import asyncio
  from unittest import mock

  agent = _bare_claude_agent()

  async def fake_create(_opts):
    raise RuntimeError("ECONNREFUSED 127.0.0.1:1234")

  agent._sdk = mock.MagicMock()
  agent._sdk.start = lambda: None
  agent._sdk.create_client = fake_create
  agent._build_options = lambda *_a, **_k: "OPTIONS"  # type: ignore[assignment]

  raised: list[BaseException] = []
  try:
    asyncio.run(agent.start("/tmp/project", "claude-opus-4-7", resume="abc"))
  except BaseException as exc:
    raised.append(exc)
  assert raised and "ECONNREFUSED" in str(raised[0])


def test_start_does_not_retry_when_no_resume():
  """If no resume id was set, an exit-1 failure isn't a resume problem
  — propagate. Otherwise we'd hide every CLI-startup bug."""
  import asyncio
  from unittest import mock

  agent = _bare_claude_agent()

  attempts = {"n": 0}

  async def fake_create(_opts):
    attempts["n"] += 1
    class _ProcessError(Exception):
      def __init__(self):
        super().__init__("Command failed with exit code 1")
        self.exit_code = 1
    _ProcessError.__name__ = "ProcessError"
    err = RuntimeError("SDK connect failed after 5 attempts")
    err.__cause__ = _ProcessError()
    raise err

  agent._sdk = mock.MagicMock()
  agent._sdk.start = lambda: None
  agent._sdk.create_client = fake_create
  agent._build_options = lambda *_a, **_k: "OPTIONS"  # type: ignore[assignment]

  raised = False
  try:
    asyncio.run(agent.start("/tmp/project", "claude-opus-4-7", resume=""))
  except RuntimeError:
    raised = True
  assert raised
  # Single attempt, no retry — there's no resume id to drop.
  assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# /btw side questions
# ---------------------------------------------------------------------------


def test_default_side_question_is_empty():
  """Non-Claude adapters inherit the no-op: side questions unsupported."""
  import asyncio
  from nemo.coding_agent import CodingAgent

  class _Stub(CodingAgent):
    async def run_turn(self, prompt, on_event):
      return 0.0, {}
    async def interrupt(self): pass
    async def start(self, project_dir, model, resume=""): pass
    async def reset(self, project_dir, model, resume=""): pass
    async def stop(self): pass

  assert asyncio.run(_Stub().side_question("hi", "sess")) == ""


def _btw_claude_agent():
  """Bare ClaudeCodingAgent with just the attrs side_question reads.

  Distinct from _bare_claude_agent above (which targets start()'s
  resume-fallback path) — do NOT merge the two; that earlier collision
  silently shadowed the start() helper and broke its tests.
  """
  from nemo.claude_agent import ClaudeCodingAgent
  from nemo.coding_agent import EndpointConfig

  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = "/proj"
  agent._model = "claude-opus-4-7"
  agent._chat_id = "chat-1"
  agent._system_prompt = ""
  agent._endpoint = EndpointConfig()
  return agent


def test_side_question_without_session_runs_fresh(monkeypatch):
  """First message of a session: no sdk_session_id yet. /btw must still
  answer (Claude Code's does) — from a FRESH session, i.e. NO resume and
  NO fork_session (forking nothing is meaningless), but still read-only."""
  import asyncio
  import claude_agent_sdk as sdk

  captured: dict[str, object] = {}

  async def fake_query(*, prompt, options=None, transport=None):
    captured["options"] = options
    yield sdk.AssistantMessage(
      content=[sdk.TextBlock(text="fresh answer")], model="m")
    yield sdk.ResultMessage(
      subtype="success", duration_ms=1, duration_api_ms=1,
      is_error=False, num_turns=1, session_id="fresh")

  monkeypatch.setattr(sdk, "query", fake_query)

  agent = _btw_claude_agent()
  answer = asyncio.run(agent.side_question("hello?", ""))

  assert answer == "fresh answer"
  opts = captured["options"]
  assert getattr(opts, "resume", None) in (None, "")
  assert getattr(opts, "fork_session", False) is False
  assert opts.allowed_tools == []
  # >1: read-only rests on the empty allow-list, not the turn cap; a cap
  # of 1 starves agentic models that burn turns on blocked tool calls.
  assert opts.max_turns > 1


def test_side_question_requires_project_dir():
  """Hard guard: no project dir means the adapter isn't initialised."""
  import asyncio
  agent = _btw_claude_agent()
  agent._project_dir = ""
  assert asyncio.run(agent.side_question("q", "sess")) == ""


def test_side_question_forks_resumes_and_is_readonly(monkeypatch):
  """The side query must resume the live session for context but fork it
  (so the answer never enters real history), with no tools and a single
  turn. Asserts the options Nemo hands the SDK enforce exactly that."""
  import asyncio
  import claude_agent_sdk as sdk

  captured: dict[str, object] = {}

  async def fake_query(*, prompt, options=None, transport=None):
    captured["prompt"] = prompt
    captured["options"] = options
    yield sdk.AssistantMessage(
      content=[sdk.TextBlock(text="It was config.toml.")], model="m")
    yield sdk.ResultMessage(
      subtype="success", duration_ms=1, duration_api_ms=1,
      is_error=False, num_turns=1, session_id="forked-throwaway")

  monkeypatch.setattr(sdk, "query", fake_query)

  agent = _btw_claude_agent()
  answer = asyncio.run(agent.side_question("which file?", "live-session"))

  assert answer == "It was config.toml."
  assert captured["prompt"] == "which file?"
  opts = captured["options"]
  assert opts.resume == "live-session"
  assert opts.fork_session is True
  assert opts.allowed_tools == []
  # >1: read-only rests on the empty allow-list, not the turn cap; a cap
  # of 1 starves agentic models that burn turns on blocked tool calls.
  assert opts.max_turns > 1
  assert "Bash" in opts.disallowed_tools and "Write" in opts.disallowed_tools


def test_side_question_survives_sdk_error(monkeypatch):
  import asyncio
  import claude_agent_sdk as sdk

  async def boom(*, prompt, options=None, transport=None):
    raise RuntimeError("connect failed")
    yield  # pragma: no cover — make this an async generator

  monkeypatch.setattr(sdk, "query", boom)
  agent = _btw_claude_agent()
  answer = asyncio.run(agent.side_question("q", "sess"))
  assert answer.startswith("⚠️ btw failed:")


def test_side_question_closes_generator_in_task(monkeypatch):
  """Regression: a daemon crash came from the SDK query generator being
  finalised across tasks on the host loop (anyio "exit cancel scope in a
  different task"). side_question must drive + explicitly aclose() the
  generator itself (on its own worker loop), never leaking it to GC."""
  import asyncio
  import claude_agent_sdk as sdk

  closed = {"v": False}

  class _Gen:
    def __aiter__(self):
      return self

    async def __anext__(self):
      yielded = getattr(self, "_done", False)
      if yielded:
        raise StopAsyncIteration
      self._done = True
      return sdk.ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1,
        is_error=False, num_turns=1, session_id="s")

    async def aclose(self):
      closed["v"] = True

  def fake_query(*, prompt, options=None, transport=None):
    return _Gen()

  monkeypatch.setattr(sdk, "query", fake_query)
  agent = _btw_claude_agent()
  answer = asyncio.run(agent.side_question("q", "sess"))
  assert closed["v"] is True, "generator was not explicitly aclose()d"
  assert "btw failed" not in answer
