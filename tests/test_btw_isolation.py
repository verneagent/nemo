"""Guards for the /btw crash class: the Claude SDK must never be driven
on the host event loop.

Background: 0.4.15's /btw ran ``claude_agent_sdk.query()`` on the daemon's
main asyncio loop. The SDK's internal anyio task group raised "Attempted
to exit cancel scope in a different task than it was entered in" when its
generator was finalised across tasks, which propagated as a fatal
CancelledError and killed ``main_loop`` — the whole chat went silent. The
unit tests missed it because they mocked ``query`` away (the broken part
was exactly the real SDK's anyio lifecycle, never the wrapper logic).

These tests close that gap with checks that DON'T depend on mocking the
SDK away:

1. ``test_sdk_driving_primitives_are_isolated`` — a static, always-on
   architectural guard: the only place allowed to construct/drive the
   SDK client or the top-level ``query()`` is the dedicated isolation
   wrappers. Trips deterministically if anyone reintroduces a
   host-loop SDK driver.
2. ``test_side_question_teardown_error_does_not_kill_host_loop`` — locks
   the contract that an SDK generator-teardown failure is contained in
   the worker and the host loop stays alive afterwards.
3. ``test_side_question_real_sdk_survives_persistent_loop`` — the only
   thing that exercises the real anyio lifecycle. Gated behind
   ``NEMO_REAL_SDK=1``; the release process MUST run it.
"""

from __future__ import annotations

import ast
import asyncio
import os
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_NEMO = _REPO / "nemo"


def _btw_claude_agent():
  from nemo.claude_agent import ClaudeCodingAgent
  from nemo.coding_agent import EndpointConfig

  agent = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  agent._project_dir = str(_REPO)
  agent._model = "claude-opus-4-7"
  agent._chat_id = "chat-iso"
  agent._system_prompt = ""
  agent._endpoint = EndpointConfig()
  return agent


# ---------------------------------------------------------------------------
# 1. Static architectural guard (always on, no SDK needed)
# ---------------------------------------------------------------------------


def test_sdk_driving_primitives_are_isolated():
  """ClaudeSDKClient construction and the top-level claude_agent_sdk
  ``query()`` are the two ways to *drive* the SDK. Each must live only in
  its isolated-loop owner, never anywhere it could run on the host loop.
  """
  py_files = sorted(_NEMO.rglob("*.py"))

  # (a) ClaudeSDKClient(...) may only be constructed in sdk_thread.py
  client_offenders = [
    p.relative_to(_REPO).as_posix()
    for p in py_files
    if "ClaudeSDKClient(" in p.read_text() and p.name != "sdk_thread.py"
  ]
  assert not client_offenders, (
    "ClaudeSDKClient must only be constructed in nemo/sdk_thread.py "
    f"(its dedicated thread/loop owner); found in: {client_offenders}"
  )

  # (b) The top-level `query` symbol may only be imported from
  #     claude_agent_sdk in claude_agent.py.
  def _imports_query(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      if isinstance(node, ast.ImportFrom) and node.module == "claude_agent_sdk":
        if any(alias.name == "query" for alias in node.names):
          return True
    return False

  query_importers = [
    p.relative_to(_REPO).as_posix() for p in py_files if _imports_query(p)
  ]
  assert query_importers == ["nemo/claude_agent.py"], (
    "Top-level claude_agent_sdk.query() may only be used by "
    f"nemo/claude_agent.py's isolated side_question; importers: {query_importers}"
  )

  # (c) In claude_agent.py every `query(...)` call must sit inside
  #     side_question, and side_question must keep the isolation markers
  #     (worker thread + fresh asyncio.run loop). This catches someone
  #     moving the call back onto the host loop or dropping the wrapper.
  ca = _NEMO / "claude_agent.py"
  tree = ast.parse(ca.read_text())
  side_q = next(
    (
      n
      for n in ast.walk(tree)
      if isinstance(n, ast.AsyncFunctionDef) and n.name == "side_question"
    ),
    None,
  )
  assert side_q is not None, "side_question not found in claude_agent.py"
  lo, hi = side_q.lineno, side_q.end_lineno

  query_calls = [
    n.lineno
    for n in ast.walk(tree)
    if isinstance(n, ast.Call)
    and isinstance(n.func, ast.Name)
    and n.func.id == "query"
  ]
  assert query_calls, "expected a query() call in claude_agent.py"
  outside = [ln for ln in query_calls if not (lo <= ln <= hi)]
  assert not outside, (
    f"query() called outside side_question at lines {outside} — it must "
    "run only inside the isolated worker"
  )

  body_src = ast.get_source_segment(ca.read_text(), side_q) or ""
  assert "asyncio.to_thread(" in body_src, (
    "side_question lost its worker-thread isolation (asyncio.to_thread) — "
    "driving the SDK on the host loop crashes the daemon"
  )
  assert "asyncio.run(" in body_src, (
    "side_question lost its dedicated worker loop (asyncio.run) — the SDK's "
    "anyio scopes must not share the host loop"
  )


# ---------------------------------------------------------------------------
# 2. Loop-survival contract (always on, deterministic)
# ---------------------------------------------------------------------------


def test_side_question_teardown_error_does_not_kill_host_loop(monkeypatch):
  """Reproduce the production *symptom* deterministically: the SDK
  generator's teardown blows up the way anyio did. side_question must
  swallow it inside the worker and the host loop must still be usable —
  the original bug killed the loop and silenced the chat."""
  import claude_agent_sdk as sdk

  class _Gen:
    def __init__(self):
      self._done = False

    def __aiter__(self):
      return self

    async def __anext__(self):
      if self._done:
        raise StopAsyncIteration
      self._done = True
      return sdk.ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1,
        is_error=False, num_turns=1, session_id="s")

    async def aclose(self):
      # The exact anyio failure shape that took the daemon down.
      raise RuntimeError(
        "Attempted to exit cancel scope in a different task than it "
        "was entered in")

  monkeypatch.setattr(sdk, "query", lambda **_kw: _Gen())

  agent = _btw_claude_agent()
  loop = asyncio.new_event_loop()
  try:
    answer = loop.run_until_complete(agent.side_question("q", "sess"))
    assert isinstance(answer, str) and answer  # graceful, never raised

    async def _probe() -> str:
      await asyncio.sleep(0)
      return "alive"

    # Production symptom was the host loop dying here.
    assert loop.run_until_complete(_probe()) == "alive"
  finally:
    loop.close()


def test_side_question_drains_generator_before_closing(monkeypatch):
  """The teardown crash (and the empty-answer symptom) came from breaking
  out of the async-for on ResultMessage and then aclose()-ing the still
  suspended generator — finalising it from a different task than the one
  that entered its anyio task group. Lock the fix: side_question must
  consume the generator to StopAsyncIteration, so the SDK closes its own
  task group on the worker task; any aclose() only runs after the
  generator is already exhausted (where it is a no-op)."""
  import claude_agent_sdk as sdk

  events: list[str] = []

  class _Gen:
    def __init__(self):
      self._msgs = iter([
        sdk.AssistantMessage(
          content=[sdk.TextBlock(text="hi there")], model="m"),
        sdk.ResultMessage(
          subtype="success", duration_ms=1, duration_api_ms=1,
          is_error=False, num_turns=1, session_id="s"),
      ])
      self._exhausted = False

    def __aiter__(self):
      return self

    async def __anext__(self):
      try:
        return next(self._msgs)
      except StopIteration:
        self._exhausted = True
        events.append("exhausted")
        raise StopAsyncIteration

    async def aclose(self):
      # Real exhausted async generators no-op here; the bug was aclose()
      # being called on a *suspended* one. Fail loudly if that regresses.
      assert self._exhausted, (
        "aclose() called before the generator drained — this is exactly "
        "the cross-task finalisation that crashes anyio")
      events.append("aclose")

  monkeypatch.setattr(sdk, "query", lambda **_kw: _Gen())

  agent = _btw_claude_agent()
  answer = asyncio.run(agent.side_question("q", "sess"))

  assert answer == "hi there"
  # Drained to completion first; the ResultMessage did NOT short-circuit
  # the loop before the generator naturally stopped.
  assert events[0] == "exhausted", events


def test_side_question_empty_answer_is_logged(monkeypatch, caplog):
  """A blank `/btw` (a result with no assistant text) used to return the
  fallback string while logging nothing — leaving zero forensic trace of
  why it was blank. Lock that it now emits a diagnostic WARNING."""
  import logging

  import claude_agent_sdk as sdk

  class _Gen:
    def __init__(self):
      self._done = False

    def __aiter__(self):
      return self

    async def __anext__(self):
      if self._done:
        raise StopAsyncIteration
      self._done = True
      return sdk.ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1,
        is_error=False, num_turns=1, session_id="s")

    async def aclose(self):
      pass

  monkeypatch.setattr(sdk, "query", lambda **_kw: _Gen())

  agent = _btw_claude_agent()
  with caplog.at_level(logging.WARNING, logger="nemo.claude_agent"):
    answer = asyncio.run(agent.side_question("q", "sess"))

  assert answer == "⚠️ btw returned no answer."
  msgs = [r.getMessage() for r in caplog.records]
  assert any("side_question empty" in m for m in msgs), msgs
  # The diagnostic must carry the shape needed to triage (no text, a
  # success result, not a timeout).
  hit = next(m for m in msgs if "side_question empty" in m)
  assert "assistant_msgs=0" in hit and "timed_out=False" in hit, hit


def test_side_question_allows_more_than_one_turn(monkeypatch):
  """Production blank `/btw`: deepseek-v4-pro, resumed into a tool-heavy
  session, attempted (denied) tool calls and the turn ended as
  `error_max_turns` before any text — because max_turns was 1. The side
  query exposes no tools, so read-only is guaranteed by the allow-list,
  not by capping turns; capping at 1 only starves agentic models of the
  turn they need to answer after their tool calls bounce. Lock max_turns
  > 1 so this can't regress."""
  import claude_agent_sdk as sdk

  captured: dict[str, object] = {}

  class _Gen:
    def __init__(self):
      self._msgs = iter([
        sdk.AssistantMessage(
          content=[sdk.TextBlock(text="ok")], model="m"),
        sdk.ResultMessage(
          subtype="success", duration_ms=1, duration_api_ms=1,
          is_error=False, num_turns=1, session_id="s"),
      ])

    def __aiter__(self):
      return self

    async def __anext__(self):
      try:
        return next(self._msgs)
      except StopIteration:
        raise StopAsyncIteration

    async def aclose(self):
      pass

  def _fake_query(*, prompt, options):  # noqa: ARG001
    captured["max_turns"] = getattr(options, "max_turns", None)
    captured["allowed_tools"] = getattr(options, "allowed_tools", None)
    return _Gen()

  monkeypatch.setattr(sdk, "query", _fake_query)

  agent = _btw_claude_agent()
  answer = asyncio.run(agent.side_question("q", "sess"))

  assert answer == "ok"
  assert isinstance(captured["max_turns"], int) and captured["max_turns"] > 1, (
    f"max_turns must be > 1 to survive blocked tool attempts; "
    f"got {captured['max_turns']!r}")
  # Read-only must still rest on the empty allow-list, not on the turn cap.
  assert captured["allowed_tools"] == [], captured["allowed_tools"]


def test_side_question_captures_cli_stderr_on_failure(monkeypatch, caplog):
  """When the forked CLI exits non-zero the SDK only says "Command failed
  with exit code N (Check stderr output for details)" — the real reason is
  in the CLI's stderr, which the SDK discards unless a callback is given.
  Lock that side_question (a) passes a stderr callback so the SDK pipes it,
  and (b) logs the captured tail on failure, so `⚠️ btw failed: …` is
  diagnosable instead of opaque."""
  import logging

  import claude_agent_sdk as sdk

  captured: dict[str, object] = {}

  class _Gen:
    def __init__(self, options):
      self._options = options
      self._started = False

    def __aiter__(self):
      return self

    async def __anext__(self):
      if not self._started:
        self._started = True
        # The SDK pipes the CLI's stderr line-by-line into this callback,
        # then surfaces the non-zero exit as ProcessError.
        if callable(self._options.stderr):
          self._options.stderr("Error: deepseek endpoint 429 rate limited")
        raise sdk.ProcessError(
          "Command failed with exit code 1", exit_code=1,
          stderr="Check stderr output for details")
      raise StopAsyncIteration

    async def aclose(self):
      pass

  def _fake_query(*, prompt, options):  # noqa: ARG001
    captured["stderr_cb"] = options.stderr
    return _Gen(options)

  monkeypatch.setattr(sdk, "query", _fake_query)

  agent = _btw_claude_agent()
  with caplog.at_level(logging.WARNING, logger="nemo.claude_agent"):
    answer = asyncio.run(agent.side_question("q", "sess"))

  assert answer.startswith("⚠️ btw failed"), answer
  assert callable(captured["stderr_cb"]), "no stderr callback passed to query()"
  msgs = [r.getMessage() for r in caplog.records]
  assert any(
    "cli stderr" in m and "429 rate limited" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 3. Real-SDK smoke (gated; release must run with NEMO_REAL_SDK=1)
# ---------------------------------------------------------------------------


@pytest.mark.realsdk
@pytest.mark.skipif(
  os.environ.get("NEMO_REAL_SDK") != "1",
  reason="real-SDK smoke: set NEMO_REAL_SDK=1 (needs claude CLI + auth)",
)
def test_side_question_real_sdk_survives_persistent_loop():
  """The only test that drives the real anyio lifecycle end to end —
  exactly what the mocked unit tests could never catch. Runs a real /btw
  on a persistent loop and asserts the loop survives afterwards."""
  agent = _btw_claude_agent()
  agent._model = os.environ.get("NEMO_REAL_SDK_MODEL", "claude-haiku-4-5-20251001")

  loop = asyncio.new_event_loop()
  try:
    answer = loop.run_until_complete(
      agent.side_question("Reply with exactly the word: pong", ""))
    assert isinstance(answer, str) and answer.strip(), answer
    assert not answer.startswith("⚠️ btw failed"), answer

    async def _probe() -> str:
      await asyncio.sleep(0)
      return "alive"

    assert loop.run_until_complete(_probe()) == "alive"
  finally:
    loop.close()
