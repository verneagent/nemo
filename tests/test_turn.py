"""Tests for nemo.turn — SDK turn execution and event emission."""

import asyncio
from dataclasses import dataclass
from unittest import mock

import pytest

from nemo.turn import (
  ProgressEvent, AnswerEvent,
  TaskStartedEvent, TaskDoneEvent, DoneEvent, ErrorEvent,
  RateLimitNoticeEvent, CompactNoticeEvent, TurnEvent,
  canonical_usage,
)
from nemo.claude_turn import run_turn, normalize_claude_usage


def test_canonical_usage_sums_total():
  assert canonical_usage(
    input_tokens=10, cache_read=20, cache_creation=5, output_tokens=3,
  ) == {
    "input_tokens": 10, "cache_read_input_tokens": 20,
    "cache_creation_input_tokens": 5, "output_tokens": 3, "total_tokens": 38,
  }


def test_normalize_claude_usage_corrects_omlx_inclusive_input():
  # omlx#1487 (v0.3.8-0.3.12): the Anthropic shim returns input_tokens
  # INCLUSIVE of cache_read + cache_creation. Detect via exact equality and
  # subtract — real Anthropic essentially never has new-uncached input that
  # matches the cached prefix sum.
  out = normalize_claude_usage({
    "input_tokens": 65619,
    "cache_read_input_tokens": 63488,
    "cache_creation_input_tokens": 2131,
    "output_tokens": 58,
  })
  # Heuristic clamps i to 0; total now reflects real prompt + output.
  assert out == {
    "input_tokens": 0,
    "cache_read_input_tokens": 63488,
    "cache_creation_input_tokens": 2131,
    "output_tokens": 58,
    "total_tokens": 65677,
  }


def test_normalize_claude_usage_leaves_real_anthropic_alone():
  # Real Anthropic shape: tiny input_tokens (just the new user message),
  # large cache_read / cache_creation. Must NOT trigger the heuristic.
  out = normalize_claude_usage({
    "input_tokens": 3,
    "cache_read_input_tokens": 9442,
    "cache_creation_input_tokens": 15010,
    "output_tokens": 5,
  })
  assert out["input_tokens"] == 3  # untouched
  assert out["total_tokens"] == 24460


def test_normalize_claude_usage_no_cache_unchanged():
  # cr + cw == 0 — heuristic must not fire even if input_tokens happens to
  # also be 0 (degenerate equality 0 == 0 must be ignored).
  out = normalize_claude_usage({
    "input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "output_tokens": 5,
  })
  assert out["input_tokens"] == 0
  assert out["total_tokens"] == 5


def test_normalize_claude_usage_drops_extras_and_adds_total():
  # Claude's native ResultMessage.usage carries cache + bookkeeping keys; only
  # the canonical five survive, and total_tokens is computed.
  out = normalize_claude_usage({
    "input_tokens": 3,
    "cache_creation_input_tokens": 15010,
    "cache_read_input_tokens": 9442,
    "output_tokens": 5,
    "server_tool_use": {"web_search_requests": 0},
    "service_tier": "standard",
  })
  assert out == {
    "input_tokens": 3, "cache_read_input_tokens": 9442,
    "cache_creation_input_tokens": 15010, "output_tokens": 5,
    "total_tokens": 24460,
  }
  # Empty usage stays empty so the card omits the line.
  assert normalize_claude_usage({}) == {}


# ---------------------------------------------------------------------------
# Mock SDK types (so we don't need claude_agent_sdk installed)
# ---------------------------------------------------------------------------

@dataclass
class FakeTextBlock:
  text: str
  type: str = "text"


@dataclass
class FakeThinkingBlock:
  thinking: str
  signature: str = ""
  type: str = "thinking"


@dataclass
class FakeToolUseBlock:
  name: str
  input: dict
  id: str = "tu_1"
  type: str = "tool_use"


@dataclass
class FakeAssistantMessage:
  content: list
  parent_tool_use_id: str | None = None


@dataclass
class FakeResultMessage:
  total_cost_usd: float = 0.01
  usage: dict = None
  is_error: bool = False
  result: str = ""

  def __post_init__(self):
    if self.usage is None:
      self.usage = {"input_tokens": 100, "output_tokens": 50}


@dataclass
class FakeTaskStartedMessage:
  task_id: str


@dataclass
class FakeTaskNotificationMessage:
  task_id: str
  status: str = "completed"


@dataclass
class FakeTaskProgressMessage:
  description: str = ""


# ---------------------------------------------------------------------------
# Helper: mock client
# ---------------------------------------------------------------------------

class FakeClient:
  def __init__(self, messages):
    self._messages = messages
    self._queried = False

  async def query(self, prompt):
    self._queried = True

  async def receive_messages(self):
    for msg in self._messages:
      yield msg

  async def stop_task(self, task_id):
    pass


class QueueClient:
  """Models the real SDK client's SINGLE underlying message stream.

  ``receive_messages()`` pulls from one shared asyncio.Queue and blocks when it
  is empty (exactly as the CLI stream goes silent between turns). Messages left
  unconsumed by one run_turn therefore persist into the NEXT run_turn's
  receive_messages() — this is what lets the test detect the "+1 turn lag"
  stranding bug. ``query()`` sets the steer flag it was given, mirroring
  SDKThread.steer.
  """

  def __init__(self, steered):
    self.q: asyncio.Queue = asyncio.Queue()
    self._steered = steered
    self.queried: list[str] = []

  def feed(self, *messages):
    for m in messages:
      self.q.put_nowait(m)

  async def query(self, prompt):
    self.queried.append(prompt)
    self._steered[0] = True

  async def receive_messages(self):
    while True:
      msg = await self.q.get()
      yield msg

  async def stop_task(self, task_id):
    pass


def _sdk_modules():
  """Create a mock for claude_agent_sdk module."""
  return {
    "claude_agent_sdk": mock.MagicMock(
      AssistantMessage=FakeAssistantMessage,
      TextBlock=FakeTextBlock,
      ThinkingBlock=FakeThinkingBlock,
      ToolUseBlock=FakeToolUseBlock,
      ResultMessage=FakeResultMessage,
      TaskStartedMessage=FakeTaskStartedMessage,
      TaskNotificationMessage=FakeTaskNotificationMessage,
      TaskProgressMessage=FakeTaskProgressMessage,
    ),
  }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_text_only_turn():
  """A turn that produces only text output."""
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="Hello world")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 200}),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      cost, usage = await run_turn(client, "say hello", events.append)
      assert cost == 0.02
      assert usage["input_tokens"] == 200
      assert usage["total_tokens"] == 200  # canonical: total computed from parts
  asyncio.run(_run())
  assert any(isinstance(e, AnswerEvent) and e.text == "Hello world" for e in events)
  assert any(isinstance(e, DoneEvent) for e in events)


def test_tool_use_turn():
  """A turn with tool use should emit ProgressEvents (first=True then first=False)."""
  messages = [
    FakeAssistantMessage(content=[FakeToolUseBlock(name="Read", input={"file_path": "/a/b.py"})]),
    FakeAssistantMessage(content=[FakeToolUseBlock(name="Edit", input={"file_path": "/a/c.py"})]),
    FakeAssistantMessage(content=[FakeTextBlock(text="Done editing")]),
    FakeResultMessage(),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "fix bug", events.append)
  asyncio.run(_run())
  tool_events = [e for e in events if isinstance(e, ProgressEvent) and e.kind == "tool"]
  assert len(tool_events) == 2
  assert tool_events[0].first is True
  assert tool_events[1].first is False


def test_task_lifecycle():
  """TaskStarted and TaskNotification should emit corresponding events."""
  messages = [
    FakeAssistantMessage(content=[FakeToolUseBlock(name="Agent", input={"description": "search"})]),
    FakeTaskStartedMessage(task_id="task_1"),
    FakeTaskNotificationMessage(task_id="task_1", status="completed"),
    FakeResultMessage(),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "search code", events.append)
  asyncio.run(_run())
  assert any(isinstance(e, TaskStartedEvent) and e.task_id == "task_1" for e in events)
  assert any(isinstance(e, TaskDoneEvent) and e.task_id == "task_1" for e in events)


def test_stale_leak_raises_for_reconnect_resume():
  """SDK #788: a stale TaskNotification leaking into the stream must raise
  StaleLeakError (a TransientAPIError) so SDKThread.run_turn_with_reconnect
  recovers via reconnect-with-resume — NOT an in-band drain.
  """
  from nemo.claude_turn import StaleLeakError, TransientAPIError

  assert issubclass(StaleLeakError, TransientAPIError), \
    "StaleLeakError must subclass TransientAPIError so the existing " \
    "reconnect-with-resume path catches it"

  stale_messages = [
    FakeTaskNotificationMessage(task_id="stale_1", status="completed"),
    FakeAssistantMessage(content=[FakeTextBlock(text="answer to a STALE")]),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  sent_prompts: list[str] = []
  events: list = []
  call_count = 0

  class Client:
    async def query(self, prompt):
      sent_prompts.append(prompt)

    async def receive_messages(self):
      nonlocal call_count
      call_count += 1
      for msg in stale_messages:
        yield msg

    async def stop_task(self, task_id):
      pass

  stale = {"stale_1"}

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(Client(), "real prompt", events.append,
                     stale_tasks=stale)

  with pytest.raises(StaleLeakError):
    asyncio.run(_run())

  # Exactly one turn attempted — no in-band drain/retry loop.
  assert call_count == 1
  assert sent_prompts == ["real prompt"]
  # The leaked stale's answer must NOT surface to the user.
  assert not any(isinstance(e, AnswerEvent) for e in events), \
    "stale-contaminated output leaked to on_event"
  # No DoneEvent — turn did not complete; the reconnect layer retries.
  assert not any(isinstance(e, DoneEvent) for e in events)
  # A visible breadcrumb MUST be emitted before the raise so the recovery
  # is not silent.
  from nemo.turn import StaleLeakNoticeEvent
  notices = [e for e in events if isinstance(e, StaleLeakNoticeEvent)]
  assert len(notices) == 1 and notices[0].task_id == "stale_1", \
    f"expected one StaleLeakNoticeEvent for stale_1, got {notices}"
  # All tracked stale ids cleared: they belong to the about-to-be-killed
  # subprocess; the resumed session starts clean.
  assert stale == set(), f"stale_tasks must be cleared on leak, got {stale}"


def test_stale_leak_detected_after_some_clean_output():
  """The leak guard fires wherever the stale appears — even after some
  legitimate output — still raising StaleLeakError and stopping the turn.
  """
  from nemo.claude_turn import StaleLeakError
  from nemo.turn import StaleLeakNoticeEvent

  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="partial work")]),
    FakeTaskNotificationMessage(task_id="ghost", status="completed"),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  events: list = []
  stale = {"ghost"}

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append,
                     stale_tasks=stale)

  with pytest.raises(StaleLeakError):
    asyncio.run(_run())
  assert stale == set()
  notices = [e for e in events if isinstance(e, StaleLeakNoticeEvent)]
  assert len(notices) == 1 and notices[0].task_id == "ghost"


def test_stop_task_circuit_breaker_on_control_timeout():
  """First 'Control request timeout' from stop_task disables subsequent calls.

  When the SDK's control channel wedges, stop_task returns this error on
  every pending id. After the first occurrence we should short-circuit all
  further stop_task invocations within the same run_turn orchestration.
  """
  # Turn yields two pending tasks, a real output, then ResultMessage. No
  # stale is marked, so the turn is "clean" — the pending tasks are both
  # promoted to stale. (The output keeps the turn non-empty; this test is
  # about the stop_task breaker, not empty responses.)
  messages = [
    FakeAssistantMessage(content=[FakeToolUseBlock(name="Agent", input={"description": "x"})]),
    FakeTaskStartedMessage(task_id="t1"),
    FakeTaskStartedMessage(task_id="t2"),
    FakeTaskStartedMessage(task_id="t3"),
    FakeAssistantMessage(content=[FakeTextBlock(text="done")]),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  stop_calls: list[str] = []

  class TimingOutClient:
    async def query(self, prompt):
      pass

    async def receive_messages(self):
      for msg in messages:
        yield msg

    async def stop_task(self, task_id):
      stop_calls.append(task_id)
      raise RuntimeError("Control request timeout: stop_task")

  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(TimingOutClient(), "real", events.append)

  asyncio.run(_run())
  # First stop_task failure trips the breaker — further ids are skipped.
  # (pending_tasks is a set so iteration order is not guaranteed; we only
  # care that exactly one stop_task was attempted.)
  assert len(stop_calls) == 1, \
    f"expected only first stop_task to be attempted, got {stop_calls}"
  assert stop_calls[0] in {"t1", "t2", "t3"}


def test_transient_api_error_in_assistant_message_raises_and_suppresses():
  """CLI-surfaced 'API Error:' text must not reach the user as an answer.

  It should instead raise TransientAPIError from _single_turn so
  SDKThread.run_turn_with_reconnect can reconnect and retry.
  """
  from nemo.claude_turn import TransientAPIError
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(
      text="API Error: Unable to connect to API (ECONNRESET)")]),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "hello", events.append)

  with pytest.raises(TransientAPIError):
    asyncio.run(_run())
  # The error text must NOT have been emitted to the user.
  assert not any(
    isinstance(e, AnswerEvent) and "API Error" in e.text
    for e in events
  ), "CLI error message leaked to user as AnswerEvent"


def test_non_retryable_402_api_error_raises_and_suppresses():
  """402/billing failures are provider/account state, not transient network."""
  from nemo.claude_turn import NonRetryableAPIError
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(
      text=(
        'API Error: 402 {"error":{"message":"Insufficient Balance",'
        '"type":"unknown_error"}}'
      ))]),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hello", events.append)

  with pytest.raises(NonRetryableAPIError):
    asyncio.run(_run())
  assert not any(
    isinstance(e, AnswerEvent) and "Insufficient Balance" in e.text
    for e in events
  ), "non-retryable API error leaked to user as AnswerEvent"
  assert any(
    isinstance(e, ErrorEvent) and "Insufficient Balance" in e.message
    for e in events
  )


def test_transient_api_error_via_result_is_error_flag():
  """ResultMessage.is_error + transient-signal body → TransientAPIError."""
  from nemo.claude_turn import TransientAPIError
  messages = [
    FakeResultMessage(
      total_cost_usd=0.0,
      is_error=True,
      result="Request failed: socket hang up",
    ),
  ]
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append)

  with pytest.raises(TransientAPIError):
    asyncio.run(_run())


def test_transient_api_error_detector_no_false_positive_on_user_topic():
  """User asking 'what is ECONNRESET?' gets a normal answer — no reconnect.

  Real model replies never START with 'API Error:' and is_error is False,
  so strict prefix matching must let this through as a regular AnswerEvent.
  """
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(
      text=(
        "ECONNRESET is a TCP error code meaning the remote peer forcibly "
        "closed the connection (RST packet). You'll see it when the server "
        "crashed, a firewall killed the connection, or a load balancer "
        "dropped the socket. It's distinct from ETIMEDOUT (no response) "
        "and EAI_AGAIN (DNS retry)."
      ))]),
    FakeResultMessage(total_cost_usd=0.02, is_error=False),
  ]
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "what is ECONNRESET?",
                     events.append)

  asyncio.run(_run())
  answers = [e.text for e in events if isinstance(e, AnswerEvent)]
  assert len(answers) == 1, f"expected one AnswerEvent, got {answers!r}"
  assert "ECONNRESET" in answers[0]


def test_leaked_tool_call_markup_is_flagged_not_shown_as_answer():
  """Opus glitch: the model emits a tool call as PLAIN TEXT — the tool never
  runs and the bare `<invoke …>` markup would otherwise render as a bogus
  "Done ✓" answer. The guard must wrap it in an anomaly notice instead.

  Reproduces the exact observed leak (chat oc_fac92663… session e51a4341):
  the final message of a simulator-driving turn was `court\\n<invoke
  name="Read">…</invoke>`, so the screenshot was never read.
  """
  leaked = (
    'court\n<invoke name="Read">\n'
    '<parameter name="file_path">/tmp/vf_agent.png</parameter>\n</invoke>'
  )
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text=leaked)]),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "怎么样了", events.append)

  asyncio.run(_run())
  answers = [e for e in events if isinstance(e, AnswerEvent)]
  assert len(answers) == 1
  body = answers[0].text
  # Must NOT be presented verbatim as if it were a real answer.
  assert body != leaked
  # The anomaly notice (with the warning marker) must be present, and the raw
  # markup preserved for debugging — fenced so the channel renders it literally.
  assert "⚠️" in body and "未完成" in body
  assert "```" in body and leaked in body
  # The turn still completes (we don't crash or reconnect — it's a model glitch,
  # not a transport failure).
  assert any(isinstance(e, DoneEvent) for e in events)


def test_leaked_tool_call_guard_no_false_positive_on_fenced_explanation():
  """A model legitimately *explaining* the tool-call format inside a code
  fence must pass through untouched — the guard only fires on bare leaks."""
  explanation = (
    "To call a tool you emit a structured block like this:\n\n"
    '```\n<invoke name="Read">\n'
    '<parameter name="file_path">/x.py</parameter>\n</invoke>\n```\n\n'
    "The SDK parses that into a tool_use block."
  )
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text=explanation)]),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "how do tool calls work?",
                     events.append)

  asyncio.run(_run())
  answers = [e for e in events if isinstance(e, AnswerEvent)]
  assert len(answers) == 1
  assert answers[0].text == explanation  # untouched
  assert "⚠️" not in answers[0].text


def test_leaked_tool_call_guard_ignores_incomplete_fragment():
  """Prose that merely names `<invoke` without a closing `</invoke>` is not a
  balanced invocation and must not trip the guard."""
  from nemo.claude_turn import _looks_like_leaked_tool_call
  assert not _looks_like_leaked_tool_call(
    "The <invoke name= opener starts an Anthropic tool call.")
  assert not _looks_like_leaked_tool_call("just a normal answer")
  # Bare, balanced, un-fenced markup IS a leak.
  assert _looks_like_leaked_tool_call(
    '<invoke name="ui_tap"><parameter name="x">1</parameter></invoke>')


def test_clean_real_turn_preserves_freshly_promoted_stales():
  """Regression: pending_tasks promoted to stale at ResultMessage must
  survive the "clean real turn" drop path.

  Previous behavior: after a clean real turn, ALL stale_tasks were cleared
  under the assumption that nothing surfaced means nothing will. But the
  ResultMessage handler had just promoted N still-pending tasks into
  stale_tasks seconds earlier — those legitimate future-stale ids got
  wiped, so the next turn saw unexpected TaskNotificationMessages that
  weren't in stale_tasks and leaked as TaskDoneEvents / contaminated the
  model context.
  """
  # Turn 1: two background tasks start but never complete before
  # ResultMessage. Neither id was in stale_tasks at turn start.
  messages = [
    FakeTaskStartedMessage(task_id="fresh_x"),
    FakeTaskStartedMessage(task_id="fresh_y"),
    FakeAssistantMessage(content=[FakeTextBlock(text="kicking off tasks")]),
    FakeResultMessage(total_cost_usd=0.1),
  ]
  events: list = []
  stale: set[str] = set()

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "do stuff",
                     events.append, stale_tasks=stale)

  asyncio.run(_run())

  # Both freshly promoted ids must remain in stale_tasks for the NEXT turn
  # to recognise late TaskNotifications as stale.
  assert stale == {"fresh_x", "fresh_y"}, \
    f"freshly promoted stales were wrongly dropped: {stale}"


def test_progress_timeout_fires_on_systemmessage_storm(monkeypatch):
  """SDK emitting SystemMessages indefinitely must not keep a dead turn alive.

  Observed in production: claude CLI enters a rate-limit retry loop that
  emits a SystemMessage every ~1 min. Each SystemMessage used to reset
  HEARTBEAT_TIMEOUT, so the turn was never declared stuck. Progress-only
  timeout should trigger after PROGRESS_TIMEOUT seconds regardless.
  """
  from nemo import claude_turn as turn_module

  # Shrink both timeouts so the test runs fast.
  monkeypatch.setattr(turn_module, "PROGRESS_TIMEOUT", 0.5)
  monkeypatch.setattr(turn_module, "HEARTBEAT_TIMEOUT", 5)

  # A fake "SystemMessage" class (outside our claude_agent_sdk imports) whose
  # class name matches the non-progress set.
  @dataclass
  class SystemMessage:
    subtype: str = "rate_limit"

  # Yield SystemMessages forever (simulating rate-limit retry loop).
  # The progress timeout should fire first and break out.
  async def forever_system():
    while True:
      await asyncio.sleep(0.2)
      yield SystemMessage()

  class StormClient:
    async def query(self, prompt):
      pass

    def receive_messages(self):
      return forever_system()

    async def stop_task(self, task_id):
      pass

  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(StormClient(), "hello", events.append)

  with pytest.raises(TimeoutError):
    asyncio.run(_run())

  # The user should see an ErrorEvent documenting the stuck turn.
  assert any(isinstance(e, ErrorEvent) for e in events), \
    "progress-timeout abort must surface as ErrorEvent"


def test_watchdog_defers_while_prompt_awaits_user(monkeypatch):
  """A pending AskUserQuestion/permission prompt must not trip the watchdog.

  Regression: while an interactive prompt is awaiting the user, the SDK is
  blocked inside its can_use_tool callback and emits no messages. That
  silence used to exhaust PROGRESS_TIMEOUT and force a reconnect, which tore
  down the in-flight question and re-asked it — discarding whatever the user
  had already selected (observed: a multi-select that "took three tries").
  With ``is_paused`` reporting True the watchdog stands down, and the turn
  completes normally once the user answers and the message stream resumes.
  """
  from nemo import claude_turn as turn_module

  # Shrink so the silent (paused) stretch spans several watchdog ticks fast.
  monkeypatch.setattr(turn_module, "PROGRESS_TIMEOUT", 0.3)
  monkeypatch.setattr(turn_module, "HEARTBEAT_TIMEOUT", 5)

  paused = {"v": True}

  # Stay silent well past PROGRESS_TIMEOUT (simulating the SDK blocked in
  # can_use_tool), then "the user answers": clear the pause and yield a real
  # output + result. (The output keeps the turn non-empty; this test is about
  # the watchdog deferral, not empty responses.)
  async def blocked_then_answer():
    await asyncio.sleep(1.0)
    paused["v"] = False
    yield FakeAssistantMessage(content=[FakeTextBlock(text="Here you go")])
    yield FakeResultMessage(total_cost_usd=0.05)

  gen = blocked_then_answer()

  class PausedClient:
    async def query(self, prompt):
      pass

    def receive_messages(self):
      return gen

    async def stop_task(self, task_id):
      pass

  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      return await run_turn(
        PausedClient(), "ask the user", events.append,
        is_paused=lambda: paused["v"])

  cost, _usage = asyncio.run(_run())

  # No TimeoutError raised, the answer was consumed, and the turn finished.
  assert cost == 0.05
  assert any(isinstance(e, DoneEvent) for e in events)
  assert not any(isinstance(e, ErrorEvent) for e in events), \
    "watchdog must not fire while a prompt is awaiting the user"


def test_watchdog_still_fires_when_not_paused(monkeypatch):
  """Sanity: the same silent stall WITHOUT a pending prompt still times out.

  Guards against is_paused accidentally disabling the watchdog for genuine
  hangs — only an actually-pending prompt (is_paused True) should defer it.
  """
  from nemo import claude_turn as turn_module

  monkeypatch.setattr(turn_module, "PROGRESS_TIMEOUT", 0.3)
  monkeypatch.setattr(turn_module, "HEARTBEAT_TIMEOUT", 5)

  async def never_answers():
    await asyncio.sleep(5.0)
    yield FakeResultMessage()

  gen = never_answers()

  class HungClient:
    async def query(self, prompt):
      pass

    def receive_messages(self):
      return gen

    async def stop_task(self, task_id):
      pass

  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      # is_paused present but always False — a real hang, not a prompt.
      await run_turn(
        HungClient(), "do work", events.append, is_paused=lambda: False)

  with pytest.raises(TimeoutError):
    asyncio.run(_run())
  assert any(isinstance(e, ErrorEvent) for e in events)


def test_reset_clears_stale_tasks_on_claude_adapter():
  """ClaudeCodingAgent.reset() must clear self._stale_tasks.

  Task ids from a dead SDK session will never produce notifications on a
  fresh session, so keeping them would risk a spurious StaleLeakError
  after every reconnect.
  """
  from nemo.claude_agent import ClaudeCodingAgent

  # Build a minimal adapter without touching the real SDK.
  adapter = ClaudeCodingAgent.__new__(ClaudeCodingAgent)
  adapter._stale_tasks = {"ghost_1", "ghost_2"}

  # Patch the parts reset() touches so we don't actually spawn a CLI.
  adapter._sdk = mock.AsyncMock()
  adapter._sdk.reconnect = mock.AsyncMock()
  adapter._build_options = mock.Mock(return_value=object())

  async def _run():
    await adapter.reset("/tmp", "claude-opus-4-6")

  asyncio.run(_run())
  assert adapter._stale_tasks == set(), \
    "reset() must clear _stale_tasks since old session ids are dead"


def test_empty_turn_raises_empty_response_error():
  """A turn that completes (ResultMessage) with no real output must NOT be
  finalized as a misleading empty Done card — it raises EmptyResponseError so
  the reconnect layer retries once and the host surfaces an explicit
  empty-response error card (observed: empty Done card from the CLI's
  "No response requested." placeholder).
  """
  from nemo.claude_turn import EmptyResponseError

  messages = [FakeResultMessage(total_cost_usd=0.0, usage={})]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "", events.append)
  with pytest.raises(EmptyResponseError):
    asyncio.run(_run())
  # The turn must never emit a DoneEvent for an empty completion.
  assert not any(isinstance(e, DoneEvent) for e in events)


def test_empty_placeholder_text_is_not_an_answer():
  """The CLI's "No response requested." placeholder as an AssistantMessage
  text must be ignored (not emitted as an AnswerEvent, not marked as the
  answer) so the turn is treated as empty and raises EmptyResponseError.
  """
  from nemo.claude_turn import EmptyResponseError

  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="No response requested.")]),
    FakeResultMessage(total_cost_usd=0.0, usage={}),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "", events.append)
  with pytest.raises(EmptyResponseError):
    asyncio.run(_run())
  assert not any(isinstance(e, AnswerEvent) for e in events), \
    "the empty-response placeholder must not be surfaced as an answer"


def test_task_progress_event():
  """TaskProgressMessage should emit ProgressEvent when working."""
  messages = [
    FakeAssistantMessage(content=[FakeToolUseBlock(name="Bash", input={"command": "ls"})]),
    FakeTaskProgressMessage(description="Searching files..."),
    FakeResultMessage(),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "test", events.append)
  asyncio.run(_run())
  progress = [e for e in events if isinstance(e, ProgressEvent) and e.kind == "tool"]
  assert any(e.summary == "Searching files..." for e in progress)


def test_mixed_text_and_tool():
  """Message with both text and tool blocks."""
  messages = [
    FakeAssistantMessage(content=[
      FakeTextBlock(text="Let me check"),
      FakeToolUseBlock(name="Read", input={"file_path": "/x.py"}),
    ]),
    FakeResultMessage(),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "test", events.append)
  asyncio.run(_run())
  # Text takes priority when present
  answer_events = [e for e in events if isinstance(e, AnswerEvent)]
  assert len(answer_events) == 1
  assert answer_events[0].text == "Let me check"


def test_trailing_thinking_compensation():
  """When the last event is thinking (no text), an AnswerEvent should be synthesized."""
  messages = [
    FakeAssistantMessage(content=[
      FakeThinkingBlock(thinking="Let me analyze this"),
      FakeToolUseBlock(name="Read", input={"file_path": "/a.py"}),
    ]),
    FakeAssistantMessage(content=[
      FakeThinkingBlock(thinking="I see the issue, the fix is to change line 42"),
    ]),
    FakeResultMessage(),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "fix bug", events.append)
  asyncio.run(_run())
  answer_events = [e for e in events if isinstance(e, AnswerEvent)]
  assert len(answer_events) == 1
  assert answer_events[0].text == "I see the issue, the fix is to change line 42"
  assert any(isinstance(e, DoneEvent) for e in events)


def test_rate_limit_event_emits_notice():
  """SDK's RateLimitEvent (matched by class name in non-progress branch) must
  surface as a RateLimitNoticeEvent carrying the upstream status fields."""

  @dataclass
  class FakeRateLimitInfo:
    status: str
    rate_limit_type: str = ""
    resets_at: int | None = None
    utilization: float | None = None

  @dataclass
  class RateLimitEvent:
    rate_limit_info: FakeRateLimitInfo

  messages = [
    RateLimitEvent(rate_limit_info=FakeRateLimitInfo(
      status="rejected",
      rate_limit_type="five_hour",
      resets_at=1_700_000_000,
      utilization=0.97,
    )),
    FakeAssistantMessage(content=[FakeTextBlock(text="back to work")]),
    FakeResultMessage(),
  ]
  events: list = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append)
  asyncio.run(_run())

  notices = [e for e in events if isinstance(e, RateLimitNoticeEvent)]
  assert len(notices) == 1
  n = notices[0]
  assert n.status == "rejected"
  assert n.rate_limit_type == "five_hour"
  assert n.resets_at == 1_700_000_000
  assert abs((n.utilization or 0.0) - 0.97) < 1e-9


def test_compact_boundary_emits_notice():
  """SystemMessage(subtype="compact_boundary") must surface as
  CompactNoticeEvent with the metadata fields extracted from data."""

  @dataclass
  class SystemMessage:
    subtype: str
    data: dict

  messages = [
    SystemMessage(
      subtype="compact_boundary",
      data={
        "type": "system",
        "subtype": "compact_boundary",
        "compact_metadata": {
          "trigger": "auto",
          "pre_tokens": 45_000,
          "post_tokens": 8_200,
          "duration_ms": 12_345,
        },
      },
    ),
    FakeAssistantMessage(content=[FakeTextBlock(text="back to work")]),
    FakeResultMessage(),
  ]
  events: list = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append)
  asyncio.run(_run())

  notices = [e for e in events if isinstance(e, CompactNoticeEvent)]
  assert len(notices) == 1
  n = notices[0]
  assert n.trigger == "auto"
  assert n.pre_tokens == 45_000
  assert n.post_tokens == 8_200
  assert n.duration_ms == 12_345
  # Compaction is real work — the downstream AssistantMessage should still
  # produce its AnswerEvent (i.e. compact_boundary doesn't short-circuit
  # the rest of the turn).
  assert any(isinstance(e, AnswerEvent) and e.text == "back to work"
             for e in events)


def test_compact_boundary_with_partial_metadata():
  """A compact_boundary with only the required fields (no post_tokens / no
  duration) must still emit a notice — those fields are documented optional
  in the Claude CLI stream-json schema."""

  @dataclass
  class SystemMessage:
    subtype: str
    data: dict

  messages = [
    SystemMessage(
      subtype="compact_boundary",
      data={
        "type": "system",
        "subtype": "compact_boundary",
        "compact_metadata": {"trigger": "manual", "pre_tokens": 30_000},
      },
    ),
    FakeAssistantMessage(content=[FakeTextBlock(text="resuming")]),
    FakeResultMessage(),
  ]
  events: list = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append)
  asyncio.run(_run())

  notices = [e for e in events if isinstance(e, CompactNoticeEvent)]
  assert len(notices) == 1
  assert notices[0].trigger == "manual"
  assert notices[0].pre_tokens == 30_000
  assert notices[0].post_tokens == 0
  assert notices[0].duration_ms == 0


def test_microcompact_boundary_is_suppressed():
  """SystemMessage(subtype="microcompact_boundary") should be silently
  dropped — the Claude CLI's UI also returns null for it. We still want
  to consume the message without crashing or surfacing a notice."""

  @dataclass
  class SystemMessage:
    subtype: str
    data: dict

  messages = [
    SystemMessage(subtype="microcompact_boundary", data={}),
    FakeAssistantMessage(content=[FakeTextBlock(text="back")]),
    FakeResultMessage(),
  ]
  events: list = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append)
  asyncio.run(_run())
  assert not any(isinstance(e, CompactNoticeEvent) for e in events)


def test_rate_limit_event_with_missing_info_is_skipped():
  """If rate_limit_info is absent, no notice should be emitted (defensive)."""
  @dataclass
  class RateLimitEvent:
    rate_limit_info: object = None

  messages = [
    RateLimitEvent(),
    FakeAssistantMessage(content=[FakeTextBlock(text="proceeding")]),
    FakeResultMessage(),
  ]
  events: list = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hi", events.append)
  asyncio.run(_run())
  assert not any(isinstance(e, RateLimitNoticeEvent) for e in events)


def test_no_compensation_when_answer_is_last():
  """No trailing thinking compensation when the last event is an AnswerEvent."""
  messages = [
    FakeAssistantMessage(content=[
      FakeThinkingBlock(thinking="thinking..."),
      FakeTextBlock(text="Here is my answer"),
    ]),
    FakeResultMessage(),
  ]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      await run_turn(client, "test", events.append)
  asyncio.run(_run())
  answer_events = [e for e in events if isinstance(e, AnswerEvent)]
  assert len(answer_events) == 1
  assert answer_events[0].text == "Here is my answer"


def test_first_message_timeout():
  """First message doesn't arrive within 30s — TimeoutError + ErrorEvent."""
  events = []

  class HangingClient:
    async def query(self, _prompt):
      pass

    async def receive_messages(self):
      # Yield nothing — the async iterator hangs forever
      await asyncio.Future()  # never resolves
      # Make this a generator
      yield  # pragma: no cover

    async def stop_task(self, _task_id):
      pass

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = HangingClient()
      # Patch asyncio.wait to simulate timeout (empty done set)
      original_wait = asyncio.wait

      async def fake_wait(fs, **kwargs):
        # Return empty done set, all pending — simulates timeout
        return set(), set(fs)

      with mock.patch("asyncio.wait", side_effect=fake_wait):
        try:
          await run_turn(client, "hello", events.append)
          assert False, "Expected TimeoutError"
        except TimeoutError:
          pass
    assert any(isinstance(e, ErrorEvent) for e in events)
    error = next(e for e in events if isinstance(e, ErrorEvent))
    assert "timed out" in error.message.lower()

  asyncio.run(_run())


def test_heartbeat_timeout():
  """First message arrives, then second one times out at 300s."""
  events = []
  wait_call_count = 0

  class OneMessageClient:
    async def query(self, _prompt):
      pass

    async def receive_messages(self):
      yield FakeAssistantMessage(content=[FakeTextBlock(text="thinking...")])
      # Second message hangs forever
      await asyncio.Future()
      yield  # pragma: no cover

    async def stop_task(self, _task_id):
      pass

  async def _run():
    nonlocal wait_call_count
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = OneMessageClient()
      original_wait = asyncio.wait

      async def fake_wait(fs, **kwargs):
        nonlocal wait_call_count
        wait_call_count += 1
        if wait_call_count == 1:
          # First call: let the message through
          return await original_wait(fs, **kwargs)
        else:
          # Second call: simulate timeout
          return set(), set(fs)

      with mock.patch("asyncio.wait", side_effect=fake_wait):
        try:
          await run_turn(client, "hello", events.append)
          assert False, "Expected TimeoutError"
        except TimeoutError:
          pass
    # Should have the text event from first message
    assert any(isinstance(e, AnswerEvent) and e.text == "thinking..." for e in events)
    # Should have error event from timeout
    assert any(isinstance(e, ErrorEvent) for e in events)

  asyncio.run(_run())


@pytest.mark.slow
def test_query_timeout():
  """Mock client.query() to be slow — anyio.fail_after(15) triggers."""
  events = []

  class SlowQueryClient:
    async def query(self, _prompt):
      await asyncio.sleep(60)  # way longer than 15s limit

    async def receive_messages(self):
      yield FakeResultMessage()  # pragma: no cover

    async def stop_task(self, _task_id):
      pass

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = SlowQueryClient()
      try:
        await run_turn(client, "hello", events.append)
        assert False, "Expected timeout"
      except (TimeoutError, Exception) as exc:
        # anyio.fail_after raises TimeoutError (via anyio.get_cancelled_exc_class
        # or builtins.TimeoutError depending on backend)
        assert "timed out" in str(type(exc).__name__).lower() or isinstance(exc, TimeoutError) or "cancel" in str(type(exc).__name__).lower(), f"Unexpected: {exc!r}"

  asyncio.run(_run())


# ---------------------------------------------------------------------------
# Steer double-Result desync ("答上一个问题" +1 turn lag) regression
#
# A steer injected mid-turn can be run by the CLI as its OWN follow-on turn
# with its OWN ResultMessage. If run_turn stops at the FIRST Result, that
# second Result + its answer strand in the receive buffer and the NEXT turn
# drains them — every question then gets the previous turn's answer. Fix B:
# once steered, keep consuming until the CLI is quiescent so the continuation
# folds into THIS run_turn and nothing is stranded.
# ---------------------------------------------------------------------------

def _answers(events):
  return [e.text for e in events if isinstance(e, AnswerEvent)]


def test_late_steer_continuation_drained_into_same_turn():
  """Late steer → second Result. Both answers belong to THIS run_turn and the
  shared stream is left EMPTY (nothing stranded for the next turn)."""
  steered = [True]  # a steer was injected into this turn (SDKThread.steer)
  client = QueueClient(steered)
  # The CLI emitted the prompt's answer + Result, then processed the steer as
  # its own turn: continuation answer + a second Result.
  client.feed(
    FakeAssistantMessage(content=[FakeTextBlock(text="first")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 100}),
    FakeAssistantMessage(content=[FakeTextBlock(text="second")]),
    FakeResultMessage(total_cost_usd=0.03, usage={"input_tokens": 100}),
  )
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      return await run_turn(client, "prompt", events.append, steered=steered)

  cost, _usage = asyncio.run(_run())
  # BOTH answers surfaced in this single turn (the steer's answer was NOT
  # stranded for the next turn).
  assert _answers(events) == ["first", "second"]
  # Cost accumulated across both Results.
  assert abs(cost - 0.05) < 1e-9
  # The shared stream is drained: no leftover Result to lag the next turn.
  assert client.q.empty(), "steer continuation left messages stranded in buffer"


def test_folded_steer_single_result_completes_without_hang():
  """Folded steer → only ONE Result ever arrives. The turn must still complete
  (via quiescence) rather than hang waiting for a second Result — the failure
  mode of the reverted fixed-counter approach."""
  steered = [True]
  client = QueueClient(steered)
  client.feed(
    FakeAssistantMessage(content=[FakeTextBlock(text="only")]),
    FakeResultMessage(total_cost_usd=0.01),
  )
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      return await run_turn(client, "prompt", events.append, steered=steered)

  cost, _usage = asyncio.run(asyncio.wait_for(_run(), timeout=5))
  assert _answers(events) == ["only"]
  assert abs(cost - 0.01) < 1e-9
  assert client.q.empty()


def test_no_lag_next_turn_gets_its_own_answer():
  """End-to-end: over ONE shared stream, a steered turn that drains its
  continuation leaves the NEXT (unsteered) turn to answer its OWN prompt — no
  +1 turn lag."""
  steered = [False]
  client = QueueClient(steered)

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      # Turn 1: a steer fired mid-turn (sets the flag) → continuation drained.
      steered[0] = True
      client.feed(
        FakeAssistantMessage(content=[FakeTextBlock(text="t1-answer")]),
        FakeResultMessage(total_cost_usd=0.01),
        FakeAssistantMessage(content=[FakeTextBlock(text="t1-steer-answer")]),
        FakeResultMessage(total_cost_usd=0.01),
      )
      ev1: list = []
      await run_turn(client, "q1", ev1.append, steered=steered)
      assert client.q.empty(), "turn 1 stranded a Result into turn 2's stream"

      # Turn 2: no steer (SDKThread resets the flag per turn).
      steered[0] = False
      client.feed(
        FakeAssistantMessage(content=[FakeTextBlock(text="t2-answer")]),
        FakeResultMessage(total_cost_usd=0.01),
      )
      ev2: list = []
      await run_turn(client, "q2", ev2.append, steered=steered)
      return _answers(ev1), _answers(ev2)

  a1, a2 = asyncio.run(_run())
  assert a1 == ["t1-answer", "t1-steer-answer"]
  # The crux: turn 2 answers q2, NOT the stranded t1-steer-answer.
  assert a2 == ["t2-answer"]


# ---------------------------------------------------------------------------
# Resumed-turn drain regression (empty Done card incident 2026-08-20)
#
# A turn resumed after a reconnect-with-resume (run_turn_with_reconnect
# attempt > 0) replays session history; the model can first finalize a
# SPURIOUS EMPTY ResultMessage (the CLI's "No response requested."
# placeholder) before processing the re-sent prompt for real. If run_turn
# stops at that first Result the real answer strands in the receive buffer →
# the NEXT turn drains it (+1 turn lag), AND the current turn finalizes a
# misleading empty "Done ✓". Fix 2: resumed turns don't break at the first
# Result; they drain past an empty first Result (bounded by
# RESUME_DRAIN_TIMEOUT) so the real continuation folds into THIS run_turn.
# ---------------------------------------------------------------------------

def test_resumed_turn_drains_past_empty_first_result():
  """Resumed turn: empty first Result, then the real continuation. The real
  answer is surfaced and the shared stream is left EMPTY (nothing stranded
  for the next turn)."""
  client = QueueClient(steered=[False])
  client.feed(
    # Spurious empty Result from the replayed session state.
    FakeResultMessage(total_cost_usd=0.0, usage={},
                      result="No response requested."),
    # The real answer to the re-sent prompt.
    FakeAssistantMessage(content=[FakeTextBlock(text="real answer")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 100}),
  )
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      return await run_turn(client, "prompt", events.append, resumed=True)

  cost, _usage = asyncio.run(asyncio.wait_for(_run(), timeout=5))
  # The real answer surfaced — NOT an empty Done card, and the empty first
  # Result's placeholder was NOT emitted as the answer.
  assert _answers(events) == ["real answer"]
  assert any(isinstance(e, DoneEvent) for e in events)
  assert client.q.empty(), "resumed turn stranded its real continuation"
  assert abs(cost - 0.02) < 1e-9


def test_resumed_turn_with_real_first_result_completes_normally():
  """A resumed turn whose first Result already carries the real answer ends
  immediately — the drain only engages when the first Result is empty (no
  hang, no spurious EmptyResponseError)."""
  client = QueueClient(steered=[False])
  client.feed(
    FakeAssistantMessage(content=[FakeTextBlock(text="normal answer")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 100}),
  )
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      return await run_turn(client, "prompt", events.append, resumed=True)

  cost, _usage = asyncio.run(asyncio.wait_for(_run(), timeout=5))
  assert _answers(events) == ["normal answer"]
  assert any(isinstance(e, DoneEvent) for e in events)
  assert client.q.empty()


def test_resumed_turn_empty_first_result_no_continuation_raises():
  """Resumed turn whose empty first Result has NO continuation within
  RESUME_DRAIN_TIMEOUT is genuinely empty → EmptyResponseError (bounded
  same-client retry) instead of spinning the progress budget."""
  from nemo.claude_turn import EmptyResponseError

  client = QueueClient(steered=[False])
  client.feed(
    FakeResultMessage(total_cost_usd=0.0, usage={},
                      result="No response requested."),
  )
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05), \
         mock.patch("nemo.claude_turn.RESUME_DRAIN_TIMEOUT", 0.2):
      await run_turn(client, "prompt", events.append, resumed=True)

  with pytest.raises(EmptyResponseError):
    asyncio.run(asyncio.wait_for(_run(), timeout=5))
  assert not any(isinstance(e, DoneEvent) for e in events), \
    "an empty resumed turn must not finalize a Done card"


def test_no_lag_after_resumed_turn_next_turn_gets_its_own_answer():
  """End-to-end over ONE shared stream: a resumed turn that drains its real
  continuation leaves the NEXT turn to answer its OWN prompt — no +1 turn
  lag (the observed failure before Fix 2)."""
  client = QueueClient(steered=[False])

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      # Turn 1: resumed (attempt > 0 after a reconnect).
      client.feed(
        FakeResultMessage(total_cost_usd=0.0, usage={},
                          result="No response requested."),
        FakeAssistantMessage(content=[FakeTextBlock(text="t1-real-answer")]),
        FakeResultMessage(total_cost_usd=0.01),
      )
      ev1: list = []
      await run_turn(client, "q1", ev1.append, resumed=True)
      assert client.q.empty(), "resumed turn 1 stranded a Result into turn 2's stream"

      # Turn 2: fresh (attempt 0, NOT resumed).
      client.feed(
        FakeAssistantMessage(content=[FakeTextBlock(text="t2-answer")]),
        FakeResultMessage(total_cost_usd=0.01),
      )
      ev2: list = []
      await run_turn(client, "q2", ev2.append)
      return _answers(ev1), _answers(ev2)

  a1, a2 = asyncio.run(_run())
  assert a1 == ["t1-real-answer"]
  # The crux: turn 2 answers q2, NOT the stranded t1 continuation.
  assert a2 == ["t2-answer"]


def test_slow_steer_continuation_survives_quiescence_window():
  """Incident 2026-07-16 18:33: the CLI dequeued the steer at the Result
  boundary but its continuation stayed SILENT for longer than
  QUIESCENCE_TIMEOUT (15s observed vs the 8s window). Pure quiescence ended
  the turn and stranded the whole continuation → permanent +1 turn lag.
  With the transcript probe reporting one started continuation batch, the
  turn must wait past the quiescence window and drain the late continuation
  into THIS run_turn."""
  steered = [True]
  client = QueueClient(steered)
  client.feed(
    FakeAssistantMessage(content=[FakeTextBlock(text="first")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 100}),
    # NOTE: continuation NOT fed yet — it arrives after the quiescence window.
  )
  events: list = []
  probe_calls: list[str] = []

  def probe(session_id: str) -> int:
    probe_calls.append(session_id)
    return 1  # transcript shows a user-row fold → one continuation running

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      async def _late_continuation():
        # Continuation first message lags 4x the quiescence window.
        await asyncio.sleep(0.2)
        client.feed(
          FakeAssistantMessage(content=[FakeTextBlock(text="steer-answer")]),
          FakeResultMessage(total_cost_usd=0.03, usage={"input_tokens": 100}),
        )
      feeder = asyncio.ensure_future(_late_continuation())
      try:
        return await run_turn(
          client, "prompt", events.append, steered=steered, steer_probe=probe)
      finally:
        await feeder

  cost, _usage = asyncio.run(asyncio.wait_for(_run(), timeout=5))
  assert probe_calls, "probe was never consulted at quiescence expiry"
  # BOTH answers folded into this turn — the late continuation was NOT
  # stranded for the next turn.
  assert _answers(events) == ["first", "steer-answer"]
  assert abs(cost - 0.05) < 1e-9
  assert client.q.empty(), "late steer continuation left messages stranded"


def test_steer_probe_zero_ends_turn_at_quiescence():
  """Probe says no continuation started (remove-fold or still-pending steer)
  → the turn ends at the quiescence window exactly as before, no hang."""
  steered = [True]
  client = QueueClient(steered)
  client.feed(
    FakeAssistantMessage(content=[FakeTextBlock(text="only")]),
    FakeResultMessage(total_cost_usd=0.01),
  )
  events: list = []

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      return await run_turn(
        client, "prompt", events.append, steered=steered,
        steer_probe=lambda sid: 0)

  asyncio.run(asyncio.wait_for(_run(), timeout=5))
  assert _answers(events) == ["only"]
  assert client.q.empty()


def test_steer_probe_failure_falls_back_to_quiescence():
  """A probe that raises must not wedge the turn — fall back to plain
  quiescence (the pre-probe behavior)."""
  steered = [True]
  client = QueueClient(steered)
  client.feed(
    FakeAssistantMessage(content=[FakeTextBlock(text="only")]),
    FakeResultMessage(total_cost_usd=0.01),
  )
  events: list = []

  def probe(session_id: str) -> int:
    raise OSError("transcript unreadable")

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()), \
         mock.patch("nemo.claude_turn.QUIESCENCE_TIMEOUT", 0.05):
      return await run_turn(
        client, "prompt", events.append, steered=steered, steer_probe=probe)

  asyncio.run(asyncio.wait_for(_run(), timeout=5))
  assert _answers(events) == ["only"]
  assert client.q.empty()
