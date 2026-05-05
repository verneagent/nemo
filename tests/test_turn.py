"""Tests for nemo.turn — SDK turn execution and event emission."""

import asyncio
from dataclasses import dataclass
from unittest import mock

import pytest

from nemo.turn import (
  run_turn, ProgressEvent, AnswerEvent,
  TaskStartedEvent, TaskDoneEvent, DoneEvent, ErrorEvent,
  RateLimitNoticeEvent, TurnEvent,
)


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

  async def receive_response(self):
    for msg in self._messages:
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


def test_stale_task_detection():
  """Stale task notifications should trigger a drain turn then a real retry.

  With the drain-prompt workaround for SDK #788, we always start in real
  mode (the prior turn's stop_task may have succeeded, in which case no
  stale ever surfaces). When a stale *does* leak at the front of the real
  turn we expect:
    attempt 0 (real, contaminated): stale discarded, rest suppressed;
    attempt 1 (drain):              clean response, mode → real;
    attempt 2 (real, clean):        real answer delivered.
  """
  from nemo.turn import DRAIN_PROMPT_MARKER
  stale_messages = [
    FakeTaskNotificationMessage(task_id="stale_1", status="completed"),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  drain_clean_messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="NEMO_DRAIN_OK xxxx")]),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  real_messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="Fresh response")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 300}),
  ]

  sent_prompts: list[str] = []
  call_count = 0
  events = []

  class DrainRetryClient:
    async def query(self, prompt):
      sent_prompts.append(prompt)

    async def receive_response(self):
      nonlocal call_count
      idx = call_count
      call_count += 1
      msgs = [stale_messages, drain_clean_messages, real_messages][idx]
      for msg in msgs:
        yield msg

    async def stop_task(self, task_id):
      pass

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = DrainRetryClient()
      cost, _usage = await run_turn(
        client, "test", events.append,
        stale_tasks={"stale_1"},
      )
      # Cost is now cumulative across contaminated+drain+real turns.
      assert cost == pytest.approx(0.01 + 0.0 + 0.02)

  asyncio.run(_run())
  assert call_count == 3, f"expected 3 turns (real+drain+real), got {call_count}"
  # Prompt ordering: real (contaminated), drain, real (clean).
  assert sent_prompts[0] == "test"
  assert sent_prompts[1].startswith(DRAIN_PROMPT_MARKER), \
    f"drain prompt must carry the drain marker, got: {sent_prompts[1][:80]!r}"
  assert sent_prompts[2] == "test"
  # Drain turn's AssistantMessage must not surface as AnswerEvent.
  answer_events = [e for e in events if isinstance(e, AnswerEvent)]
  assert any(e.text == "Fresh response" for e in answer_events)
  assert not any(e.text.startswith("NEMO_DRAIN_OK") for e in answer_events), \
    "drain turn's reply must not leak to on_event"


def test_drain_does_not_amplify_stale_tasks():
  """Drain turns use a drain prompt that must not emit TaskStartedEvents.

  Even if the SDK delivers a TaskStartedMessage during a drain turn (which
  shouldn't happen because the drain prompt forbids tools, but we verify
  defensively), the event must be suppressed so the consumer doesn't think
  the user turn is spawning tasks.
  """
  from nemo.turn import DRAIN_PROMPT_MARKER
  stale_messages = [
    FakeTaskNotificationMessage(task_id="stale_1", status="completed"),
    FakeTaskStartedMessage(task_id="rogue_in_drain"),
    FakeResultMessage(total_cost_usd=0.0),
  ]
  clean_messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="answer")]),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  sent_prompts: list[str] = []
  events: list = []
  call_count = 0

  class Client:
    async def query(self, prompt):
      sent_prompts.append(prompt)

    async def receive_response(self):
      nonlocal call_count
      idx = call_count
      call_count += 1
      msgs = stale_messages if idx == 0 else clean_messages
      for msg in msgs:
        yield msg

    async def stop_task(self, task_id):
      pass

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(Client(), "test", events.append,
                     stale_tasks={"stale_1"})
  asyncio.run(_run())

  # Drain prompt must have been used (identified by its marker).
  assert any(p.startswith(DRAIN_PROMPT_MARKER) for p in sent_prompts), \
    "no drain-marked prompt was sent"
  # No TaskStartedEvent must reach the consumer
  assert not any(isinstance(e, TaskStartedEvent) for e in events), \
    "drain-turn TaskStartedMessage leaked as TaskStartedEvent"


def test_multiple_stales_drained_without_amplification():
  """N accumulated stales should converge in N+1 drain turns + 1 real turn.

  Previous naive re-query workaround amplified when the user prompt itself
  spawned subagents. Drain prompt cannot spawn tasks, so each drain turn
  clears exactly one stale notification, and after all are drained the
  real prompt runs cleanly. Cost is cumulative across all turns.
  """
  from nemo.turn import DRAIN_PROMPT_MARKER
  ids = ["stale_a", "stale_b", "stale_c"]
  # Turn 0 (real, contaminated): sees first stale, suppresses rest.
  # Turns 1-2 (drain): each sees one more stale.
  # Turn 3 (drain): clean (no stale) → switch back to real.
  # Turn 4 (real): clean final answer.
  per_turn_messages: list[list] = [
    [FakeTaskNotificationMessage(task_id=ids[0], status="completed"),
     FakeResultMessage(total_cost_usd=0.01)],
    [FakeTaskNotificationMessage(task_id=ids[1], status="completed"),
     FakeResultMessage(total_cost_usd=0.0)],
    [FakeTaskNotificationMessage(task_id=ids[2], status="completed"),
     FakeResultMessage(total_cost_usd=0.0)],
    [FakeAssistantMessage(content=[FakeTextBlock(text="NEMO_DRAIN_OK xxxx")]),
     FakeResultMessage(total_cost_usd=0.0)],
    [FakeAssistantMessage(content=[FakeTextBlock(text="final answer")]),
     FakeResultMessage(total_cost_usd=0.05, usage={"input_tokens": 50})],
  ]
  sent_prompts: list[str] = []
  call_count = 0
  events: list = []

  class MultiStaleClient:
    async def query(self, prompt):
      sent_prompts.append(prompt)

    async def receive_response(self):
      nonlocal call_count
      idx = call_count
      call_count += 1
      for msg in per_turn_messages[idx]:
        yield msg

    async def stop_task(self, task_id):
      pass

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      cost, _ = await run_turn(MultiStaleClient(), "real", events.append,
                               stale_tasks=set(ids))
      assert cost == pytest.approx(0.01 + 0.05)

  asyncio.run(_run())
  assert call_count == 5, f"expected 5 turns (real+3 drain+real), got {call_count}"
  # First prompt is real (contaminated), next three are drain, last is real.
  assert sent_prompts[0] == "real"
  drain_prompts = sent_prompts[1:4]
  assert all(p.startswith(DRAIN_PROMPT_MARKER) for p in drain_prompts), \
    f"drain prompts must carry the drain marker, got: {drain_prompts!r}"
  # Each drain must be UNIQUE — that's the whole point of fix 1: no fixed
  # answer template can poison the model into replying the same way to the
  # next real user message.
  assert len(set(drain_prompts)) == len(drain_prompts), \
    "drain prompts must be unique per call (nonce-bearing)"
  assert sent_prompts[4] == "real"
  # Only the final real answer must surface.
  answer_texts = [e.text for e in events if isinstance(e, AnswerEvent)]
  assert answer_texts == ["final answer"]
  # Exactly one DoneEvent.
  done = [e for e in events if isinstance(e, DoneEvent)]
  assert len(done) == 1
  assert done[0].cost == pytest.approx(0.01 + 0.05)


def test_stop_task_circuit_breaker_on_control_timeout():
  """First 'Control request timeout' from stop_task disables subsequent calls.

  When the SDK's control channel wedges, stop_task returns this error on
  every pending id. After the first occurrence we should short-circuit all
  further stop_task invocations within the same run_turn orchestration.
  """
  # Turn yields two pending tasks then ResultMessage. No stale is marked,
  # so the turn is "clean" — the pending tasks are both promoted to stale.
  messages = [
    FakeTaskStartedMessage(task_id="t1"),
    FakeTaskStartedMessage(task_id="t2"),
    FakeTaskStartedMessage(task_id="t3"),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  stop_calls: list[str] = []

  class TimingOutClient:
    async def query(self, prompt):
      pass

    async def receive_response(self):
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
  from nemo.turn import TransientAPIError
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


def test_transient_api_error_via_result_is_error_flag():
  """ResultMessage.is_error + transient-signal body → TransientAPIError."""
  from nemo.turn import TransientAPIError
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


def test_clean_real_turn_drops_inherited_never_surfaced_stales():
  """Inherited stale ids that never surface ARE dropped on clean real turn.

  This is the anti-amplification guard: if the prior turn left phantom
  ids (e.g. dead-session ids) and a clean real turn goes through without
  any stale notification appearing, we should not carry them forever.
  """
  # Inherited {ghost}; current turn sees neither a stale notification
  # nor any new TaskStarted. Clean turn → inherited 'ghost' must be dropped.
  messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="hi")]),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  stale = {"ghost"}

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      await run_turn(FakeClient(messages), "hello", lambda _e: None,
                     stale_tasks=stale)

  asyncio.run(_run())
  assert stale == set(), "inherited 'ghost' must be dropped after clean turn"


def test_progress_timeout_fires_on_systemmessage_storm(monkeypatch):
  """SDK emitting SystemMessages indefinitely must not keep a dead turn alive.

  Observed in production: claude CLI enters a rate-limit retry loop that
  emits a SystemMessage every ~1 min. Each SystemMessage used to reset
  HEARTBEAT_TIMEOUT, so the turn was never declared stuck. Progress-only
  timeout should trigger after PROGRESS_TIMEOUT seconds regardless.
  """
  from nemo import turn as turn_module

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

    def receive_response(self):
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


def test_reset_clears_stale_tasks_on_claude_adapter():
  """ClaudeCodingAgent.reset() must clear self._stale_tasks.

  Task ids from a dead SDK session will never produce notifications on a
  fresh session, so keeping them would force a pointless drain turn after
  every reconnect.
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


def test_empty_turn():
  """A turn with only ResultMessage."""
  messages = [FakeResultMessage(total_cost_usd=0.0, usage={})]
  events = []
  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = FakeClient(messages)
      cost, _usage = await run_turn(client, "", events.append)
      assert cost == 0.0
  asyncio.run(_run())
  assert len(events) == 1
  assert isinstance(events[0], DoneEvent)


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


def test_rate_limit_event_with_missing_info_is_skipped():
  """If rate_limit_info is absent, no notice should be emitted (defensive)."""
  @dataclass
  class RateLimitEvent:
    rate_limit_info: object = None

  messages = [
    RateLimitEvent(),
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


def test_stale_task_retry_exhaustion():
  """When MAX_RETRIES is exhausted, stale detection should give up and emit DoneEvent."""
  call_count = 0
  events = []

  class AlwaysStaleClient:
    """Client where every response is contaminated by a stale notification.

    On each call, we inject a fresh stale task ID into the stale_tasks set
    (via a TaskNotification whose ID we pre-add to stale_tasks before
    each receive_response iteration).
    """
    async def query(self, _prompt: str) -> None:
      pass

    async def receive_response(self):
      nonlocal call_count
      call_count += 1
      # Each retry yields a stale notification with a unique task_id
      tid = f"stale_{call_count}"
      yield FakeTaskNotificationMessage(task_id=tid, status="completed")
      yield FakeResultMessage(total_cost_usd=0.01)

    async def stop_task(self, _task_id: str) -> None:
      pass

  # Use a custom set subclass that always re-populates with the next stale id
  class PerpetualStaleSet(set):  # type: ignore[type-arg]
    """A set that, after being discarded/checked, injects the next stale id."""
    def __init__(self, *args: object, **kwargs: object) -> None:
      super().__init__(*args, **kwargs)  # type: ignore[arg-type]
      self._counter = 1

    def discard(self, elem: object) -> None:
      super().discard(elem)
      # Pre-add the next stale task so the next retry also sees stale
      self._counter += 1
      self.add(f"stale_{self._counter}")

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      from nemo.turn import MAX_RETRIES  # noqa: F811
      stale = PerpetualStaleSet({"stale_1"})
      client = AlwaysStaleClient()
      await run_turn(
        client, "test", events.append,
        stale_tasks=stale,
      )
      # Should have retried MAX_RETRIES times then given up
      assert call_count == MAX_RETRIES + 1
      # Should still emit a DoneEvent at the end
      assert any(isinstance(e, DoneEvent) for e in events)

  asyncio.run(_run())


def test_first_message_timeout():
  """First message doesn't arrive within 30s — TimeoutError + ErrorEvent."""
  events = []

  class HangingClient:
    async def query(self, _prompt):
      pass

    async def receive_response(self):
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

    async def receive_response(self):
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

    async def receive_response(self):
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
