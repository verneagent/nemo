"""Tests for nemo.turn — SDK turn execution and event emission."""

import asyncio
from dataclasses import dataclass
from unittest import mock

import pytest

from nemo.turn import (
  run_turn, ToolStartEvent, ToolProgressEvent, TextEvent,
  TaskStartedEvent, TaskDoneEvent, DoneEvent, ErrorEvent,
  TurnEvent,
)
from nemo.cards import ToolRecord


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
  assert any(isinstance(e, TextEvent) and e.text == "Hello world" for e in events)
  assert any(isinstance(e, DoneEvent) for e in events)


def test_tool_use_turn():
  """A turn with tool use should emit ToolStartEvent then ToolProgressEvent."""
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
  tool_events = [e for e in events if isinstance(e, (ToolStartEvent, ToolProgressEvent))]
  assert len(tool_events) == 2
  assert isinstance(tool_events[0], ToolStartEvent)
  assert isinstance(tool_events[1], ToolProgressEvent)


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
  """Stale task notifications should trigger re-query."""
  # First turn: stale notification contaminates
  stale_messages = [
    FakeTaskNotificationMessage(task_id="stale_1", status="completed"),
    FakeResultMessage(total_cost_usd=0.01),
  ]
  # Re-query turn: clean
  clean_messages = [
    FakeAssistantMessage(content=[FakeTextBlock(text="Fresh response")]),
    FakeResultMessage(total_cost_usd=0.02, usage={"input_tokens": 300}),
  ]

  call_count = 0
  events = []

  class ReQueryClient:
    async def query(self, prompt):
      pass

    async def receive_response(self):
      nonlocal call_count
      msgs = stale_messages if call_count == 0 else clean_messages
      call_count += 1
      for msg in msgs:
        yield msg

    async def stop_task(self, task_id):
      pass

  async def _run():
    with mock.patch.dict("sys.modules", _sdk_modules()):
      client = ReQueryClient()
      cost, _usage = await run_turn(
        client, "test", events.append,
        stale_tasks={"stale_1"},
      )
      assert cost == 0.02
  asyncio.run(_run())
  assert call_count == 2  # original + re-query
  text_events = [e for e in events if isinstance(e, TextEvent)]
  assert any(e.text == "Fresh response" for e in text_events)


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
  """TaskProgressMessage should emit ToolProgressEvent when working."""
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
  progress = [e for e in events if isinstance(e, ToolProgressEvent)]
  assert any(e.tool.summary == "Searching files..." for e in progress)


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
  text_events = [e for e in events if isinstance(e, TextEvent)]
  assert len(text_events) == 1
  assert text_events[0].text == "Let me check"


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
    assert any(isinstance(e, TextEvent) and e.text == "thinking..." for e in events)
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
