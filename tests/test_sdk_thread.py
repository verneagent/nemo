"""Tests for nemo.sdk_thread — dedicated SDK thread with its own event loop."""

import asyncio
import sys
from unittest import mock

import pytest

from claude_agent_sdk import CLIConnectionError
from nemo.sdk_thread import SDKThread, MAX_CONNECT_ATTEMPTS


def _ensure_real_sdk():
  """Restore the real claude_agent_sdk if other tests contaminated sys.modules."""
  mod = sys.modules.get("claude_agent_sdk")
  if mod is None:
    # Not yet imported — import it fresh
    import claude_agent_sdk  # noqa: F401
    return
  if hasattr(mod, "ClaudeSDKClient"):
    return  # Already the real module
  # Force re-import from disk
  for key in list(sys.modules):
    if key == "claude_agent_sdk" or key.startswith("claude_agent_sdk."):
      del sys.modules[key]
  import claude_agent_sdk  # noqa: F401


_ensure_real_sdk()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
  """Run an async function in a fresh event loop (test helper)."""
  loop = asyncio.new_event_loop()
  try:
    return loop.run_until_complete(coro)
  finally:
    loop.close()


@pytest.fixture()
def sdk_thread():
  """Provide a started SDKThread that is stopped after the test."""
  _ensure_real_sdk()
  t = SDKThread()
  t.start()
  yield t
  t.stop()


# ---------------------------------------------------------------------------
# 1. Thread starts and event loop is ready
# ---------------------------------------------------------------------------

class TestStart:
  def test_start_sets_loop_and_thread(self, sdk_thread: SDKThread):
    assert sdk_thread._loop is not None
    assert sdk_thread._loop.is_running()
    assert sdk_thread._thread is not None
    assert sdk_thread._thread.is_alive()

  def test_start_fails_if_loop_not_set(self):
    """If _run never sets _loop (simulated by patching), start raises."""
    t = SDKThread()

    # Replace _run so the thread sets _ready but never creates a loop
    def bad_run():
      t._ready.set()

    with mock.patch.object(t, "_run", bad_run):
      with pytest.raises(RuntimeError, match="SDK thread failed to start"):
        t.start()


# ---------------------------------------------------------------------------
# 2. create_client retries on failure
# ---------------------------------------------------------------------------

class TestCreateClient:
  def test_retries_then_succeeds(self, sdk_thread: SDKThread):
    mock_client = mock.AsyncMock()
    # First call: __aenter__ raises, second call: succeeds
    fail_client = mock.AsyncMock()
    fail_client.__aenter__ = mock.AsyncMock(side_effect=RuntimeError("boom"))
    fail_client.__aexit__ = mock.AsyncMock()

    ok_client = mock.AsyncMock()
    ok_client.__aenter__ = mock.AsyncMock()
    ok_client._transport = None  # no process check

    call_count = 0

    def make_client(options=None):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return fail_client
      return ok_client

    with mock.patch("nemo.sdk_thread.asyncio.sleep", new_callable=mock.AsyncMock):
      with mock.patch("claude_agent_sdk.ClaudeSDKClient", side_effect=make_client):
        _run(sdk_thread.create_client(options=mock.MagicMock()))

    assert call_count == 2
    assert sdk_thread._client is ok_client

  # ---------------------------------------------------------------------------
  # 3. create_client raises after MAX_CONNECT_ATTEMPTS
  # ---------------------------------------------------------------------------

  def test_raises_after_max_attempts(self, sdk_thread: SDKThread):
    fail_client = mock.AsyncMock()
    fail_client.__aenter__ = mock.AsyncMock(side_effect=RuntimeError("fail"))
    fail_client.__aexit__ = mock.AsyncMock()

    with mock.patch("nemo.sdk_thread.asyncio.sleep", new_callable=mock.AsyncMock):
      with mock.patch("claude_agent_sdk.ClaudeSDKClient", return_value=fail_client):
        with pytest.raises(RuntimeError, match=f"after {MAX_CONNECT_ATTEMPTS} attempts"):
          _run(sdk_thread.create_client(options=mock.MagicMock()))

    # Verify it was called MAX_CONNECT_ATTEMPTS times
    assert fail_client.__aenter__.await_count == MAX_CONNECT_ATTEMPTS

  def test_raises_on_cli_exited(self, sdk_thread: SDKThread):
    """If the CLI process exits during connect, it should raise and retry."""
    mock_proc = mock.MagicMock()
    mock_proc.returncode = 1
    mock_transport = mock.MagicMock()
    mock_transport._process = mock_proc

    client = mock.AsyncMock()
    client.__aenter__ = mock.AsyncMock()
    client._transport = mock_transport
    client.__aexit__ = mock.AsyncMock()

    with mock.patch("nemo.sdk_thread.asyncio.sleep", new_callable=mock.AsyncMock):
      with mock.patch("claude_agent_sdk.ClaudeSDKClient", return_value=client):
        with pytest.raises(RuntimeError, match="after .* attempts"):
          _run(sdk_thread.create_client(options=mock.MagicMock()))


# ---------------------------------------------------------------------------
# 4. run_turn dispatches to SDK thread and returns result
# ---------------------------------------------------------------------------

class TestRunTurn:
  def test_dispatches_and_returns(self, sdk_thread: SDKThread):
    sdk_thread._client = mock.MagicMock()
    expected = (0.5, {"input_tokens": 100, "output_tokens": 50})

    with mock.patch("nemo.sdk_thread.run_turn", new_callable=mock.AsyncMock) as mock_rt:
      mock_rt.return_value = expected
      result = _run(sdk_thread.run_turn("hello", on_event=lambda e: None))

    assert result == expected
    mock_rt.assert_awaited_once()

  def test_raises_if_no_client(self, sdk_thread: SDKThread):
    sdk_thread._client = None
    with pytest.raises(RuntimeError, match="not connected"):
      _run(sdk_thread.run_turn("hello", on_event=lambda e: None))

  def test_passes_stale_tasks(self, sdk_thread: SDKThread):
    sdk_thread._client = mock.MagicMock()
    stale = {"task_1", "task_2"}

    with mock.patch("nemo.sdk_thread.run_turn", new_callable=mock.AsyncMock) as mock_rt:
      mock_rt.return_value = (0.0, {})
      _run(sdk_thread.run_turn("hi", on_event=lambda e: None, stale_tasks=stale))

    _, kwargs = mock_rt.call_args
    assert kwargs.get("stale_tasks") == stale


# ---------------------------------------------------------------------------
# 5. run_turn_with_reconnect retries on TimeoutError
# ---------------------------------------------------------------------------

class TestRunTurnWithReconnect:
  def test_retries_on_timeout(self, sdk_thread: SDKThread):
    expected = (0.5, {})
    call_count = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      if call_count < 2:
        raise TimeoutError("hung")
      return expected

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock):
        result = _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    assert result == expected
    assert call_count == 2

  def test_retries_on_stale_leak_with_resume(self, sdk_thread: SDKThread):
    """SDK #788: StaleLeakError must route through the same reconnect path
    and reconnect with the options_factory result (resume=<session_id>),
    then retry the SAME real prompt — no interrupt anywhere.
    """
    from nemo.claude_turn import StaleLeakError, TransientAPIError

    assert issubclass(StaleLeakError, TransientAPIError)

    expected = (0.7, {"input_tokens": 9})
    call_count = 0
    seen_prompts: list[str] = []

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      seen_prompts.append(prompt)
      if call_count < 2:
        raise StaleLeakError("stale task X leaked into turn stream")
      return expected

    resume_options = mock.MagicMock(name="resumed_options")
    factory_calls = 0

    def options_factory():
      nonlocal factory_calls
      factory_calls += 1
      return resume_options

    reconnect_args: list = []

    async def fake_reconnect(options):
      reconnect_args.append(options)

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "interrupt") as interrupt_mock:
        with mock.patch.object(sdk_thread, "reconnect",
                               side_effect=fake_reconnect):
          result = _run(sdk_thread.run_turn_with_reconnect(
            "real prompt", on_event=lambda e: None,
            options=mock.MagicMock(),
            options_factory=options_factory,
            max_attempts=3,
          ))

    assert result == expected
    assert call_count == 2
    # Same real prompt retried (never a drain prompt).
    assert seen_prompts == ["real prompt", "real prompt"]
    # Reconnected exactly once, with the resume-bearing options.
    assert reconnect_args == [resume_options]
    assert factory_calls == 1
    # Recovery must NOT interrupt — the wedged control channel is dead.
    interrupt_mock.assert_not_called()

  def test_raises_without_options(self, sdk_thread: SDKThread):
    """If options is None, TimeoutError propagates immediately."""
    sdk_thread._client = mock.MagicMock()

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      raise TimeoutError("hung")

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with pytest.raises(TimeoutError):
        _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None, options=None,
        ))

  # ---------------------------------------------------------------------------
  # 6. Off-by-one: last reconnect actually runs a turn
  # ---------------------------------------------------------------------------

  def test_no_reconnect_on_last_attempt(self, sdk_thread: SDKThread):
    """With max_attempts=3, the last attempt raises without reconnecting.

    Attempts 0 and 1 timeout → reconnect. Attempt 2 (last) timeout →
    raises immediately, no wasted reconnect.
    """
    turn_calls = 0
    reconnect_calls = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal turn_calls
      turn_calls += 1
      raise TimeoutError("hung")

    async def fake_reconnect(options):
      nonlocal reconnect_calls
      reconnect_calls += 1

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", side_effect=fake_reconnect):
        with pytest.raises(TimeoutError):
          _run(sdk_thread.run_turn_with_reconnect(
            "hello", on_event=lambda e: None,
            options=mock.MagicMock(), max_attempts=3,
          ))

    # 3 turn attempts total
    assert turn_calls == 3
    # Only 2 reconnects (after attempts 0 and 1, NOT after the last)
    assert reconnect_calls == 2

  def test_succeeds_on_last_attempt(self, sdk_thread: SDKThread):
    """If the turn succeeds on the last attempt, result is returned."""
    turn_calls = 0
    expected = (1.0, {"ok": True})

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal turn_calls
      turn_calls += 1
      if turn_calls < 3:
        raise TimeoutError("hung")
      return expected

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock):
        result = _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    assert result == expected
    assert turn_calls == 3

  def test_retries_on_not_connected(self, sdk_thread: SDKThread):
    """Should reconnect when client is not connected (RuntimeError)."""
    expected = (0.5, {})
    call_count = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      if call_count < 2:
        raise RuntimeError("SDK client not connected")
      return expected

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock):
        result = _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    assert result == expected
    assert call_count == 2

  def test_non_retryable_api_error_closes_without_retry(self, sdk_thread: SDKThread):
    """402/billing failures should close the wedged CLI and not reconnect."""
    from nemo.claude_turn import NonRetryableAPIError

    call_count = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      raise NonRetryableAPIError("API Error: 402 Insufficient Balance")

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn), \
         mock.patch.object(sdk_thread, "close_client", new_callable=mock.AsyncMock) as close_client, \
         mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock) as reconnect:
      with pytest.raises(NonRetryableAPIError):
        _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    assert call_count == 1
    close_client.assert_awaited_once()
    reconnect.assert_not_awaited()

  def test_not_connected_raises_after_max_attempts(self, sdk_thread: SDKThread):
    """Should raise after exhausting all reconnect attempts for not-connected."""
    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      raise RuntimeError("SDK client not connected")

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock):
        with pytest.raises(RuntimeError, match="not connected"):
          _run(sdk_thread.run_turn_with_reconnect(
            "hello", on_event=lambda e: None,
            options=mock.MagicMock(), max_attempts=3,
          ))

  def test_cancel_aborts_reconnect_loop(self, sdk_thread: SDKThread):
    """cancel() should abort the reconnect loop between attempts."""
    call_count = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      raise TimeoutError("hung")

    async def fake_reconnect(options):
      # Simulate cancel being called during reconnect
      sdk_thread.cancel()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", side_effect=fake_reconnect):
        with pytest.raises(asyncio.CancelledError):
          _run(sdk_thread.run_turn_with_reconnect(
            "hello", on_event=lambda e: None,
            options=mock.MagicMock(), max_attempts=3,
          ))

    # Should have only attempted once before cancel took effect
    assert call_count == 1

  def test_retries_on_terminated_process(self, sdk_thread: SDKThread):
    """Should reconnect when subprocess terminated (CLIConnectionError)."""
    expected = (0.5, {})
    call_count = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      if call_count < 2:
        raise CLIConnectionError("Cannot write to terminated process (exit code: 1)")
      return expected

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock):
        result = _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    assert result == expected
    assert call_count == 2

  def test_reconnect_uses_options_factory(self, sdk_thread: SDKThread):
    """When options_factory is provided, reconnect uses its return value
    instead of the static `options` snapshot. This is what lets the host
    inject `resume=<latest_session_id>` so a watchdog-forced reconnect
    preserves conversation context (chat-amnesia regression).
    """
    expected = (0.5, {})
    turn_calls = 0
    factory_calls = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal turn_calls
      turn_calls += 1
      if turn_calls < 2:
        raise TimeoutError("hung")
      return expected

    def factory():
      nonlocal factory_calls
      factory_calls += 1
      return "FRESH_OPTIONS"

    reconnect_options: list[object] = []

    async def fake_reconnect(opts):
      reconnect_options.append(opts)

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", side_effect=fake_reconnect):
        result = _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options="STATIC", options_factory=factory, max_attempts=3,
        ))

    assert result == expected
    assert factory_calls == 1
    assert reconnect_options == ["FRESH_OPTIONS"]

  def test_factory_returning_none_raises_immediately(self, sdk_thread: SDKThread):
    """If options_factory returns None, treat it like no-options and raise."""
    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      raise TimeoutError("hung")

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock) as mock_reconn:
        with pytest.raises(TimeoutError):
          _run(sdk_thread.run_turn_with_reconnect(
            "hello", on_event=lambda e: None,
            options=None, options_factory=lambda: None,
          ))

    mock_reconn.assert_not_awaited()

  def test_other_runtime_error_not_retried(self, sdk_thread: SDKThread):
    """RuntimeError without 'not connected' should not be retried."""
    call_count = 0

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                      steer_probe=None, resumed=False):
      nonlocal call_count
      call_count += 1
      raise RuntimeError("some other error")

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn):
      with mock.patch.object(sdk_thread, "reconnect", new_callable=mock.AsyncMock) as mock_reconn:
        with pytest.raises(RuntimeError, match="some other error"):
          _run(sdk_thread.run_turn_with_reconnect(
            "hello", on_event=lambda e: None,
            options=mock.MagicMock(), max_attempts=3,
          ))

    assert call_count == 1
    mock_reconn.assert_not_awaited()

  def test_empty_response_retried_on_same_client_without_reconnect(
      self, sdk_thread: SDKThread):
    """An EmptyResponseError (the model returned nothing — the CLI's "No
    response requested." placeholder) is retried on the SAME client WITHOUT
    a reconnect: the subprocess is healthy, and reconnecting would replay the
    session and re-introduce the same empty completion."""
    from nemo.claude_turn import EmptyResponseError

    calls: list[tuple[str, bool]] = []

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                        steer_probe=None, resumed=False):
      calls.append((prompt, resumed))
      if len(calls) == 1:
        raise EmptyResponseError("Model returned an empty response")
      return (0.02, {})

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn), \
         mock.patch.object(sdk_thread, "reconnect",
                           new_callable=mock.AsyncMock) as mock_reconn:
      result = _run(sdk_thread.run_turn_with_reconnect(
        "hello", on_event=lambda e: None,
        options=mock.MagicMock(), max_attempts=3,
      ))

    # First attempt empty → retried once (attempt 1, now "resumed" since a
    # retry pass must drain past an empty first Result), then succeeded.
    assert [c[0] for c in calls] == ["hello", "hello"]
    assert [c[1] for c in calls] == [False, True]
    assert result == (0.02, {})
    mock_reconn.assert_not_awaited()

  def test_empty_response_twice_gives_up_without_reconnect(
      self, sdk_thread: SDKThread):
    """Two consecutive empty completions give up (bounded) so the host can
    surface an explicit empty-response error card — never a reconnect."""
    from nemo.claude_turn import EmptyResponseError

    calls: list[tuple[str, bool]] = []

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                        steer_probe=None, resumed=False):
      calls.append((prompt, resumed))
      raise EmptyResponseError("Model returned an empty response")

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn), \
         mock.patch.object(sdk_thread, "reconnect",
                           new_callable=mock.AsyncMock) as mock_reconn:
      with pytest.raises(EmptyResponseError):
        _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    # Bounded: initial + one retry, then give up. No reconnect.
    assert [c[0] for c in calls] == ["hello", "hello"]
    mock_reconn.assert_not_awaited()

  def test_incomplete_turn_retried_on_same_client_without_reconnect(
      self, sdk_thread: SDKThread):
    """An IncompleteTurnError (the model's response ended on a thinking/tool-
    only tail — final text cut off mid-generation) is retried on the SAME
    client WITHOUT a reconnect: the subprocess is healthy, the model's partial
    work is already in session history, and reconnecting would replay the
    session and re-introduce the same broken completion."""
    from nemo.claude_turn import IncompleteTurnError

    calls: list[tuple[str, bool]] = []

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                        steer_probe=None, resumed=False):
      calls.append((prompt, resumed))
      if len(calls) == 1:
        raise IncompleteTurnError("Incomplete turn: only thinking/tool output")
      return (0.02, {})

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn), \
         mock.patch.object(sdk_thread, "reconnect",
                           new_callable=mock.AsyncMock) as mock_reconn:
      result = _run(sdk_thread.run_turn_with_reconnect(
        "hello", on_event=lambda e: None,
        options=mock.MagicMock(), max_attempts=3,
      ))

    # First attempt incomplete → retried once (attempt 1, now "resumed" since
    # a retry pass must drain past a broken first Result), then succeeded.
    assert [c[0] for c in calls] == ["hello", "hello"]
    assert [c[1] for c in calls] == [False, True]
    assert result == (0.02, {})
    mock_reconn.assert_not_awaited()

  def test_incomplete_turn_twice_gives_up_without_reconnect(
      self, sdk_thread: SDKThread):
    """Two consecutive incomplete turns give up (bounded) so the host can
    surface an explicit incomplete-turn error card with recovery guidance —
    never a reconnect."""
    from nemo.claude_turn import IncompleteTurnError

    calls: list[tuple[str, bool]] = []

    async def fake_turn(prompt, on_event, stale_tasks=None, is_paused=None,
                        steer_probe=None, resumed=False):
      calls.append((prompt, resumed))
      raise IncompleteTurnError("Incomplete turn: only thinking/tool output")

    sdk_thread._client = mock.MagicMock()

    with mock.patch.object(sdk_thread, "run_turn", side_effect=fake_turn), \
         mock.patch.object(sdk_thread, "reconnect",
                           new_callable=mock.AsyncMock) as mock_reconn:
      with pytest.raises(IncompleteTurnError):
        _run(sdk_thread.run_turn_with_reconnect(
          "hello", on_event=lambda e: None,
          options=mock.MagicMock(), max_attempts=3,
        ))

    # Bounded: initial + one retry, then give up. No reconnect.
    assert [c[0] for c in calls] == ["hello", "hello"]
    mock_reconn.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. interrupt dispatches to SDK thread
# ---------------------------------------------------------------------------

class TestInterrupt:
  def test_interrupt_calls_client(self, sdk_thread: SDKThread):
    mock_client = mock.AsyncMock()
    sdk_thread._client = mock_client

    _run(sdk_thread.interrupt())

    mock_client.interrupt.assert_awaited_once()

  def test_interrupt_noop_without_client(self, sdk_thread: SDKThread):
    sdk_thread._client = None
    # Should not raise
    _run(sdk_thread.interrupt())

  def test_interrupt_suppresses_error(self, sdk_thread: SDKThread):
    mock_client = mock.AsyncMock()
    mock_client.interrupt = mock.AsyncMock(side_effect=RuntimeError("fail"))
    sdk_thread._client = mock_client

    # Should not raise — errors are logged but suppressed
    _run(sdk_thread.interrupt())


class TestSteer:
  def test_steer_calls_query_and_returns_true(self, sdk_thread: SDKThread):
    # Mid-turn steer writes a user message into the live stream via query()
    # — it is folded into the running turn, no extra Result to absorb.
    mock_client = mock.AsyncMock()
    sdk_thread._client = mock_client

    assert _run(sdk_thread.steer("also add tests")) is True
    mock_client.query.assert_awaited_once_with("also add tests")

  def test_steer_noop_without_client(self, sdk_thread: SDKThread):
    sdk_thread._client = None
    assert _run(sdk_thread.steer("hello")) is False

  def test_steer_returns_false_on_error(self, sdk_thread: SDKThread):
    mock_client = mock.AsyncMock()
    mock_client.query = mock.AsyncMock(side_effect=RuntimeError("boom"))
    sdk_thread._client = mock_client

    # Error is suppressed; steer reports failure so the host queues instead.
    assert _run(sdk_thread.steer("hello")) is False


# ---------------------------------------------------------------------------
# 8. close_client cleans up
# ---------------------------------------------------------------------------

class TestCloseClient:
  def test_close_calls_aexit(self, sdk_thread: SDKThread):
    mock_client = mock.AsyncMock()
    sdk_thread._client = mock_client

    _run(sdk_thread.close_client())

    mock_client.__aexit__.assert_awaited_once_with(None, None, None)
    assert sdk_thread._client is None

  def test_close_noop_without_client(self, sdk_thread: SDKThread):
    sdk_thread._client = None
    _run(sdk_thread.close_client())
    assert sdk_thread._client is None

  def test_close_suppresses_error(self, sdk_thread: SDKThread):
    mock_client = mock.AsyncMock()
    mock_client.__aexit__ = mock.AsyncMock(side_effect=RuntimeError("fail"))
    sdk_thread._client = mock_client

    _run(sdk_thread.close_client())
    assert sdk_thread._client is None


# ---------------------------------------------------------------------------
# 9. stop stops the event loop and joins thread
# ---------------------------------------------------------------------------

class TestStop:
  def test_stop_joins_thread(self):
    t = SDKThread()
    t.start()
    assert t._thread.is_alive()
    assert t._loop.is_running()

    t.stop()

    assert not t._thread.is_alive()
    assert not t._loop.is_running()

  def test_stop_idempotent(self):
    """Calling stop when not started should not raise."""
    t = SDKThread()
    t.stop()  # no-op

  def test_loop_stopped_after_stop(self):
    t = SDKThread()
    t.start()
    loop = t._loop
    t.stop()

    # Loop is no longer running after stop
    assert not loop.is_running()


# ---------------------------------------------------------------------------
# reconnect
# ---------------------------------------------------------------------------

class TestReconnect:
  def test_reconnect_closes_then_creates(self, sdk_thread: SDKThread):
    calls = []

    async def fake_close():
      calls.append("close")

    async def fake_create(options):
      calls.append("create")

    with mock.patch.object(sdk_thread, "close_client", side_effect=fake_close):
      with mock.patch.object(sdk_thread, "create_client", side_effect=fake_create):
        _run(sdk_thread.reconnect(options=mock.MagicMock()))

    assert calls == ["close", "create"]


# ---------------------------------------------------------------------------
# Regression: __aenter__ and __aexit__ MUST run on the same asyncio.Task.
#
# anyio cancel scopes are task-bound; mixing them across tasks raised
# "Attempted to exit cancel scope in a different task than it was entered in"
# and left orphan tasks spinning on sdk-loop forever (30–40% idle CPU).
# ---------------------------------------------------------------------------

class TestLifecycleSameTask:
  def test_aenter_and_aexit_run_on_same_task(self, sdk_thread: SDKThread):
    """Open then close — both calls must observe the same current Task."""
    captured: dict[str, object] = {}

    class TaskCapturingClient:
      def __init__(self, *args, **kwargs):
        self._transport = None

      async def __aenter__(self):
        captured["aenter"] = asyncio.current_task()
        return self

      async def __aexit__(self, exc_type, exc, tb):
        captured["aexit"] = asyncio.current_task()
        return False

    with mock.patch("claude_agent_sdk.ClaudeSDKClient", TaskCapturingClient):
      _run(sdk_thread.create_client(options=mock.MagicMock()))
      _run(sdk_thread.close_client())

    assert captured["aenter"] is not None
    assert captured["aenter"] is captured["aexit"], (
      "aenter and aexit ran on different tasks — anyio cancel scope will "
      "raise and leak orphan tasks"
    )

  def test_aexit_on_owner_task_after_failed_close(self, sdk_thread: SDKThread):
    """Even if a previous __aexit__ raised, the next open/close pair still
    binds aenter+aexit to the (same) owner task. Verifies the owner task
    survives exceptions and keeps processing.
    """
    seq: list[tuple[str, object]] = []

    class SometimesFailingClient:
      _instances = 0

      def __init__(self, *args, **kwargs):
        self._transport = None
        SometimesFailingClient._instances += 1
        self._idx = SometimesFailingClient._instances

      async def __aenter__(self):
        seq.append(("aenter", asyncio.current_task()))
        return self

      async def __aexit__(self, exc_type, exc, tb):
        seq.append(("aexit", asyncio.current_task()))
        if self._idx == 1:
          raise RuntimeError("Attempted to exit cancel scope in a different task")
        return False

    with mock.patch("claude_agent_sdk.ClaudeSDKClient", SometimesFailingClient):
      _run(sdk_thread.create_client(options=mock.MagicMock()))
      _run(sdk_thread.close_client())  # first __aexit__ raises (logged)
      _run(sdk_thread.create_client(options=mock.MagicMock()))
      _run(sdk_thread.close_client())

    aenter_tasks = [t for k, t in seq if k == "aenter"]
    aexit_tasks = [t for k, t in seq if k == "aexit"]
    # All aenters and aexits run on the same single owner task.
    all_tasks = set(aenter_tasks + aexit_tasks)
    assert len(all_tasks) == 1, f"expected one owner task, got {all_tasks}"
