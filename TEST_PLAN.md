# Nemo Test Plan

Generated from code review on 2026-04-07.

## Review Summary

| Module | Lines | Test File | Coverage | Priority Issues |
|--------|-------|-----------|----------|-----------------|
| `nemo/sdk_thread.py` | 165 | **NONE** | 0% | Thread lifecycle, reconnect off-by-one |
| `nemo/agent.py` | 706 | **NONE** (test_main.py covers CLI only) | ~5% | Thread safety on `_on_event`, permission queue cross-loop |
| `nemo/turn.py` | 229 | test_turn.py (309L) | ~70% | Timeout path untested, anyio.fail_after |
| `nemo/permissions.py` | 130 | test_permissions.py (64L) | ~30% | `can_use_tool` flow untested, cross-loop await |
| `nemo/db.py` | 150 | test_db.py (120L) | ~80% | Cross-thread access untested |
| `nemo/relay_events.py` | 277 | test_relay_events.py (507L) | ~85% | push_back ordering |
| `relay/relay.py` | 1053 | relay/test_relay.py (1051L) | ~75% | WS fan-out, malformed request, int() ValueError |
| `nemo/lark/events.py` | 260 | test_lark_events.py (280L) | ~80% | stop() doesn't stop WS thread |
| `nemo/lark/api.py` | 210 | test_lark_api.py (130L) | ~60% | Retry, pagination, error handling |
| `nemo/commands.py` | 165 | test_commands.py (189L) | ~85% | autoapprove "on" substring match |
| `nemo/cards.py` | 354 | test_cards.py | ~90% | — |
| `nemo/config.py` | 60 | test_config.py (86L) | ~90% | — |
| `nemo/workspace.py` | 306 | test_workspace.py (306L) | ~85% | — |

## Critical Issues Found

### C1: Permission handler uses wrong event loop
**File:** `nemo/permissions.py:97-122`
**Problem:** `can_use_tool` is an async function called from the SDK thread. It
awaits `events_source.next_message()` which calls `asyncio.Queue.get()`. That
queue is bound to the main event loop, not the SDK thread's loop. Will hang or
raise.
**Fix:** Bridge the call back to the main loop via `asyncio.run_coroutine_threadsafe`.
**Test:** Verify permission prompt/response works across threads.

### C2: `_on_event` thread safety
**File:** `nemo/agent.py:415-479`
**Problem:** `_on_event` runs on SDK thread, mutates `_turn_card_id` (line 427).
Main loop reads `_turn_card_id` (line 582). No lock.
**Fix:** Use `threading.Lock` or make `_on_event` post to main loop via `call_soon_threadsafe`.
**Test:** Concurrent event emission + signal watching.

### C3: `run_turn_with_reconnect` off-by-one
**File:** `nemo/sdk_thread.py:135-143`
**Problem:** After the last reconnect, the loop exits without retrying `run_turn`.
The final reconnect is wasted.
**Fix:** `range(max_attempts)` → `range(max_attempts)` with retry after reconnect, or restructure loop.
**Test:** Verify all reconnect attempts actually run a turn.

## Test Plan: What to Add

### Phase 1: New test files (zero coverage)

#### T1: `tests/test_sdk_thread.py`
- [ ] Thread starts and event loop is ready
- [ ] `create_client` retries on failure (mock ClaudeSDKClient)
- [ ] `create_client` raises after MAX_CONNECT_ATTEMPTS
- [ ] `run_turn` dispatches to SDK thread and returns result
- [ ] `run_turn_with_reconnect` retries on TimeoutError
- [ ] `run_turn_with_reconnect` off-by-one — all attempts run a turn (C3)
- [ ] `interrupt` dispatches to SDK thread
- [ ] `close_client` cleans up
- [ ] `stop` stops the event loop and joins thread

#### T2: `tests/test_agent_loop.py`
- [ ] Basic message → SDK turn → card lifecycle (mock SDK + Lark API)
- [ ] Command dispatch (/clear, /model, /exit)
- [ ] Signal detection (esc, exit, dissolve) during turn
- [ ] Error handling — exception in turn doesn't crash loop
- [ ] Cleanup on exit (SDK closed, events closed, db deactivated)
- [ ] Relay heartbeat runs periodically (mock relay)
- [ ] `_on_event` thread safety (C2) — concurrent card updates

### Phase 2: Fill gaps in existing tests

#### T3: `tests/test_turn.py` — add timeout tests
- [ ] First message timeout (30s) triggers TimeoutError
- [ ] Heartbeat timeout (300s) triggers TimeoutError
- [ ] ErrorEvent emitted on timeout
- [ ] `anyio.fail_after(15)` on `query()` — mock slow query

#### T4: `tests/test_permissions.py` — add can_use_tool flow
- [ ] `build_permission_handler` returns callable
- [ ] Approve flow: card sent → user replies "y" → returns True
- [ ] Deny flow: card sent → user replies "n" → returns False
- [ ] Timeout flow: no reply → returns False
- [ ] "always" response sets autoapprove in db
- [ ] Cross-thread safety (C1) — verify correct event loop usage

#### T5: `tests/test_db.py` — add thread safety
- [ ] Concurrent reads from multiple threads
- [ ] Concurrent write + read from different threads
- [ ] `check_same_thread=False` actually works

#### T6: `relay/test_relay.py` — add missing coverage
- [ ] WebSocket fan-out: push → connected WS clients receive
- [ ] WebSocket ack via WS message
- [ ] Malformed request body (non-JSON) → 400
- [ ] `/poll` with non-numeric timeout → graceful handling
- [ ] Cleanup loop removes expired messages
- [ ] Concurrent push + poll thread safety

### Phase 3: Bug fixes to verify

#### T7: Minor bugs
- [ ] `commands.py:131` — autoapprove "on" substring match (e.g., "/autoapprove tone" should NOT enable)
- [ ] `lark/api.py:195` — `get_chat_members` pagination has no limit
- [ ] `lark/events.py:229` — `stop()` should actually stop WS thread
- [ ] `relay_events.py:267` — `push_back` ordering (events go to back, not front)

## Execution Order

1. **Phase 1** — Write T1, T2 (highest impact, zero coverage)
2. **Phase 2** — Fill T3-T6 (increase coverage on critical paths)
3. **Phase 3** — Fix bugs, add regression tests T7
4. **Integration** — End-to-end test with real relay server

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_sdk_thread.py -v

# With coverage
python -m pytest tests/ --cov=nemo --cov-report=term-missing

# Relay server tests
python -m pytest relay/test_relay.py -v
```
