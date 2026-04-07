# Nemo E2E Test Plan

Automated end-to-end tests against a live nemo instance. Sends real Lark
messages as a user, verifies responses via the Lark IM API.

## Quick Start

```bash
cd ~/code/verneagent/nemo
python3 scripts/e2e_test.py                    # full run (all 6 phases)
python3 scripts/e2e_test.py --skip-sdk         # commands only (fast)
python3 scripts/e2e_test.py --stress           # stale-task stress test only
python3 scripts/e2e_test.py --project          # multi-turn project test only
python3 scripts/e2e_test.py --verbose          # debug logging
python3 scripts/e2e_test.py --chat <CHAT_ID>   # custom group
```

The script handles token refresh, nemo start/stop, eviction of stale
instances, and response verification — no manual setup needed beyond
the prerequisites below.

## Prerequisites

- `~/.nemo/config.json` — app_id, app_secret, relay_url
- `~/.nemo/user_token.json` — user OAuth token (2h TTL, auto-refreshed)
- Relay server running (`relay_url` in config, default: `http://47.95.232.145`)
- Test group: `oc_8183e1682019ddc0857a29074b3e2858` (nemo-test-1)

### Token Setup (first time only)

Run the device flow to obtain `user_token.json` — see CLAUDE.md
"Self-Debugging with User Identity" section. The token refreshes
automatically thereafter (refresh_token lasts 30 days).

## Test Matrix

| # | Test | Phase | Wait | What it checks |
|---|------|-------|------|----------------|
| T01 | ping | 1 Commands | 5s | Basic command response (card) |
| T02 | /help | 1 Commands | 5s | Help card |
| T03 | /model | 1 Commands | 5s | Model info card |
| T04 | /cost | 1 Commands | 5s | Cost tracking card |
| T05 | /diag | 1 Commands | 8s | Diagnostics card |
| T06 | Simple question | 2 SDK | 15s | Full SDK turn lifecycle |
| T07 | Bash tool | 2 SDK | 20s | Tool use + working card |
| T08 | Read tool | 2 SDK | 20s | File read tool |
| T09 | Multi-tool | 2 SDK | 25s | Multiple tool calls in one turn |
| T10 | /esc interrupt | 3 Signals | 15s | Cancel in-progress turn |
| T11 | /clear | 3 Signals | 15s | Session reset + SDK reconnect |
| T12 | Post-clear turn | 3 Signals | 15s | SDK works after clear |
| T13 | /exit | 3 Signals | 25s | Graceful shutdown |
| T14 | Restart | 4 Recovery | 30s | Fresh start after exit |
| T15 | Post-recovery turn | 4 Recovery | 15s | SDK works after restart |
| T16 | Empty message | 3 Signals | 3s | Whitespace ignored silently |
| T17 | /dissolve | — | — | Always skipped (manual only) |
| T20 | autoapprove on | 5 Stress | 5s | Enable auto-approve for agents |
| T21 | Multi-agent turn (x3) | 5 Stress | 180s | Parallel Agent spawns complete |
| T22 | Follow-up after agents (x3) | 5 Stress | 60s | No stale task hang, correct response |
| T23 | No duplicate responses (x3) | 5 Stress | — | Only 1-2 bot messages per turn |
| T24 | Stale handling summary | 5 Stress | — | Log shows proper stale task handling |
| T30 | /cd to temp dir | 6 Project | 15s | Switch working directory |
| T31 | Create calculator | 6 Project | 60s | main.py created with argparse CLI |
| T32 | Add history feature | 6 Project | 60s | history.py created and integrated |
| T33 | Write tests | 6 Project | 60s | test_calc.py with pytest tests |
| T34 | Run tests | 6 Project | 120s | pytest passes (or fixes failures) |
| T35 | Add verbose flag | 6 Project | 60s | Error handling + --verbose flag |
| T36 | Project files check | 6 Project | — | main.py exists in temp dir |
| T37 | Context continuity | 6 Project | 30s | Remembers what files it created |
| T38 | Rapid-fire (3 msgs) | 6 Project | 45s | 3 quick messages → 3 responses |
| T40 | Permission approve | 7 Perms | 60s | "y" approves Write, file created |
| T41 | Permission deny | 7 Perms | 45s | "n" denies Write, file not created |
| T42 | Permission always | 7 Perms | 60s | "always" approves + sets autoapprove |
| T43 | Auto-approve verify | 7 Perms | 45s | Subsequent write needs no prompt |
| T50 | Create group B | 8 Dual | — | Temp Lark group for second instance |
| T51 | Both instances start | 8 Dual | 30s | Two nemo on same dir, different groups |
| T52 | Both respond to ping | 8 Dual | 5s | Independent command handling |
| T53 | Concurrent reads | 8 Dual | 30s | Both read same file simultaneously |
| T54 | Concurrent writes | 8 Dual | 60s | Both write different files in same dir |
| T55 | Log isolation | 8 Dual | — | Each PID log contains only its own chat |
| T56 | No errors | 8 Dual | — | Neither log has ERROR entries |

## Verification Method

Three verification layers, used by different test phases:

1. **Lark API** (primary) — `GET /im/v1/messages` with `sort_type=ByCreateTimeDesc`.
   Checks that a bot response card appears after the test message timestamp.
   Used by all phases.

2. **Log analysis** (`LogAnalyzer` class) — structured search over
   `~/.nemo/logs/nemo-<PID>.log`. Used for permission flow detection
   (`Permission request:`), stale task counting, error checking, and
   startup readiness. The `wait_for()` method polls the log for a pattern.

3. **File system** — direct file existence checks after write operations.
   Used by Phase 6 (project files) and Phase 7 (permission approve/deny).

## Known Pitfalls

Issues discovered during initial E2E development (2026-04-07). These are
all fixed in the current codebase, but documented here so future
developers understand why certain code exists.

### 1. ResultMessage must break the loop (turn.py)

**Symptom:** Main loop hangs for 300s between turns.

The SDK's `receive_response()` iterator doesn't raise `StopAsyncIteration`
after `ResultMessage` — it continues indefinitely (SDK docs confirm this).
Without an explicit `break` after handling `ResultMessage`, the loop waited
for `HEARTBEAT_TIMEOUT` (300s) before giving up.

**Fix:** `break` after `ResultMessage` in `turn.py` line ~214.

### 2. Log handlers don't flush by default (__main__.py)

**Symptom:** Log file appears empty or stale during E2E monitoring.

Python uses block buffering for stderr when redirected to a file (which
happens when nemo runs in background). Both `StreamHandler` and
`RotatingFileHandler` inherit this behavior.

**Fix:** Wrap every handler's `emit()` to call `flush()` afterward via
`_make_flushing()` helper.

### 3. Competing nemo processes (workspace.py)

**Symptom:** Two Pong responses to a single ping; duplicate cards.

When starting nemo with `--chat`, the process skipped idle detection
entirely (that code only ran in `discover_or_create_chat()`). A second
instance would happily connect to the same group.

**Fix:** `evict_existing()` runs before `claim_group()` — checks local
PID first, then relay heartbeat. SIGTERM with SIGKILL fallback.

### 4. User API messages DO trigger webhooks

**Claim that was wrong:** "Messages sent via user token API don't trigger
Lark webhook callbacks to the relay."

**Reality:** They DO trigger webhooks. The initial test failure was caused
by a truncated `sender_id` (`ou_1f03ce275afdf` vs full
`ou_1f03ce275afdf3486d658740a39d0d8a`), which made nemo filter the message
as unauthorized.

**Implication:** The E2E test script works correctly by sending messages
through the Lark user API — they arrive at nemo via the normal
relay → WebSocket path.

### 5. Sender ID must be exact

Nemo's `is_authorized()` compares the sender's `open_id` from the relay
event against the operator's full `open_id`. Any mismatch (truncation,
wrong user) causes silent filtering with no error log at INFO level.
Use `--verbose` to see `Skipping: unauthorized sender` debug messages.

### 6. Shutdown takes ~15s

`/exit` triggers SDK client cleanup (thread stop + WS close + group
release). The 15s delay is normal — the E2E script allows 25s.

### 7. Commands don't log "Processing:"

Only SDK turns log `Processing: <message>`. Built-in commands (ping,
help, model, etc.) respond directly without that log line. Don't use
log-based verification for command tests — use the Lark API.

### 8. Stale path also needs break (turn.py)

**Symptom:** After a stale TaskNotification is detected, the turn hangs
for 300s even though ResultMessage was received.

The stale handling path (`if found_stale: continue`) captured the
ResultMessage cost but didn't `break` — it continued the loop waiting for
StopAsyncIteration, which might never come (same root cause as pitfall #1).

**Fix:** Added `break` after capturing ResultMessage in the stale path.

### 9. Agent tool required for task spawning

Background tasks (and the stale notification leak) only happen when the
Agent tool is available. Without it in `allowed_tools`, Claude Code
won't spawn subagents and TaskStartedMessage/TaskNotificationMessage
won't appear.

**Fix:** Added "Agent" to `allowed_tools` in agent.py.

### 10. Permissions block advanced tests

Write/Edit/Bash-write operations require user approval in `default`
permission mode. The permission handler sends a card to Lark and waits
up to 300s for reply. In automated testing, no one is there to approve.

**Fix:** Send `autoapprove on` before Phase 5, 6, and 8 tests. Phase 7
deliberately leaves autoapprove OFF to test the permission flow itself.

### 11. Permission approval via user API messages

The permission handler reads replies from the same event queue as the
main loop (`events.next_message()`). User API messages arrive via relay
→ WebSocket → queue, so sending "y" or "n" as a Lark message works as
a permission reply.

**Timing requirement:** The "y"/"n" must arrive AFTER the permission card
is sent. The test uses `LogAnalyzer.wait_for("Permission request:")` to
detect when the card was sent before replying. If sent too early, the
message is consumed by `_watch_signals()` and re-queued as a regular
message, not a permission reply.

### 12. Dual-instance same-dir is safe for different files

Two nemo instances sharing the same project directory work independently
as long as they write different files. Each has its own:
- PID log file (`nemo-<PID>.log`)
- Lark group and relay WebSocket
- SDK client session

File conflicts (both writing the same file) are possible but not guarded
against — this mirrors real-world behavior where two developers might
edit the same codebase.

## Maintenance

When adding new commands or features:

1. Add a test case to the appropriate phase in `scripts/e2e_test.py`
2. Update the test matrix above
3. Run the full suite: `python3 scripts/e2e_test.py`

For quick smoke tests after minor changes, `--skip-sdk` runs only
command tests in ~30 seconds.
