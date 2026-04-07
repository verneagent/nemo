# Nemo E2E Test Plan

Automated end-to-end tests against a live nemo instance. Sends real Lark
messages as a user, verifies responses via the Lark IM API.

## Quick Start

```bash
cd ~/code/verneagent/nemo
python3 scripts/e2e_test.py                    # full run
python3 scripts/e2e_test.py --skip-sdk         # commands only (fast)
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
| T01 | ping | Commands | 5s | Basic command response (card) |
| T02 | /help | Commands | 5s | Help card |
| T03 | /model | Commands | 5s | Model info card |
| T04 | /cost | Commands | 5s | Cost tracking card |
| T05 | /diag | Commands | 8s | Diagnostics card |
| T06 | Simple question | SDK | 15s | Full SDK turn lifecycle |
| T07 | Bash tool | SDK | 20s | Tool use + working card |
| T08 | Read tool | SDK | 20s | File read tool |
| T09 | Multi-tool | SDK | 25s | Multiple tool calls in one turn |
| T10 | /esc interrupt | Signals | 15s | Cancel in-progress turn |
| T11 | /clear | Signals | 15s | Session reset + SDK reconnect |
| T12 | Post-clear turn | SDK | 15s | SDK works after clear |
| T13 | /exit | Signals | 25s | Graceful shutdown |
| T14 | Restart | Recovery | 30s | Fresh start after exit |
| T15 | Post-recovery turn | SDK | 15s | SDK works after restart |
| T16 | Empty message | Edge | 3s | Whitespace ignored silently |
| T17 | /dissolve | Destructive | — | Always skipped (manual only) |

## Verification Method

All tests verify via **Lark bot API** (`GET /im/v1/messages` with
`sort_type=ByCreateTimeDesc`), not log files. The script checks that a
bot response card (`msg_type=interactive`) appears after the test message
timestamp.

Log files (`~/.nemo/logs/nemo-<PID>.log`) are used only for startup
readiness detection (`SDK client connected`).

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

## Maintenance

When adding new commands or features:

1. Add a test case to the appropriate phase in `scripts/e2e_test.py`
2. Update the test matrix above
3. Run the full suite: `python3 scripts/e2e_test.py`

For quick smoke tests after minor changes, `--skip-sdk` runs only
command tests in ~30 seconds.
