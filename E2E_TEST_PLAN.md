# Nemo End-to-End Test Plan

Real-world integration tests against a live nemo instance. Tests are executed
by sending messages as a real Lark user and verifying responses via logs and
Lark API.

## Prerequisites

- Nemo config at `~/.nemo/config.json` (app_id, app_secret, relay_url)
- User token at `~/.nemo/user_token.json` (2h TTL, refresh before testing)
- Test group: `oc_8183e1682019ddc0857a29074b3e2858` (nemo-test-1)
- Relay server running at `http://47.95.232.145`

## Phase 0: Setup

### 0.1 Refresh user token

```bash
python3 -c "
import json, requests
cfg = json.load(open('$HOME/.nemo/config.json'))
tok = json.load(open('$HOME/.nemo/user_token.json'))
r = requests.post('https://open.larksuite.com/open-apis/authen/v2/oauth/token',
  json={'grant_type': 'refresh_token', 'refresh_token': tok['refresh_token'],
        'client_id': cfg['app_id'], 'client_secret': cfg['app_secret']})
import time; d = r.json(); d['saved_at'] = time.time()
json.dump(d, open('$HOME/.nemo/user_token.json', 'w'), indent=2)
print('Refreshed, expires_in:', d.get('expires_in'))
"
```

If refresh_token is also expired, use device flow (see CLAUDE.md).

### 0.2 Start nemo

```bash
cd ~/code/verneagent/nemo
python3 -m nemo --chat oc_8183e1682019ddc0857a29074b3e2858 2>&1 &
NEMO_PID=$!
sleep 5
tail -5 ~/.nemo/logs/nemo-$NEMO_PID.log
# Verify: "Start card sent: om_xxx"
```

### 0.3 Helper: send message as user

```bash
send_msg() {
  python3 -c "
import json, requests
t = json.load(open('$HOME/.nemo/user_token.json'))['access_token']
r = requests.post('https://open.larksuite.com/open-apis/im/v1/messages',
  params={'receive_id_type': 'chat_id'},
  headers={'Authorization': f'Bearer {t}', 'Content-Type': 'application/json'},
  json={'receive_id': 'oc_8183e1682019ddc0857a29074b3e2858', 'msg_type': 'text',
        'content': json.dumps({'text': '$1'})})
print(r.json().get('data',{}).get('message_id','?'))
"
}
```

### 0.4 Helper: check logs

```bash
check_log() {
  tail -${2:-30} ~/.nemo/logs/nemo-$NEMO_PID.log | grep -i "${1:-.}"
}
```

---

## Phase 1: Commands (no SDK)

Built-in commands that don't trigger an SDK turn. Fast, deterministic.

### T01: /ping

```bash
send_msg "ping"
sleep 3
check_log "Pong"
```

**Expect:**
- Log: `Processing: ping`
- Response card with Pong, model name, uptime, message count

### T02: /help

```bash
send_msg "/help"
sleep 3
check_log "Commands"
```

**Expect:**
- Log: `Processing: /help`
- Response with command table

### T03: /model (show)

```bash
send_msg "/model"
sleep 3
check_log "model"
```

**Expect:**
- Response showing current model name (e.g. `claude-sonnet-4-6`)

### T04: /cost

```bash
send_msg "/cost"
sleep 3
check_log "Cost"
```

**Expect:**
- Response with `$0.0000` (no turns yet)

### T05: /diag

```bash
send_msg "/diag"
sleep 5
check_log "Diagnostics\|diag"
```

**Expect:**
- Token refresh: OK
- Send card: OK
- Workspace tag: OK or MISSING

---

## Phase 2: SDK Turns

Messages that trigger Claude Agent SDK execution. Verify the full
Working card → tool calls → Done card lifecycle.

### T06: Simple question

```bash
send_msg "What is 2+2?"
sleep 15
check_log "Processing\|turn msg\|DoneEvent"
```

**Expect:**
- Log: `Processing: What is 2+2?`
- Log: `turn msg: AssistantMessage` (SDK response)
- Log: no errors
- Group: Working card → Done card with answer

### T07: Bash tool call

```bash
send_msg "List files in the current directory using ls"
sleep 15
check_log "ToolStart\|Bash"
```

**Expect:**
- Log: `ToolStartEvent` with Bash tool
- Working card shows `$ ls` or similar
- Done card with file listing

### T08: Read tool call

```bash
send_msg "Read the first 3 lines of pyproject.toml and tell me the project name"
sleep 15
check_log "ToolStart\|Read"
```

**Expect:**
- Log: `ToolStartEvent` with Read tool
- Done card with project name from pyproject.toml

### T09: Multi-tool turn

```bash
send_msg "How many Python files are in the nemo/ directory? Use a command to count them."
sleep 20
check_log "ToolStart\|DoneEvent"
```

**Expect:**
- Multiple tool calls in log
- Collapsible panel in done card showing tool history
- Correct file count in response

---

## Phase 3: Signals & Control

Test interruption, session reset, and graceful exit.

### T10: /esc during turn

```bash
send_msg "Write a detailed 500-word essay about the history of Python programming"
sleep 3
send_msg "/esc"
sleep 10
check_log "interrupt\|cancel"
```

**Expect:**
- Log: signal detected = esc
- Log: `interrupt()` called
- Response: "Operation cancelled."
- Agent continues running (not exited)

### T11: /clear (session reset)

```bash
send_msg "/clear"
sleep 10
check_log "Session Cleared\|create_client\|reconnect"
```

**Expect:**
- SDK client recreated
- Card: "Session Cleared"
- Subsequent messages work normally

### T12: Verify post-clear

```bash
send_msg "Say hello"
sleep 15
check_log "Processing: Say hello"
```

**Expect:**
- Normal SDK turn after clear
- Proves session reset worked

### T13: /exit (graceful shutdown)

```bash
send_msg "/exit"
sleep 5
check_log "Agent stopped"
# Verify process exited
ps -p $NEMO_PID > /dev/null 2>&1 && echo "FAIL: still running" || echo "OK: exited"
```

**Expect:**
- Card: "Nemo — Stopped"
- Log: `Agent stopped.`
- Process exits cleanly
- Group remains (not dissolved)

---

## Phase 4: Recovery & Edge Cases

### T14: Stale session cleanup

```bash
# Start nemo again on the same group
python3 -m nemo --chat oc_8183e1682019ddc0857a29074b3e2858 2>&1 &
NEMO_PID=$!
sleep 5
check_log "Cleaning stale session\|Start card sent"
```

**Expect:**
- Log: `Cleaning stale session <old_session_id>`
- Start card sent successfully
- Agent fully functional

### T15: Post-recovery turn

```bash
send_msg "What is 3+3?"
sleep 15
check_log "Processing\|DoneEvent"
```

**Expect:**
- Normal SDK turn works after stale session cleanup

### T16: Empty/whitespace message

```bash
send_msg "   "
sleep 3
check_log "Skipping: empty"
```

**Expect:**
- Log: `Skipping: empty text`
- No card sent, no error

### T17: /dissolve (shutdown + dissolve group)

> **WARNING:** This dissolves the test group. Only run if you're prepared to
> recreate it. Skip in routine testing.

```bash
send_msg "/dissolve"
sleep 5
check_log "Dissolved\|Agent stopped"
```

**Expect:**
- Card: "Nemo — Dissolved"
- Group dissolved via API
- Process exits

---

## Verification Checklist

| # | Test | Status |
|---|------|--------|
| T01 | /ping | ☐ |
| T02 | /help | ☐ |
| T03 | /model | ☐ |
| T04 | /cost | ☐ |
| T05 | /diag | ☐ |
| T06 | Simple question | ☐ |
| T07 | Bash tool call | ☐ |
| T08 | Read tool call | ☐ |
| T09 | Multi-tool turn | ☐ |
| T10 | /esc interrupt | ☐ |
| T11 | /clear reset | ☐ |
| T12 | Post-clear turn | ☐ |
| T13 | /exit shutdown | ☐ |
| T14 | Stale session cleanup | ☐ |
| T15 | Post-recovery turn | ☐ |
| T16 | Empty message | ☐ |
| T17 | /dissolve (optional) | ☐ |

## Notes

- **Token TTL:** User token expires in 2 hours. Refresh before a full run.
- **SDK turn timeout:** Allow 15-20s for SDK turns. Simple questions are faster.
- **Log location:** `~/.nemo/logs/nemo-<PID>.log` (per-process).
- **Reading group messages:** Use bot token to `GET /im/v1/messages` if
  log verification is insufficient.
- **T17 is destructive** — only run when specifically testing dissolve flow.
  The test group must be recreated manually afterward.
