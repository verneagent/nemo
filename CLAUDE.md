# Nemo

Lark-connected coding agent daemon powered by Claude Agent SDK.

Named after Captain Nemo — autonomous, persistent, remotely commanded.

## What It Is

A standalone executable that:
1. Receives Lark events via a Cloudflare Worker relay (or direct Lark 长连接 fallback)
2. Runs Claude Agent SDK to process coding tasks
3. Sends responses back via Lark IM API

```
python -m nemo --chat-id <ID> --project-dir <DIR> [--model claude-opus-4-6]
```

Dev install (editable, code changes take effect immediately):
```
pip install -e .
nemo --chat-id <ID> --project-dir <DIR>
```

Do NOT use `pipx install captain-nemo` on the dev machine — pipx freezes
the published version and uses a separate venv (different SDK version).
`pipx` is for end-user installs only.

## Publishing to PyPI

```bash
# 1. Bump version in pyproject.toml
# 2. Build and upload
python3 -m build && twine upload dist/captain_nemo-X.Y.Z*
```

PyPI credentials are in `~/.pypirc`. Package name: `captain-nemo`.
Users install via `pipx install captain-nemo`.

## What It Is NOT

- Not a Claude Code skill (no SKILL.md, no hooks)
- Not a library — it's a daemon process

## Architecture

```
Lark Group
  ↕ Lark Webhook (HTTP callback)
Relay Server (Aliyun SWAS 47.95.232.145, /opt/nemo-relay/relay.py)
  ↕ aiohttp + SQLite — stores messages, supports WS + long-poll
Nemo daemon
  ↕ RelayEventStream (WS /ws/chat:{chatId}) — receive events
  ↕ Lark IM API — send/update messages
  ↕ Claude Agent SDK — coding execution
```

Lark's persistent connection (长连接) only allows one connection per app.
The relay server (`/opt/nemo-relay/relay.py` on Aliyun) receives Lark
webhooks, stores messages in SQLite, and fans out to nemo agents via
WebSocket + long-poll. Single Python process, drop-in replacement for
the CF Worker + Durable Objects stack used by handoff.

If no relay is configured (`relay_url` absent from config), Nemo falls back
to direct Lark 长连接 via `lark-oapi` SDK — but this blocks all other
consumers of the same app's events.

## Key Design Decisions

1. **Aliyun relay server** — Lark 长连接 is single-consumer per app.
   The relay receives Lark webhooks, stores in SQLite, fans out to nemo
   agents via WebSocket + long-poll. Code: `/opt/nemo-relay/relay.py`
   on Aliyun SWAS (47.95.232.145). Access via `aliyun swas-open run-command`.

2. **One card per turn** — A single Card V2 evolves through Working →
   Done via PATCH. Tool history lives in a `collapsible_panel`, always
   available. No message filter levels needed.

3. **Single event callback** — `run_turn()` emits typed events (ToolStart,
   Text, Done) through one callback, replacing the old dual send_fn/working_fn.

4. **Event-driven loop** — Messages arrive via WebSocket push, not polling.
   The main loop waits on `events.next_message()` rather than
   `while True: poll()`.

## Config

Uses `~/.nemo/config.json` for Lark credentials and relay config:
- `app_id`, `app_secret`, `email` — Lark app credentials
- `relay_url`, `relay_api_key` — Aliyun relay server (or env: `NEMO_RELAY_URL`, `NEMO_RELAY_API_KEY`)

## Module Layout

```
nemo/
├── __main__.py      # CLI entry point
├── agent.py         # Main loop & orchestration
├── turn.py          # SDK turn execution & event streaming
├── sdk_thread.py    # Dedicated thread for SDK (isolates anyio)
├── cards.py         # Card V2 builder (Working/Done)
├── commands.py      # Built-in commands (/clear, /model, /cd, etc.)
├── config.py        # Credentials & configuration
├── db.py            # SQLite session & message storage
├── messages.py      # Message filtering & prompt building
├── permissions.py   # Text-based permission bridge
├── relay.py         # Relay client — heartbeat & message registration
├── relay_events.py  # RelayEventStream — WS/poll from Cloudflare Worker
├── workspace.py     # Workspace tag, group discovery, claim/release
├── status_tab.py    # Lark group status tab management
├── monitor.py       # Signal detection (/esc, /exit, /dissolve)
├── group_config.py  # Group-level config (pinned messages)
├── guests.py        # Guest user handling
├── norms.py         # Group norms
├── preflight.py     # Startup checks
├── channel.py       # Channel interface (abstract)
├── coding_agent.py  # Coding agent interface (abstract)
└── lark/            # Lark API layer
    ├── api.py       # Lark IM API client (send, update, download)
    ├── auth.py      # Tenant token management
    └── events.py    # LarkEventStream (direct 长连接 fallback)
```

## Lark WebSocket (Verified)

### Connection
- Use `lark-oapi` Python SDK: `lark.ws.Client(app_id, app_secret, event_handler=handler, domain=lark.LARK_DOMAIN)`
- Must set `domain=lark.LARK_DOMAIN` (international version), otherwise defaults to feishu.cn and fails with "Incorrect domain name"
- Connects to `wss://msg-frontier-sg.larksuite.com/ws/v2`

### Event & Callback Registration
- Events (messages, reactions): `register_p2_im_message_receive_v1(handler)`
- Card actions (button clicks): `register_p2_card_action_trigger(handler)` — note: p2 not p1
- Both arrive through the same WebSocket connection

### Console Configuration (via lark-console)
- Event subscription mode: `POST /developers/v1/event/switch/{appId}` with `{"eventMode": 4}`
- Card callback mode: `POST /developers/v1/callback/switch/{appId}` with `{"callbackMode": 4}`
- Mode 4 = persistent connection (WebSocket). Mode 1 = HTTP. (Not mode 2 as documented elsewhere)
- These are two separate endpoints — both must be set independently

### Card V2 Notes
- V2 does NOT support `{"tag": "action"}` wrapper — put buttons directly in elements or inside `column_set`
- Buttons go inside `{"tag": "column_set"}` → `{"tag": "column"}` → `{"tag": "button"}`
- V2 does NOT support `{"tag": "note"}` — use `{"tag": "markdown", "text_size": "notation"}` with `<font color='grey'>` instead
- `collapsible_panel` header must use `{"title": {"tag": "plain_text", ...}}`, NOT a bare `{"tag": "markdown", ...}`
- `get_message` API returns degraded content for cards — the original body is lost. Do NOT store data in card body if you need to read it back. Use text messages for persistent data.

## Self-Debugging with User Identity

When debugging nemo, you need to send messages **as the user** (not the bot) to trigger `im.message.receive_v1` events. Bot-sent messages do NOT trigger WS events for the same bot.

### Setup: lark-cli user OAuth

```bash
# 1. Ensure app has im:message.send_as_user scope (via lark-console)
node ~/.claude/skills/lark-console/scripts/console_api.mjs scopes find <appId> send_as_user
node ~/.claude/skills/lark-console/scripts/console_api.mjs scopes add <appId> <scopeId>
node ~/.claude/skills/lark-console/scripts/console_api.mjs version publish <appId> --version X.Y.Z --notes "Add send_as_user"

# 2. Login as user with the scope
lark-cli auth login --scope "im:message.send_as_user"
# This opens a browser link — user must authorize

# 3. Verify scope
lark-cli auth check --scope "im:message.send_as_user"
```

### Sending messages as user

**Known bug**: `lark-cli api --as user POST /im/v1/messages` silently crashes (exit 1, no output). Use the Python device flow workaround instead:

```python
# Device flow to get user access token (bypasses lark-cli encryption)
import json, urllib.request, urllib.parse, time

with open("~/.nemo/config.json") as f:
    nemo = json.load(f)
app_id, app_secret = nemo["app_id"], nemo["app_secret"]

# Step 1: Device authorization
body = f"client_id={app_id}&client_secret={app_secret}&scope=im:message.send_as_user".encode()
req = urllib.request.Request(
    "https://accounts.larksuite.com/oauth/v1/device_authorization",
    data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
)
resp = json.loads(urllib.request.urlopen(req).read())
device_code = resp["device_code"]
# Show verification_uri_complete to user — they must open it and authorize

# Step 2: Poll for token (after user authorizes)
token_url = "https://open.larksuite.com/open-apis/authen/v2/oauth/token"
payload = json.dumps({
    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    "device_code": device_code,
    "client_id": app_id, "client_secret": app_secret,
}).encode()
# Poll every 5s until access_token appears in response

# Step 3: Send message as user
req = urllib.request.Request(
    "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
    data=json.dumps({
        "receive_id": "<chat_id>",
        "msg_type": "text",
        "content": json.dumps({"text": "hello nemo"})
    }).encode(),
    headers={
        "Authorization": f"Bearer {user_access_token}",
        "Content-Type": "application/json"
    }
)
```

### What to verify during self-debug

1. **WS connection**: nemo log shows `= connection is OPEN`
2. **Event receipt**: Send user message, log shows `WS parsed: chat=... sender=... text=...`
3. **Start card**: Log shows `Start card sent: om_xxx`
4. **SDK turn**: Log shows Claude response within ~5s
5. **Done card update**: No `230099` error (Card V2 compat)
6. **Config persistence**: Only 1 pinned config message (text type, not card)

### Nemo App
- App ID: `cli_a9583021bef89ed4`
- P2P chat_id (with owner): `oc_6731728a1d02fcb97c67a16806d5c6b0`
- Test group chat_id: `oc_8183e1682019ddc0857a29074b3e2858` (nemo-test-1)

## Quick Test: Simulate User Messages

User token at `~/.nemo/user_token.json` (2h TTL, refresh with device flow above).

### 1. Start nemo

```bash
cd ~/code/verneagent/nemo
python3 -m nemo --chat oc_8183e1682019ddc0857a29074b3e2858 2>&1 &
NEMO_PID=$!
# Wait for "Start card sent" in log
tail -f ~/.nemo/logs/nemo-$NEMO_PID.log
```

### 2. Send message as user

```python
import json, requests
token = json.load(open("~/.nemo/user_token.json"))["access_token"]
chat_id = "oc_8183e1682019ddc0857a29074b3e2858"
requests.post(
    "https://open.larksuite.com/open-apis/im/v1/messages",
    params={"receive_id_type": "chat_id"},
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"receive_id": chat_id, "msg_type": "text",
          "content": json.dumps({"text": "your message here"})},
)
```

Or as a one-liner from another Claude session:
```bash
python3 -c "
import json, requests
t = json.load(open('$HOME/.nemo/user_token.json'))['access_token']
requests.post('https://open.larksuite.com/open-apis/im/v1/messages',
  params={'receive_id_type': 'chat_id'},
  headers={'Authorization': f'Bearer {t}', 'Content-Type': 'application/json'},
  json={'receive_id': 'oc_8183e1682019ddc0857a29074b3e2858', 'msg_type': 'text',
        'content': json.dumps({'text': 'What is 2+2?'})})
"
```

### 3. Check results

```bash
tail -20 ~/.nemo/logs/nemo-$NEMO_PID.log
```

Key log lines to look for:
- `Processing: <message>` — message received from relay
- `query() prompt=N chars` — sent to SDK
- `turn msg: AssistantMessage` — got response
- Card created/updated — visible in Lark group

### Token refresh

Token lasts 2 hours. If expired, re-run device flow (see above) or:
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

## Reference: Current handoff_agent.py

The existing implementation lives in the handoff skill repo at
`~/code/verneagent/handoff/scripts/handoff_agent.py` (1,329 lines).
It depends on 7+ shared modules (lark_im, handoff_db, handoff_config,
handoff_worker, handoff_lifecycle, group_config, permission_core).
See `plan.md` for the full migration strategy.
