# Nemo

Lark-connected coding agent daemon powered by Claude Agent SDK.

Named after Captain Nemo — autonomous, persistent, remotely commanded.

## What It Is

A standalone executable that:
1. Connects to a Lark group via WebSocket long connection (长连接)
2. Receives messages directly from Lark — no Cloudflare Worker, no webhook relay
3. Runs Claude Agent SDK to process coding tasks
4. Sends responses back via Lark IM API

```
python -m nemo --chat-id <ID> --project-dir <DIR> [--model claude-opus-4-6]
```

Or installed via pip:
```
pip install -e .
nemo --chat-id <ID> --project-dir <DIR>
```

## What It Is NOT

- Not a Claude Code skill (no SKILL.md, no hooks)
- Not dependent on Cloudflare Worker or any external relay
- Not a library — it's a daemon process

## Architecture

```
Lark Group
  ↕ Lark IM API (send/update messages)
Nemo daemon
  ↕ Lark WebSocket Gateway (receive events via 长连接)
  ↕ Claude Agent SDK (coding execution)
```

Zero infrastructure. The daemon connects directly to Lark's persistent
connection gateway for events (`im.message.receive_v1`, reactions) AND
card action callbacks (`card.action.trigger`). Uses the IM API for sending
responses. No public URL, no webhook, no Worker needed.

## Key Design Decisions

1. **No Cloudflare Worker** — Lark WebSocket 长连接 replaces the entire
   Worker + Durable Object + polling stack. Events arrive directly.

2. **Card buttons via persistent connection** — Lark's persistent connection
   mode supports card action callbacks (`card.action.trigger`), so interactive
   buttons (Approve/Deny, Stop) work without any HTTP webhook endpoint.
   Permission cards and Stop buttons behave the same as in handoff.

3. **One card per turn** — A single Card V2 evolves through Working → Response
   → Done via PATCH. Tool history lives in a `collapsible_panel`, always
   available. No message filter levels needed.

4. **Single event callback** — `run_turn()` emits typed events (ToolStart,
   Text, Done) through one callback, replacing the old dual send_fn/working_fn.

5. **Event-driven loop** — Messages arrive via WebSocket push, not polling.
   The main loop is `async for event in lark_ws:` rather than
   `while True: poll()`.

## Config

Uses `~/.nemo/config.json` for Lark credentials (app_id, app_secret, email).
No worker_url or worker_api_key needed.

## Module Layout

Agent and Channel are decoupled — core orchestration depends on abstract
interfaces, not on Lark or Claude SDK directly.

```
nemo/
├── __main__.py      # CLI entry point
├── core.py          # Main loop & orchestration (channel/agent agnostic)
├── channel.py       # Channel interface (abstract)
├── agent.py         # Agent interface (abstract)
├── lark/            # Lark channel implementation
│   ├── channel.py   # LarkChannel (implements Channel)
│   ├── cards.py     # Card V2 builder (Working/Response/Done)
│   ├── api.py       # Lark IM API client (send, update, download)
│   ├── auth.py      # Tenant token management
│   └── events.py    # Lark WebSocket event subscription (长连接)
├── claude/          # Claude agent implementation
│   ├── agent.py     # ClaudeAgent (implements Agent)
│   └── turn.py      # SDK turn execution & event streaming
├── commands.py      # Built-in commands (/clear, /model, /cd, etc.)
├── messages.py      # Message filtering & prompt building
├── db.py            # SQLite session & message storage
└── config.py        # Credentials & configuration
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

## Reference: Current handoff_agent.py

The existing implementation lives in the handoff skill repo at
`~/code/verneagent/handoff/scripts/handoff_agent.py` (1,329 lines).
It depends on 7+ shared modules (lark_im, handoff_db, handoff_config,
handoff_worker, handoff_lifecycle, group_config, permission_core).
See `plan.md` for the full migration strategy.
