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

### Nemo App
- App ID: `cli_a9583021bef89ed4`
- P2P chat_id (with owner): `oc_6731728a1d02fcb97c67a16806d5c6b0`

## Reference: Current handoff_agent.py

The existing implementation lives in the handoff skill repo at
`~/code/verneagent/handoff/scripts/handoff_agent.py` (1,329 lines).
It depends on 7+ shared modules (lark_im, handoff_db, handoff_config,
handoff_worker, handoff_lifecycle, group_config, permission_core).
See `plan.md` for the full migration strategy.
