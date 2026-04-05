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

Zero infrastructure. The daemon connects directly to Lark's WebSocket gateway
for event subscription (`im.message.receive_v1`, reactions) and uses the IM API
for sending responses. No public URL, no webhook, no Worker needed.

## Key Design Decisions

1. **No Cloudflare Worker** — Lark WebSocket 长连接 replaces the entire
   Worker + Durable Object + polling stack. Events arrive directly.

2. **No card buttons** — Card action callbacks require an HTTP webhook URL,
   which reintroduces the need for infrastructure. Instead:
   - Permission approval: user replies "y" / "n" to permission messages
   - Stop: user sends `/esc`
   - Cards are still used for display (Working, Done, status) but are read-only

3. **One card per turn** — A single Card V2 evolves through Working → Response
   → Done via PATCH. Tool history lives in a `collapsible_panel`, always
   available. No message filter levels needed.

4. **Single event callback** — `run_turn()` emits typed events (ToolStart,
   Text, Done) through one callback, replacing the old dual send_fn/working_fn.

5. **Event-driven loop** — Messages arrive via WebSocket push, not polling.
   The main loop is `async for event in lark_ws:` rather than
   `while True: poll()`.

## Config

Reuses `~/.handoff/config.json` for Lark credentials (app_id, app_secret, email).
No worker_url or worker_api_key needed.

## Module Layout

```
nemo/
├── __main__.py      # CLI entry point
├── agent.py         # Main event loop & orchestration
├── turn.py          # SDK turn execution & message streaming
├── cards.py         # Unified turn card (V2 with collapsible panels)
├── monitor.py       # Concurrent signal watcher during SDK turns
├── permissions.py   # Text-based permission bridge (no card buttons)
├── commands.py      # Built-in commands (/clear, /model, /cd, etc.)
├── messages.py      # Message filtering & prompt building
├── lark/
│   ├── api.py       # Lark IM API client (send, update, download)
│   ├── auth.py      # Tenant token management
│   └── events.py    # Lark WebSocket event subscription (长连接)
├── db.py            # SQLite session & message storage
└── config.py        # Credentials & configuration
```

## Reference: Current handoff_agent.py

The existing implementation lives in the handoff skill repo at
`~/code/verneagent/handoff/scripts/handoff_agent.py` (1,329 lines).
It depends on 7+ shared modules (lark_im, handoff_db, handoff_config,
handoff_worker, handoff_lifecycle, group_config, permission_core).
See `plan.md` for the full migration strategy.
