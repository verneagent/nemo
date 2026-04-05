# Nemo Implementation Plan

## Goal

Rewrite `handoff_agent.py` as a standalone executable. Key changes:
- Replace Cloudflare Worker relay with Lark WebSocket 长连接
- Replace 3-card system (Working + Response + Done) with unified turn card
- Replace card button interactions with text-based commands
- Clean module separation (currently 1,329-line monolith + 7 shared modules)

## Architecture Comparison

### Before (handoff_agent.py)

```
Lark → Webhook → Cloudflare Worker → Durable Objects → WS/HTTP poll → Agent
Agent → Lark IM API → Lark Group
Card buttons → Worker /card-action → Durable Objects → Agent polls
```

Dependencies: Cloudflare Worker, Worker API key, Durable Objects, custom WS
polling, `handoff_worker.py`, `permission_core.py`, `permission_bridge.py`

### After (Nemo)

```
Lark → WebSocket Gateway (长连接) → Agent (direct)
Agent → Lark IM API → Lark Group
```

Dependencies: Lark app credentials only. Zero infrastructure.

## Phase 1: Lark Event Subscription (`nemo/lark/events.py`)

Replace the entire Worker polling stack with Lark's native WebSocket 长连接.

### How It Works

Lark provides a WebSocket gateway for event subscription. The client:
1. Obtains a WebSocket URL via REST API (`/callback/ws/endpoint`)
2. Connects and receives events as JSON frames
3. Auto-reconnects on disconnect

### Events to Subscribe

| Event | Purpose |
|---|---|
| `im.message.receive_v1` | New messages in the group |
| `im.message.reaction.created_v1` | Emoji reactions |

### Interface

```python
class LarkEventStream:
  """Async iterator over Lark events via WebSocket 长连接."""

  def __init__(self, app_id: str, app_secret: str):
    ...

  async def __aiter__(self):
    """Yields LarkEvent objects."""
    ...

  async def close(self):
    ...

@dataclass
class LarkEvent:
  event_type: str    # "im.message.receive_v1"
  chat_id: str
  sender_id: str
  message_id: str
  msg_type: str      # "text", "image", "file", etc.
  text: str
  mentions: list
  image_key: str
  file_key: str
  file_name: str
  parent_id: str
  create_time: str
  raw: dict          # Full event payload
```

### Research Needed

- Verify Lark WebSocket gateway API: endpoint URL, auth flow, message format
- Check `lark-cli event +subscribe` source for implementation reference
  (lives at `~/.agents/skills/lark-event/`)
- Determine if `card.action.trigger` can be subscribed (unlikely per research,
  but worth confirming with a live test)
- Handle reconnection, heartbeat, and token refresh

### Source Reference

The existing Worker-based polling lives in:
- `handoff_worker.py`: `poll_worker_ws()`, `poll_worker()`, `_WebSocket` class
- `handoff_agent.py`: `wait_for_reply_inline()` (lines 104-193)
- `handoff_agent.py`: `_message_monitor_sync()` (lines 610-719)

## Phase 2: Lark API Client (`nemo/lark/api.py`, `nemo/lark/auth.py`)

Port the Lark IM API functions. These are stable and straightforward.

### auth.py

```python
class LarkAuth:
  def get_token(self, app_id, app_secret) -> str:
    """Cached tenant access token with auto-refresh."""
```

Source: `lark_auth.py` (simple, port as-is)

### api.py

Functions to port from `lark_im.py`:

| Function | Purpose |
|---|---|
| `send_card(token, chat_id, card)` | Send interactive card → message_id |
| `update_card(token, message_id, card)` | PATCH existing card |
| `send_markdown(token, chat_id, text)` | Send Card V2 markdown |
| `get_bot_info(token)` | Bot's open_id |
| `lookup_open_id_by_email(token, email)` | Resolve email → open_id |
| `get_chat_info(token, chat_id)` | Group info |
| `get_chat_members(token, chat_id)` | Member list |
| `download_image(token, message_id, image_key)` | Download image |
| `download_file(token, message_id, file_key, name)` | Download file |
| `add_reaction(token, message_id, emoji_type)` | Add emoji reaction |

No changes to API surface — just clean extraction from `lark_im.py`.

## Phase 3: Unified Turn Card (`nemo/cards.py`)

Replace the 3-card system with one evolving card.

### Current System (handoff_agent.py)

| Card | When | Builder |
|---|---|---|
| Working (grey, V1) | Tool use starts | `build_card("Working...")` + Stop button |
| Response (V2 markdown) | Agent text output | `build_markdown_card(text)` |
| Done (green, V1) | Turn completes | `build_card("Done ✓")` + token note |

Three separate messages. Working and Done use Card V1. Response uses Card V2.
The PostToolUse hook in CLI mode adds a 4th "Working" V2 card.

### New System (Nemo)

**One Card V2 message** that PATCHes through phases:

#### Phase: Working

```
┌─────────────────────────────┐
│ [grey] Working...           │
├─────────────────────────────┤
│ `Grep: pattern`             │  ← current tool
│                             │
│ ▸ Previous tools (3)        │  ← collapsible_panel (collapsed)
│   - `Read: config.py`      │
│   - `Edit: main.py`        │
│   - `Bash: npm test`       │
└─────────────────────────────┘
```

- Grey header with escalating title (Working → Working hard → ...)
- Current tool action in body
- Past tools in `collapsible_panel` (collapsed by default)
- No Stop button (user sends `/esc` instead)

#### Phase: Response

```
┌─────────────────────────────┐
│                             │  ← no header
│ Here's what I found...      │  ← response markdown
│ ```python                   │
│ def foo():                  │
│ ```                         │
│                             │
│ ▸ Tools used (4)            │  ← collapsible_panel
└─────────────────────────────┘
```

- No colored header (clean look)
- Response text in body
- All tools folded into collapsible panel

#### Phase: Done

```
┌─────────────────────────────┐
│ [green] Done ✓              │
├─────────────────────────────┤
│ Here's what I found...      │  ← same response text
│                             │
│ ▸ Tools used (4)            │  ← collapsible_panel
│                             │
│ 12s | in: 45,230 | out: 892 │  ← note (grey footnote)
└─────────────────────────────┘
```

- Green header "Done ✓"
- Response text persists
- Tools in collapsible panel
- Duration + token usage in note element

### Card V2 Structure

```json
{
  "schema": "2.0",
  "config": {"update_multi": true},
  "header": {"title": {"tag": "plain_text", "content": "Working..."}, "template": "grey"},
  "body": {
    "direction": "vertical",
    "elements": [
      {"tag": "markdown", "content": "`current tool`"},
      {
        "tag": "collapsible_panel",
        "expanded": false,
        "header": {"tag": "markdown", "content": "**Previous tools (3)**"},
        "vertical_spacing": "8px",
        "elements": [
          {"tag": "markdown", "content": "- `Read: config.py`\n- `Edit: main.py`"}
        ]
      }
    ]
  }
}
```

### Tool Summary Function

Port `_tool_use_summary()` from handoff_agent.py (lines 388-413) as
`tool_use_summary()`. No changes needed — it already produces clean one-liners.

### Source Reference

- Card V1 builder: `lark_im.py` `build_card()` (lines 105-158)
- Card V2 builder: `lark_im.py` `build_markdown_card()` (lines 161-187)
- Working card: `lark_im.py` `build_working_card()` (lines 190-220)
- Current _working_fn: `handoff_agent.py` (lines 1150-1203)
- Current _send_lark: `handoff_agent.py` (lines 1109-1138)

## Phase 4: SDK Turn Execution (`nemo/turn.py`)

Simplify the dual-callback pattern into a single event stream.

### Current Pattern (handoff_agent.py)

```python
# Two separate callbacks
async def run_agent_turn(client, prompt, send_fn=None, working_fn=None, ...):
  # ToolUseBlock → working_fn("start", summary)
  # ToolUseBlock → working_fn("progress", summary)
  # TextBlock → send_fn(text, task_id=..., pending_tasks=...)
  # ResultMessage → working_fn("done", usage=usage)
```

### New Pattern (Nemo)

```python
# Single typed event callback
async def run_turn(client, prompt, on_event, stale_tasks=None):
  # ToolUseBlock → on_event(ToolStartEvent(tool=...))
  # ToolUseBlock → on_event(ToolProgressEvent(tool=...))
  # TextBlock → on_event(TextEvent(text=..., task_id=...))
  # ResultMessage → on_event(DoneEvent(cost=..., usage=...))
```

Event types:
- `ToolStartEvent` — first tool use, create Working card
- `ToolProgressEvent` — subsequent tool, update card
- `TextEvent` — agent text output, transition to Response phase
- `TaskStartedEvent` / `TaskDoneEvent` — sub-agent lifecycle
- `DoneEvent` — turn complete, transition to Done phase

### Stale Task Workaround

Keep the SDK bug #788 workaround as-is. When a stale `TaskNotificationMessage`
is detected, drop the contaminated response and re-query. Up to 5 retries.

Source: `handoff_agent.py` lines 459-480, 554-561

## Phase 5: Main Agent Loop (`nemo/agent.py`)

Event-driven instead of poll-based.

### Current Flow (handoff_agent.py)

```python
while running:
  data = wait_for_reply_inline(...)  # HTTP/WS poll Worker, 300s timeout
  if data.get("timeout"): continue
  if data.get("takeover"): break
  # dispatch commands
  # run SDK + concurrent monitor
```

### New Flow (Nemo)

```python
async def main_loop(...):
  events = LarkEventStream(app_id, app_secret)
  while running:
    msg = await receive_next_message(events, session)  # filters, etc.
    if is_command(msg):
      handle_command(msg)
      continue
    # Run SDK turn with concurrent signal watching
    await run_sdk_turn(client, msg, events)
```

Key differences:
- **Event-driven**: messages arrive via push, not poll
- **Signal monitoring**: the event stream itself detects /esc and handback
  during SDK execution (no separate WebSocket monitor thread)
- **No takeover via Worker**: takeover is handled by checking session DB
  on startup (cleanup stale sessions, same as now)

### Concurrent Signal Monitoring

During a SDK turn, we need to watch for /esc, handback, and takeover signals.
Currently this runs in a thread via `_message_monitor_sync()` with its own WS.

In Nemo, the Lark WebSocket event stream is the single source of events.
During a turn, we fork it:
- SDK turn runs as an async task
- Event stream continues reading in parallel
- If /esc or handback arrives, interrupt the SDK task

```python
sdk_task = asyncio.create_task(run_turn(client, prompt, on_event))
async for event in events:
  if is_esc(event): await client.interrupt(); break
  if is_handback(event): await client.interrupt(); break
# or sdk_task completes naturally
```

No thread, no separate WebSocket — just async task coordination.

### Source Reference

- Main loop: `handoff_agent.py` `main_loop()` (lines 722-1309)
- Session setup: lines 722-866 (credentials, operator, bot, stale cleanup, activate)
- Message wait: `wait_for_reply_inline()` (lines 104-193)
- Command dispatch: lines 946-1089
- SDK + monitor concurrency: lines 1094-1280
- Signal monitor: `_message_monitor_sync()` (lines 610-719)

## Phase 6: Permission Bridge (`nemo/permissions.py`)

Text-based instead of card-button-based.

### Current Flow (handoff_agent.py)

1. Send Card V1 with Approve/Deny/Approve All buttons
2. Card button click → Worker `/card-action` → Durable Object
3. Agent polls Worker for button response via WebSocket
4. PATCH card to show decision (green/red)

Depends on: `permission_core.py`, `permission_bridge.py`, Worker, card action callback URL

### New Flow (Nemo)

1. Send a **read-only card** showing the tool and asking for approval
2. Wait for the user's text reply: "y", "n", "always"
3. PATCH card to show decision
4. No card buttons, no Worker callback, no polling

```python
async def request_permission(tool_name, tool_input, events, token, chat_id):
  # Send permission info card (read-only, no buttons)
  card = build_permission_card(tool_name, tool_input)
  msg_id = send_card(token, chat_id, card)

  # Wait for text reply
  reply = await events.next_message(timeout=300)
  decision = parse_decision(reply.text)  # "y"→allow, "n"→deny, "always"→always

  # Update card
  if decision == "allow" or decision == "always":
    update_card(token, msg_id, approved_card(tool_name, tool_input))
  else:
    update_card(token, msg_id, denied_card(tool_name, tool_input))

  return decision
```

### Source Reference

- `handoff_agent.py` `_build_permission_handler()` (lines 216-316)
- `permission_core.py`: `prepare_permission_request()`, `run_permission_poll_loop()`, `update_permission_card()`

## Phase 7: Commands & Messages (`nemo/commands.py`, `nemo/messages.py`)

### commands.py

Port built-in commands from `handoff_agent.py` lines 946-1089:

| Command | Action |
|---|---|
| `/clear` | Restart SDK client |
| `/model <name>` | Switch model, restart client |
| `/esc` | Interrupt SDK, cancel current operation |
| `/cd <dir>` | Change working directory, restart client |
| `/ping` | Status (uptime, cost, messages, model) |
| `/cost` | Session API cost |
| `/usage` | Link to usage dashboard |
| `/help` | Command list |
| `autoapprove on/off` | Toggle auto-approve |
| `handback` | Stop agent |

Clean extraction — the logic is simple string matching and response formatting.

### messages.py

Port message processing from `handoff_agent.py` lines 904-925 and `wait_for_reply.py` filter functions:

- `build_prompt(replies)` — JSON if media/parent_id, plain text otherwise
- `strip_mentions(text, replies)` — remove @-mention markers
- `filter_self_bot(replies, bot_id)` — exclude bot's own messages
- `filter_by_operator(replies, operator_id)` — keep operator only
- `filter_by_allowed_senders(replies, operator_id, member_roles)` — operator + guests
- `filter_bot_interactions(replies, bot_id)` — need_mention mode

## Phase 8: Database (`nemo/db.py`)

Minimal SQLite for session state and message history.

### Tables

Same schema as `handoff_db.py` but accessed through a single `Database` class
instead of module-level functions:

```python
class Database:
  def __init__(self, project_dir: str): ...

  # Sessions
  def activate(self, session_id, chat_id, model, **kwargs): ...
  def deactivate(self, session_id) -> str | None: ...
  def get_session(self, session_id) -> dict | None: ...
  def get_chat_owner(self, chat_id) -> str | None: ...
  def set_last_checked(self, session_id, ts): ...
  def set_autoapprove(self, chat_id, enabled): ...

  # Messages
  def record_received(self, chat_id, text, ...): ...
  def record_sent(self, message_id, text, ...): ...

  # Working state
  def set_working(self, session_id, message_id): ...
  def clear_working(self, session_id): ...
  def get_working(self, session_id) -> str | None: ...
```

### DB Location

Same path as handoff: `~/.handoff/projects/<hash>/handoff-data.db`
This allows coexistence — nemo and handoff can share the same DB.

### Source Reference

- `handoff_db.py`: full schema and all functions

## Phase 9: Config (`nemo/config.py`)

Reuse `~/.handoff/config.json` format. Only fields needed:

| Field | Required | Purpose |
|---|---|---|
| `app_id` | Yes | Lark app ID |
| `app_secret` | Yes | Lark app secret |
| `email` | Yes | Operator email (for sender filtering) |

**Not needed**: `worker_url`, `worker_api_key` (no Worker dependency).

Profile support: load from `~/.handoff/profiles/<name>.json` when `--profile` is set.

### Source Reference

- `handoff_config.py`: `load_credentials()`, `load_worker_url()`, `resolve_profile()`

## Phase 10: Entry Point (`nemo/__main__.py`, `pyproject.toml`)

### CLI

```bash
# Direct
python -m nemo --chat-id <ID> --project-dir <DIR> [--model MODEL] [--profile NAME] [-v]

# Installed
nemo --chat-id <ID> --project-dir <DIR>
```

### SDK Auto-detection

Port `_ensure_sdk()` from `handoff_agent.py` lines 22-53. Searches PATH for a
Python with `claude_agent_sdk` installed and re-execs if current Python lacks it.

### pyproject.toml

```toml
[project]
name = "nemo"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["claude-agent-sdk"]

[project.scripts]
nemo = "nemo.__main__:main"
```

## Implementation Order

1. **Phase 1** (events.py) — This is the critical new component. Everything
   else is porting. Start here and verify Lark WebSocket 长连接 works.
2. **Phase 2** (api.py, auth.py) — Straightforward port from lark_im.py.
3. **Phase 8** (db.py) — Clean extraction from handoff_db.py.
4. **Phase 9** (config.py) — Trivial.
5. **Phase 3** (cards.py) — The main UX improvement. Build + test the
   collapsible_panel card.
6. **Phase 4** (turn.py) — Port run_agent_turn with new event pattern.
7. **Phase 6** (permissions.py) — Text-based permission bridge.
8. **Phase 7** (commands.py, messages.py) — Simple port.
9. **Phase 5** (agent.py) — Wire everything together.
10. **Phase 10** (__main__.py, pyproject.toml) — Entry point and packaging.

## Open Questions

1. **Lark WebSocket 长连接 API**: Need to verify the exact endpoint and auth
   flow. `lark-cli event +subscribe` uses it — check its implementation for
   reference. The REST endpoint may be `/callback/ws/endpoint` or similar.

2. **Card action via WebSocket**: Research confirms card action callbacks
   (`card.action.trigger`) are HTTP-only. If we later want interactive buttons,
   we'd need to add a callback server. For now, text-based is the design choice.

3. **Multi-session takeover**: Without the Worker's takeover endpoint, we rely
   on the DB. On startup, check if another session owns the chat_id, deactivate
   it, and proceed. The old agent detects deactivation via DB check on next
   poll. May need a brief sleep or a file-based signal.

4. **`collapsible_panel` mobile rendering**: Verify that Lark mobile app
   renders `collapsible_panel` correctly. If not, fall back to `hr` + smaller
   text for tool history.

5. **SKILL-agent.md**: The current agent loads SKILL-agent.md as a system
   prompt append telling the SDK client how to interact with Lark (download
   images, send files, etc. via `handoff_ops.py`). Nemo needs its own version
   of this, or the SDK client needs a way to call nemo's Lark API directly.
