"""Nemo Relay Server — receives Feishu/Lark HTTP callbacks, serves long-poll,
WebSocket, and management endpoints to nemo agents.

Drop-in replacement for the CF Worker + Durable Objects + KV backend,
using a single Python process + SQLite + aiohttp.

Endpoints:
  POST /webhook            — Feishu event callback
  POST /card-action        — Feishu card action callback
  GET  /ws/<key>           — WebSocket upgrade (preferred over long-poll)
  GET  /poll/<key>         — Long-poll for new messages
  GET  /replies/<key>      — Non-blocking get (instant return, no long-poll)
  POST /replies/<key>/ack  — Remove processed messages (create_time <= before)
  POST /takeover/<key>     — Signal takeover to polling/WS clients
  POST /relay              — Cross-group relay messaging
  POST /register-message   — Register message→chat mapping for reaction routing
  GET  /stop/<key>         — Check and consume stop flag
  POST /stop/<key>         — Signal stop to a nemo agent
  POST /heartbeat/<key>    — Upsert heartbeat (agent is alive)
  GET  /heartbeat/<key>    — Check if agent is alive (TTL-based)
  DELETE /heartbeat/<key>  — Explicit release (agent shutting down)
  GET  /status/<key>       — Status check (always ok, no DO quota concept)
  GET  /health             — Health check

Auth: All endpoints except /webhook and /card-action require Bearer token (API_KEY).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from aiohttp import web, WSMsgType

log = logging.getLogger("relay")

# --- Config ---
API_KEY = os.environ.get("RELAY_API_KEY", "")
VERIFY_TOKENS: set[str] = set()
for env_key in ("VERIFY_TOKEN", "VERIFY_TOKENS"):
    for t in os.environ.get(env_key, "").split(","):
        t = t.strip()
        if t:
            VERIFY_TOKENS.add(t)

PORT = int(os.environ.get("RELAY_PORT", "9800"))
DB_PATH = os.environ.get("RELAY_DB", "/opt/nemo-relay/relay.db")
REPLY_TTL = 72 * 3600  # 72 hours

# --- SQLite setup ---
_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            payload TEXT NOT NULL,
            create_time TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT (unixepoch('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_key ON messages(key, id);

        CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires_at REAL
        );
    """)
    conn.commit()
    conn.close()


def _cleanup_expired():
    """Remove expired messages and KV entries."""
    with _db_lock:
        conn = _get_db()
        cutoff = time.time() - REPLY_TTL
        conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
        conn.execute(
            "DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )
        conn.commit()
        conn.close()


# --- Waiters for long-poll ---
_waiters: dict[str, list[asyncio.Event]] = {}
_waiters_lock = threading.Lock()

# --- WebSocket subscribers ---
_ws_clients: dict[str, list[web.WebSocketResponse]] = {}
_ws_lock = threading.Lock()


def _notify_waiters(key: str):
    """Wake all long-poll waiters and push to all WebSocket clients for this key."""
    with _waiters_lock:
        events = _waiters.pop(key, [])
    for e in events:
        e.set()


def _register_waiter(key: str, event: asyncio.Event):
    with _waiters_lock:
        _waiters.setdefault(key, []).append(event)


def _remove_waiter(key: str, event: asyncio.Event):
    with _waiters_lock:
        lst = _waiters.get(key, [])
        try:
            lst.remove(event)
        except ValueError:
            pass


def _register_ws(key: str, ws: web.WebSocketResponse):
    with _ws_lock:
        _ws_clients.setdefault(key, []).append(ws)


def _unregister_ws(key: str, ws: web.WebSocketResponse):
    with _ws_lock:
        lst = _ws_clients.get(key, [])
        try:
            lst.remove(ws)
        except ValueError:
            pass
        if not lst and key in _ws_clients:
            del _ws_clients[key]


async def _broadcast_ws(key: str, data: dict):
    """Send JSON message to all WebSocket clients subscribed to key."""
    with _ws_lock:
        clients = list(_ws_clients.get(key, []))
    if not clients:
        return
    message = json.dumps(data, ensure_ascii=False)
    for ws in clients:
        try:
            await ws.send_str(message)
        except Exception:
            pass


# --- Message storage ---

def _push_message_sync(key: str, payload: dict) -> dict:
    """Insert message into DB. Returns the payload with _id set. Must be called with care for thread safety."""
    create_time = payload.get("create_time", str(int(time.time() * 1000)))
    payload_json = json.dumps(payload, ensure_ascii=False)
    with _db_lock:
        conn = _get_db()
        cur = conn.execute(
            "INSERT INTO messages (key, payload, create_time) VALUES (?, ?, ?)",
            (key, payload_json, create_time),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
    enriched = dict(payload)
    enriched["_id"] = row_id
    return enriched


async def _push_message(key: str, payload: dict):
    """Push a message to a single key, notify waiters and WS clients."""
    loop = asyncio.get_event_loop()
    enriched = await loop.run_in_executor(None, _push_message_sync, key, payload)
    _notify_waiters(key)
    await _broadcast_ws(key, {"replies": [enriched], "count": 1})


async def _push_to_keys(keys: list[str], payload: dict):
    """Push message to multiple keys (chat + root + nonce)."""
    create_time = payload.get("create_time", str(int(time.time() * 1000)))
    payload_json = json.dumps(payload, ensure_ascii=False)

    def _insert_all():
        with _db_lock:
            conn = _get_db()
            ids = {}
            for key in keys:
                if key:
                    cur = conn.execute(
                        "INSERT INTO messages (key, payload, create_time) VALUES (?, ?, ?)",
                        (key, payload_json, create_time),
                    )
                    ids[key] = cur.lastrowid
            conn.commit()
            conn.close()
            return ids

    loop = asyncio.get_event_loop()
    ids = await loop.run_in_executor(None, _insert_all)

    for key in keys:
        if key and key in ids:
            enriched = dict(payload)
            enriched["_id"] = ids[key]
            _notify_waiters(key)
            await _broadcast_ws(key, {"replies": [enriched], "count": 1})



def _get_messages_since_time(key: str, since: str = "") -> list[dict]:
    """Get messages with create_time > since (string comparison)."""
    conn = _get_db()
    if since:
        rows = conn.execute(
            "SELECT id, payload, create_time FROM messages WHERE key = ? AND create_time > ? ORDER BY id",
            (key, since),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, payload, create_time FROM messages WHERE key = ? ORDER BY id",
            (key,),
        ).fetchall()
    conn.close()
    results = []
    for row in rows:
        msg = json.loads(row["payload"])
        msg["_id"] = row["id"]
        results.append(msg)
    return results


def _ack_messages(key: str, before: str) -> tuple[int, int]:
    """Remove messages with create_time <= before. Returns (removed, remaining)."""
    with _db_lock:
        conn = _get_db()
        # Count before deletion
        old_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE key = ?", (key,)
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM messages WHERE key = ? AND create_time <= ?",
            (key, before),
        )
        new_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE key = ?", (key,)
        ).fetchone()[0]
        conn.commit()
        conn.close()
    return old_count - new_count, new_count


# --- KV helpers ---

def _kv_get(key: str) -> str | None:
    conn = _get_db()
    row = conn.execute(
        "SELECT value FROM kv WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
        (key, time.time()),
    ).fetchone()
    conn.close()
    return row["value"] if row else None


def _kv_set(key: str, value: str, ttl: int | None = None):
    expires_at = time.time() + ttl if ttl else None
    with _db_lock:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, expires_at),
        )
        conn.commit()
        conn.close()


def _kv_delete(key: str):
    with _db_lock:
        conn = _get_db()
        conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        conn.commit()
        conn.close()


# --- Webhook handler ---

def _parse_message(event: dict) -> tuple[dict, str, str | None] | None:
    """Parse event into (reply_dict, chat_id, root_id)."""
    message = event.get("message", {})
    sender = event.get("sender", {})
    chat_id = message.get("chat_id")
    root_id = message.get("root_id") or None
    if not chat_id:
        return None

    msg_type = message.get("message_type", "unknown")
    text = ""
    image_key = ""
    file_key = ""
    file_name = ""

    try:
        content = json.loads(message.get("content", "{}"))
    except (json.JSONDecodeError, TypeError):
        content = {}

    if msg_type == "text":
        text = content.get("text", "")
    elif msg_type == "image":
        image_key = content.get("image_key", "")
        text = "[image]"
    elif msg_type == "file":
        file_key = content.get("file_key", "")
        file_name = content.get("file_name", "")
        text = f"[file: {file_name or 'unknown'}]"
    elif msg_type == "post":
        # Worker: if (!Array.isArray(content.content)) → locale-keyed
        raw = content.get("content")
        if isinstance(raw, list):
            paragraphs = raw
        else:
            locale = next(iter(content), None)
            if locale and isinstance(content.get(locale), dict):
                paragraphs = content[locale].get("content", [])
            else:
                paragraphs = []
        parts = []
        image_keys = []
        for para in paragraphs:
            if not isinstance(para, list):
                continue
            for elem in para:
                if elem.get("text"):
                    parts.append(elem["text"])
                elif elem.get("tag") == "img" and elem.get("image_key"):
                    parts.append("[image]")
                    image_keys.append(elem["image_key"])
        text = "\n".join(parts) or "[post]"
        if image_keys:
            image_key = ",".join(image_keys)
    elif msg_type == "sticker":
        file_key = content.get("file_key", "")
        text = "[sticker]"
    elif msg_type == "merge_forward":
        text = "[merge_forward]"
    else:
        text = f"[{msg_type} message]"

    mentions = []
    for m in message.get("mentions", []):
        sender_id_obj = m.get("id", {})
        open_id = sender_id_obj.get("open_id", "") if isinstance(sender_id_obj, dict) else ""
        mentions.append({
            "key": m.get("key", ""),
            "id": open_id,
            "name": m.get("name", ""),
        })

    reply: dict = {
        "text": text,
        "msg_type": msg_type,
        "sender_type": sender.get("sender_type", "unknown"),
        "sender_id": (sender.get("sender_id") or {}).get("open_id", ""),
        "create_time": message.get("create_time", ""),
        "message_id": message.get("message_id", ""),
    }
    if image_key:
        reply["image_key"] = image_key
    if file_key:
        reply["file_key"] = file_key
    if file_name:
        reply["file_name"] = file_name
    if message.get("parent_id"):
        reply["parent_id"] = message["parent_id"]
    if mentions:
        reply["mentions"] = mentions

    return reply, chat_id, root_id


def _extract_action_info(action: dict) -> tuple[str, str]:
    """Extract action text and msg_type from card action. Mirrors CF Worker logic."""
    value = action.get("value", {})
    form_value = action.get("form_value")
    if form_value and isinstance(form_value, dict) and form_value:
        entries = list(form_value.items())
        text = str(entries[0][1]) if len(entries) == 1 else json.dumps(form_value)
        return text, "form_action"
    if action.get("option") is not None:
        return action.get("option", ""), "select_action"
    if action.get("input_value") is not None:
        return action.get("input_value", ""), "input_action"
    return value.get("action", ""), "button_action"


async def _handle_card_action(event: dict) -> tuple[int, dict]:
    action = event.get("action", {})
    value = action.get("value", {})
    operator = event.get("operator", {})
    chat_id = value.get("chat_id", "")
    root_id = value.get("root_id", "")
    nonce = value.get("nonce", "")

    action_text, msg_type = _extract_action_info(action)

    # Stop button
    if action_text == "__stop__":
        approvers = value.get("approvers")
        if isinstance(approvers, list) and approvers:
            operator_id = operator.get("open_id", "")
            if operator_id not in approvers:
                return 200, {
                    "toast": {"type": "error", "content": "Only the owner or coowners can stop"},
                }
        if chat_id:
            _kv_set(f"stop:chat:{chat_id}", "1", ttl=300)
            await _push_message(f"chat:{chat_id}", {
                "text": "__stop__",
                "msg_type": "stop_signal",
                "create_time": str(int(time.time() * 1000)),
            })
        return 200, {
            "toast": {"type": "warning", "content": "Stopping..."},
            "card": {
                "type": "raw",
                "data": {
                    "schema": "2.0",
                    "config": {"update_multi": True},
                    "header": {
                        "title": {"tag": "plain_text", "content": "Stopping..."},
                        "template": "orange",
                    },
                    "body": {
                        "direction": "vertical",
                        "elements": [
                            {"tag": "markdown", "content": "Stop requested. Waiting for current tool to finish..."},
                        ],
                    },
                },
            },
        }

    # Authorization check for regular actions
    approvers = value.get("approvers")
    if isinstance(approvers, list) and approvers:
        operator_id = operator.get("open_id", "")
        if operator_id not in approvers:
            return 200, {
                "toast": {"type": "error", "content": "Only the operator or coowners can decide"},
            }

    # Store as reply — push to chat, root, and nonce keys
    if (chat_id or root_id or nonce) and action_text:
        reply = {
            "text": action_text,
            "msg_type": msg_type,
            "sender_type": "user",
            "sender_id": operator.get("open_id", ""),
            "create_time": str(int(time.time() * 1000)),
            "message_id": "",
        }
        keys = []
        if chat_id:
            keys.append(f"chat:{chat_id}")
        if root_id:
            keys.append(root_id)
        if nonce:
            keys.append(nonce)
        await _push_to_keys(keys, reply)

    # Return updated card
    DENY_TEXTS = {"n", "no", "deny", "reject", "0"}
    is_deny = action_text.lower() in DENY_TEXTS
    template = "red" if is_deny else "green"
    toast_type = "warning" if is_deny else "success"
    toast_verb = "Denied" if is_deny else "Got it"

    title = value.get("title", "Confirmed")
    body = value.get("body", "")
    elements = []
    if body:
        elements.append({"tag": "div", "text": {"content": body, "tag": "lark_md"}})
    elements.append({
        "tag": "div",
        "text": {"content": f"> Selected: **{action_text}**", "tag": "lark_md"},
    })

    return 200, {
        "toast": {"type": toast_type, "content": f"{toast_verb}: {action_text}"},
        "card": {
            "type": "raw",
            "data": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": template,
                },
                "elements": elements,
            },
        },
    }


async def _handle_webhook(data: dict) -> tuple[int, dict]:
    # URL verification challenge
    if data.get("type") == "url_verification":
        return 200, {"challenge": data.get("challenge", "")}

    header = data.get("header", {})
    event = data.get("event", {})

    # Verify token
    if VERIFY_TOKENS and header.get("token") not in VERIFY_TOKENS:
        log.warning("Token mismatch: %s", header.get("token", ""))
        return 403, {"error": "forbidden"}

    # Idempotency
    event_id = header.get("event_id")
    if event_id:
        existing = _kv_get(f"seen:{event_id}")
        if existing:
            return 200, {"ok": True, "duplicate": True}
        _kv_set(f"seen:{event_id}", "1", ttl=3600)

    event_type = header.get("event_type", "")

    if event_type == "im.message.receive_v1":
        result = _parse_message(event)
        if result:
            reply, chat_id, root_id = result
            keys = []
            if chat_id:
                keys.append(f"chat:{chat_id}")
            if root_id:
                keys.append(root_id)
            if keys:
                await _push_to_keys(keys, reply)
                log.info(
                    "Message pushed: chat=%s msg_type=%s text=%s",
                    chat_id,
                    reply.get("msg_type"),
                    reply.get("text", "")[:60],
                )

    elif event_type == "im.message.reaction.created_v1":
        message_id = event.get("message_id", "")
        reaction_type = (event.get("reaction_type") or {}).get("emoji_type", "")
        operator_id = (event.get("user_id") or {}).get("open_id", "")
        if message_id and reaction_type:
            chat_id = _kv_get(f"msgchat:{message_id}")
            if chat_id:
                reply = {
                    "text": reaction_type,
                    "msg_type": "reaction",
                    "target_message_id": message_id,
                    "sender_type": "user",
                    "sender_id": operator_id,
                    "create_time": event.get("action_time", str(int(time.time() * 1000))),
                    "message_id": "",
                }
                await _push_message(f"chat:{chat_id}", reply)

    elif event_type == "card.action.trigger":
        return await _handle_card_action(event)

    return 200, {"ok": True}


# --- Auth middleware ---

def _check_auth(request: web.Request) -> web.Response | None:
    """Return an error response if auth fails, or None if auth passes."""
    if not API_KEY:
        return None
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {API_KEY}":
        return None
    return web.json_response({"error": "unauthorized"}, status=401)


# --- Route handlers ---

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "verify_tokens": len(VERIFY_TOKENS)})


async def handle_webhook(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except (json.JSONDecodeError, TypeError, Exception):
        return web.json_response({"error": "bad request"}, status=400)
    status, resp = await _handle_webhook(data)
    return web.json_response(resp, status=status)


async def handle_card_action(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except (json.JSONDecodeError, TypeError, Exception):
        return web.json_response({"error": "bad request"}, status=400)
    status, resp = await _handle_webhook(data)
    return web.json_response(resp, status=status)


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    _register_ws(key, ws)
    try:
        # Check takeover flag
        takeover_key = f"takeover:{key}"
        loop = asyncio.get_event_loop()
        takeover_val = await loop.run_in_executor(None, _kv_get, takeover_key)
        if takeover_val:
            await loop.run_in_executor(None, _kv_delete, takeover_key)
            await ws.send_json({"replies": [], "count": 0, "takeover": True})
        else:
            # Send initial messages filtered by since
            since = request.query.get("since", "")
            messages = await loop.run_in_executor(None, _get_messages_since_time, key, since)
            if messages:
                await ws.send_json({"replies": messages, "count": len(messages)})

        # Process incoming messages from client
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("ack"):
                        before = data["ack"]
                        await loop.run_in_executor(None, _ack_messages, key, before)
                    if data.get("ping"):
                        await ws.send_json({"pong": True})
                except (json.JSONDecodeError, Exception):
                    pass
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    finally:
        _unregister_ws(key, ws)

    return ws


async def handle_poll(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key", "replies": [], "count": 0})

    since = request.query.get("since", "")
    try:
        timeout = min(int(request.query.get("timeout", "25")), 55)
    except (ValueError, TypeError):
        timeout = 25

    loop = asyncio.get_event_loop()

    # Check for takeover flag
    takeover_key = f"takeover:{key}"
    takeover_val = await loop.run_in_executor(None, _kv_get, takeover_key)
    if takeover_val:
        await loop.run_in_executor(None, _kv_delete, takeover_key)
        return web.json_response({"replies": [], "count": 0, "takeover": True})

    # Check for stop signal
    stop_key = f"stop:{key}"
    stop_val = await loop.run_in_executor(None, _kv_get, stop_key)
    if stop_val:
        await loop.run_in_executor(None, _kv_delete, stop_key)
        return web.json_response({"replies": [], "count": 0, "stop": True})

    # For poll, we support both since (create_time string) and since as an integer id.
    # The CF Worker uses create_time string comparison for poll.
    messages = await loop.run_in_executor(None, _get_messages_since_time, key, since)
    if messages:
        return web.json_response({"replies": messages, "count": len(messages)})

    # Wait for new messages
    event = asyncio.Event()
    _register_waiter(key, event)
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    finally:
        _remove_waiter(key, event)

    # Check takeover again after wake
    takeover_val = await loop.run_in_executor(None, _kv_get, takeover_key)
    if takeover_val:
        await loop.run_in_executor(None, _kv_delete, takeover_key)
        return web.json_response({"replies": [], "count": 0, "takeover": True})

    # Check stop again after wake
    stop_val = await loop.run_in_executor(None, _kv_get, stop_key)
    if stop_val:
        await loop.run_in_executor(None, _kv_delete, stop_key)
        return web.json_response({"replies": [], "count": 0, "stop": True})

    messages = await loop.run_in_executor(None, _get_messages_since_time, key, since)
    return web.json_response({"replies": messages, "count": len(messages)})


async def handle_get_replies(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key", "replies": [], "count": 0})

    loop = asyncio.get_event_loop()

    # Check for takeover flag
    takeover_key = f"takeover:{key}"
    takeover_val = await loop.run_in_executor(None, _kv_get, takeover_key)
    if takeover_val:
        await loop.run_in_executor(None, _kv_delete, takeover_key)
        return web.json_response({"replies": [], "count": 0, "takeover": True})

    since = request.query.get("since", "")
    messages = await loop.run_in_executor(None, _get_messages_since_time, key, since)
    return web.json_response({"replies": messages, "count": len(messages)})


async def handle_ack_replies(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    before = request.query.get("before")
    if not before:
        return web.json_response({"error": "missing before parameter"}, status=400)

    loop = asyncio.get_event_loop()
    removed, remaining = await loop.run_in_executor(None, _ack_messages, key, before)
    return web.json_response({"removed": removed, "remaining": remaining})


async def handle_takeover(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    loop = asyncio.get_event_loop()

    # Store takeover flag in KV
    takeover_key = f"takeover:{key}"
    await loop.run_in_executor(None, _kv_set, takeover_key, "1", 300)

    # Wake HTTP long-poll waiters
    _notify_waiters(key)

    # Push to WebSocket clients
    await _broadcast_ws(key, {"replies": [], "count": 0, "takeover": True})

    return web.json_response({"ok": True})


async def handle_relay(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    try:
        data = await request.json()
    except (json.JSONDecodeError, TypeError, Exception):
        return web.json_response({"error": "bad request"}, status=400)

    to_chat_id = data.get("to_chat_id")
    message = data.get("message", "")
    from_chat_id = data.get("from_chat_id", "")
    from_chat_name = data.get("from_chat_name", "")
    from_workspace = data.get("from_workspace", "")

    if not to_chat_id or not message:
        return web.json_response({"error": "missing to_chat_id or message"}, status=400)

    reply = {
        "text": message,
        "msg_type": "relay",
        "from_chat_id": from_chat_id,
        "from_chat_name": from_chat_name,
        "from_workspace": from_workspace,
        "sender_type": "relay",
        "sender_id": "",
        "create_time": str(int(time.time() * 1000)),
        "message_id": "",
    }

    await _push_message(f"chat:{to_chat_id}", reply)
    return web.json_response({"ok": True})


async def handle_register_message(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    try:
        data = await request.json()
    except (json.JSONDecodeError, TypeError, Exception):
        return web.json_response({"error": "bad request"}, status=400)

    msg_id = data.get("message_id", "")
    chat_id = data.get("chat_id", "")
    if msg_id and chat_id:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _kv_set, f"msgchat:{msg_id}", chat_id, 604800)
        return web.json_response({"ok": True})
    else:
        return web.json_response({"error": "missing message_id or chat_id"}, status=400)


async def handle_get_stop(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    stop_key = f"stop:{key}"
    loop = asyncio.get_event_loop()
    val = await loop.run_in_executor(None, _kv_get, stop_key)
    if val:
        await loop.run_in_executor(None, _kv_delete, stop_key)
        return web.json_response({"stop": True})
    return web.json_response({"stop": False})


async def handle_post_stop(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _kv_set, f"stop:{key}", "1", 300)
    await _push_message(key, {
        "text": "__stop__",
        "msg_type": "stop_signal",
        "create_time": str(int(time.time() * 1000)),
    })
    return web.json_response({"ok": True})


async def handle_status(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    return web.json_response({"ok": True, "do_quota_exhausted": False, "exhausted_at": None})


# --- Heartbeat endpoints ---

HEARTBEAT_TTL = 90  # seconds — agent sends every ~30s, 3x margin

async def handle_post_heartbeat(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    try:
        data = await request.json()
    except Exception:
        data = {}

    # Store heartbeat with TTL
    hb_key = f"heartbeat:{key}"
    value = json.dumps({
        "ts": time.time(),
        "pid": data.get("pid", 0),
        "model": data.get("model", ""),
        "machine": data.get("machine", ""),
    })
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _kv_set, hb_key, value, HEARTBEAT_TTL)
    return web.json_response({"ok": True})


async def handle_get_heartbeat(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    hb_key = f"heartbeat:{key}"
    loop = asyncio.get_event_loop()
    val = await loop.run_in_executor(None, _kv_get, hb_key)
    if val:
        info = json.loads(val)
        return web.json_response({"alive": True, **info})
    return web.json_response({"alive": False})


async def handle_delete_heartbeat(request: web.Request) -> web.Response:
    denied = _check_auth(request)
    if denied:
        return denied

    key = request.match_info["key"]
    if not key:
        return web.json_response({"error": "missing key"}, status=400)

    hb_key = f"heartbeat:{key}"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _kv_delete, hb_key)
    return web.json_response({"ok": True})


# --- Cleanup background task ---

async def _cleanup_loop():
    while True:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _cleanup_expired)
        except Exception as e:
            log.error("Cleanup error: %s", e)
        await asyncio.sleep(3600)


async def on_startup(app: web.Application):
    app["cleanup_task"] = asyncio.create_task(_cleanup_loop())


async def on_cleanup(app: web.Application):
    task = app.get("cleanup_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# --- App factory ---

def create_app() -> web.Application:
    app = web.Application()

    # Public endpoints (no auth — Feishu verifies via token)
    app.router.add_post("/webhook", handle_webhook)
    app.router.add_post("/card-action", handle_card_action)

    # Health check (no auth for simplicity, matches existing relay behavior)
    app.router.add_get("/health", handle_health)

    # WebSocket endpoint
    app.router.add_get("/ws/{key:.+}", handle_ws)

    # Long-poll endpoint
    app.router.add_get("/poll/{key:.+}", handle_poll)

    # Non-blocking get replies
    # Note: ack must be registered before the general get to match /replies/<key>/ack first
    app.router.add_post("/replies/{key:.+}/ack", handle_ack_replies)
    app.router.add_get("/replies/{key:.+}", handle_get_replies)

    # Takeover
    app.router.add_post("/takeover/{key:.+}", handle_takeover)

    # Relay
    app.router.add_post("/relay", handle_relay)

    # Register message for reaction routing
    app.router.add_post("/register-message", handle_register_message)

    # Stop
    app.router.add_get("/stop/{key:.+}", handle_get_stop)
    app.router.add_post("/stop/{key:.+}", handle_post_stop)

    # Heartbeat
    app.router.add_post("/heartbeat/{key:.+}", handle_post_heartbeat)
    app.router.add_get("/heartbeat/{key:.+}", handle_get_heartbeat)
    app.router.add_delete("/heartbeat/{key:.+}", handle_delete_heartbeat)

    # Status
    app.router.add_get("/status/{key:.+}", handle_status)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


# --- Main ---

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[relay] %(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not VERIFY_TOKENS:
        log.warning("No VERIFY_TOKENS set — webhook verification disabled")
    if not API_KEY:
        log.warning("No RELAY_API_KEY set — poll endpoints unprotected")

    _init_db()

    app = create_app()
    log.info("Relay server listening on :%d", PORT)
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
