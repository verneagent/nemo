"""Lark IM API client — send, update, download messages."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "https://open.larksuite.com/open-apis"


def _request(url: str, token: str, payload: dict[str, Any] | None = None,
             method: str = "GET", timeout: int = 30) -> dict[str, Any]:
  """Make an authenticated request to Lark API."""
  headers = {"Authorization": f"Bearer {token}"}
  data = None
  if payload is not None:
    headers["Content-Type"] = "application/json"
    data = json.dumps(payload).encode()
    if method == "GET":
      method = "POST"
  req = urllib.request.Request(url, data=data, headers=headers, method=method)
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return json.loads(resp.read())
  except urllib.error.HTTPError as e:
    try:
      return json.loads(e.read())
    except Exception:
      raise e


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------

def send_card(token: str, chat_id: str, card: dict[str, Any]) -> str:
  """Send an interactive card message. Returns message_id."""
  url = f"{BASE_URL}/im/v1/messages?receive_id_type=chat_id"
  payload = {
    "receive_id": chat_id,
    "msg_type": "interactive",
    "content": json.dumps(card),
  }
  data: dict[str, Any] = {}
  for attempt in range(3):
    data = _request(url, token, payload)
    if data.get("code") == 0:
      return data["data"]["message_id"]
    if attempt < 2:
      time.sleep(1)
  raise RuntimeError(f"Failed to send card: {data}")


def update_card(token: str, message_id: str, card: dict[str, Any]) -> None:
  """PATCH an existing card message."""
  url = f"{BASE_URL}/im/v1/messages/{message_id}"
  payload = {"msg_type": "interactive", "content": json.dumps(card)}
  data = _request(url, token, payload, method="PATCH")
  if data.get("code") != 0:
    raise RuntimeError(f"Failed to update card: {data}")


def send_text(token: str, chat_id: str, text: str) -> str:
  """Send a plain text message. Returns message_id."""
  url = f"{BASE_URL}/im/v1/messages?receive_id_type=chat_id"
  payload = {
    "receive_id": chat_id,
    "msg_type": "text",
    "content": json.dumps({"text": text}),
  }
  data = _request(url, token, payload)
  if data.get("code") == 0:
    return data["data"]["message_id"]
  raise RuntimeError(f"Failed to send text: {data}")


# ---------------------------------------------------------------------------
# User & chat info
# ---------------------------------------------------------------------------

def get_bot_info(token: str) -> dict[str, Any]:
  """Get bot's own info (open_id, etc.)."""
  url = f"{BASE_URL}/bot/v3/info"
  data = _request(url, token)
  if data.get("code") != 0:
    raise RuntimeError(f"Bot info error: {data}")
  return data.get("data", {}).get("bot", {})


def lookup_open_id_by_email(token: str, email: str) -> str | None:
  """Resolve email to Lark open_id."""
  url = f"{BASE_URL}/contact/v3/users/batch_get_id?user_id_type=open_id"
  data = _request(url, token, {"emails": [email]})
  if data.get("code") != 0:
    return None
  users = data.get("data", {}).get("user_list", [])
  if users and users[0].get("user_id"):
    return users[0]["user_id"]
  return None


def get_chat_info(token: str, chat_id: str) -> dict[str, Any]:
  """Get chat group info."""
  url = f"{BASE_URL}/im/v1/chats/{chat_id}"
  data = _request(url, token)
  if data.get("code") != 0:
    return {}
  return data.get("data", {})


def get_message(token: str, message_id: str) -> dict[str, Any]:
  """Fetch a message by ID."""
  url = f"{BASE_URL}/im/v1/messages/{message_id}"
  data = _request(url, token)
  if data.get("code") != 0:
    return {}
  return data.get("data", {})


def delete_message(token: str, message_id: str) -> None:
  """Delete a message."""
  url = f"{BASE_URL}/im/v1/messages/{message_id}"
  _request(url, token, method="DELETE")


def create_pin(token: str, message_id: str) -> None:
  """Pin a message in its chat."""
  url = f"{BASE_URL}/im/v1/pins"
  data = _request(url, token, {"message_id": message_id})
  if data.get("code") != 0:
    raise RuntimeError(f"Failed to pin message: {data}")


def delete_pin(token: str, message_id: str) -> None:
  """Unpin a message."""
  url = f"{BASE_URL}/im/v1/pins/{message_id}"
  _request(url, token, method="DELETE")


def list_pins(token: str, chat_id: str) -> list[dict[str, Any]]:
  """List all pinned messages in a chat."""
  url = f"{BASE_URL}/im/v1/pins?chat_id={chat_id}"
  pins: list[dict[str, Any]] = []
  page_token = ""
  for _ in range(10):
    req_url = url + (f"&page_token={page_token}" if page_token else "")
    data = _request(req_url, token)
    if data.get("code") != 0:
      break
    items = data.get("data", {}).get("items", [])
    pins.extend(items)
    if not data.get("data", {}).get("has_more"):
      break
    page_token = data["data"].get("page_token", "")
  return pins


def update_chat_info(token: str, chat_id: str,
                     fields: dict[str, str]) -> None:
  """Update chat group info (e.g. description)."""
  url = f"{BASE_URL}/im/v1/chats/{chat_id}"
  data = _request(url, token, fields, method="PUT")
  if data.get("code") != 0:
    raise RuntimeError(f"Failed to update chat: {data}")


def list_bot_chats(token: str) -> list[dict[str, Any]]:
  """List all chat groups the bot belongs to."""
  url = f"{BASE_URL}/im/v1/chats?user_id_type=open_id"
  chats: list[dict[str, Any]] = []
  page_token = ""
  for _ in range(50):  # safety limit
    req_url = url + (f"&page_token={page_token}" if page_token else "")
    data = _request(req_url, token)
    if data.get("code") != 0:
      break
    items = data.get("data", {}).get("items", [])
    chats.extend(items)
    if not data.get("data", {}).get("has_more"):
      break
    page_token = data["data"].get("page_token", "")
  return chats


def get_chat_members(token: str, chat_id: str) -> list[dict[str, Any]]:
  """Get all members of a chat group."""
  url = f"{BASE_URL}/im/v1/chats/{chat_id}/members?member_id_type=open_id"
  members: list[dict[str, Any]] = []
  page_token = ""
  while True:
    req_url = url + (f"&page_token={page_token}" if page_token else "")
    data = _request(req_url, token)
    if data.get("code") != 0:
      break
    items = data.get("data", {}).get("items", [])
    members.extend(items)
    if not data.get("data", {}).get("has_more"):
      break
    page_token = data["data"].get("page_token", "")
  return members


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

def download_image(token: str, message_id: str, image_key: str) -> str:
  """Download an image from a message. Returns local file path."""
  from ..config import tmp_dir
  url = f"{BASE_URL}/im/v1/messages/{message_id}/resources/{image_key}?type=image"
  img_dir = os.path.join(tmp_dir(), "nemo-images")
  os.makedirs(img_dir, exist_ok=True)
  path = os.path.join(img_dir, f"{image_key}.png")
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  with urllib.request.urlopen(req, timeout=30) as resp:
    with open(path, "wb") as f:
      f.write(resp.read())
  return path


def download_file(token: str, message_id: str, file_key: str,
                  file_name: str = "") -> str:
  """Download a file from a message. Returns local file path."""
  from ..config import tmp_dir
  url = f"{BASE_URL}/im/v1/messages/{message_id}/resources/{file_key}?type=file"
  file_dir = os.path.join(tmp_dir(), "nemo-files")
  os.makedirs(file_dir, exist_ok=True)
  name = os.path.basename(file_name) if file_name else file_key
  path = os.path.join(file_dir, name)
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  with urllib.request.urlopen(req, timeout=60) as resp:
    with open(path, "wb") as f:
      f.write(resp.read())
  return path


def add_reaction(token: str, message_id: str, emoji_type: str) -> None:
  """Add an emoji reaction to a message."""
  url = f"{BASE_URL}/im/v1/messages/{message_id}/reactions"
  try:
    _request(url, token, {"reaction_type": {"emoji_type": emoji_type}})
  except Exception:
    pass
