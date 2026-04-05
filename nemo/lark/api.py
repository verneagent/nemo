"""Lark IM API client — send, update, download messages."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://open.larksuite.com/open-apis"


def _im_post(url: str, token: str, payload: dict) -> dict:
  """POST to Lark IM API."""
  req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={
      "Content-Type": "application/json",
      "Authorization": f"Bearer {token}",
    },
  )
  with urllib.request.urlopen(req, timeout=30) as resp:
    return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------

def send_card(token: str, chat_id: str, card: dict) -> str:
  """Send an interactive card message. Returns message_id."""
  url = f"{BASE_URL}/im/v1/messages"
  payload = {
    "receive_id": chat_id,
    "msg_type": "interactive",
    "content": json.dumps(card),
  }
  data = None
  for attempt in range(3):
    data = _im_post(f"{url}?receive_id_type=chat_id", token, payload)
    if data.get("code") == 0:
      return data["data"]["message_id"]
    if attempt < 2:
      time.sleep(1)
  raise RuntimeError(f"Failed to send card: {data}")


def update_card(token: str, message_id: str, card: dict) -> None:
  """PATCH an existing card message."""
  url = f"{BASE_URL}/im/v1/messages/{message_id}"
  payload = {"msg_type": "interactive", "content": json.dumps(card)}
  req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={
      "Content-Type": "application/json",
      "Authorization": f"Bearer {token}",
    },
    method="PATCH",
  )
  try:
    with urllib.request.urlopen(req, timeout=30) as resp:
      data = json.loads(resp.read())
  except urllib.error.HTTPError as e:
    try:
      data = json.loads(e.read())
    except Exception:
      raise e
  if data.get("code") != 0:
    raise RuntimeError(f"Failed to update card: {data}")


def send_markdown(token: str, chat_id: str, content: str) -> str:
  """Send a Card V2 markdown message. Returns message_id."""
  from ..cards import build_markdown_card
  card = build_markdown_card(content)
  return send_card(token, chat_id, card)


# ---------------------------------------------------------------------------
# User & chat info
# ---------------------------------------------------------------------------

def get_bot_info(token: str) -> dict:
  """Get bot's own info (open_id, etc.)."""
  url = f"{BASE_URL}/bot/v3/info"
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read())
  if data.get("code") != 0:
    raise RuntimeError(f"Bot info error: {data}")
  return data.get("data", {}).get("bot", {})


def lookup_open_id_by_email(token: str, email: str) -> str | None:
  """Resolve email to Lark open_id."""
  url = f"{BASE_URL}/contact/v3/users/batch_get_id"
  payload = {"emails": [email]}
  data = _im_post(f"{url}?user_id_type=open_id", token, payload)
  if data.get("code") != 0:
    return None
  users = data.get("data", {}).get("user_list", [])
  if users and users[0].get("user_id"):
    return users[0]["user_id"]
  return None


def get_chat_info(token: str, chat_id: str) -> dict:
  """Get chat group info."""
  url = f"{BASE_URL}/im/v1/chats/{chat_id}"
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read())
  if data.get("code") != 0:
    return {}
  return data.get("data", {})


def get_chat_members(token: str, chat_id: str) -> list[dict]:
  """Get all members of a chat group."""
  url = f"{BASE_URL}/im/v1/chats/{chat_id}/members?member_id_type=open_id"
  members: list[dict] = []
  page_token = ""
  while True:
    req_url = url + (f"&page_token={page_token}" if page_token else "")
    req = urllib.request.Request(req_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
      data = json.loads(resp.read())
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
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  img_dir = os.path.join(tmp_dir(), "handoff-images")
  os.makedirs(img_dir, exist_ok=True)
  path = os.path.join(img_dir, f"{image_key}.png")
  with urllib.request.urlopen(req, timeout=30) as resp:
    with open(path, "wb") as f:
      f.write(resp.read())
  return path


def download_file(token: str, message_id: str, file_key: str, file_name: str = "") -> str:
  """Download a file from a message. Returns local file path."""
  from ..config import tmp_dir
  url = f"{BASE_URL}/im/v1/messages/{message_id}/resources/{file_key}?type=file"
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  file_dir = os.path.join(tmp_dir(), "handoff-files")
  os.makedirs(file_dir, exist_ok=True)
  name = file_name or file_key
  path = os.path.join(file_dir, name)
  with urllib.request.urlopen(req, timeout=60) as resp:
    with open(path, "wb") as f:
      f.write(resp.read())
  return path


def add_reaction(token: str, message_id: str, emoji_type: str) -> None:
  """Add an emoji reaction to a message."""
  url = f"{BASE_URL}/im/v1/messages/{message_id}/reactions"
  payload = {"reaction_type": {"emoji_type": emoji_type}}
  try:
    _im_post(url, token, payload)
  except Exception:
    pass
