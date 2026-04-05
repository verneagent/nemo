"""Configuration management — loads credentials from ~/.handoff/config.json.

Nemo reuses the same config format as handoff for compatibility.
"""

from __future__ import annotations

import json
import os

CONFIG_FILE = os.path.expanduser("~/.handoff/config.json")
DB_BASE = os.path.expanduser("~/.handoff/projects")
TMP_DIR = os.environ.get("HANDOFF_TMP_DIR", "/tmp/handoff")


def load_config(profile: str | None = None) -> dict:
  """Load the full config dict for a profile."""
  if profile and profile != "default":
    path = os.path.expanduser(f"~/.handoff/profiles/{profile}.json")
  else:
    path = CONFIG_FILE
  if not os.path.isfile(path):
    return {}
  with open(path) as f:
    return json.load(f)


def load_credentials(profile: str | None = None) -> dict | None:
  """Load app_id, app_secret, email from config."""
  cfg = load_config(profile)
  app_id = cfg.get("app_id")
  app_secret = cfg.get("app_secret")
  if not app_id or not app_secret:
    return None
  return {
    "app_id": app_id,
    "app_secret": app_secret,
    "email": cfg.get("email", ""),
  }


def load_worker_url(profile: str | None = None) -> str:
  """Load the Cloudflare Worker URL."""
  cfg = load_config(profile)
  return cfg.get("worker_url", "")


def load_api_key(profile: str | None = None) -> str:
  """Load the Worker API key."""
  cfg = load_config(profile)
  return cfg.get("worker_api_key", "")


def resolve_profile(explicit: str | None = None) -> str:
  """Resolve the active profile name."""
  if explicit:
    return explicit
  env = os.environ.get("HANDOFF_PROFILE")
  if env:
    return env
  default_file = os.path.expanduser("~/.handoff/default_profile")
  if os.path.isfile(default_file):
    with open(default_file) as f:
      return f.read().strip() or "default"
  return "default"


def tmp_dir() -> str:
  """Return the handoff temp directory, creating it if needed."""
  os.makedirs(TMP_DIR, exist_ok=True)
  return TMP_DIR
