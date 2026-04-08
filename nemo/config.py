"""Configuration management — loads credentials from ~/.nemo/<profile>.json.

Profile files live directly in ~/.nemo/:
  ~/.nemo/default.json   (used when --profile is omitted)
  ~/.nemo/alice.json
  ~/.nemo/bob.json
"""

from __future__ import annotations

import json
import os

from .types import JsonObject

CONFIG_DIR = os.path.expanduser("~/.nemo")
DB_BASE = os.path.join(CONFIG_DIR, "projects")
TMP_DIR = os.environ.get("NEMO_TMP_DIR", "/tmp/nemo")
RELAY_URL = os.environ.get("NEMO_RELAY_URL", "")
RELAY_API_KEY = os.environ.get("NEMO_RELAY_API_KEY", "")

# Active profile — set once at startup via set_profile()
_profile: str = "default"


def set_profile(name: str) -> None:
  """Set the active profile. Called once from __main__."""
  global _profile
  _profile = name


def profile_path(name: str | None = None) -> str:
  """Return the path to a profile config file."""
  return os.path.join(CONFIG_DIR, f"{name or _profile}.json")


def load_config() -> JsonObject:
  """Load the active profile's config dict."""
  path = profile_path()
  if os.path.isfile(path):
    with open(path) as f:
      return json.load(f)
  return {}


def load_relay_config() -> tuple[str, str]:
  """Load relay URL and API key from env or config.

  Returns (relay_url, api_key). Either may be empty if not configured.
  """
  if RELAY_URL:
    return RELAY_URL, RELAY_API_KEY
  cfg = load_config()
  return cfg.get("relay_url", ""), cfg.get("relay_api_key", "")


def load_credentials() -> dict[str, str] | None:
  """Load app_id, app_secret, email from the active profile.

  Returns dict with keys {app_id, app_secret, email} or None if missing.
  """
  cfg = load_config()
  app_id = cfg.get("app_id")
  app_secret = cfg.get("app_secret")
  if not app_id or not app_secret:
    return None
  return {
    "app_id": app_id,
    "app_secret": app_secret,
    "email": cfg.get("email", ""),
  }


def tmp_dir() -> str:
  """Return the nemo temp directory, creating it if needed."""
  os.makedirs(TMP_DIR, exist_ok=True)
  return TMP_DIR
