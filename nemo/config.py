"""Configuration management — loads credentials from ~/.nemo/config.json."""

from __future__ import annotations

import json
import os
from typing import Any

CONFIG_DIR = os.path.expanduser("~/.nemo")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
# Fallback to legacy handoff config for migration
_LEGACY_CONFIG = os.path.expanduser("~/.handoff/config.json")
DB_BASE = os.path.join(CONFIG_DIR, "projects")
TMP_DIR = os.environ.get("NEMO_TMP_DIR", "/tmp/nemo")


def load_config() -> dict[str, Any]:
  """Load the full config dict. Falls back to legacy ~/.handoff/ path."""
  for path in (CONFIG_FILE, _LEGACY_CONFIG):
    if os.path.isfile(path):
      with open(path) as f:
        return json.load(f)
  return {}


def load_credentials() -> dict[str, str] | None:
  """Load app_id, app_secret, email from config.

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
