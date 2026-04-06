"""Preflight checks — verify configuration before starting."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_preflight(credentials: dict[str, str], chat_id: str = "") -> list[str]:
  """Run all checks. Returns list of error strings (empty = all good)."""
  errors: list[str] = []

  # Check 1: credentials exist
  app_id = credentials.get("app_id", "")
  app_secret = credentials.get("app_secret", "")
  if not app_id:
    errors.append("Missing app_id in credentials")
  if not app_secret:
    errors.append("Missing app_secret in credentials")
  if errors:
    return errors  # Can't continue without credentials

  # Check 2: can get token
  from .lark.auth import get_token
  try:
    token = get_token(app_id, app_secret)
  except Exception as e:
    errors.append(f"Token acquisition failed: {e}")
    return errors

  # Check 3: can get bot info
  from .lark import api as lark_api
  try:
    bot_info = lark_api.get_bot_info(token)
    if not bot_info.get("open_id"):
      errors.append("Bot info returned no open_id")
  except Exception as e:
    errors.append(f"Bot info check failed: {e}")

  # Check 4: if chat_id provided, can access chat
  if chat_id:
    try:
      lark_api.get_chat_info(token, chat_id)
    except Exception as e:
      errors.append(f"Chat access check failed for {chat_id}: {e}")

  return errors
