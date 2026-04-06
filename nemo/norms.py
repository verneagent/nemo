"""Group norms — persistent rules injected into the system prompt."""

from __future__ import annotations

from .group_config import load_config, save_config


def get_norms(token: str, chat_id: str) -> dict[str, str]:
  """Get all norms from group config."""
  config = load_config(token, chat_id)
  return dict(config.get("rules", {}))


def add_norm(token: str, chat_id: str, name: str, text: str) -> None:
  """Add or update a norm."""
  config = load_config(token, chat_id)
  rules = config.get("rules", {})
  rules[name] = text
  config["rules"] = rules
  save_config(token, chat_id, config)


def remove_norm(token: str, chat_id: str, name: str) -> bool:
  """Remove a norm. Returns True if found."""
  config = load_config(token, chat_id)
  rules = config.get("rules", {})
  if name not in rules:
    return False
  del rules[name]
  config["rules"] = rules
  save_config(token, chat_id, config)
  return True


def format_norms_prompt(norms: dict[str, str]) -> str:
  """Format norms as text for system prompt injection."""
  if not norms:
    return ""
  lines = ["Group norms:"]
  for name, text in norms.items():
    lines.append(f"- {name}: {text}")
  return "\n".join(lines)
