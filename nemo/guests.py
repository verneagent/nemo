"""Guest management — control who can interact with the agent in group chats.

Roles:
- operator: the primary user (from config email)
- coowner: full access, can approve permissions
- guest: can send messages to agent

Guests are stored in the group config pinned card.
"""

from __future__ import annotations

from . import group_config


def list_guests(token: str, chat_id: str) -> list[dict[str, str]]:
  """List all guests from group config."""
  config = group_config.load_config(token, chat_id)
  return config.get("guests", [])


def add_guest(
  token: str, chat_id: str, open_id: str, name: str = "", role: str = "guest",
) -> None:
  """Add a guest to the group config.

  If the open_id already exists, update name and role instead of duplicating.
  """
  if role not in ("guest", "coowner"):
    raise ValueError(f"Invalid role: {role!r} (must be 'guest' or 'coowner')")

  config = group_config.load_config(token, chat_id)
  guests: list[dict[str, str]] = config.get("guests", [])

  # Update existing entry if present
  for g in guests:
    if g["open_id"] == open_id:
      g["name"] = name or g.get("name", "")
      g["role"] = role
      config["guests"] = guests
      group_config.save_config(token, chat_id, config)
      return

  guests.append({"open_id": open_id, "name": name, "role": role})
  config["guests"] = guests
  group_config.save_config(token, chat_id, config)


def remove_guest(token: str, chat_id: str, open_id: str) -> None:
  """Remove a guest from the group config."""
  config = group_config.load_config(token, chat_id)
  guests: list[dict[str, str]] = config.get("guests", [])
  config["guests"] = [g for g in guests if g["open_id"] != open_id]
  group_config.save_config(token, chat_id, config)


def get_member_roles(token: str, chat_id: str) -> dict[str, str]:
  """Get a mapping of open_id -> role for all guests."""
  guests = list_guests(token, chat_id)
  return {g["open_id"]: g.get("role", "guest") for g in guests}


def is_authorized_sender(
  open_id: str, operator_open_id: str, member_roles: dict[str, str],
) -> bool:
  """Check if a sender is authorized (operator, coowner, or guest)."""
  if open_id == operator_open_id:
    return True
  return open_id in member_roles
