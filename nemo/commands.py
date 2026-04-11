"""Built-in agent commands — dispatched before SDK processing."""

from __future__ import annotations

import os
import re
import time


class AgentContext:
  """Minimal context for command handlers."""

  def __init__(self, model: str, project_dir: str, start_time: float):
    self.model = model
    self.project_dir = project_dir
    self.start_time = start_time
    self.msg_count = 0
    self.total_cost = 0.0


# Each handler returns (handled: bool, response_text: str | None).
# If handled=True the main loop skips SDK processing.
# Special responses starting with __ are action codes handled by agent.py.


def try_dispatch(text: str, ctx: AgentContext) -> tuple[bool, str | None]:
  """Check text against all commands. Returns (handled, response)."""
  t = text.strip().lower()

  # /clear
  if t in ("/clear", "clear", "清空", "重置"):
    return True, "__clear__"

  # /model
  if t.startswith("/model"):
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      new_model = parts[1].strip()
      return True, f"__model__:{new_model}"
    return True, f"Current model: **{ctx.model}**\n\nUsage: `/model claude-sonnet-4-6`"

  # /esc
  if t in ("/esc", "esc", "cancel", "取消"):
    return True, "__esc__"

  # /cd
  if t.startswith("/cd "):
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      new_dir = os.path.expanduser(parts[1].strip())
      if os.path.isdir(new_dir):
        return True, f"__cd__:{os.path.abspath(new_dir)}"
      return True, f"Directory not found: `{new_dir}`"
    return True, None

  # /ping
  if t in ("/ping", "ping"):
    uptime = int(time.time() - ctx.start_time)
    h, m = divmod(uptime, 3600)
    mins, secs = divmod(m, 60)
    return True, (
      f"🏓 Pong!\n"
      f"- Model: **{ctx.model}**\n"
      f"- Uptime: {h}h {mins}m {secs}s\n"
      f"- Messages: {ctx.msg_count}\n"
      f"- Cost: ${ctx.total_cost:.4f}\n"
      f"- CWD: `{ctx.project_dir}`"
    )

  # /cost
  if t in ("/cost", "cost"):
    return True, (
      f"💰 Session Cost\n"
      f"- Total: **${ctx.total_cost:.4f}**\n"
      f"- Messages: {ctx.msg_count}\n"
      f"- Model: {ctx.model}"
    )

  # /usage
  if t in ("/usage", "usage"):
    return True, "Plan usage: [claude.ai/settings/usage](https://claude.ai/settings/usage)"

  # /help
  if t in ("/help", "help", "帮助"):
    return True, (
      "**Agent Commands**\n\n"
      "| Command | Description |\n"
      "|---|---|\n"
      "| `/model` | Show current model |\n"
      "| `/model <name>` | Switch model |\n"
      "| `/clear` | Reset conversation |\n"
      "| `/cd <dir>` | Change working directory |\n"
      "| `/esc` | Cancel current operation |\n"
      "| `/ping` | Status check |\n"
      "| `/cost` | Session API cost |\n"
      "| `/usage` | Plan usage limits |\n"
      "| `/mention` | Toggle @mention requirement |\n"
      "| `/name <name>` | Rename this group |\n"
      "| `/norm` | Manage group norms |\n"
      "| `/guest` | Manage guests |\n"
      "| `/diag` | Run diagnostics |\n"
      "| `/exit` | Stop agent, keep group |\n"
      "| `/dissolve` | Stop agent, dissolve group |\n"
      "| `/help` | This help |\n"
      "| `/autoapprove` | Toggle auto-approve |"
    )

  # /norm
  if t.startswith("/norm"):
    parts = text.strip().split(None, 3)
    if len(parts) >= 2:
      sub = parts[1].lower()
      if sub == "list":
        return True, "__norm_list__"
      if sub == "add" and len(parts) >= 4:
        return True, f"__norm_add__:{parts[2]}:{parts[3]}"
      if sub == "remove" and len(parts) >= 3:
        return True, f"__norm_remove__:{parts[2]}"
    return True, (
      "**Norm Commands**\n\n"
      "| Command | Description |\n"
      "|---|---|\n"
      "| `/norm add <name> <text>` | Add or update a norm |\n"
      "| `/norm remove <name>` | Remove a norm |\n"
      "| `/norm list` | List all norms |"
    )

  # /mention
  if t in ("/mention", "mention"):
    return True, "__mention_toggle__"
  if t in ("/mention on", "mention on"):
    return True, "__mention__:on"
  if t in ("/mention off", "mention off"):
    return True, "__mention__:off"

  # /name <new name>
  if t.startswith("/name"):
    parts = text.strip().split(None, 1)
    if len(parts) < 2 or not parts[1].strip():
      return True, "Usage: `/name <new group name>`"
    new_name = parts[1].strip()
    return True, f"__name__:{new_name}"

  # /diag
  if t in ("/diag", "diag"):
    return True, "__diag__"

  # /autoapprove
  if t in ("/autoapprove", "autoapprove"):
    return True, "__autoapprove_toggle__"
  if re.match(r"/?(?:auto[\s\-]*approve|autoapprove)\s+(on|off)", t):
    enabled = "on" in t.split()[-1]
    return True, f"__autoapprove__:{'on' if enabled else 'off'}"

  # /guest
  if t.startswith("/guest"):
    parts = text.strip().split()
    if len(parts) < 2:
      return True, (
        "**Guest Commands**\n\n"
        "| Command | Description |\n"
        "|---|---|\n"
        "| `/guest list` | List all guests |\n"
        "| `/guest add <name> [coowner]` | Add a guest (optionally as coowner) |\n"
        "| `/guest add all [coowner]` | Add every member (except operator) |\n"
        "| `/guest remove <name>` | Remove a guest |"
      )
    sub = parts[1].strip().lower()
    if sub == "list":
      return True, "__guest_list__"
    if sub == "add" and len(parts) >= 3:
      name = parts[2].strip()
      role = "guest"
      if len(parts) >= 4 and parts[3].strip().lower() == "coowner":
        role = "coowner"
      if name.lower() == "all":
        return True, f"__guest_add_all__:{role}"
      return True, f"__guest_add__:{role}:{name}"
    if sub == "remove" and len(parts) >= 3:
      name = parts[2].strip()
      return True, f"__guest_remove__:{name}"
    return True, "Usage: `/guest list`, `/guest add <name>`, `/guest remove <name>`"

  # /dissolve — stop agent, dissolve group (check before /exit to avoid prefix match)
  if t in ("/dissolve", "dissolve"):
    return True, "__dissolve__"

  # /exit — stop agent, keep group
  if t in ("/exit", "exit"):
    return True, "__exit__"

  return False, None


# Commands that require SDK restart — NOT safe during a turn.
_NEEDS_SDK = ("__clear__", "__esc__", "__model__:", "__cd__:")


def is_inline_safe(response: str | None) -> bool:
  """Check if a command response can be handled during an active turn.

  Returns True for commands that don't interact with the SDK client:
  /ping, /cost, /help, /mention, /autoapprove, /norm, /guest, /diag, etc.
  Returns False for /clear, /esc, /model, /cd (need SDK restart).
  """
  if not response:
    return False
  return not any(response.startswith(p) for p in _NEEDS_SDK)
