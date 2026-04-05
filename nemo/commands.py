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
  if t.startswith("/model") or t.startswith("model "):
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      new_model = parts[1].strip()
      return True, f"__model__:{new_model}"
    return True, f"Current model: **{ctx.model}**\n\nUsage: `/model claude-sonnet-4-6`"

  # /esc
  if t in ("/esc", "esc", "cancel", "取消"):
    return True, "__esc__"

  # /cd
  if t.startswith("/cd ") or t.startswith("cd "):
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
      "| `/help` | This help |\n"
      "| `handback` | Stop agent |\n"
      "| `autoapprove on/off` | Toggle auto-approve |"
    )

  # autoapprove
  if re.match(r"(auto[\s\-]*approve|autoapprove)\s+(on|off)", t):
    enabled = "on" in t.split()[-1]
    return True, f"__autoapprove__:{'on' if enabled else 'off'}"

  # handback
  if t in ("handback", "hand back", "handback dissolve", "hand back dissolve"):
    dissolve = "dissolve" in t
    return True, f"__handback__:{'dissolve' if dissolve else 'normal'}"

  return False, None
