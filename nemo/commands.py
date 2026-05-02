"""Built-in agent commands — dispatched before SDK processing."""

from __future__ import annotations

import os
import re
import time


_EFFORT_LEVELS = ("low", "medium", "high", "max")


def _format_model_catalog(catalog) -> str:
  """Render a ModelCatalog as a markdown block grouped by visibility + aliases."""
  lines: list[str] = []
  if catalog.visible:
    lines.append("Available: " + ", ".join(f"`{m}`" for m in catalog.visible))
  if getattr(catalog, "api_only", ()):
    lines.append(
      "API-only (ChatGPT account rejects these): "
      + ", ".join(f"`{m}`" for m in catalog.api_only)
    )
  if catalog.hidden:
    lines.append("Legacy: " + ", ".join(f"`{m}`" for m in catalog.hidden))
  if catalog.aliases:
    pairs = ", ".join(f"`{a}` → `{full}`" for a, full in catalog.aliases.items())
    lines.append("Aliases: " + pairs)
  if getattr(catalog, "note", ""):
    lines.append("Note: " + catalog.note)
  return "\n".join(lines) if lines else "(no models configured)"

# What each effort level actually does per provider.
_EFFORT_DETAIL: dict[str, dict[str, str]] = {
  "claude": {
    "": "high (SDK default)",
    "low": "low effort",
    "medium": "medium effort",
    "high": "high effort",
    "max": "max effort (no token cap)",
  },
  "codex": {
    "": "medium in Codex",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high (Codex tops out at high)",
  },
  "opencode": {
    "": "normal reasoning",
    "low": "lighter prompt guidance",
    "medium": "default",
    "high": "stronger prompt guidance",
    "max": "stronger prompt guidance (max → high)",
  },
}


class AgentContext:
  """Minimal context for command handlers."""

  def __init__(self, model: str, project_dir: str, start_time: float):
    from .agent_factory import AgentProvider
    self.model = model
    self.project_dir = project_dir
    self.start_time = start_time
    self.msg_count = 0
    self.total_cost = 0.0
    self.effort = ""
    self.provider: AgentProvider = "claude"


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
    from .agent_factory import is_model_compatible, model_catalog_for_provider
    catalog = model_catalog_for_provider(ctx.provider, ctx.project_dir)
    listing = _format_model_catalog(catalog)
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      new_model = parts[1].strip()
      if not is_model_compatible(ctx.provider, new_model, ctx.project_dir):
        return True, (
          f"Unknown model `{new_model}` for provider **{ctx.provider}**.\n\n"
          f"{listing}"
        )
      return True, f"__model__:{new_model}"
    return True, (
      f"Current model: **{ctx.model}** (provider **{ctx.provider}**)\n\n"
      f"{listing}\n\nUsage: `/model <name>`"
    )

  # /effort
  if t.startswith("/effort"):
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      arg = parts[1].strip().lower()
      if arg in ("off", "none", "clear", "default"):
        return True, "__effort__:"
      if arg in _EFFORT_LEVELS:
        return True, f"__effort__:{arg}"
      return True, (
        f"Unknown effort level: `{arg}`. "
        f"Use `/effort low|medium|high|max|default`."
      )
    current = ctx.effort or "default"
    detail_map = _EFFORT_DETAIL.get(ctx.provider, {})
    detail = detail_map.get(ctx.effort, "")
    hint = f" — {detail}" if detail else ""
    return True, (
      f"Current effort: **{current}**{hint}\n\n"
      f"Usage: `/effort low|medium|high|max|default`"
    )

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
    if ctx.provider == "claude":
      return True, "Plan usage: [claude.ai/settings/usage](https://claude.ai/settings/usage)"
    if ctx.provider == "opencode":
      return True, "Usage is provider-specific under OpenCode. Run `opencode stats` locally for totals."
    return True, "Usage is provider-specific for this agent. Check the local CLI/account UI for totals."

  # /help
  if t in ("/help", "help", "帮助"):
    return True, (
      "**Agent Commands**\n\n"
      "| Command | Description |\n"
      "|---|---|\n"
      "| `/model` | Show current model |\n"
      "| `/model <name>` | Switch model |\n"
      "| `/effort` | Show current reasoning effort |\n"
      "| `/effort <low\\|medium\\|high\\|max\\|default>` | Set reasoning effort |\n"
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
