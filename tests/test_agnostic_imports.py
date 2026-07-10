"""Guardrail: agent-agnostic modules must stay agent-agnostic.

Nemo's design (AGENTS.md) keeps orchestration and the turn-event vocabulary
free of any concrete coding-agent: ``agent.py`` is channel- and agent-agnostic,
and ``turn.py`` holds only the typed events every adapter emits plus the shared
usage schema. SDK-specific logic belongs in the concrete adapters
(``claude_agent`` / ``claude_turn`` / ``codex_agent`` / ``opencode_agent`` and
the SDK plumbing in ``sdk_thread``).

This once leaked: the Claude-SDK turn consumer (``run_turn`` / ``_single_turn``,
which ``import claude_agent_sdk``) lived inside the neutrally-named ``turn.py``.
This test statically forbids that class of leak so it cannot regress: it parses
each agnostic module's AST and asserts it imports NONE of the forbidden
agent-specific modules — anywhere, including deferred/inline imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent / "nemo"

# Modules that MUST NOT depend on any specific coding agent.
AGNOSTIC_MODULES = (
  "turn.py",         # agent-agnostic event vocabulary + usage schema
  "agent.py",        # channel- and agent-agnostic orchestration
  "channel.py",      # abstract channel boundary
  "coding_agent.py",  # abstract coding-agent boundary
)

# Import targets that make a module agent-specific. Includes the vendor SDKs
# and every concrete adapter / SDK-plumbing module. ``agent_factory`` is the
# ONE sanctioned place that maps an agent kind to its adapter, so it is not
# listed here (and is itself not agnostic).
FORBIDDEN = frozenset({
  # vendor SDKs
  "claude_agent_sdk",
  "codex_sdk",
  # concrete adapters + SDK plumbing
  "claude_agent",
  "claude_turn",
  "claude_cli_agent",
  "codex_agent",
  "opencode_agent",
  "sdk_thread",
})


def _imported_bases(source: str) -> set[str]:
  """Every top-level module base name imported by ``source`` (any nesting).

  ``import a.b`` -> "a"; ``from a.b import c`` -> "a"; ``from .x import y`` ->
  "x" (relative imports resolve to the sibling module ``x``).
  """
  bases: set[str] = set()
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        bases.add(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
      if node.level and node.module:
        # relative: `from .claude_turn import x` -> base "claude_turn"
        bases.add(node.module.split(".")[0])
      elif node.level and not node.module:
        # `from . import claude_turn` -> the imported *names* are the modules
        for alias in node.names:
          bases.add(alias.name.split(".")[0])
      elif node.module:
        bases.add(node.module.split(".")[0])
  return bases


@pytest.mark.parametrize("module", AGNOSTIC_MODULES)
def test_agnostic_module_has_no_agent_specific_imports(module):
  path = _PKG / module
  assert path.exists(), f"agnostic module not found: {path}"
  leaked = _imported_bases(path.read_text()) & FORBIDDEN
  assert not leaked, (
    f"{module} is agent-agnostic but imports agent-specific module(s) "
    f"{sorted(leaked)}. Move that logic into the concrete adapter "
    f"(claude_turn / claude_agent / codex_agent / opencode_agent) — "
    f"agnostic modules speak only the shared Channel/CodingAgent/turn-event "
    f"abstractions. See tests/test_agnostic_imports.py."
  )
