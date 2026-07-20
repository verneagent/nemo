"""Hermetic test env: strip host-shell vars that nemo reads at runtime.

Dev shells (especially inside Claude Code) export ANTHROPIC_*/CLAUDE_CODE_*
and provider keys; nemo's _build_options passes them through to the SDK
subprocess, so any test that builds options without an explicit patch.dict
silently inherits the host shell (e.g. ANTHROPIC_DEFAULT_OPUS_MODEL=k3
broke test_claude_shell_env_passthrough_overlay). Tests that need a var
set it via mock.patch.dict, which overlays on top of this sanitized env.
"""

import os

import pytest

_STRIP_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_", "NEMO_")
_STRIP_EXACT = {
  "CLAUDECODE",
  "CLAUDE_AGENT_SDK_VERSION",
  "DEEPSEEK_API_KEY",
  "KIMI_API_KEY",
  "BAILIAN_API_KEY",
}


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
  for key in list(os.environ):
    if key.startswith(_STRIP_PREFIXES) or key in _STRIP_EXACT:
      monkeypatch.delenv(key, raising=False)
