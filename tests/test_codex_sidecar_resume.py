"""Run the codex sidecar's JS-level resume-fallback tests.

The fallback decision lives in nemo/codex_sidecar/resume.mjs (JS so it
can be wired into the @openai/codex-sdk call site directly), so the
authoritative tests are also JS. This wrapper shells out to node so the
checks run in pytest like everything else; if node is missing we skip.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest


_SIDE_CAR_DIR = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  "nemo",
  "codex_sidecar",
)
_TEST_SCRIPT = os.path.join(_SIDE_CAR_DIR, "test_resume.mjs")


def test_codex_sidecar_resume_fallback():
  if not os.path.isfile(_TEST_SCRIPT):
    pytest.fail(f"missing JS test file: {_TEST_SCRIPT}")
  node = shutil.which("node")
  if not node:
    pytest.skip("node not on PATH — sidecar JS tests can't run")

  result = subprocess.run(
    [node, _TEST_SCRIPT],
    cwd=_SIDE_CAR_DIR,
    capture_output=True,
    text=True,
    timeout=20,
  )
  combined = (result.stdout or "") + (result.stderr or "")
  assert result.returncode == 0, (
    f"node test_resume.mjs failed (rc={result.returncode}):\n{combined}"
  )
  assert "OK:" in result.stdout, f"unexpected stdout:\n{combined}"
