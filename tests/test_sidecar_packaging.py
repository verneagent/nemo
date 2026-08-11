"""Wheel-packaging guard: every runtime sidecar file must ship in the wheel.

nemo/opencode_agent.py spawns ``node run_turn.mjs`` from the *installed*
package directory, so any ``.mjs`` module it imports must be declared in
``[tool.setuptools.package-data]``. Local editable installs read the repo
directly and hide gaps; a pipx/wheel install crashes with
``ERR_MODULE_NOT_FOUND`` the first time opencode (or codex) runs. This test
keeps the package-data list in sync with the files the sidecars actually use.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REL_IMPORT_RE = re.compile(r"""from\s+"(\./[A-Za-z0-9_./-]+\.mjs)"\s*""")


def _package_data() -> list[str]:
  with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
    data = tomllib.load(f)
  return list(data["tool"]["setuptools"]["package-data"]["nemo"])


def test_every_sidecar_runtime_file_is_packaged() -> None:
  packaged = set(_package_data())
  for dirname in ("codex_sidecar", "opencode_sidecar"):
    sidecar_dir = _REPO_ROOT / "nemo" / dirname
    for path in sorted(sidecar_dir.glob("*.mjs")):
      if path.name.startswith("test_"):
        continue
      assert f"{dirname}/{path.name}" in packaged, (
        f"{dirname}/{path.name} runs on the daemon host but is not in "
        "[tool.setuptools.package-data] — wheel installs crash with "
        "ERR_MODULE_NOT_FOUND (see opencode_sidecar events.mjs incident)"
      )


def test_every_relative_sidecar_import_is_packaged() -> None:
  packaged = set(_package_data())
  for dirname in ("codex_sidecar", "opencode_sidecar"):
    sidecar_dir = _REPO_ROOT / "nemo" / dirname
    for path in sorted(sidecar_dir.glob("*.mjs")):
      for imported in _REL_IMPORT_RE.findall(path.read_text()):
        assert f"{dirname}/{imported.lstrip('./')}" in packaged, (
          f"{dirname}/{path.name} imports {imported}, which is not in "
          "[tool.setuptools.package-data]"
        )
