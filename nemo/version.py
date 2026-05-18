"""Version helpers for the running Nemo code."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import (
  PackageNotFoundError,
  distribution,
  version as metadata_version,
)
from pathlib import Path


@dataclass(frozen=True)
class VersionInfo:
  version: str
  source: str
  path: str
  metadata_version: str


def _pyproject_info(start: Path) -> tuple[str, Path] | None:
  """Read the version/root from the nearest ancestor pyproject.toml."""
  import tomllib

  for directory in (start, *start.parents):
    pyproject = directory / "pyproject.toml"
    if not pyproject.is_file():
      continue
    try:
      data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
      return None
    project = data.get("project")
    if not isinstance(project, dict):
      return None
    name = project.get("name")
    version = project.get("version")
    if name == "captain-nemo" and isinstance(version, str):
      return version, directory
    return None
  return None


def _version_from_pyproject(start: Path) -> str:
  """Read the project version from the nearest ancestor pyproject.toml."""
  info = _pyproject_info(start)
  return info[0] if info is not None else ""


def _metadata_install_info() -> tuple[str, str, str]:
  """Return (version, source, path) from installed package metadata."""
  try:
    dist = distribution("captain-nemo")
  except PackageNotFoundError:
    return "unknown", "not installed", ""

  source = "installed package"
  path = str(dist.locate_file(""))
  try:
    direct_url = dist.read_text("direct_url.json")
  except OSError:
    direct_url = None
  if direct_url:
    import json
    try:
      data = json.loads(direct_url)
    except json.JSONDecodeError:
      data = {}
    if isinstance(data, dict):
      dir_info = data.get("dir_info")
      if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        source = "editable install"
      url = data.get("url")
      if isinstance(url, str) and url.startswith("file://"):
        from urllib.parse import unquote, urlparse
        parsed = urlparse(url)
        if parsed.path:
          path = unquote(parsed.path)
  try:
    return metadata_version("captain-nemo"), source, path
  except PackageNotFoundError:
    return "unknown", source, path


def get_version_info() -> VersionInfo:
  """Return version plus where the running Nemo code appears to come from."""
  metadata_ver, metadata_source, metadata_path = _metadata_install_info()
  source_info = _pyproject_info(Path(__file__).resolve().parent)
  if source_info is not None:
    source_version, root = source_info
    return VersionInfo(
      version=source_version,
      source="source checkout",
      path=str(root),
      metadata_version=metadata_ver,
    )
  return VersionInfo(
    version=metadata_ver,
    source=metadata_source,
    path=metadata_path,
    metadata_version=metadata_ver,
  )


def get_version() -> str:
  """Return the version for the checked-out code, falling back to metadata.

  Development installs can leave stale distribution metadata behind. The
  running module path is the more useful source of truth for start cards and
  `/version`, because those describe the code that is actually executing.
  """
  return get_version_info().version
