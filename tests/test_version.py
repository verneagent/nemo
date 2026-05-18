from pathlib import Path

from nemo.version import _version_from_pyproject, get_version, get_version_info


def test_version_from_pyproject_prefers_captain_nemo_project(tmp_path):
  pyproject = tmp_path / "pyproject.toml"
  pyproject.write_text(
    '[project]\nname = "captain-nemo"\nversion = "9.8.7"\n',
    encoding="utf-8",
  )
  package_dir = tmp_path / "nemo"
  package_dir.mkdir()

  assert _version_from_pyproject(package_dir) == "9.8.7"


def test_get_version_prefers_running_source_over_stale_metadata(monkeypatch):
  monkeypatch.setattr(
    "nemo.version._pyproject_info",
    lambda _start: ("0.4.20", Path("/repo/nemo")),
  )
  monkeypatch.setattr(
    "nemo.version._metadata_install_info",
    lambda: ("0.4.0", "installed package", "/site-packages"),
  )
  monkeypatch.setattr(
    "nemo.version.metadata_version", lambda _name: "0.4.0")

  assert get_version() == "0.4.20"


def test_get_version_info_reports_source_path_and_metadata(monkeypatch):
  monkeypatch.setattr(
    "nemo.version._pyproject_info",
    lambda _start: ("0.4.20", Path("/repo/nemo")),
  )
  monkeypatch.setattr(
    "nemo.version._metadata_install_info",
    lambda: ("0.4.0", "installed package", "/site-packages"),
  )

  info = get_version_info()

  assert info.version == "0.4.20"
  assert info.source == "source checkout"
  assert info.path == "/repo/nemo"
  assert info.metadata_version == "0.4.0"
