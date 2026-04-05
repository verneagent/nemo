"""Tests for nemo.config."""

import json
import os
import tempfile
from unittest import mock

from nemo.config import load_config, load_credentials, tmp_dir, CONFIG_FILE, _LEGACY_CONFIG


def test_load_config_missing_file():
  with mock.patch("nemo.config.CONFIG_FILE", "/nonexistent/config.json"):
    with mock.patch("nemo.config._LEGACY_CONFIG", "/nonexistent/legacy.json"):
      assert load_config() == {}


def test_load_config_legacy_fallback(tmp_path):
  """Falls back to legacy ~/.handoff/config.json."""
  cfg = {"app_id": "legacy_app", "app_secret": "legacy_secret"}
  legacy = tmp_path / "legacy.json"
  legacy.write_text(json.dumps(cfg))
  with mock.patch("nemo.config.CONFIG_FILE", "/nonexistent/config.json"):
    with mock.patch("nemo.config._LEGACY_CONFIG", str(legacy)):
      result = load_config()
  assert result["app_id"] == "legacy_app"


def test_load_config_nemo_takes_priority(tmp_path):
  """~/.nemo/config.json takes priority over legacy path."""
  nemo_cfg = {"app_id": "nemo_app", "app_secret": "s"}
  legacy_cfg = {"app_id": "legacy_app", "app_secret": "s"}
  nemo = tmp_path / "nemo.json"
  legacy = tmp_path / "legacy.json"
  nemo.write_text(json.dumps(nemo_cfg))
  legacy.write_text(json.dumps(legacy_cfg))
  with mock.patch("nemo.config.CONFIG_FILE", str(nemo)):
    with mock.patch("nemo.config._LEGACY_CONFIG", str(legacy)):
      result = load_config()
  assert result["app_id"] == "nemo_app"


def test_load_config_reads_json(tmp_path):
  cfg = {"app_id": "cli_test", "app_secret": "secret123", "email": "a@b.com"}
  path = tmp_path / "config.json"
  path.write_text(json.dumps(cfg))
  with mock.patch("nemo.config.CONFIG_FILE", str(path)):
    result = load_config()
  assert result == cfg


def test_load_credentials_success(tmp_path):
  cfg = {"app_id": "cli_test", "app_secret": "secret123", "email": "a@b.com"}
  path = tmp_path / "config.json"
  path.write_text(json.dumps(cfg))
  with mock.patch("nemo.config.CONFIG_FILE", str(path)):
    creds = load_credentials()
  assert creds is not None
  assert creds["app_id"] == "cli_test"
  assert creds["app_secret"] == "secret123"
  assert creds["email"] == "a@b.com"


def test_load_credentials_missing_secret(tmp_path):
  cfg = {"app_id": "cli_test"}
  path = tmp_path / "config.json"
  path.write_text(json.dumps(cfg))
  with mock.patch("nemo.config.CONFIG_FILE", str(path)):
    assert load_credentials() is None


def test_load_credentials_no_email(tmp_path):
  cfg = {"app_id": "cli_test", "app_secret": "secret123"}
  path = tmp_path / "config.json"
  path.write_text(json.dumps(cfg))
  with mock.patch("nemo.config.CONFIG_FILE", str(path)):
    creds = load_credentials()
  assert creds is not None
  assert creds["email"] == ""


def test_tmp_dir_creates_directory(tmp_path):
  d = str(tmp_path / "nemo-tmp")
  with mock.patch("nemo.config.TMP_DIR", d):
    result = tmp_dir()
  assert result == d
  assert os.path.isdir(d)
