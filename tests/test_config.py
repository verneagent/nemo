"""Tests for nemo.config — multi-profile configuration."""

import json
import os
from unittest import mock

from nemo.config import (
  load_config, load_credentials, tmp_dir, set_profile, profile_path,
)


def test_profile_path_default():
  set_profile("default")
  assert profile_path().endswith("/default.json")


def test_profile_path_custom():
  set_profile("alice")
  assert profile_path().endswith("/alice.json")
  assert profile_path("bob").endswith("/bob.json")
  set_profile("default")  # reset


def test_load_config_missing_file():
  set_profile("nonexistent_test")
  with mock.patch("nemo.config.CONFIG_DIR", "/nonexistent"):
    assert load_config() == {}
  set_profile("default")


def test_load_config_reads_json(tmp_path):
  cfg = {"app_id": "cli_test", "app_secret": "secret123", "email": "a@b.com"}
  path = tmp_path / "myprofile.json"
  path.write_text(json.dumps(cfg))
  set_profile("myprofile")
  with mock.patch("nemo.config.CONFIG_DIR", str(tmp_path)):
    result = load_config()
  assert result == cfg
  set_profile("default")


def test_load_credentials_success(tmp_path):
  cfg = {"app_id": "cli_test", "app_secret": "secret123", "email": "a@b.com"}
  path = tmp_path / "default.json"
  path.write_text(json.dumps(cfg))
  set_profile("default")
  with mock.patch("nemo.config.CONFIG_DIR", str(tmp_path)):
    creds = load_credentials()
  assert creds is not None
  assert creds["app_id"] == "cli_test"
  assert creds["app_secret"] == "secret123"
  assert creds["email"] == "a@b.com"


def test_load_credentials_missing_secret(tmp_path):
  cfg = {"app_id": "cli_test"}
  path = tmp_path / "default.json"
  path.write_text(json.dumps(cfg))
  set_profile("default")
  with mock.patch("nemo.config.CONFIG_DIR", str(tmp_path)):
    assert load_credentials() is None


def test_load_credentials_no_email(tmp_path):
  cfg = {"app_id": "cli_test", "app_secret": "secret123"}
  path = tmp_path / "default.json"
  path.write_text(json.dumps(cfg))
  set_profile("default")
  with mock.patch("nemo.config.CONFIG_DIR", str(tmp_path)):
    creds = load_credentials()
  assert creds is not None
  assert creds["email"] == ""


def test_tmp_dir_creates_directory(tmp_path):
  d = str(tmp_path / "nemo-tmp")
  with mock.patch("nemo.config.TMP_DIR", d):
    result = tmp_dir()
  assert result == d
  assert os.path.isdir(d)
