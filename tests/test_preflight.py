"""Tests for nemo.preflight — startup verification checks."""

from unittest.mock import patch, MagicMock
from nemo.preflight import run_preflight


def test_missing_app_id():
  errors = run_preflight({"app_secret": "secret"})
  assert any("app_id" in e for e in errors)


def test_missing_app_secret():
  errors = run_preflight({"app_id": "id"})
  assert any("app_secret" in e for e in errors)


def test_missing_both():
  errors = run_preflight({})
  assert len(errors) == 2


@patch("nemo.lark.auth.get_token", side_effect=RuntimeError("bad creds"))
def test_token_failure(mock_token):
  errors = run_preflight({"app_id": "id", "app_secret": "secret"})
  assert any("Token" in e for e in errors)


@patch("nemo.lark.api.get_bot_info", side_effect=RuntimeError("no bot"))
@patch("nemo.lark.auth.get_token", return_value="tok123")
def test_bot_info_failure(mock_token, mock_bot):
  errors = run_preflight({"app_id": "id", "app_secret": "secret"})
  assert any("Bot info" in e for e in errors)


@patch("nemo.lark.api.get_bot_info", return_value={})
@patch("nemo.lark.auth.get_token", return_value="tok123")
def test_bot_info_no_open_id(mock_token, mock_bot):
  errors = run_preflight({"app_id": "id", "app_secret": "secret"})
  assert any("open_id" in e for e in errors)


@patch("nemo.lark.api.get_chat_info", side_effect=RuntimeError("no access"))
@patch("nemo.lark.api.get_bot_info", return_value={"open_id": "ou_bot"})
@patch("nemo.lark.auth.get_token", return_value="tok123")
def test_chat_access_failure(mock_token, mock_bot, mock_chat):
  errors = run_preflight({"app_id": "id", "app_secret": "secret"}, chat_id="oc_123")
  assert any("Chat access" in e for e in errors)


@patch("nemo.lark.api.get_chat_info", return_value={"chat_id": "oc_123"})
@patch("nemo.lark.api.get_bot_info", return_value={"open_id": "ou_bot"})
@patch("nemo.lark.auth.get_token", return_value="tok123")
def test_all_pass(mock_token, mock_bot, mock_chat):
  errors = run_preflight({"app_id": "id", "app_secret": "secret"}, chat_id="oc_123")
  assert errors == []


@patch("nemo.lark.api.get_bot_info", return_value={"open_id": "ou_bot"})
@patch("nemo.lark.auth.get_token", return_value="tok123")
def test_all_pass_no_chat(mock_token, mock_bot):
  errors = run_preflight({"app_id": "id", "app_secret": "secret"})
  assert errors == []
