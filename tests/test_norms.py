"""Tests for nemo.norms — group norm management."""

from unittest.mock import patch, MagicMock
from nemo.norms import get_norms, add_norm, remove_norm, format_norms_prompt


def _mock_config(rules=None):
  """Return a config dict with given rules."""
  return {
    "guests": [],
    "autoapprove": False,
    "filter": "concise",
    "rules": dict(rules or {}),
  }


@patch("nemo.norms.save_config")
@patch("nemo.norms.load_config")
def test_get_norms_empty(mock_load, mock_save):
  mock_load.return_value = _mock_config()
  result = get_norms("tok", "chat1")
  assert result == {}


@patch("nemo.norms.save_config")
@patch("nemo.norms.load_config")
def test_get_norms_with_rules(mock_load, mock_save):
  mock_load.return_value = _mock_config({"brevity": "Keep it short"})
  result = get_norms("tok", "chat1")
  assert result == {"brevity": "Keep it short"}


@patch("nemo.norms.save_config")
@patch("nemo.norms.load_config")
def test_add_norm(mock_load, mock_save):
  mock_load.return_value = _mock_config()
  add_norm("tok", "chat1", "brevity", "Keep it short")
  mock_save.assert_called_once()
  saved_config = mock_save.call_args[0][2]
  assert saved_config["rules"]["brevity"] == "Keep it short"


@patch("nemo.norms.save_config")
@patch("nemo.norms.load_config")
def test_add_norm_update_existing(mock_load, mock_save):
  mock_load.return_value = _mock_config({"brevity": "Old text"})
  add_norm("tok", "chat1", "brevity", "New text")
  saved_config = mock_save.call_args[0][2]
  assert saved_config["rules"]["brevity"] == "New text"


@patch("nemo.norms.save_config")
@patch("nemo.norms.load_config")
def test_remove_norm_exists(mock_load, mock_save):
  mock_load.return_value = _mock_config({"brevity": "Keep it short"})
  result = remove_norm("tok", "chat1", "brevity")
  assert result is True
  saved_config = mock_save.call_args[0][2]
  assert "brevity" not in saved_config["rules"]


@patch("nemo.norms.save_config")
@patch("nemo.norms.load_config")
def test_remove_norm_not_found(mock_load, mock_save):
  mock_load.return_value = _mock_config()
  result = remove_norm("tok", "chat1", "nonexistent")
  assert result is False
  mock_save.assert_not_called()


def test_format_norms_prompt_empty():
  assert format_norms_prompt({}) == ""


def test_format_norms_prompt_with_norms():
  norms = {"brevity": "Keep it short", "lang": "Use English"}
  result = format_norms_prompt(norms)
  assert "Group norms:" in result
  assert "brevity: Keep it short" in result
  assert "lang: Use English" in result
