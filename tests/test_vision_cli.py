"""Tests for nemo.vision_cli content-block selection and response parsing."""

from unittest import mock

import pytest

from nemo.agent_factory import MediaVision, model_media_vision
from nemo.presets import Preset
from nemo.vision_cli import _media_block, _extract_content


def test_media_vision_builtin_model_sees_image_not_video():
  # A non-preset is the agent's own built-in model (Claude/Codex): sees images
  # natively, never video. Patch resolve_preset so the result is independent
  # of any ambient ~/.nemo/models.json.
  with mock.patch("nemo.presets.resolve_preset", return_value=None):
    assert model_media_vision("claude", "claude-opus-4-7") == MediaVision(True, False)
    assert model_media_vision("codex", "gpt-5.5") == MediaVision(True, False)


def test_media_vision_preset_uses_declared_flags():
  vl = Preset(name="qwen-vl", openai_url="https://x", sees_image=True, sees_video=True)
  with mock.patch("nemo.presets.resolve_preset", return_value=vl):
    assert model_media_vision("codex", "qwen-vl") == MediaVision(True, True)


def test_media_vision_text_preset_is_routed_to_nemo_vision():
  # deepseek/kimi: a preset with no vision block defaults to text-only.
  text = Preset(name="deepseek-v4-pro", anthropic_url="https://x")
  with mock.patch("nemo.presets.resolve_preset", return_value=text):
    assert model_media_vision("claude", "deepseek-v4-pro") == MediaVision(False, False)


def _write(tmp_path, name: str) -> str:
  p = tmp_path / name
  p.write_bytes(b"\x00\x01\x02\x03")
  return str(p)


def test_media_block_video(tmp_path):
  block = _media_block(_write(tmp_path, "clip.mp4"))
  assert block["type"] == "video_url"
  url = block["video_url"]["url"]
  assert isinstance(url, str) and url.startswith("data:video/mp4;base64,")


def test_media_block_image(tmp_path):
  block = _media_block(_write(tmp_path, "shot.png"))
  assert block["type"] == "image_url"
  url = block["image_url"]["url"]
  assert isinstance(url, str) and url.startswith("data:image/png;base64,")


def test_media_block_image_jpg(tmp_path):
  block = _media_block(_write(tmp_path, "a.JPG"))  # case-insensitive ext
  assert block["type"] == "image_url"
  assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_media_block_unsupported_ext(tmp_path):
  with pytest.raises(SystemExit):
    _media_block(_write(tmp_path, "notes.txt"))


def test_extract_content_ok():
  body = {"choices": [{"message": {"content": "  a person typing  "}}]}
  assert _extract_content(body) == "a person typing"


@pytest.mark.parametrize("body", [
  {},                                   # no choices
  {"choices": []},                      # empty choices
  {"choices": [{"message": {}}]},       # no content
  {"choices": [{"message": {"content": "  "}}]},  # blank content
  "not a dict",                          # wrong top-level type
])
def test_extract_content_malformed(body):
  with pytest.raises(SystemExit):
    _extract_content(body)
