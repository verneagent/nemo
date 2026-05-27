"""Tests for nemo.vision_cli content-block selection and response parsing."""

import json
import os
from unittest import mock

import pytest

from nemo.agent_factory import MediaVision, model_media_vision
from nemo.presets import Preset
from nemo.vision_cli import (
  _DEFAULT_BASE_URL, _DEFAULT_MODEL, _media_block, _extract_content,
  helper_available, load_config,
)


def _write_cfg(tmp_path, obj) -> str:
  p = tmp_path / "vision.json"
  p.write_text(json.dumps(obj))
  return str(p)


def test_load_config_reads_all_fields(tmp_path):
  cfg = _write_cfg(tmp_path, {
    "baseURL": "https://x/v1/", "apiKey": "sk-literal", "model": "m1"})
  base, key, model = load_config(cfg)
  assert base == "https://x/v1"   # trailing slash trimmed
  assert key == "sk-literal"
  assert model == "m1"


def test_load_config_apikey_env_ref(tmp_path):
  cfg = _write_cfg(tmp_path, {"apiKey": "{env:MY_VISION_KEY}"})
  with mock.patch.dict(os.environ, {"MY_VISION_KEY": "sk-env"}, clear=False):
    base, key, model = load_config(cfg)
  assert key == "sk-env"
  assert (base, model) == (_DEFAULT_BASE_URL, _DEFAULT_MODEL)  # defaults fill gaps


def test_load_config_absent_apikey_falls_back_to_bailian_env(tmp_path):
  cfg = _write_cfg(tmp_path, {"model": "m2"})  # no apiKey field
  with mock.patch.dict(os.environ, {"BAILIAN_API_KEY": "sk-bailian"}, clear=False):
    _, key, model = load_config(cfg)
  assert key == "sk-bailian"
  assert model == "m2"


def test_load_config_missing_file_uses_defaults(tmp_path):
  with mock.patch.dict(os.environ, {"BAILIAN_API_KEY": "sk-b"}, clear=False):
    base, key, model = load_config(str(tmp_path / "nope.json"))
  assert (base, model) == (_DEFAULT_BASE_URL, _DEFAULT_MODEL)
  assert key == "sk-b"


def test_helper_available_true_when_key_resolves(tmp_path):
  cfg = _write_cfg(tmp_path, {"apiKey": "sk-literal"})
  assert helper_available(cfg) is True


def test_helper_unavailable_when_no_file_and_no_env(tmp_path):
  missing = str(tmp_path / "nope.json")
  with mock.patch.dict(os.environ, {}, clear=True):  # no BAILIAN_API_KEY
    assert helper_available(missing) is False


def test_helper_unavailable_when_env_ref_unset(tmp_path):
  cfg = _write_cfg(tmp_path, {"apiKey": "{env:UNSET_VISION_KEY}"})
  with mock.patch.dict(os.environ, {}, clear=True):
    assert helper_available(cfg) is False


def test_media_vision_builtin_model_sees_image_not_video():
  # A non-preset is the agent's own built-in model (Claude/Codex): sees images
  # natively, never video. Patch resolve_preset + load_presets so the result is
  # independent of any ambient ~/.nemo/models.json.
  with mock.patch("nemo.presets.resolve_preset", return_value=None), \
       mock.patch("nemo.presets.load_presets", return_value={}):
    assert model_media_vision("claude", "claude-opus-4-7") == MediaVision(True, False)
    assert model_media_vision("codex", "gpt-5.5") == MediaVision(True, False)


def test_media_vision_resolves_preset_by_remote_id():
  # After a preset /model switch the live model is the resolved remote id
  # (e.g. deepseek-v4-pro[1m]), not the preset name. Capability must still
  # resolve — else a text-only preset reads as a vision model and loses its
  # nemo-vision hint.
  ds = Preset(name="deepseek-v4-pro", anthropic_url="https://x",
              anthropic_remote="deepseek-v4-pro[1m]")  # sees_* default False
  with mock.patch("nemo.presets.resolve_preset", return_value=None), \
       mock.patch("nemo.presets.load_presets",
                  return_value={"deepseek-v4-pro": ds}):
    assert model_media_vision("claude", "deepseek-v4-pro[1m]") == MediaVision(False, False)


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
