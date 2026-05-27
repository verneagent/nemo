"""Tests for nemo.vision_cli content-block selection and response parsing."""

import pytest

from nemo.vision_cli import _media_block, _extract_content


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
