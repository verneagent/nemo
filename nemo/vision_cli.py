"""nemo-vision: describe a local image or video with a multimodal model.

Invoked from the coding agent's shell when it meets a ``[video: <path>]`` (or
``[image: <path>]``) marker that the active model cannot see on its own. It
wraps an OpenAI-compatible multimodal endpoint (Alibaba DashScope / Qwen by
default) and prints a plain-text description to stdout, so any coding agent —
vision-capable or not — can "look at" the media by running this command and
reading its output.

Config via environment:
  NEMO_VISION_API_KEY   (required) bearer key for the endpoint
  NEMO_VISION_BASE_URL  endpoint base (default: DashScope compatible-mode)
  NEMO_VISION_MODEL     model name (default: qwen3.5-35b-a3b)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

from .types import JsonObject, JsonValue

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen3-vl-flash"
_DEFAULT_QUESTION = "详细描述这个媒体文件的内容。"
_TIMEOUT_S = 300

# Mime by extension. Video and image are the only content types the endpoint
# accepts; everything else is rejected up-front rather than sent and 400'd.
_VIDEO_MIME = {
  ".mp4": "video/mp4", ".mov": "video/quicktime", ".m4v": "video/x-m4v",
  ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".webm": "video/webm",
  ".flv": "video/x-flv", ".wmv": "video/x-ms-wmv",
}
_IMAGE_MIME = {
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
  ".heic": "image/heic", ".heif": "image/heif",
}


def _media_block(path: str) -> JsonObject:
  """Build the chat-completions content block for a local media file."""
  ext = os.path.splitext(path)[1].lower()
  with open(path, "rb") as fh:
    data = base64.b64encode(fh.read()).decode()
  if ext in _VIDEO_MIME:
    url = f"data:{_VIDEO_MIME[ext]};base64,{data}"
    return {"type": "video_url", "video_url": {"url": url}}
  if ext in _IMAGE_MIME:
    url = f"data:{_IMAGE_MIME[ext]};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url}}
  raise SystemExit(
    f"nemo-vision: unsupported media extension {ext or '(none)'} "
    f"(want image or video)")


def describe(path: str, question: str) -> str:
  """Send the media + question to the endpoint and return its text answer."""
  key = os.environ.get("NEMO_VISION_API_KEY", "").strip()
  if not key:
    raise SystemExit("nemo-vision: NEMO_VISION_API_KEY is not set")
  base = os.environ.get("NEMO_VISION_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
  model = os.environ.get("NEMO_VISION_MODEL", _DEFAULT_MODEL)
  payload: JsonObject = {
    "model": model,
    "messages": [{
      "role": "user",
      "content": [_media_block(path), {"type": "text", "text": question}],
    }],
  }
  req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
      "Authorization": f"Bearer {key}",
      "Content-Type": "application/json",
    },
    method="POST",
  )
  with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
    body: JsonValue = json.loads(resp.read().decode())
  return _extract_content(body)


def _extract_content(body: JsonValue) -> str:
  """Pull choices[0].message.content out of a chat-completions response."""
  if not isinstance(body, dict):
    raise SystemExit(f"nemo-vision: unexpected response: {body!r}")
  choices = body.get("choices")
  if not isinstance(choices, list) or not choices:
    raise SystemExit(f"nemo-vision: no choices in response: {body!r}")
  first = choices[0]
  message = first.get("message") if isinstance(first, dict) else None
  content = message.get("content") if isinstance(message, dict) else None
  if not isinstance(content, str) or not content.strip():
    raise SystemExit(f"nemo-vision: empty content in response: {body!r}")
  return content.strip()


def main() -> None:
  args = sys.argv[1:]
  if not args or args[0] in ("-h", "--help"):
    raise SystemExit("usage: nemo-vision <image-or-video-path> [question]")
  path = os.path.expanduser(args[0])
  if not os.path.isfile(path):
    raise SystemExit(f"nemo-vision: file not found: {path}")
  question = " ".join(args[1:]).strip() or _DEFAULT_QUESTION
  try:
    print(describe(path, question))
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode(errors="replace")[:500]
    raise SystemExit(f"nemo-vision: HTTP {exc.code}: {detail}")
  except urllib.error.URLError as exc:
    raise SystemExit(f"nemo-vision: request failed: {exc.reason}")


if __name__ == "__main__":
  main()
