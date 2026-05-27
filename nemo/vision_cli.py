"""nemo-vision: describe a local image or video with a multimodal model.

Invoked from the coding agent's shell when it meets a ``[video: <path>]`` (or
``[image: <path>]``) marker that the active model cannot see on its own. It
wraps an OpenAI-compatible multimodal endpoint (Alibaba DashScope / Qwen by
default) and prints a plain-text description to stdout, so any coding agent —
vision-capable or not — can "look at" the media by running this command and
reading its output.

Config lives in ``~/.nemo/vision.json`` (same home as ``models.json``)::

    {
      "baseURL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "apiKey":  "{env:BAILIAN_API_KEY}",
      "model":   "qwen3-vl-flash"
    }

All fields are optional — missing ``baseURL`` / ``model`` fall back to the
defaults below. ``apiKey`` follows the models.json convention: ``{env:VAR}``
reads the secret from the environment (so it stays out of the file), any other
non-empty string is a literal key, and an absent ``apiKey`` falls back to
``$BAILIAN_API_KEY``. The endpoint is any OpenAI-compatible multimodal chat API
(Alibaba Cloud Bailian / 百炼 by default).
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request

from .types import JsonObject, JsonValue

_CONFIG_PATH = "~/.nemo/vision.json"
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen3-vl-flash"
_DEFAULT_QUESTION = "详细描述这个媒体文件的内容。"
_TIMEOUT_S = 300

# Mirrors presets.py: an apiKey of "{env:VARNAME}" resolves from the env.
_ENV_REF = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")

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


def _resolve_key(raw: JsonValue) -> str:
  """Resolve an apiKey value from vision.json.

  ``{env:VAR}`` reads ``$VAR``; any other non-empty string is a literal key;
  an absent/blank value falls back to ``$BAILIAN_API_KEY``.
  """
  if isinstance(raw, str) and raw.strip():
    token = raw.strip()
    m = _ENV_REF.match(token)
    return os.environ.get(m.group(1), "").strip() if m else token
  return os.environ.get("BAILIAN_API_KEY", "").strip()


def load_config(path: str = _CONFIG_PATH) -> tuple[str, str, str]:
  """Read (base_url, api_key, model) from vision.json, filling defaults."""
  raw: JsonObject = {}
  try:
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
      loaded = json.load(fh)
    if isinstance(loaded, dict):
      raw = loaded
  except FileNotFoundError:
    pass
  except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"nemo-vision: cannot read {path}: {exc}")
  base = raw.get("baseURL")
  model = raw.get("model")
  base_url = base.strip() if isinstance(base, str) and base.strip() else _DEFAULT_BASE_URL
  model_id = model.strip() if isinstance(model, str) and model.strip() else _DEFAULT_MODEL
  return base_url.rstrip("/"), _resolve_key(raw.get("apiKey")), model_id


def helper_available(path: str = _CONFIG_PATH) -> bool:
  """Whether a usable API key resolves, i.e. running nemo-vision would have
  credentials. Some machines have no vision helper configured (no vision.json
  and no $BAILIAN_API_KEY); callers gate the [image:]/[video:] hints on this so
  the agent isn't told to run a tool that can't work."""
  return bool(load_config(path)[1])


def describe(path: str, question: str) -> str:
  """Send the media + question to the endpoint and return its text answer."""
  base, key, model = load_config()
  if not key:
    raise SystemExit(
      "nemo-vision: no API key — set apiKey in ~/.nemo/vision.json "
      "or export $BAILIAN_API_KEY")
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
