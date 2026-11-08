"""nemo-search: client-side web search, independent of the model endpoint.

Claude Code's ``WebSearch`` is an Anthropic *server* tool
(``web_search_20250305`` / ``web_search_20260209``): the CLI sends the tool
definition with the request and the API backend runs the search. A gateway
that speaks the Anthropic wire protocol but does not implement that tool
therefore fails in the worst possible way — it returns FABRICATED results.
Observed 2026-11-08 on opencode.ai/zen/go: one query came back as the tool
call echoed verbatim, another as a submodel's prose ("I don't have a web
search tool available in this environment"). The model cannot tell those
apart from real search results.

This module sidesteps the endpoint entirely: a normal CLI process talks to
a search API and prints text, exactly like ``nemo-vision`` does for images.
It works on every model because tool execution is client-side.

Config lives in ``~/.nemo/search.json``::

    {
      "backend": "brave",
      "brave_api_key": "{env:BRAVE_API_KEY}",
      "searxng_url": "http://127.0.0.1:8080"
    }

``backend`` is ``brave`` (default), ``ddg`` (keyless, HTML scrape, fragile)
or ``searxng`` (any instance with ``format=json`` enabled). A secret written
as ``{env:VAR}`` is read from the environment, mirroring ``presets.py`` and
``vision_cli.py`` so the key itself stays out of the file.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .types import JsonObject, JsonValue

_CONFIG_PATH = "~/.nemo/search.json"
_DEFAULT_TIMEOUT_S = 20
_DEFAULT_COUNT = 8
_MAX_COUNT = 20
_MAX_SNIPPET = 400

_ENV_REF = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
_USER_AGENT = (
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class SearchError(RuntimeError):
  """A search could not be performed. Never swallowed into empty results."""


@dataclass(frozen=True)
class SearchResult:
  title: str
  url: str
  description: str


def _resolve_secret(value: JsonValue, what: str) -> str:
  """Resolve a config value; ``{env:VAR}`` reads the secret from the env."""
  if not isinstance(value, str) or not value.strip():
    raise SearchError(f"{what} 未配置")
  match = _ENV_REF.match(value.strip())
  if match is None:
    return value.strip()
  resolved = os.environ.get(match.group(1), "")
  if not resolved:
    raise SearchError(f"{what} 引用了环境变量 {match.group(1)}，但它是空的")
  return resolved


def _load_config() -> JsonObject:
  path = os.path.expanduser(_CONFIG_PATH)
  try:
    with open(path, encoding="utf-8") as fh:
      data = json.load(fh)
  except FileNotFoundError:
    return {}
  except (OSError, json.JSONDecodeError) as exc:
    raise SearchError(f"读不了 {_CONFIG_PATH}: {exc}") from exc
  if not isinstance(data, dict):
    raise SearchError(f"{_CONFIG_PATH} 必须是一个 JSON 对象")
  return data


def _backend(config: JsonObject) -> str:
  raw = os.environ.get("NEMO_SEARCH_BACKEND") or config.get("backend") or "brave"
  if not isinstance(raw, str):
    raise SearchError("backend 配置必须是字符串")
  name = raw.strip().lower()
  if name not in {"brave", "ddg", "searxng"}:
    raise SearchError(f"未知 backend: {name}（可用 brave / ddg / searxng）")
  return name


def _http_get(url: str, headers: dict[str, str]) -> str:
  req = urllib.request.Request(url, headers=headers)
  try:
    with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT_S) as resp:
      return resp.read().decode("utf-8", errors="replace")
  except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")[:300]
    raise SearchError(f"搜索 API 返回 HTTP {exc.code}: {body}") from exc
  except urllib.error.URLError as exc:
    raise SearchError(f"连不上搜索 API: {exc.reason}") from exc


def _parse_json(body: str, what: str) -> JsonValue:
  try:
    return json.loads(body)
  except json.JSONDecodeError as exc:
    raise SearchError(
      f"{what} 不是合法 JSON（{exc.msg}）：{body[:200]!r}") from exc


def _as_object(value: JsonValue, what: str) -> JsonObject:
  if not isinstance(value, dict):
    raise SearchError(f"{what} 格式异常")
  return value


def _as_list(value: JsonValue, what: str) -> list[JsonValue]:
  if not isinstance(value, list):
    raise SearchError(f"{what} 格式异常")
  return value


def _str_field(obj: JsonObject, key: str) -> str:
  value = obj.get(key)
  return value if isinstance(value, str) else ""


def _trim(text: str) -> str:
  collapsed = re.sub(r"<[^>]+>", "", text)
  collapsed = re.sub(r"\s+", " ", collapsed).strip()
  if len(collapsed) <= _MAX_SNIPPET:
    return collapsed
  return collapsed[: _MAX_SNIPPET - 1] + "…"


def _brave(query: str, count: int, config: JsonObject) -> list[SearchResult]:
  key = _resolve_secret(
    config.get("brave_api_key") or os.environ.get("BRAVE_API_KEY", ""),
    "brave_api_key（~/.nemo/search.json 或 $BRAVE_API_KEY）")
  params = urllib.parse.urlencode({
    "q": query, "count": count, "extra_snippets": "false",
  })
  body = _http_get(
    f"https://api.search.brave.com/res/v1/web/search?{params}",
    {"Accept": "application/json", "X-Subscription-Token": key})
  data = _as_object(_parse_json(body, "Brave 响应"), "Brave 响应")
  web = _as_object(data.get("web", {}), "Brave web 字段")
  results: list[SearchResult] = []
  for raw in _as_list(web.get("results", []), "Brave results"):
    item = _as_object(raw, "Brave 结果项")
    url = _str_field(item, "url")
    if not url:
      continue
    results.append(SearchResult(
      title=_trim(_str_field(item, "title")) or url,
      url=url,
      description=_trim(_str_field(item, "description")),
    ))
  return results


_DDG_LINK = re.compile(
  r"<a[^>]*class=[\"']result-link[\"'][^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
  re.IGNORECASE | re.DOTALL)


def _ddg(query: str, count: int, config: JsonObject) -> list[SearchResult]:
  del config
  params = urllib.parse.urlencode({"q": query})
  html = _http_get(
    f"https://lite.duckduckgo.com/lite/?{params}",
    {"User-Agent": _USER_AGENT, "Accept": "text/html"})
  results: list[SearchResult] = []
  for match in _DDG_LINK.finditer(html):
    url = urllib.parse.unquote(match.group(1).strip())
    title = _trim(match.group(2))
    if not url or not title:
      continue
    results.append(SearchResult(title=title, url=url, description=""))
    if len(results) >= count:
      break
  return results


def _searxng(query: str, count: int, config: JsonObject) -> list[SearchResult]:
  base = _resolve_secret(config.get("searxng_url"), "searxng_url")
  params = urllib.parse.urlencode({"q": query, "format": "json"})
  body = _http_get(
    f"{base.rstrip('/')}/search?{params}",
    {"Accept": "application/json", "User-Agent": _USER_AGENT})
  data = _as_object(_parse_json(body, "SearXNG 响应"), "SearXNG 响应")
  results: list[SearchResult] = []
  for raw in _as_list(data.get("results", []), "SearXNG results"):
    item = _as_object(raw, "SearXNG 结果项")
    url = _str_field(item, "url")
    if not url:
      continue
    results.append(SearchResult(
      title=_trim(_str_field(item, "title")) or url,
      url=url,
      description=_trim(_str_field(item, "content")),
    ))
    if len(results) >= count:
      break
  return results


_BACKENDS = {"brave": _brave, "ddg": _ddg, "searxng": _searxng}


def search(query: str, count: int = _DEFAULT_COUNT) -> list[SearchResult]:
  """Run one query through the configured backend.

  Raises ``SearchError`` on any failure — an empty list means the backend
  genuinely returned no results, never "something went wrong".
  """
  query = query.strip()
  if not query:
    raise SearchError("query 不能为空")
  count = max(1, min(count, _MAX_COUNT))
  config = _load_config()
  return _BACKENDS[_backend(config)](query, count, config)


def format_results(results: list[SearchResult]) -> str:
  if not results:
    return "（没有搜索结果）"
  blocks: list[str] = []
  for index, item in enumerate(results, start=1):
    block = f"{index}. {item.title}\n   {item.url}"
    if item.description:
      block += f"\n   {item.description}"
    blocks.append(block)
  return "\n\n".join(blocks)
