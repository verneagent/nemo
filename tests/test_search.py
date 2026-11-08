"""nemo-search: client-side search that does not depend on the model gateway.

The failure this guards against is subtle: a non-Anthropic endpoint answers
``WebSearch`` with fabricated results instead of an error (2026-11-08), so
the tool "works" while lying. The CLI is the replacement, and it must fail
LOUDLY — an empty list for a broken key would reproduce the same bug.
"""

from __future__ import annotations

import json

import pytest

from nemo import search as search_mod
from nemo.search import (
  SearchError,
  SearchResult,
  _resolve_secret,
  format_results,
  search,
)


_BRAVE_BODY = json.dumps({
  "web": {
    "results": [
      {"title": "A <b>title</b>", "url": "https://a.example/1",
       "description": "first   snippet"},
      {"title": "B", "url": "https://b.example/2", "description": ""},
      {"title": "no url"},
    ]
  }
})


def test_brave_results_are_parsed_and_filtered(monkeypatch):
  monkeypatch.setenv("BRAVE_API_KEY", "k")
  monkeypatch.setattr(search_mod, "_load_config", lambda: {})
  seen: list[str] = []

  def fake_get(url: str, headers: dict[str, str]) -> str:
    seen.append(url)
    assert headers["X-Subscription-Token"] == "k"
    return _BRAVE_BODY

  monkeypatch.setattr(search_mod, "_http_get", fake_get)
  results = search("q", 5)
  assert results == [
    SearchResult(title="A title", url="https://a.example/1",
                 description="first snippet"),
    SearchResult(title="B", url="https://b.example/2", description=""),
  ]
  # A result without a URL cannot be cited or fetched — it is dropped, not
  # emitted as a title-only entry.
  assert all(r.url for r in results)
  assert "count=5" in seen[0]


def test_a_missing_key_is_an_error_not_an_empty_result(monkeypatch):
  """No silent failure: a broken key must not look like "no results"."""
  monkeypatch.delenv("BRAVE_API_KEY", raising=False)
  monkeypatch.setenv("NEMO_SEARCH_BACKEND", "brave")
  monkeypatch.setattr(search_mod, "_load_config", lambda: {})
  with pytest.raises(SearchError, match="brave_api_key"):
    search("q")


def test_env_ref_in_config_is_resolved(monkeypatch):
  monkeypatch.setenv("MY_BRAVE", "secret")
  assert _resolve_secret("{env:MY_BRAVE}", "key") == "secret"
  assert _resolve_secret("literal", "key") == "literal"
  monkeypatch.delenv("MY_BRAVE")
  with pytest.raises(SearchError, match="MY_BRAVE"):
    _resolve_secret("{env:MY_BRAVE}", "key")


def test_env_backend_overrides_the_config(monkeypatch):
  monkeypatch.setenv("NEMO_SEARCH_BACKEND", "searxng")
  monkeypatch.setattr(
    search_mod, "_load_config",
    lambda: {"backend": "brave", "searxng_url": "http://127.0.0.1:9"})
  called: list[str] = []

  def fake_get(url: str, headers: dict[str, str]) -> str:
    called.append(url)
    return json.dumps({"results": [
      {"title": "t", "url": "https://x.example", "content": "c"}]})

  monkeypatch.setattr(search_mod, "_http_get", fake_get)
  results = search("q", 3)
  assert results[0].url == "https://x.example"
  assert called[0].startswith("http://127.0.0.1:9/search?")


def test_unknown_backend_is_an_error(monkeypatch):
  monkeypatch.setenv("NEMO_SEARCH_BACKEND", "google")
  monkeypatch.setattr(search_mod, "_load_config", lambda: {})
  with pytest.raises(SearchError, match="google"):
    search("q")


def test_malformed_response_is_an_error_not_empty(monkeypatch):
  monkeypatch.setenv("BRAVE_API_KEY", "k")
  monkeypatch.setattr(search_mod, "_load_config", lambda: {})
  monkeypatch.setattr(
    search_mod, "_http_get", lambda url, headers: "<html>blocked</html>")
  with pytest.raises(SearchError, match="JSON"):
    search("q")


def test_format_results_is_readable_markdown():
  out = format_results([
    SearchResult(title="T", url="https://x.example", description="D")])
  assert out == "1. T\n   https://x.example\n   D"
  assert format_results([]) == "（没有搜索结果）"
