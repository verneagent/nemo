"""nemo-search: print real web search results on stdout.

The agent runs this as a Bash tool when ``WebSearch`` is denied (the deny
reason names the command). Output is plain markdown — title, URL, snippet
per result — so the model can read it without any parsing.

    nemo-search "豆包实时语音 价格" --count 5
    nemo-search --json "qwen omni pricing"
"""

from __future__ import annotations

import json
import sys

from .search import (
  _DEFAULT_COUNT,
  _MAX_COUNT,
  SearchError,
  SearchResult,
  format_results,
  search,
)

_USAGE = (
  'usage: nemo-search "<query>" [--count N] [--json]\n'
  "  backend/config: ~/.nemo/search.json (brave | ddg | searxng),\n"
  "  override with NEMO_SEARCH_BACKEND"
)


def _parse_args(args: list[str]) -> tuple[str, int, bool]:
  query_parts: list[str] = []
  count = _DEFAULT_COUNT
  as_json = False
  index = 0
  while index < len(args):
    arg = args[index]
    if arg == "--json":
      as_json = True
    elif arg in ("-n", "--count"):
      index += 1
      if index >= len(args):
        raise SystemExit(_USAGE)
      try:
        count = int(args[index])
      except ValueError as exc:
        raise SystemExit(f"nemo-search: --count 需要整数: {exc}") from exc
    elif arg.startswith("--count="):
      try:
        count = int(arg.split("=", 1)[1])
      except ValueError as exc:
        raise SystemExit(f"nemo-search: --count 需要整数: {exc}") from exc
    elif arg in ("-h", "--help"):
      raise SystemExit(_USAGE)
    else:
      query_parts.append(arg)
    index += 1
  query = " ".join(query_parts).strip()
  if not query:
    raise SystemExit(_USAGE)
  return query, max(1, min(count, _MAX_COUNT)), as_json


def _to_json(results: list[SearchResult]) -> str:
  return json.dumps(
    [{"title": r.title, "url": r.url, "description": r.description}
     for r in results],
    ensure_ascii=False, indent=2)


def main() -> None:
  query, count, as_json = _parse_args(sys.argv[1:])
  try:
    results = search(query, count)
  except SearchError as exc:
    raise SystemExit(f"nemo-search: {exc}") from exc
  print(_to_json(results) if as_json else format_results(results))


if __name__ == "__main__":
  main()
