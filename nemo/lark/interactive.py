"""Extract readable text from Lark interactive (card) message content.

Lark delivers interactive messages in two shapes:

* **Simplified** — returned by ``get_message`` / messages-list, and by many
  webhook deliveries::

      {"title": "Done ✓",
       "elements": [[{"tag": "img", "image_key": "..."},
                     {"tag": "text", "text": " "}], ...]}

  ``elements`` is a list of rows, each row a list of ``{tag, ...}`` dicts.

* **Schema 2.0** — the format nemo itself sends through the card API::

      {"schema": "2.0",
       "header": {"title": {"tag": "plain_text", "content": "Done ✓"}},
       "body": {"direction": "vertical", "elements": [{"tag": "markdown", ...}]}}

  ``elements`` is a flat list of block dicts that nest (column_set / column /
  collapsible_panel / form / button / ...).

Both are walked recursively. Anything unrecognised degrades to ``[interactive]``
so a bare card still surfaces to the model as an opaque marker rather than as
empty text.

This is the single source of truth for card-text extraction on the daemon side.
The relay embeds a stdlib-only copy in ``relay/relay.py`` (``_extract_card_text``)
so the remote relay does not need nemo installed — keep the two in sync.
"""

from nemo.types import JsonObject

_TEXT_TAGS = frozenset({"text", "plain_text", "lark_md", "markdown", "md", "a"})
_IMG_TAGS = frozenset({"img", "image", "icon"})
# Container keys to descend into for any non-leaf block. "header"/"title" lead
# so a section (e.g. collapsible_panel) reads its title before its body.
_CONTAINER_KEYS = (
  "header", "title", "text", "content", "elements", "columns",
  "children", "fields", "options",
)


def extract_interactive_text(content: JsonObject) -> str:
  """Return readable text from a card content dict (either shape)."""
  title = _extract_title(content)
  elements = _extract_elements(content)
  body_text = "\n".join(p for p in _walk(elements) if p.strip())
  if title and body_text:
    return f"{title}\n{body_text}"
  return str(title) or body_text or "[interactive]"


def _extract_title(content: JsonObject) -> str:
  """Card title from schema 2.0 header or the simplified ``title`` field."""
  header = content.get("header")
  if isinstance(header, dict):
    t = header.get("title")
    if isinstance(t, dict):
      val = t.get("content") or t.get("text")
      if val:
        return str(val)
  title = content.get("title")
  if isinstance(title, dict):
    return str(title.get("content") or title.get("text") or "")
  return str(title or "")


def _extract_elements(content: JsonObject):
  """Element list from either shape (simplified top-level or schema-2.0 body)."""
  elements = content.get("elements")
  if not isinstance(elements, list):
    body = content.get("body")
    if isinstance(body, dict):
      elements = body.get("elements")
  return elements if isinstance(elements, list) else []


def _walk(node) -> list[str]:
  """Recursively collect leaf text, ``[image]`` markers, and container titles."""
  out: list[str] = []
  if isinstance(node, list):
    for child in node:
      out.extend(_walk(child))
  elif isinstance(node, dict):
    tag = node.get("tag", "")
    if tag in _TEXT_TAGS:
      val = node.get("text") or node.get("content")
      if val:
        out.append(str(val))
      children = node.get("children")
      if isinstance(children, list):
        out.extend(_walk(children))
    elif tag in _IMG_TAGS:
      out.append("[image]")
    else:
      for key in _CONTAINER_KEYS:
        val = node.get(key)
        if isinstance(val, (list, dict)):
          out.extend(_walk(val))
        elif isinstance(val, str) and val.strip():
          out.append(val)
  elif isinstance(node, str) and node.strip():
    out.append(node)
  return out
