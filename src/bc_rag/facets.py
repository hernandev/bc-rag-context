"""Facets: the one label format, from `.bc-rag.json` to search results.

A facet is a key with one value or a list of values:

    {"area": "engine", "scope": "internal", "provider": ["olo", "toast"]}

The same object, always named `facets`, appears on config groups, `groupsCommand`
output, files, chunks, the corpus sidecar, the Qdrant payload, MCP search and
listing, and the CLI. bc-rag stores and returns every value as a list.

Search semantics: values under one key are any-of; different keys must all match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

Facets = dict[str, list[str]]

FACET_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# Keys bc-rag sets itself. A config facet with one of these names is rejected.
RESERVED_KEYS: dict[str, str] = {
    "group": "the config group that claimed the file",
    "specSlug": "one OpenAPI spec file",
    "method": "the HTTP method of one OpenAPI operation",
    "apiPath": "the path of one OpenAPI operation (prefer search_sparse for it)",
    "operationId": "the operationId of one OpenAPI operation (prefer search_sparse for it)",
    "apiTag": "the OpenAPI tags of the operation",
    "at": "a chat transcript turn's timestamp",
    "day": "a chat transcript turn's day, YYYY-MM-DD",
    "role": "a chat transcript turn's speaker, user or agent",
}


def normalize_facets(raw: Any, *, where: str = "facets") -> Facets:
    """Validate the shape and return key -> list of values. Raises ValueError naming the problem."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object of key -> value or list of values")
    result: Facets = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not FACET_KEY.fullmatch(key):
            raise ValueError(
                f"{where}: facet key {key!r} must start with a letter and use only "
                "letters, digits, _ and -"
            )
        items = value if isinstance(value, list) else [value]
        values: list[str] = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{where}: facet {key!r} has value {item!r}; values are non-empty strings"
                )
            text = item.strip()
            if text not in values:
                values.append(text)
        if not values:
            raise ValueError(f"{where}: facet {key!r} has no values")
        result[key] = values
    return result


def check_not_reserved(facets: Facets, *, where: str = "facets") -> None:
    for key in facets:
        if key in RESERVED_KEYS:
            raise ValueError(f"{where}: facet key {key!r} is set by bc-rag; pick another name")


def with_facet(facets: Facets, key: str, values: str | list[str]) -> Facets:
    """A copy of `facets` with `key` set to `values`."""
    items = [values] if isinstance(values, str) else list(values)
    return {**facets, key: [item for item in items if item]}


def facets_line(facets: Facets) -> str:
    """`area=engine, provider=olo|toast`, for the dense embedding header."""
    return ", ".join(f"{key}={'|'.join(values)}" for key, values in facets.items())


def parse_facet_options(items: list[str] | None, *, option: str) -> Facets:
    """`--facet key=value` flags, repeated, into one facets object. Raises ValueError."""
    raw: dict[str, list[str]] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"{option} {item!r} must look like key=value")
        raw.setdefault(key.strip(), []).append(value.strip())
    return normalize_facets(raw, where=option)


def facets_match(have: Facets, wanted: Facets | None, excluded: Facets | None = None) -> bool:
    """Search semantics on one object: any-of within a key, all keys, none of `excluded`."""
    for key, values in (wanted or {}).items():
        if not set(have.get(key, [])) & set(values):
            return False
    for key, values in (excluded or {}).items():
        if set(have.get(key, [])) & set(values):
            return False
    return True


def distinct_facet_values(groups: list) -> dict[str, list[str]]:
    """Every facet key on enabled groups, with its distinct values, plus `group` names."""
    values: dict[str, set[str]] = {}
    for group in groups:
        if not getattr(group, "enabled", True):
            continue
        values.setdefault("group", set()).add(group.name)
        for key, items in (getattr(group, "facets", None) or {}).items():
            values.setdefault(key, set()).update(items)
    return {key: sorted(values[key]) for key in sorted(values)}


# -- facets bc-rag sets on OpenAPI chunks -----------------------------------------

_OPERATION_HEADING = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\s+(\S+)$")
_OPERATION_ID = re.compile(r"\*\*Operation ID\*\*:\s*`([^`]+)`")
_OPERATION_TAGS = re.compile(r"\*\*Tags\*\*:\s*(.+)")


def spec_slug_from_rel(rel_path: str) -> str | None:
    """`olo--ordering-api-1.1.bundle.openapi.json` -> `ordering-api-1.1.bundle`."""
    name = Path(rel_path).name
    if name in {"spec.md", "INDEX.md"}:
        name = Path(rel_path).parent.name
    name = re.sub(r"\.openapi(\.json)?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.json$", "", name, flags=re.IGNORECASE)
    if "--" not in name:
        return None
    return name.split("--", 1)[1] or None


def operation_from_heading_path(heading_path: str | None) -> tuple[str, str] | None:
    if not heading_path:
        return None
    for part in heading_path.split(" > "):
        match = _OPERATION_HEADING.match(part.strip())
        if match:
            return match.group(1), match.group(2)
    return None


def operation_id_and_tags(text: str) -> tuple[str | None, list[str]]:
    """The operationId and every OpenAPI tag listed under one operation heading."""
    op_id = _OPERATION_ID.search(text)
    found = _OPERATION_TAGS.search(text)
    tags: list[str] = []
    if found:
        for item in found.group(1).split(","):
            tag = item.strip().strip("`").strip()
            if tag and tag not in tags:
                tags.append(tag)
    return (op_id.group(1) if op_id else None), tags


def enrich_openapi_chunks(rel_path: str, chunks: list) -> None:
    """Set specSlug, method, apiPath, operationId and apiTag on the chunks of one spec.md.

    Chunks under one operation's sub-headings carry that operation's facets too, so
    `{"apiPath": "/baskets/create"}` gathers every chunk of the operation.
    """
    slug = spec_slug_from_rel(rel_path)
    base: Facets = {"specSlug": [slug]} if slug else {}
    current: Facets = dict(base)
    for chunk in chunks:
        heading = getattr(chunk, "heading_path", None) or ""
        found = operation_from_heading_path(heading)
        if found is None:
            current = dict(base)
        else:
            method, api_path = found
            if current.get("method") != [method] or current.get("apiPath") != [api_path]:
                # a new operation. Its sub-heading chunks keep these facets.
                current = {**base, "method": [method], "apiPath": [api_path]}
            op_id, tags = operation_id_and_tags(getattr(chunk, "text", "") or "")
            if op_id:
                current["operationId"] = [op_id]
            if tags:
                current["apiTag"] = tags
        chunk.facets = {**chunk.facets, **current}
