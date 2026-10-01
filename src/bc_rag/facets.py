"""Bedrock sidecar keys, stored as `key:value` tags.

Same names and values as bc-project-bundler metadataAttributes.
"""

from __future__ import annotations

import re
from pathlib import Path

SIDECAR_KEYS = (
    "scope",
    "system",
    "lifecycle",
    "area",
    "content_type",
    "package",
    "layer",
    "vendor",
    "provider",
    "section",
    "corpus",
    "method",
    "apiPath",
    "operationId",
    "tag",
    "specSlug",
    "domain",
    "aka",
    "traceStatusClass",
    "typeSafetyClass",
    "at",
    "day",
    "role",
)

_OPERATION_HEADING = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\s+(\S+)$"
)
_OPERATION_ID = re.compile(r"\*\*Operation ID\*\*:\s*`([^`]+)`")
_OPERATION_TAG = re.compile(r"\*\*Tags\*\*:\s*(.+)")


_NX_KEY = re.compile(r"^nx-[A-Za-z0-9_-]+$")


def is_filter_key(key: str) -> bool:
    """Sidecar keys, plus raw Nx tags stored as `nx-group`, `nx-layer`, and so on."""
    return key == "group" or key in SIDECAR_KEYS or _NX_KEY.fullmatch(key) is not None


def parse_tag_clause(raw: str) -> tuple[str, str] | None:
    """Split `vendor:olo` or `vendor=olo` into a filter key and value."""
    text = raw.strip()
    if not text:
        return None
    for separator in (":", "="):
        if separator not in text:
            continue
        key, value = text.split(separator, 1)
        key = key.strip()
        value = value.strip()
        if is_filter_key(key) and value:
            return key, value
    return None


def distinct_tag_values(groups: list) -> list[tuple[str, list[str]]]:
    """Tag keys and their distinct values, from enabled groups only."""
    values: dict[str, set[str]] = {}
    for group in groups:
        if not getattr(group, "enabled", True):
            continue
        for tag in getattr(group, "tags", None) or []:
            if ":" in tag:
                key, value = tag.split(":", 1)
            else:
                key, value = tag, ""
            values.setdefault(key, set()).add(value)
    return [(key, sorted(values[key])) for key in sorted(values)]


def tag_branches(tags: list | None) -> list[list[str]] | None:
    """Outer list is OR. A nested list, or a comma-joined string, is AND.

    ["a:b", ["a:c", "z:y"]] matches a:b, or both a:c and z:y.
    """
    if not tags:
        return None
    branches: list[list[str]] = []
    for item in tags:
        if isinstance(item, str):
            parts = [part.strip() for part in item.split(",") if part.strip()]
            if parts:
                branches.append(parts)
            continue
        if isinstance(item, (list, tuple)):
            terms: list[str] = []
            for sub in item:
                if not isinstance(sub, str):
                    raise ValueError("a tag group must be a list of strings")
                terms.extend(part.strip() for part in sub.split(",") if part.strip())
            if terms:
                branches.append(terms)
            continue
        raise ValueError("tags must be strings or lists of strings")
    return branches or None


def _one_branch_matches(have: list[str], terms: list[str]) -> bool:
    owned = list(have)
    wanted: dict[str, set[str]] = {}
    rejected: dict[str, set[str]] = {}
    raw: list[str] = []
    raw_rejected: list[str] = []
    for query in terms:
        negated = query.startswith("-") and len(query) > 1
        body = query[1:] if negated else query
        clause = parse_tag_clause(body)
        if clause is None:
            (raw_rejected if negated else raw).append(body.strip())
            continue
        key, value = clause
        (rejected if negated else wanted).setdefault(key, set()).add(value)
    present: dict[str, set[str]] = {}
    for tag in owned:
        clause = parse_tag_clause(tag)
        if clause is None:
            continue
        key, value = clause
        present.setdefault(key, set()).add(value)
    for key, values in wanted.items():
        if not present.get(key, set()) & values:
            return False
    for key, values in rejected.items():
        if present.get(key, set()) & values:
            return False
    if any(item in owned for item in raw_rejected):
        return False
    return all(item in owned for item in raw)


def tags_match(have: list[str] | None, queries: list | None) -> bool:
    """True when any top-level alternative matches. A nested list must all match."""
    branches = tag_branches(queries)
    if not branches:
        return True
    owned = list(have or [])
    return any(_one_branch_matches(owned, branch) for branch in branches)


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


def operation_id_and_tag(text: str) -> tuple[str | None, str | None]:
    op_id = _OPERATION_ID.search(text)
    op_tag = _OPERATION_TAG.search(text)
    tag_value = None
    if op_tag:
        tag_value = op_tag.group(1).strip().split(",")[0].strip().strip("`")
    return (op_id.group(1) if op_id else None, tag_value or None)


def apply_tag(tags: list[str], key: str, value: str) -> None:
    item = f"{key}:{value}"
    if item not in tags:
        tags.append(item)


def enrich_openapi_chunks(rel_path: str, chunks: list, metadata: dict[str, str] | None = None) -> None:
    """Add specSlug / method / apiPath / operationId / tag onto OpenAPI heading chunks."""
    slug = spec_slug_from_rel(rel_path)
    current: dict[str, str] = {}
    if slug:
        current["specSlug"] = slug
    for chunk in chunks:
        extras = dict(current)
        found = operation_from_heading_path(getattr(chunk, "heading_path", None))
        if found is not None:
            extras["method"], extras["apiPath"] = found
            op_id, op_tag = operation_id_and_tag(getattr(chunk, "text", "") or "")
            if op_id:
                extras["operationId"] = op_id
            if op_tag:
                extras["tag"] = op_tag
            current = dict(extras)
        else:
            needle_method = current.get("method")
            needle_path = current.get("apiPath")
            heading = getattr(chunk, "heading_path", None) or ""
            if needle_method and needle_path and f"{needle_method} {needle_path}" in heading:
                extras = dict(current)
            else:
                extras = {"specSlug": slug} if slug else {}
                current = dict(extras)
        tags = list(getattr(chunk, "tags", None) or [])
        meta = dict(getattr(chunk, "metadata", None) or {})
        if metadata:
            meta.update(metadata)
        for key, value in extras.items():
            if not value:
                continue
            apply_tag(tags, key, value)
            meta[key] = value
        chunk.tags = tags
        chunk.metadata = meta
