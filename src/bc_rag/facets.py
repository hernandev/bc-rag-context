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
    "method",
    "path",
    "operationId",
    "tag",
    "specSlug",
    "domain",
    "aka",
    "traceStatusClass",
    "typeSafetyClass",
)

_OPERATION_HEADING = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\s+(\S+)$"
)
_OPERATION_ID = re.compile(r"\*\*Operation ID\*\*:\s*`([^`]+)`")
_OPERATION_TAG = re.compile(r"\*\*Tags\*\*:\s*(.+)")


def parse_tag_clause(raw: str) -> tuple[str, str] | None:
    """Split `vendor:olo` or `vendor=olo` into a sidecar key and value."""
    text = raw.strip()
    if not text:
        return None
    for separator in (":", "="):
        if separator not in text:
            continue
        key, value = text.split(separator, 1)
        key = key.strip()
        value = value.strip()
        if key in SIDECAR_KEYS and value:
            return key, value
    return None


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
    """Add specSlug / method / path / operationId / tag onto OpenAPI heading chunks."""
    slug = spec_slug_from_rel(rel_path)
    current: dict[str, str] = {}
    if slug:
        current["specSlug"] = slug
    for chunk in chunks:
        extras = dict(current)
        found = operation_from_heading_path(getattr(chunk, "heading_path", None))
        if found is not None:
            extras["method"], extras["path"] = found
            op_id, op_tag = operation_id_and_tag(getattr(chunk, "text", "") or "")
            if op_id:
                extras["operationId"] = op_id
            if op_tag:
                extras["tag"] = op_tag
            current = dict(extras)
        else:
            needle_method = current.get("method")
            needle_path = current.get("path")
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
