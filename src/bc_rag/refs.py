"""Internal JSON Pointer resolver for leftover OpenAPI $ref values.

External HTTP refs are left as-is. Redocly is the split-file bundler.
This only walks `#/components/...` inside an already-bundled document.
"""

from __future__ import annotations

from typing import Any

MAX_REF_DEPTH = 12


def resolve_refs(node: Any, document: dict[str, Any], stack: tuple[str, ...] = ()) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            if ref in stack or len(stack) >= MAX_REF_DEPTH:
                return {"$ref": ref, "circular": True}
            target = json_pointer(document, ref)
            if target is None:
                return node
            resolved = resolve_refs(target, document, stack + (ref,))
            extras = {k: v for k, v in node.items() if k != "$ref"}
            if extras and isinstance(resolved, dict):
                merged = dict(resolved)
                merged.update(resolve_refs(extras, document, stack))
                return merged
            return resolved
        return {key: resolve_refs(value, document, stack) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve_refs(item, document, stack) for item in node]
    return node


def json_pointer(document: Any, pointer: str) -> Any | None:
    if pointer == "#":
        return document
    if not pointer.startswith("#/"):
        return None
    current = document
    for raw in pointer[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
        elif isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current
