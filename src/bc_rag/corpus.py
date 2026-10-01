"""Prepared corpus under ~/.bc-rag/{project}/cache/corpus/.

Each indexed document is a file plus a Bedrock sidecar, the same pair
bc-project-bundler writes:

  {path}
  {path}.metadata.json   {"metadataAttributes": {...}}

The sidecar's metadataAttributes is the file's facets. Index skip-hash is the
pair: a facet change rewrites the sidecar, the sidecar hash misses, and that file
is re-embedded. The tree can be synced to S3 later.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from bc_rag.cache import cache_dir
from bc_rag.defaults import CORPUS_DIRNAME
from bc_rag.discover import SourceFile
from bc_rag.facets import Facets, spec_slug_from_rel
from bc_rag.manifest import file_sha256

METADATA_SUFFIX = ".metadata.json"
DOC_PREFIX_TO_STRIP = "docs/bigcolony/"
FENCE_BY_SUFFIX = {
    ".ts": "ts",
    ".tsx": "tsx",
    ".mts": "ts",
    ".cts": "ts",
    ".js": "js",
    ".jsx": "jsx",
    ".mjs": "js",
    ".json": "json",
    ".vue": "vue",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".css": "css",
    ".html": "html",
}


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    key: str
    path: Path
    sidecar_path: Path
    content_sha256: str
    sidecar_sha256: str


def corpus_dir(project_dir: Path) -> Path:
    return cache_dir(project_dir) / CORPUS_DIRNAME


def sidecar_path_for(document_path: Path) -> Path:
    return Path(str(document_path) + METADATA_SUFFIX)


def attributes_from_source(source: SourceFile) -> Facets:
    """The sidecar's metadataAttributes: the file's facets plus `group`, keys sorted."""
    attributes: Facets = dict(source.facets)
    if source.group:
        attributes["group"] = [source.group]
    return {key: list(attributes[key]) for key in sorted(attributes)}


def sidecar_json(attributes: Facets) -> str:
    return json.dumps({"metadataAttributes": attributes}, indent=2) + "\n"


def sidecar_sha256(source: SourceFile) -> str:
    """The hash the written sidecar file would have, computed without touching disk."""
    text = sidecar_json(attributes_from_source(source))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first(attributes: Facets, key: str) -> str | None:
    values = attributes.get(key) or []
    return values[0] if values else None


def corpus_key(source: SourceFile) -> str:
    attributes = attributes_from_source(source)
    scope = _first(attributes, "scope") or "internal"
    if scope == "external":
        vendor = _first(attributes, "vendor") or _first(attributes, "provider") or "unknown"
        slug = spec_slug_from_rel(source.rel_path) or Path(source.rel_path).stem
        return f"external/{vendor}/{slug}/{Path(source.path).name}"
    system = _first(attributes, "system") or "bigcolony-workspaces"
    lifecycle = _first(attributes, "lifecycle") or "current"
    rel = source.rel_path
    if rel.startswith("openapi-md/"):
        rel = rel[len("openapi-md/") :]
    if rel.startswith(DOC_PREFIX_TO_STRIP):
        rel = "docs/" + rel[len(DOC_PREFIX_TO_STRIP) :]
    if source.language != "markdown" and not rel.endswith(".md"):
        rel = f"{rel}.md"
    return f"internal/{lifecycle}/{system}/{rel}"


def prepared_body(source: SourceFile) -> str:
    text = source.path.read_text(encoding="utf-8")
    suffix = source.path.suffix.lower()
    if source.language == "markdown" or suffix in {".md", ".mdx"}:
        return text
    fence = FENCE_BY_SUFFIX.get(suffix, "")
    return f"# {source.rel_path}\n\n```{fence}\n{text}\n```\n"


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return
    path.write_bytes(encoded)


def delete_corpus_key(project_dir: Path, key: str) -> list[Path]:
    if not key:
        return []
    document = corpus_dir(project_dir) / key
    removed: list[Path] = []
    for path in (document, sidecar_path_for(document)):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def prune_corpus(project_dir: Path, keep_keys: set[str]) -> list[str]:
    """Delete corpus documents whose group is no longer enabled."""
    root = corpus_dir(project_dir)
    if not root.is_dir():
        return []
    pruned: list[str] = []
    emptied: set[str] = set()
    # one os.walk: it lists each folder once and knows files from folders without a
    # stat per entry. The corpus holds two files per indexed file.
    for current, _folders, names in os.walk(root):
        prefix = os.path.relpath(current, root).replace(os.sep, "/")
        prefix = "" if prefix == "." else f"{prefix}/"
        for name in names:
            rel = prefix + name
            key = rel[: -len(METADATA_SUFFIX)] if rel.endswith(METADATA_SUFFIX) else rel
            if key in keep_keys:
                continue
            os.unlink(os.path.join(current, name))
            pruned.append(rel)
            emptied.add(current)
    # only the folders that just lost a file can have become empty; deepest first.
    for folder in sorted(emptied, key=len, reverse=True):
        path = Path(folder)
        while path != root and path.is_relative_to(root):
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent
    return pruned


def prepare_document(project_dir: Path, source: SourceFile) -> PreparedDocument:
    key = corpus_key(source)
    document_path = corpus_dir(project_dir) / key
    sidecar = sidecar_path_for(document_path)
    _write_if_changed(document_path, prepared_body(source))
    _write_if_changed(sidecar, sidecar_json(attributes_from_source(source)))
    return PreparedDocument(
        key=key,
        path=document_path,
        sidecar_path=sidecar,
        content_sha256=file_sha256(document_path),
        sidecar_sha256=file_sha256(sidecar),
    )
