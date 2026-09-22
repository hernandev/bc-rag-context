"""Prepared corpus under ~/.bc-rag/{project}/corpus/.

Each indexed document is a file plus a Bedrock sidecar, the same pair
bc-project-bundler writes:

  {path}
  {path}.metadata.json   {"metadataAttributes": {...}}

Index skip-hash is the pair. A tag change rewrites the sidecar, the sidecar
hash misses, and that file is re-embedded. The tree can be synced to S3 later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bc_rag.defaults import CORPUS_DIRNAME
from bc_rag.discover import SourceFile
from bc_rag.facets import SIDECAR_KEYS, parse_tag_clause, spec_slug_from_rel
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
    return project_dir / CORPUS_DIRNAME


def sidecar_path_for(document_path: Path) -> Path:
    return Path(str(document_path) + METADATA_SUFFIX)


def attributes_from_source(source: SourceFile) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for key, value in source.metadata.items():
        if key and value:
            attributes[key] = value
    for item in source.tags:
        clause = parse_tag_clause(item)
        if clause is not None:
            attributes[clause[0]] = clause[1]
    ordered: dict[str, str] = {}
    for key in SIDECAR_KEYS:
        if key in attributes:
            ordered[key] = attributes[key]
    for key, value in attributes.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def sidecar_json(attributes: dict[str, str]) -> str:
    return json.dumps({"metadataAttributes": attributes}, indent=2) + "\n"


def corpus_key(source: SourceFile) -> str:
    attributes = attributes_from_source(source)
    scope = attributes.get("scope") or "internal"
    if scope == "external":
        vendor = attributes.get("vendor") or attributes.get("provider") or "unknown"
        slug = spec_slug_from_rel(source.rel_path) or Path(source.rel_path).stem
        return f"external/{vendor}/{slug}/{Path(source.path).name}"
    system = attributes.get("system") or "bigcolony-workspaces"
    lifecycle = attributes.get("lifecycle") or "current"
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


def bundle_sources(root: Path, config) -> tuple[list[PreparedDocument], list[str]]:
    from bc_rag.discover import iter_source_groups
    from bc_rag.split_md import materialize_openapi_sources

    written: list[PreparedDocument] = []
    keep: set[str] = set()
    project_dir = config.project_dir(root)
    for _group, files in iter_source_groups(root, config):
        for source in materialize_openapi_sources(root, config, files):
            try:
                prepared = prepare_document(project_dir, source)
            except (OSError, UnicodeDecodeError):
                continue
            written.append(prepared)
            keep.add(prepared.key)
    pruned = prune_corpus(project_dir, keep)
    return written, pruned


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
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        key = rel[: -len(METADATA_SUFFIX)] if rel.endswith(METADATA_SUFFIX) else rel
        if key in keep_keys:
            continue
        path.unlink()
        pruned.append(rel)
    for directory in sorted(root.rglob("*"), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass
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
