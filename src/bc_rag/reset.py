"""Wipe local index state so the next `bc-rag index` is a first run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from bc_rag.cache import cache_dir
from bc_rag.config import RagConfig
from bc_rag.defaults import CORPUS_DIRNAME, OPENAPI_MD_DIRNAME


@dataclass
class ResetReport:
    removed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    # One row per space: (collection name, "deleted" | "already gone" | error text).
    collections: list[tuple[str, str]] = field(default_factory=list)


def reset_collections(root: Path, configuration: RagConfig) -> list[str]:
    """The Qdrant collection of every configured space."""
    return [configuration.qdrant_collection(root, space) for space in configuration.spaces]


def reset_targets(root: Path, configuration: RagConfig) -> list[Path]:
    project_dir = configuration.project_dir(root)
    targets = [configuration.store_dir(root, space) for space in configuration.spaces]
    targets.append(cache_dir(project_dir))
    # the places the cache lived before it had its own folder.
    targets.extend(
        path
        for path in (project_dir / CORPUS_DIRNAME, project_dir / OPENAPI_MD_DIRNAME)
        if path.exists()
    )
    return targets


def reset_project(root: Path, configuration: RagConfig) -> ResetReport:
    report = ResetReport()
    url = configuration.qdrant_http_url()
    for collection in reset_collections(root, configuration):
        report.collections.append((collection, _delete_http_collection(url, collection)))
    for path in reset_targets(root, configuration):
        if not path.exists():
            report.missing.append(str(path))
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        report.removed.append(str(path))
    return report


def _delete_http_collection(url: str, collection: str) -> str:
    try:
        from qdrant_client import QdrantClient

        from bc_rag.store import qdrant_api_key

        client = QdrantClient(
            url=url, api_key=qdrant_api_key(), timeout=30, check_compatibility=False
        )
        try:
            if not client.collection_exists(collection):
                return "already gone"
            client.delete_collection(collection)
            return "deleted"
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as error:
        return str(error)
