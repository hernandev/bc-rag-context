"""Wipe local index state so the next `bc-rag index` is a first run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from bc_rag.config import RagConfig
from bc_rag.defaults import CORPUS_DIRNAME, OPENAPI_MD_DIRNAME


@dataclass
class ResetReport:
    removed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    collection: str | None = None
    collection_deleted: bool = False
    collection_error: str | None = None


def reset_targets(root: Path, configuration: RagConfig) -> list[Path]:
    project_dir = configuration.project_dir(root)
    spaces: list[str] = list(configuration.spaces)
    targets = [configuration.store_dir(root, space) for space in spaces]
    targets.extend(
        [
            project_dir / CORPUS_DIRNAME,
            project_dir / OPENAPI_MD_DIRNAME,
        ]
    )
    return targets


def reset_project(root: Path, configuration: RagConfig) -> ResetReport:
    report = ResetReport()
    url = configuration.qdrant_http_url()
    spaces: list[str] = list(configuration.spaces)
    names = [configuration.qdrant_collection(root, space) for space in spaces]
    report.collection = names[0] if names else None
    if url:
        for collection in names:
            _delete_http_collection(url, collection, report)
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


def _delete_http_collection(url: str, collection: str, report: ResetReport) -> None:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, timeout=30, check_compatibility=False)
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
                report.collection_deleted = True
            else:
                report.missing.append(f"qdrant collection {collection}")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as error:
        report.collection_error = str(error)
