"""Per-file hash ledger so index runs only rewrite what changed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileRecord:
    sha256: str
    size: int
    chunks: int
    sidecar_sha256: str = ""
    corpus_key: str = ""
    # the source file's modification time when it was hashed. Same size and same time
    # means unchanged without reading the file, the rule `git status` uses. 0 = unknown.
    mtime_ns: int = 0


@dataclass
class Manifest:
    dense_model: str
    sparse_model: str
    files: dict[str, FileRecord] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
            "files": {
                path: {
                    "sha256": rec.sha256,
                    "sidecar_sha256": rec.sidecar_sha256,
                    "corpus_key": rec.corpus_key,
                    "size": rec.size,
                    "mtime_ns": rec.mtime_ns,
                    "chunks": rec.chunks,
                }
                for path, rec in sorted(self.files.items())
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> Manifest:
        files = {
            path: FileRecord(
                sha256=str(row["sha256"]),
                size=int(row["size"]),
                chunks=int(row["chunks"]),
                sidecar_sha256=str(row.get("sidecar_sha256") or ""),
                corpus_key=str(row.get("corpus_key") or ""),
                mtime_ns=int(row.get("mtime_ns") or 0),
            )
            for path, row in (data.get("files") or {}).items()
        }
        return cls(
            dense_model=str(data.get("dense_model") or ""),
            sparse_model=str(data.get("sparse_model") or ""),
            files=files,
        )


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> Manifest | None:
    """The ledger, or None when there is none yet.

    A file that cannot be read raises instead of returning None: None means "first
    run", which recreates the collection and throws away every paid-for vector.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return Manifest.from_json(data)
    except (ValueError, KeyError, TypeError) as error:
        raise ManifestError(
            f"{path} is not a valid manifest: {error}. Move the file aside to re-hash every "
            "file, or run bc-rag index --force"
        ) from error


def save_manifest(path: Path, manifest: Manifest) -> None:
    from bc_rag.fileio import write_text_atomic

    write_text_atomic(
        path, json.dumps(manifest.to_json(), separators=(",", ":"), ensure_ascii=False) + "\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
