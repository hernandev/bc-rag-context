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
            )
            for path, row in (data.get("files") or {}).items()
        }
        return cls(
            dense_model=str(data.get("dense_model") or ""),
            sparse_model=str(data.get("sparse_model") or ""),
            files=files,
        )


def load_manifest(path: Path) -> Manifest | None:
    if not path.is_file():
        return None
    return Manifest.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_json(), separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
