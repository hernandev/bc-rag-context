import json
import os
from pathlib import Path

from tests.support import default_files_config, entry_for

import bc_rag.indexer as indexer_module
from bc_rag.chunking import Chunk
from bc_rag.corpus import corpus_dir
from bc_rag.embeddings import SparseVec
from bc_rag.indexer import Indexer, changed_paths
from bc_rag.manifest import load_manifest


class FakeEmbedder:
    dense_model = "fake-dense"
    sparse_model = "fake-sparse"
    dense_dim = 4

    def embed_docs(self, dense_texts: list[str], sparse_texts: list[str]):
        count = len(dense_texts)
        dense = [[0.1, 0.2, 0.3, 0.4] for _ in range(count)]
        sparse = [SparseVec(indices=[1], values=[1.0]) for _ in range(count)]
        return dense, sparse


class FakeStore:
    def __init__(self) -> None:
        self.points: dict[str, list[Chunk]] = {}
        self.recreated = 0

    def recreate_collection(self, dense_dim: int) -> None:
        self.recreated += 1
        self.points = {}

    def ensure_collection(self, dense_dim: int) -> None:
        return None

    def delete_paths(self, paths: list[str]) -> None:
        for path in paths:
            self.points.pop(path, None)

    def upsert_chunks(self, chunks: list[Chunk], dense, sparse) -> None:
        for chunk in chunks:
            self.points.setdefault(chunk.path, []).append(chunk)

    def close(self) -> None:
        return None


def _write_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "alpha.ts").write_text("export const alpha = 1;\n", encoding="utf-8")
    (root / "src" / "beta.ts").write_text("export const beta = 2;\n", encoding="utf-8")


def test_second_index_skips_unchanged_files(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path),default_files_config(), FakeEmbedder(), store)
    first = indexer.run()
    second = indexer.run()

    assert first.indexed_files == 2
    assert second.indexed_files == 0
    assert second.skipped_unchanged == 2


def _skips(config, root: Path) -> dict[str, str]:
    """path -> how the last run decided it was unchanged: stat or hash."""
    lines = (config.project_dir(root) / "index.jsonl").read_text(encoding="utf-8").splitlines()
    metas = [json.loads(line)["meta"] for line in lines]
    return {meta["path"]: meta["by"] for meta in metas if meta["event"] == "skip_hash"}


def test_unchanged_files_are_skipped_without_being_read(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    config = default_files_config()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), FakeStore())
    indexer.run()
    manifest = load_manifest(config.manifest_path(tmp_path))
    assert manifest.files["src/alpha.ts"].mtime_ns > 0

    opened: list[str] = []
    real_prepare = indexer_module.prepare_document

    def watching_prepare(project_dir, source):
        opened.append(source.rel_path)
        return real_prepare(project_dir, source)

    monkeypatch.setattr(indexer_module, "prepare_document", watching_prepare)
    stats = indexer.run()

    assert stats.skipped_unchanged == 2
    assert opened == []
    assert _skips(config, tmp_path) == {"src/alpha.ts": "stat", "src/beta.ts": "stat"}


def test_a_touched_file_is_hashed_once_then_skipped_by_time(tmp_path: Path) -> None:
    _write_project(tmp_path)
    config = default_files_config()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), FakeStore())
    indexer.run()
    alpha = tmp_path / "src" / "alpha.ts"
    stamp = alpha.stat().st_mtime_ns + 5_000_000_000
    os.utime(alpha, ns=(stamp, stamp))

    touched = indexer.run()
    assert touched.indexed_files == 0
    assert _skips(config, tmp_path)["src/alpha.ts"] == "hash"
    assert load_manifest(config.manifest_path(tmp_path)).files["src/alpha.ts"].mtime_ns == stamp

    indexer.run()
    assert _skips(config, tmp_path)["src/alpha.ts"] == "stat"


def test_a_missing_corpus_copy_is_rewritten(tmp_path: Path) -> None:
    _write_project(tmp_path)
    config = default_files_config()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), FakeStore())
    indexer.run()
    key = load_manifest(config.manifest_path(tmp_path)).files["src/alpha.ts"].corpus_key
    document = corpus_dir(config.project_dir(tmp_path)) / key
    document.unlink()

    stats = indexer.run()
    assert stats.skipped_unchanged == 2
    assert _skips(config, tmp_path)["src/alpha.ts"] == "hash"
    assert document.is_file()


def test_changed_file_is_rewritten_alone(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path),default_files_config(), FakeEmbedder(), store)
    indexer.run()
    (tmp_path / "src" / "alpha.ts").write_text("export const alpha = 99;\n", encoding="utf-8")
    stats = indexer.run(only_paths=["src/alpha.ts"])

    assert stats.indexed_files == 1
    assert stats.skipped_unchanged == 0
    assert "src/beta.ts" in store.points


def test_changed_paths_matches_index_skip_hash(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    config = default_files_config()
    store = FakeStore()
    Indexer(entry_for(tmp_path),config, FakeEmbedder(), store).run()
    assert changed_paths(tmp_path, config, ["src/alpha.ts", "src/beta.ts"]) == []
    (tmp_path / "src" / "alpha.ts").write_text("export const alpha = 99;\n", encoding="utf-8")
    assert changed_paths(tmp_path, config, ["src/alpha.ts", "src/beta.ts"]) == ["src/alpha.ts"]
    assert "src/alpha.ts" in store.points


def test_manifest_is_written_before_the_run_finishes(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    store = FakeStore()
    config = default_files_config()
    Indexer(entry_for(tmp_path),config, FakeEmbedder(), store).run()
    manifest = load_manifest(config.manifest_path(tmp_path))
    assert manifest is not None
    assert "src/alpha.ts" in manifest.files
    assert "src/beta.ts" in manifest.files


def test_file_that_becomes_empty_drops_its_points(tmp_path: Path) -> None:
    _write_project(tmp_path)
    config = default_files_config()
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), store)
    indexer.run()
    assert "src/alpha.ts" in store.points
    (tmp_path / "src" / "alpha.ts").write_text("\n", encoding="utf-8")
    indexer.run()
    manifest = load_manifest(config.manifest_path(tmp_path))
    assert "src/alpha.ts" not in store.points
    assert "src/alpha.ts" not in manifest.files
    assert "src/beta.ts" in store.points


def test_deleted_file_is_removed_from_store(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path),default_files_config(), FakeEmbedder(), store)
    indexer.run()
    (tmp_path / "src" / "beta.ts").unlink()
    stats = indexer.run()

    assert stats.deleted_files == 1
    assert "src/beta.ts" not in store.points
    assert "src/alpha.ts" in store.points
    manifest = load_manifest(default_files_config().manifest_path(tmp_path))
    assert manifest is not None
    assert "src/beta.ts" not in manifest.files
