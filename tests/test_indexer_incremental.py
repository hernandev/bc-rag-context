from pathlib import Path

from bc_rag.chunking import Chunk
from tests.support import default_files_config
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
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    _write_project(tmp_path)
    store = FakeStore()
    indexer = Indexer(tmp_path, default_files_config(), FakeEmbedder(), store)
    first = indexer.run()
    second = indexer.run()

    assert first.indexed_files == 2
    assert second.indexed_files == 0
    assert second.skipped_unchanged == 2


def test_changed_file_is_rewritten_alone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    _write_project(tmp_path)
    store = FakeStore()
    indexer = Indexer(tmp_path, default_files_config(), FakeEmbedder(), store)
    indexer.run()
    (tmp_path / "src" / "alpha.ts").write_text("export const alpha = 99;\n", encoding="utf-8")
    stats = indexer.run(only_paths=["src/alpha.ts"])

    assert stats.indexed_files == 1
    assert stats.skipped_unchanged == 0
    assert "src/beta.ts" in store.points


def test_changed_paths_matches_index_skip_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    _write_project(tmp_path)
    config = default_files_config()
    store = FakeStore()
    Indexer(tmp_path, config, FakeEmbedder(), store).run()
    assert changed_paths(tmp_path, config, ["src/alpha.ts", "src/beta.ts"]) == []
    (tmp_path / "src" / "alpha.ts").write_text("export const alpha = 99;\n", encoding="utf-8")
    assert changed_paths(tmp_path, config, ["src/alpha.ts", "src/beta.ts"]) == ["src/alpha.ts"]
    assert "src/alpha.ts" in store.points


def test_manifest_is_written_before_the_run_finishes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    _write_project(tmp_path)
    store = FakeStore()
    config = default_files_config()
    Indexer(tmp_path, config, FakeEmbedder(), store).run()
    manifest = load_manifest(config.manifest_path(tmp_path))
    assert manifest is not None
    assert "src/alpha.ts" in manifest.files
    assert "src/beta.ts" in manifest.files


def test_deleted_file_is_removed_from_store(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    _write_project(tmp_path)
    store = FakeStore()
    indexer = Indexer(tmp_path, default_files_config(), FakeEmbedder(), store)
    indexer.run()
    (tmp_path / "src" / "beta.ts").unlink()
    stats = indexer.run()

    assert stats.deleted_files == 1
    assert "src/beta.ts" not in store.points
    assert "src/alpha.ts" in store.points
    manifest = load_manifest(default_files_config().manifest_path(tmp_path))
    assert manifest is not None
    assert "src/beta.ts" not in manifest.files
