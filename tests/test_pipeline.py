"""The one pipeline: chunk -> post -> upsert, shared by every space of a pass."""

import threading
from pathlib import Path

import pytest
from tests.support import entry_for, space_config

import bc_rag.indexer as indexer_module
from bc_rag.chunking import Chunk
from bc_rag.config import SourceGroup
from bc_rag.embeddings import SparseVec
from bc_rag.index_log import IndexLog
from bc_rag.indexer import Indexer, index_spaces
from bc_rag.manifest import load_manifest


class FakeEmbedder:
    dense_model = "fake-dense"
    sparse_model = "fake-sparse"
    dense_dim = 4

    def embed_docs(self, dense_texts: list[str], sparse_texts: list[str]):
        count = len(dense_texts)
        return [[0.1, 0.2, 0.3, 0.4]] * count, [SparseVec(indices=[1], values=[1.0])] * count


class FakeStore:
    def __init__(self, trail: list | None = None) -> None:
        self.points: dict[str, int] = {}
        self.trail = trail if trail is not None else []

    def recreate_collection(self, dense_dim: int) -> None:
        self.points = {}

    def ensure_collection(self, dense_dim: int) -> None:
        return None

    def delete_paths(self, paths: list[str]) -> None:
        for path in paths:
            self.points.pop(path, None)

    def upsert_chunks(self, chunks: list[Chunk], dense, sparse) -> None:
        for chunk in chunks:
            self.points[chunk.path] = self.points.get(chunk.path, 0) + 1
        self.trail.append(("upsert", sorted({chunk.path for chunk in chunks})))

    def close(self) -> None:
        return None


def _two_spaces(root: Path):
    (root / "docs").mkdir()
    (root / "src").mkdir()
    for index in range(3):
        (root / "docs" / f"d{index}.md").write_text(f"# doc {index}\n\nhello\n", encoding="utf-8")
        (root / "src" / f"c{index}.ts").write_text(f"export const c{index} = 1;\n", "utf-8")
    return space_config(
        groups=[
            SourceGroup(name="docs", space="prose", include=["docs/**/*.md"]),
            SourceGroup(name="src", space="code", include=["src/**/*.ts"]),
        ],
    )


def _five_chunks(source, text, config) -> list[Chunk]:
    return [
        Chunk(
            path=source.rel_path,
            language=source.language,
            kind="file",
            symbol=None,
            heading_path=None,
            start_line=index,
            end_line=index,
            start_byte=0,
            end_byte=0,
            text=f"{source.rel_path} part {index}",
        )
        for index in range(5)
    ]


def test_two_spaces_walk_the_project_once(tmp_path: Path, monkeypatch) -> None:
    config = _two_spaces(tmp_path)
    walks = {"n": 0}
    real_walk = indexer_module.iter_source_groups

    def counting_walk(*args, **kwargs):
        walks["n"] += 1
        return real_walk(*args, **kwargs)

    monkeypatch.setattr(indexer_module, "iter_source_groups", counting_walk)
    entry = entry_for(tmp_path)
    stats = index_spaces(
        [
            Indexer(entry, config, FakeEmbedder(), FakeStore(), space="prose"),
            Indexer(entry, config, FakeEmbedder(), FakeStore(), space="code"),
        ]
    )

    assert walks["n"] == 1
    assert stats.indexed_files == 6


def test_chunking_does_not_wait_for_posts(tmp_path: Path, monkeypatch) -> None:
    config = _two_spaces(tmp_path)
    code_chunked = threading.Event()
    waited: list[bool] = []
    real_chunk = indexer_module._SpaceRun.chunk

    def chunk(self, job):
        batches = real_chunk(self, job)
        if self.space == "code":
            code_chunked.set()
        return batches

    class SlowProse(FakeEmbedder):
        def embed_docs(self, dense_texts, sparse_texts):
            # a prose post that only returns once code files are being chunked.
            waited.append(code_chunked.wait(timeout=5))
            return super().embed_docs(dense_texts, sparse_texts)

    monkeypatch.setattr(indexer_module._SpaceRun, "chunk", chunk)
    entry = entry_for(tmp_path)
    index_spaces(
        [
            Indexer(entry, config, SlowProse(), FakeStore(), space="prose"),
            Indexer(entry, config, FakeEmbedder(), FakeStore(), space="code"),
        ]
    )

    assert waited
    assert all(waited)


def test_file_complete_comes_after_the_files_last_upsert(tmp_path: Path, monkeypatch) -> None:
    config = _two_spaces(tmp_path)
    trail: list = []
    real_event = IndexLog.event

    def recording_event(self, event, **fields):
        if event == "file_complete":
            trail.append(("file_complete", fields["path"]))
        real_event(self, event, **fields)

    monkeypatch.setattr(IndexLog, "event", recording_event)
    monkeypatch.setattr(indexer_module, "chunks_for", _five_chunks)
    # two chunks per request, so each file's five chunks span three upserts.
    monkeypatch.setattr(indexer_module, "flat_http_limits", lambda embedder: (10**9, 2))
    store = FakeStore(trail)
    Indexer(entry_for(tmp_path), config, FakeEmbedder(), store, space="code").run()

    completed = [step[1] for step in trail if step[0] == "file_complete"]
    assert sorted(completed) == ["src/c0.ts", "src/c1.ts", "src/c2.ts"]
    for path in completed:
        done_at = trail.index(("file_complete", path))
        upserts = [i for i, step in enumerate(trail) if step[0] == "upsert" and path in step[1]]
        assert len(upserts) >= 2
        assert max(upserts) < done_at
        assert store.points[path] == 5


def test_a_failed_post_still_writes_the_batches_already_embedded(
    tmp_path: Path, monkeypatch
) -> None:
    config = _two_spaces(tmp_path)

    class FailsOnOne(FakeEmbedder):
        def embed_docs(self, dense_texts, sparse_texts):
            if any("c1" in text for text in dense_texts):
                raise RuntimeError("HTTP 500")
            return super().embed_docs(dense_texts, sparse_texts)

    # one file per request, so the failure hits only src/c1.ts.
    monkeypatch.setattr(indexer_module, "flat_http_limits", lambda embedder: (10**9, 1))
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path), config, FailsOnOne(), store, space="code")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        indexer.run()

    assert set(store.points) == {"src/c0.ts", "src/c2.ts"}
    manifest = load_manifest(config.manifest_path(tmp_path, "code"))
    assert set(manifest.files) == {"src/c0.ts", "src/c2.ts"}
