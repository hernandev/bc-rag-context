import re
from pathlib import Path

from bc_rag.chunking import Chunk
from bc_rag.config import RagConfig, SourceGroup
from bc_rag.embeddings import SparseVec
from bc_rag.indexer import Indexer, take_http_batch
from bc_rag.query import search
from bc_rag.store import HybridStore

TS = """
export class LocationProjectorService {
  firstPresent(value: string | null): string | undefined {
    if (value === null) {
      return undefined;
    }
    return value;
  }
}
"""


class FakeEmbedder:
    dense_model = "fake-dense"
    sparse_model = "fake-sparse"
    dense_dim = 8

    def embed_docs(
        self, dense_texts: list[str], sparse_texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVec]]:
        return [_dense(text) for text in dense_texts], [_sparse(text) for text in sparse_texts]

    def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        return _dense(text), _sparse(text)


def _dense(text: str) -> list[float]:
    vec = [0.0] * 8
    for token in _tokens(text):
        vec[abs(hash(token)) % 8] += 1.0
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def _sparse(text: str) -> SparseVec:
    indices: list[int] = []
    values: list[float] = []
    for token in sorted(set(_tokens(text))):
        indices.append(abs(hash(token)) % 50_000)
        values.append(1.0)
    return SparseVec(indices=indices, values=values)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def test_index_then_query_finds_symbol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Location.ts").write_text(TS, encoding="utf-8")
    config = RagConfig(follow_gitignore=False)
    store = HybridStore(config.qdrant_path(tmp_path))
    embedder = FakeEmbedder()
    try:
        stats = Indexer(tmp_path, config, embedder, store).run()
        assert stats.indexed_files == 1
        assert stats.chunks >= 1
        assert store.count() == stats.chunks

        again = Indexer(tmp_path, config, embedder, store).run()
        assert again.skipped_unchanged == 1
        assert again.indexed_files == 0

        result = search(
            query="firstPresent null overlay",
            config=config,
            embedder=embedder,
            store=store,
            reranker=None,
            use_rerank=False,
        )
        assert result.hits
        assert any("firstPresent" in hit.text for hit in result.hits)
        assert result.hits[0].path == "src/Location.ts"

        empty = search(
            query="firstPresent null overlay",
            config=config,
            embedder=embedder,
            store=store,
            reranker=None,
            use_rerank=False,
            groups=["no-such-group"],
        )
        assert empty.hits == []
        grouped = search(
            query="firstPresent null overlay",
            config=config,
            embedder=embedder,
            store=store,
            reranker=None,
            use_rerank=False,
            groups=["default"],
        )
        assert grouped.hits
        names = [name for name, _count in store.facet_values("group")]
        assert "default" in names
    finally:
        store.close()


def test_many_tiny_files_index_through_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    src = tmp_path / "docs"
    src.mkdir()
    for index in range(20):
        (src / f"n{index}.md").write_text(f"# note {index}\n\none line.\n", encoding="utf-8")
    config = RagConfig(follow_gitignore=False, include=["docs/**/*.md"])
    store = HybridStore(config.qdrant_path(tmp_path))
    embedder = FakeEmbedder()
    try:
        stats = Indexer(tmp_path, config, embedder, store).run()
        assert stats.indexed_files == 20
        assert stats.chunks >= 20
    finally:
        store.close()


def test_failed_post_does_not_skip_on_next_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Location.ts").write_text(TS, encoding="utf-8")
    config = RagConfig(follow_gitignore=False)
    store = HybridStore(config.qdrant_path(tmp_path))

    class BoomEmbedder(FakeEmbedder):
        def embed_docs(self, dense_texts, sparse_texts):
            raise RuntimeError("post failed")

    first = RagConfig(
        follow_gitignore=False,
        groups=[SourceGroup(name="src", include=["src/**/*.ts"], tags=["scope:internal"])],
    )
    second = RagConfig(
        follow_gitignore=False,
        groups=[
            SourceGroup(
                name="src",
                include=["src/**/*.ts"],
                tags=["scope:internal", "area:engine"],
            )
        ],
    )
    try:
        Indexer(tmp_path, first, FakeEmbedder(), store).run()
        try:
            Indexer(tmp_path, second, BoomEmbedder(), store).run()
        except Exception:
            pass
        again = Indexer(tmp_path, second, FakeEmbedder(), store).run()
        assert again.skipped_unchanged == 0
        assert again.indexed_files == 1
    finally:
        store.close()


def test_http_batch_keeps_long_chunk_alone() -> None:
    short = Chunk(
        path="a.ts",
        language="typescript",
        kind="file",
        symbol=None,
        heading_path=None,
        start_line=1,
        end_line=2,
        text="hi",
        start_byte=0,
        end_byte=2,
    )
    long = Chunk(
        path="b.ts",
        language="typescript",
        kind="file",
        symbol=None,
        heading_path=None,
        start_line=1,
        end_line=2,
        text="x" * 30_000,
        start_byte=0,
        end_byte=30_000,
    )
    assert take_http_batch([long, short], max_chars=24_000, max_texts=8) == [long]
    packed = take_http_batch([short, short, short], max_chars=96_000, max_texts=64)
    assert len(packed) == 3
    many = [short] * 80
    assert len(take_http_batch(many, max_chars=96_000, max_texts=64)) == 64


def test_chunk_embed_prefix_does_not_enter_sparse() -> None:
    chunk = Chunk(
        path="src/a.ts",
        language="typescript",
        kind="function",
        symbol="firstPresent",
        heading_path=None,
        start_line=1,
        end_line=3,
        start_byte=0,
        end_byte=10,
        text="function firstPresent() {}",
    )
    assert "File: src/a.ts" in chunk.embed_text()
    assert "File:" not in chunk.sparse_text()
