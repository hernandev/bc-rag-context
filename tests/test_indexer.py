import re
from pathlib import Path

from bc_rag.chunking import Chunk
from bc_rag.config import SourceGroup
from tests.support import default_files_config, space_config
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
    config = default_files_config(follow_gitignore=False)
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
    config = space_config(
        follow_gitignore=False,
        groups=[SourceGroup(name="docs", space="prose", include=["docs/**/*.md"])],
    )
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
    config = default_files_config(follow_gitignore=False)
    store = HybridStore(config.qdrant_path(tmp_path))

    class BoomEmbedder(FakeEmbedder):
        def embed_docs(self, dense_texts, sparse_texts):
            raise RuntimeError("post failed")

    first = space_config(
        follow_gitignore=False,
        groups=[SourceGroup(name="src", space="prose", include=["src/**/*.ts"], tags=["scope:internal"])],
    )
    second = space_config(
        follow_gitignore=False,
        groups=[
            SourceGroup(
                name="src",
                space="prose",
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


def test_oversized_context_chunk_splits_under_char_budget() -> None:
    from bc_rag.indexer import _context_char_budget, _halve_chunk, _shrink_context_batch

    budget = _context_char_budget()
    chunk = Chunk(
        path="docs/vendor/primevue/virtualscroller.md",
        language="markdown",
        kind="section",
        symbol=None,
        heading_path="VirtualScroller",
        start_line=1,
        end_line=400,
        start_byte=0,
        end_byte=0,
        text="word " * (budget // 2),
    )
    pieces = _shrink_context_batch([chunk])
    assert len(pieces) >= 2
    assert all(len(piece.embed_text()) <= budget for piece in pieces)
    assert "".join(piece.text for piece in pieces) == chunk.text

    short = Chunk(
        path="docs/short.md",
        language="markdown",
        kind="section",
        symbol=None,
        heading_path=None,
        start_line=4,
        end_line=6,
        start_byte=0,
        end_byte=10,
        text="abcdefghij",
    )
    halved = _halve_chunk(short)
    assert [piece.text for piece in halved] == ["abcde", "fghij"]
    assert halved[1].start_line == 4
    assert len(_shrink_context_batch([short])) == 2


def test_two_spaces_share_one_index_panel(tmp_path: Path, monkeypatch) -> None:
    import json

    from rich.console import Console

    from bc_rag.index_log import IndexLog
    from bc_rag.indexer import index_spaces

    monkeypatch.setenv("BC_RAG_HOME", str(tmp_path / "home"))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n\nhello\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text(TS, encoding="utf-8")
    config = space_config(
        follow_gitignore=False,
        groups=[
            SourceGroup(name="docs", space="prose", include=["docs/**/*.md"]),
            SourceGroup(name="src", space="code", include=["src/**/*.ts"]),
        ],
    )
    console = Console(force_terminal=False)
    prose = HybridStore(tmp_path / "prose-store")
    code = HybridStore(tmp_path / "code-store")
    starts = {"n": 0}
    real_start = IndexLog.start

    def counting_start(self) -> None:
        starts["n"] += 1
        real_start(self)

    monkeypatch.setattr(IndexLog, "start", counting_start)
    try:
        stats = index_spaces(
            [
                Indexer(tmp_path, config, FakeEmbedder(), prose, console, space="prose"),
                Indexer(tmp_path, config, FakeEmbedder(), code, console, space="code"),
            ]
        )
        assert stats.indexed_files == 2
        assert stats.errors == []
        assert starts["n"] == 1
        log_path = config.project_dir(tmp_path) / "index.jsonl"
        events = [
            json.loads(line)["meta"]["event"]
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert events.count("done") == 1
        assert "space" in events
    finally:
        prose.close()
        code.close()
