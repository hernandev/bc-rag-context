import json
from pathlib import Path

import pytest

from bc_rag.config import load_config
from bc_rag.query import search_projects
from bc_rag.voyage_api import split_token_spans


def test_jina_http_reranker_id_does_not_need_use_jina_api() -> None:
    from bc_rag.embeddings import Reranker

    reranker = Reranker("jina-reranker-v3.5")
    assert reranker.jina_api is True
    assert reranker.voyage_api is False
    assert reranker._encoder is None


def test_backend_id_uses_the_default_space_when_unnamed(tmp_path: Path) -> None:
    from tests.support import space_config

    configuration = space_config()
    assert configuration.backend_id() == configuration.backend_id("prose")
    assert configuration.qdrant_collection(tmp_path) == configuration.qdrant_collection(
        tmp_path, "prose"
    )
    assert configuration.backend_id().startswith("prose--")


def test_spaces_get_separate_collections(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "prose",
                "spaces": {
                    "prose": {
                        "dense": {"provider": "voyage", "model": "voyage-context-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "dimensions": 1024,
                        "chunk": {"max_chars": 80000, "min_chars": 40},
                    },
                    "code": {
                        "dense": {"provider": "voyage", "model": "voyage-code-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "dimensions": 1024,
                        "chunk": {"max_chars": 80000, "min_chars": 40},
                    },
                },
                "groups": [
                    {"name": "docs", "space": "prose", "include": ["docs/**/*.md"]},
                    {"name": "libs", "space": "code", "include": ["libs/**/*.ts"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    configuration, _path = load_config(tmp_path)
    assert configuration.default_space == "prose"
    assert configuration.groups[0].space == "prose"
    prose = configuration.qdrant_collection(tmp_path, "prose")
    code = configuration.qdrant_collection(tmp_path, "code")
    assert prose != code
    assert "prose" in prose
    assert "code" in code
    assert configuration.store_dir(tmp_path, "prose") != configuration.store_dir(tmp_path, "code")
    assert "voyage-context-4" in configuration.backend_id("prose")
    assert "voyage-code-4" in configuration.backend_id("code")


def test_unknown_group_space_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "prose",
                "spaces": {
                    "prose": {
                        "dense": {"provider": "voyage", "model": "voyage-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    },
                },
                "groups": [{"name": "docs", "space": "missing", "include": ["**/*.md"]}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        load_config(tmp_path)


def test_auto_input_requires_a_context_model(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "code",
                "spaces": {
                    "code": {
                        "dense": {"provider": "voyage", "model": "voyage-code-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "input": "auto",
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input"):
        load_config(tmp_path)


def test_search_requires_project_and_space() -> None:
    with pytest.raises(ValueError, match="project"):
        search_projects("query", project="", space="default")
    with pytest.raises(ValueError, match="space"):
        search_projects("query", project="bc-rag-context", space="")


def test_split_keeps_a_file_together_until_the_ceiling() -> None:
    spans = split_token_spans([10, 10, 10], max_tokens=25, max_items=100)
    assert spans == [(0, 2), (2, 3)]


def test_flat_voyage_packs_short_files_from_more_than_one_path() -> None:
    from types import SimpleNamespace

    from bc_rag.indexer import flat_http_limits, take_http_batch

    class _Chunk:
        def __init__(self, path: str, text: str) -> None:
            self.path = path
            self.text = text

        def embed_text(self) -> str:
            return self.text

    embedder = SimpleNamespace(voyage_api=True, contextual=False)
    max_chars, max_texts = flat_http_limits(embedder)
    assert max_texts > 64
    chunks = [
        _Chunk("src/a.ts", "export const a = 1\n"),
        _Chunk("src/b.ts", "export const b = 2\n"),
        _Chunk("src/c.ts", "export const c = 3\n"),
    ]
    batch = take_http_batch(chunks, max_chars=max_chars, max_texts=max_texts)
    assert [item.path for item in batch] == ["src/a.ts", "src/b.ts", "src/c.ts"]


def test_one_oversized_chunk_is_its_own_request() -> None:
    spans = split_token_spans([100, 5], max_tokens=32, max_items=100)
    assert spans == [(0, 1), (1, 2)]
