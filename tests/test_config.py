import json
from pathlib import Path

import pytest

from bc_rag.config import ChunkConfig, ModelSpec, RagConfig, SpaceConfig, dump_default_config, load_config


def _spaces(model: str = "jina-embeddings-v5-text-small", provider: str = "jina") -> dict:
    return {
        "default": SpaceConfig(
            dense=ModelSpec(provider=provider, model=model),
            sparse=ModelSpec(provider="local", model="Qdrant/bm25"),
            rerank=ModelSpec(provider="jina", model="jina-reranker-v3.5"),
            chunk=ChunkConfig(),
        )
    }


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="spaces and defaultSpace are required"):
        load_config(tmp_path)


def test_extra_top_level_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "include": ["**/*.md"],
                "defaultSpace": "default",
                "spaces": {
                    "default": {
                        "dense": {"provider": "local", "model": "jinaai/jina-embeddings-v2-base-en"},
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="include is not a top-level key"):
        load_config(tmp_path)


def test_init_dump_puts_the_dense_model_on_a_space(tmp_path: Path) -> None:
    text = dump_default_config()
    raw = json.loads(text)
    assert raw["defaultSpace"] == "default"
    raw = json.loads(text)
    assert '"chunk": { "min_chars": 40, "max_chars": 2400 }' in text
    assert '"dense": { "provider": "jina", "model": "jina-embeddings-v5-text-small" }' in text
    assert '"rerank": { "provider": "voyage", "model": "rerank-2.5" }' in text
    assert raw["spaces"]["default"]["dense"] == {
        "provider": "jina",
        "model": "jina-embeddings-v5-text-small",
    }
    assert "embed" not in raw
    assert "chunk" not in raw
    assert "openapi" not in raw
    assert "**/node_modules/**" in raw["exclude"]
    assert "**/.git/**" in raw["exclude"]
    assert "**/dist/**" in raw["exclude"]
    (tmp_path / ".bc-rag.json").write_text(dump_default_config(), encoding="utf-8")
    configuration, _path = load_config(tmp_path)
    assert configuration.spaces["default"].dense.provider == "jina"
    assert configuration.spaces["default"].rerank is not None
    assert configuration.spaces["default"].rerank.provider == "voyage"
    assert configuration.spaces["default"].rerank.model == "rerank-2.5"


def test_jina_and_local_spaces_use_different_stores(tmp_path: Path) -> None:
    local = RagConfig(default_space="default", spaces=_spaces("jinaai/jina-embeddings-v2-base-en", "local"))
    jina = RagConfig(default_space="default", spaces=_spaces())
    assert local.store_dir(tmp_path).parent == jina.project_dir(tmp_path)
    assert local.qdrant_path(tmp_path) != jina.qdrant_path(tmp_path)
    assert local.backend_id().startswith("default--local--")
    assert jina.backend_id().startswith("default--jina--")
    assert "jina-embeddings-v5-text-small" in jina.backend_id()
    assert str(local.spaces["default"].chunk.max_chars) in local.backend_id()


def test_different_dense_model_gets_a_new_store(tmp_path: Path) -> None:
    first = RagConfig(default_space="default", spaces=_spaces("jina-embeddings-v4"))
    second = RagConfig(default_space="default", spaces=_spaces("jina-embeddings-v5-text-small"))
    assert first.store_dir(tmp_path) != second.store_dir(tmp_path)
    assert "jina-embeddings-v4" in str(first.qdrant_path(tmp_path))
    assert "jina-embeddings-v5-text-small" in str(second.qdrant_path(tmp_path))
