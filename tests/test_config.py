import json
from pathlib import Path

from bc_rag.config import RagConfig, load_config, with_jina_api
from bc_rag.defaults import DEFAULT_DENSE_MODEL, DEFAULT_RERANK_MODEL


def test_missing_file_uses_semantic_defaults(tmp_path: Path) -> None:
    configuration, path = load_config(tmp_path)

    assert path is None
    assert configuration.embed.dense == DEFAULT_DENSE_MODEL
    assert configuration.embed.rerank == DEFAULT_RERANK_MODEL
    assert configuration.retrieve.rerank is True


def test_partial_json_fills_the_rest(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps({"include": ["**/*.md"], "openapi": {"enabled": False}}),
        encoding="utf-8",
    )
    configuration, path = load_config(tmp_path)

    assert path == tmp_path / ".bc-rag.json"
    assert configuration.include == ["**/*.md"]
    assert configuration.openapi.enabled is False
    assert configuration.embed.dense == DEFAULT_DENSE_MODEL
    assert configuration.embed.jina_api is False
    assert isinstance(configuration, RagConfig)


def test_jina_store_is_a_sibling_of_local(tmp_path: Path) -> None:
    local = RagConfig()
    jina = with_jina_api(local, True)
    assert local.store_dir(tmp_path).parent == jina.project_dir(tmp_path)
    assert jina.store_dir(tmp_path).parent == jina.project_dir(tmp_path)
    assert local.qdrant_path(tmp_path) != jina.qdrant_path(tmp_path)
    assert local.backend_id().startswith("local--")
    assert jina.backend_id().startswith("jina--")
    assert "jina-embeddings-v5-text-small" in jina.backend_id()
    assert str(local.chunk.max_chars) in local.backend_id()


def test_different_dense_model_gets_a_new_store(tmp_path: Path) -> None:
    a = RagConfig()
    b = a.model_copy(
        update={
            "embed": a.embed.model_copy(update={"jina_api": True, "jina_dense": "jina-embeddings-v4"})
        }
    )
    c = a.model_copy(
        update={
            "embed": a.embed.model_copy(
                update={"jina_api": True, "jina_dense": "jina-embeddings-v5-text-small"}
            )
        }
    )
    assert b.store_dir(tmp_path) != c.store_dir(tmp_path)
    assert b.qdrant_path(tmp_path).parent == b.store_dir(tmp_path)
    assert c.qdrant_path(tmp_path).parent == c.store_dir(tmp_path)
    assert "jina-embeddings-v4" in str(b.qdrant_path(tmp_path))
    assert "jina-embeddings-v5-text-small" in str(c.qdrant_path(tmp_path))
