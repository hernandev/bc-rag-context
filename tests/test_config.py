import json
from pathlib import Path

import pytest

from bc_rag.config import (
    ChunkConfig,
    ModelSpec,
    RagConfig,
    SpaceConfig,
    dump_default_config,
    load_config,
    public_config,
)


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
                        "dense": {
                            "provider": "local",
                            "model": "jinaai/jina-embeddings-v2-base-en",
                        },
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


def _write(tmp_path: Path, **extra: object) -> None:
    raw = json.loads(dump_default_config())
    raw.update(extra)
    (tmp_path / ".bc-rag.json").write_text(json.dumps(raw), encoding="utf-8")


@pytest.mark.parametrize("key", ["qdrant_url", "qdrantUrl"])
def test_qdrant_url_is_not_a_project_key(tmp_path: Path, key: str) -> None:
    _write(tmp_path, **{key: "http://127.0.0.1:6333"})
    with pytest.raises(ValueError, match="bc-rag config set qdrant-url URL --global"):
        load_config(tmp_path)


def test_schedule_round_trip(tmp_path: Path) -> None:
    _write(tmp_path, schedule={"every": " 15m "})
    configuration, _path = load_config(tmp_path)
    assert configuration.schedule is not None
    assert configuration.schedule.every == "15m"
    assert public_config(configuration)["schedule"] == {"every": "15m"}


def test_no_schedule_is_omitted(tmp_path: Path) -> None:
    _write(tmp_path)
    configuration, _path = load_config(tmp_path)
    assert configuration.schedule is None
    assert "schedule" not in public_config(configuration)


@pytest.mark.parametrize(
    "schedule",
    [{"every": "0"}, {"every": "soon"}, {"every": "5m", "at": "noon"}],
)
def test_schedule_rejects_bad_values(tmp_path: Path, schedule: dict) -> None:
    _write(tmp_path, schedule=schedule)
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_jina_and_local_spaces_use_different_stores(tmp_path: Path) -> None:
    local = RagConfig(
        default_space="default", spaces=_spaces("jinaai/jina-embeddings-v2-base-en", "local")
    )
    jina = RagConfig(default_space="default", spaces=_spaces())
    assert local.store_dir(tmp_path).parent == jina.project_dir(tmp_path)
    assert local.store_dir(tmp_path) != jina.store_dir(tmp_path)
    assert local.backend_id().startswith("default--local--")
    assert jina.backend_id().startswith("default--jina--")
    assert "jina-embeddings-v5-text-small" in jina.backend_id()
    assert str(local.spaces["default"].chunk.max_chars) in local.backend_id()


def test_different_dense_model_gets_a_new_store(tmp_path: Path) -> None:
    first = RagConfig(default_space="default", spaces=_spaces("jina-embeddings-v4"))
    second = RagConfig(default_space="default", spaces=_spaces("jina-embeddings-v5-text-small"))
    assert first.store_dir(tmp_path) != second.store_dir(tmp_path)
    assert "jina-embeddings-v4" in str(first.store_dir(tmp_path))
    assert "jina-embeddings-v5-text-small" in str(second.store_dir(tmp_path))


def _group(**extra: object) -> dict:
    return {"name": "docs", "space": "default", "include": ["docs/**"], **extra}


def test_group_facets_round_trip(tmp_path: Path) -> None:
    _write(tmp_path, groups=[_group(facets={"area": "engine", "provider": ["olo", "toast"]})])
    configuration, _path = load_config(tmp_path)
    assert configuration.groups[0].facets == {"area": ["engine"], "provider": ["olo", "toast"]}
    public = public_config(configuration)
    assert public["groups"][0]["facets"] == {"area": ["engine"], "provider": ["olo", "toast"]}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"facets": {"group": "x"}}, "set by bc-rag"),
        ({"facets": {"apiPath": "/x"}}, "set by bc-rag"),
        ({"facets": {"bad key": "x"}}, "'bad key'"),
        ({"tags": ["area:engine"]}, "tags"),
        ({"metadata": {"area": "engine"}}, "metadata"),
    ],
)
def test_group_label_errors(tmp_path: Path, extra: dict, message: str) -> None:
    _write(tmp_path, groups=[_group(**extra)])
    with pytest.raises(ValueError, match=message):
        load_config(tmp_path)


def test_facet_keys_and_search_hints_round_trip(tmp_path: Path) -> None:
    _write(
        tmp_path,
        facetKeys={"provider": "integration slug, on the spec and on code"},
        searchHints=['Olo spec: {"vendor": "olo"}'],
    )
    configuration, _path = load_config(tmp_path)
    assert configuration.facet_keys == {"provider": "integration slug, on the spec and on code"}
    assert configuration.search_hints == ['Olo spec: {"vendor": "olo"}']
    public = public_config(configuration)
    assert public["facetKeys"] == configuration.facet_keys
    assert public["searchHints"] == configuration.search_hints


def test_facet_keys_cannot_describe_reserved_keys(tmp_path: Path) -> None:
    _write(tmp_path, facetKeys={"group": "x"})
    with pytest.raises(ValueError, match="described by bc-rag"):
        load_config(tmp_path)
