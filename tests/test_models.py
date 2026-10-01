from pathlib import Path

import pytest

from bc_rag.config import dump_default_config
from bc_rag.models import (
    cache_directory,
    cache_folder_name,
    configured_model_names,
    download_model,
    find_cached_path,
)


def test_download_defaults_skip_api_models(tmp_path: Path) -> None:
    # the starter config: Jina API dense, local BM25, Voyage rerank.
    (tmp_path / ".bc-rag.json").write_text(dump_default_config(), encoding="utf-8")
    assert configured_model_names(tmp_path) == [
        "jina-embeddings-v5-text-small",
        "Qdrant/bm25",
        "rerank-2.5",
    ]
    assert configured_model_names(tmp_path, local_only=True) == ["Qdrant/bm25"]


@pytest.mark.parametrize(
    "name", ["rerank-2.5", "voyage-context-4", "jina-embeddings-v5-text-small"]
)
def test_download_refuses_api_ids(name: str) -> None:
    with pytest.raises(ValueError, match="API model"):
        download_model(name)


def test_cache_folder_name_uses_huggingface_layout() -> None:
    assert (
        cache_folder_name("jinaai/jina-embeddings-v2-base-en")
        == "models--jinaai--jina-embeddings-v2-base-en"
    )


def test_find_cached_path_follows_xenova_onnx_repo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BC_RAG_MODELS", str(tmp_path))
    xenova = tmp_path / "models--xenova--jina-embeddings-v2-base-en"
    xenova.mkdir()
    found = find_cached_path("jinaai/jina-embeddings-v2-base-en")
    assert found == xenova


def test_find_cached_path_none_when_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BC_RAG_MODELS", str(tmp_path))
    assert cache_directory() == tmp_path.resolve()
    assert find_cached_path("jinaai/jina-embeddings-v2-base-en") is None
