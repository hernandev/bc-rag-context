from pathlib import Path

from bc_rag.models import cache_directory, cache_folder_name, find_cached_path


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
