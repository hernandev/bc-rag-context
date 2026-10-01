import io
import urllib.error
from unittest.mock import patch

from bc_rag.voyage_api import (
    VoyageApiError,
    _post_json,
    embed_contextual_chunks,
    embed_contextual_query,
    embed_texts,
)


def test_flat_embed_posts_input_type(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "voyage_test")
    seen: dict = {}

    def fake_post(url, payload, *, api_key, model, tokens):
        seen["url"] = url
        seen["payload"] = payload
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"total_tokens": 4},
        }

    with patch("bc_rag.voyage_api._post_json", fake_post):
        vectors, tokens = embed_texts(model="voyage-code-4", texts=["a", "b"], dimensions=1024)
    assert seen["url"].endswith("/v1/embeddings")
    assert seen["payload"]["input_type"] == "document"
    assert seen["payload"]["output_dimension"] == 1024
    assert vectors == [[1.0, 0.0], [0.0, 2.0]]
    assert tokens == 4


def test_contextual_chunks_are_one_inner_list(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "voyage_test")
    seen: dict = {}

    def fake_post(url, payload, *, api_key, model, tokens):
        seen["url"] = url
        seen["payload"] = payload
        return {
            "data": [
                {
                    "index": 0,
                    "data": [
                        {"index": 0, "embedding": [1.0]},
                        {"index": 1, "embedding": [2.0]},
                    ],
                }
            ],
            "usage": {"total_tokens": 9},
        }

    with patch("bc_rag.voyage_api._post_json", fake_post):
        vectors, tokens = embed_contextual_chunks(
            model="voyage-context-4", chunk_texts=["one", "two"]
        )
    assert seen["url"].endswith("/v1/contextualizedembeddings")
    assert seen["payload"]["inputs"] == [["one", "two"]]
    assert seen["payload"]["input_type"] == "document"
    assert "enable_auto_chunking" not in seen["payload"]
    assert vectors == [[1.0], [2.0]]
    assert tokens == 9


def test_contextual_query_uses_the_same_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "voyage_test")
    seen: dict = {}

    def fake_post(url, payload, *, api_key, model, tokens):
        seen["url"] = url
        seen["payload"] = payload
        return {
            "data": [{"index": 0, "data": [{"index": 0, "embedding": [3.0]}]}],
            "usage": {"total_tokens": 2},
        }

    with patch("bc_rag.voyage_api._post_json", fake_post):
        vector, _tokens = embed_contextual_query(model="voyage-context-4", text="where")
    assert seen["url"].endswith("/v1/contextualizedembeddings")
    assert seen["payload"]["inputs"] == ["where"]
    assert seen["payload"]["input_type"] == "query"
    assert vector == [3.0]


class _Body:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._raw


def test_connection_timeout_is_retried(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "voyage_test")
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        del request, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError(TimeoutError(60, "Operation timed out"))
        return _Body(b'{"ok": true}')

    with patch("bc_rag.voyage_api.urllib.request.urlopen", fake_urlopen):
        with patch("bc_rag.voyage_api.time.sleep"):
            body = _post_json(
                "https://api.voyageai.com/v1/embeddings",
                {"input": ["a"]},
                api_key="voyage_test",
                model="voyage-code-4",
                tokens=1,
            )
    assert calls["n"] == 2
    assert body == {"ok": True}


def test_http_503_is_retried_and_400_is_not(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "voyage_test")
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        del request, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "https://api.voyageai.com/v1/embeddings",
                503,
                "unavailable",
                hdrs=None,
                fp=io.BytesIO(b"busy"),
            )
        return _Body(b'{"ok": true}')

    with patch("bc_rag.voyage_api.urllib.request.urlopen", fake_urlopen):
        with patch("bc_rag.voyage_api.time.sleep"):
            body = _post_json(
                "https://api.voyageai.com/v1/embeddings",
                {"input": ["a"]},
                api_key="voyage_test",
                model="voyage-code-4",
                tokens=1,
            )
    assert calls["n"] == 2
    assert body == {"ok": True}

    def reject(request, timeout=0):
        del request, timeout
        raise urllib.error.HTTPError(
            "https://api.voyageai.com/v1/embeddings",
            400,
            "bad",
            hdrs=None,
            fp=io.BytesIO(b"too many tokens"),
        )

    with patch("bc_rag.voyage_api.urllib.request.urlopen", reject):
        try:
            _post_json(
                "https://api.voyageai.com/v1/embeddings",
                {"input": ["a"]},
                api_key="voyage_test",
                model="voyage-code-4",
                tokens=1,
            )
        except VoyageApiError as error:
            assert "400" in str(error)
        else:
            raise AssertionError("400 must not be retried into success")


def test_voyage_panel_does_not_say_jina(tmp_path) -> None:
    from io import StringIO

    from rich.console import Console

    from bc_rag.index_log import IndexLog

    log = IndexLog(tmp_path, Console(force_terminal=False), catalog_name="t", root=tmp_path)
    log.set_embed_label("voyage")
    log.files_total = 10
    log.files_done = 10
    try:
        buffer = StringIO()
        Console(file=buffer, force_terminal=False, width=80).print(log._render())
        text = buffer.getvalue().lower()
    finally:
        log.close()
    assert "jina" not in text
