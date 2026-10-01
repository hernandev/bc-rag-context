import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bc_rag.config import load_config
from bc_rag.jina_api import (
    JinaApiError,
    RateLimiter,
    api_model_name,
    embed_texts,
    parse_rate_headers,
    rerank_texts,
    resolve_api_key,
)


def test_use_jina_api_alias_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps({"embed": {"useJinaApi": True}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="embed is not a config key"):
        load_config(tmp_path)


def test_top_level_use_jina_api_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps({"useJinaApi": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="useJinaApi"):
        load_config(tmp_path)


def test_api_model_name_strips_fastembed_prefix() -> None:
    assert api_model_name("jinaai/jina-embeddings-v2-base-en") == "jina-embeddings-v2-base-en"
    assert api_model_name("jina-reranker-v1-turbo-en") == "jina-reranker-v1-turbo-en"


def test_missing_key_errors(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    try:
        resolve_api_key()
        raise AssertionError("expected JinaApiError")
    except JinaApiError as error:
        assert "JINA_API_KEY" in str(error)


def test_embed_texts_orders_by_index(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina_test")

    def fake_post(url, payload, *, api_key, **_kwargs):
        assert url.endswith("/v1/embeddings")
        assert payload["model"] == "jina-embeddings-v2-base-en"
        assert payload["input"] == ["a", "b"]
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"total_tokens": 12},
        }

    with patch("bc_rag.jina_api._post_json", fake_post):
        vectors, tokens = embed_texts(model="jinaai/jina-embeddings-v2-base-en", texts=["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert tokens == 12


def test_rerank_texts_fills_original_order(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina_test")

    def fake_post(url, payload, *, api_key, **_kwargs):
        assert url.endswith("/v1/rerank")
        return {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ]
        }

    with patch("bc_rag.jina_api._post_json", fake_post):
        scores, tokens = rerank_texts(
            model="jinaai/jina-reranker-v1-turbo-en",
            query="q",
            documents=["first", "second"],
        )
    assert scores == [0.1, 0.9]
    assert tokens == 0


def test_parse_rate_headers_remaining() -> None:
    parsed = parse_rate_headers(
        {
            "x-ratelimit-remaining-requests": "251",
            "X-RateLimit-Remaining-Tokens": "1850000",
            "X-RateLimit-Limit-Requests": "500",
            "X-RateLimit-Limit-Tokens": "2000000",
        }
    )
    assert parsed["remaining_requests"] == 251
    assert parsed["remaining_tokens"] == 1_850_000
    assert parsed["limit_requests"] == 500
    assert parsed["limit_tokens"] == 2_000_000
    limiter = RateLimiter(rpm=500, tpm=2_000_000)
    limiter.note_headers(
        {"X-RateLimit-Remaining-Requests": "10", "X-RateLimit-Remaining-Tokens": "99"}
    )
    snap = limiter.snapshot()
    assert snap["remaining_requests"] == 10
    assert snap["remaining_tokens"] == 99


def test_rate_limiter_records_requests() -> None:
    limiter = RateLimiter(rpm=500, tpm=2_000_000)
    limiter.acquire(10)
    limiter.acquire(10)
    assert len(limiter._requests) == 2
    assert sum(tokens for _, tokens in limiter._tokens) == 20


def test_shared_429_cooldown_blocks_then_clears() -> None:
    limiter = RateLimiter()
    delay = limiter.note_429(0.05)
    assert delay >= 0.05
    assert limiter.snapshot()["retries"] == 1
    limiter.note_success(ms=12)
    assert limiter.snapshot()["last_ok_ms"] == 12
