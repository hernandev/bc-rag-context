"""Voyage HTTP client. No SDK. Context models take a different request shape."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from bc_rag.defaults import (
    VOYAGE_API_KEY_ENV,
    VOYAGE_API_URL,
    VOYAGE_AUTO_TOKENS,
    VOYAGE_MAX_CHUNKS,
    VOYAGE_PRECHUNK_TOKENS,
)
from bc_rag.jina_api import RateLimiter, _retry_after_seconds, estimate_tokens

# Tier-1 tokens per minute from the Voyage rate-limit table.
_TPM = {
    "voyage-4": 8_000_000,
    "voyage-code-4": 8_000_000,
    "voyage-3.5": 8_000_000,
    "voyage-4-large": 3_000_000,
    "voyage-context-4": 3_000_000,
    "voyage-context-3": 3_000_000,
    "voyage-4-lite": 16_000_000,
    "voyage-3.5-lite": 16_000_000,
    "rerank-3": 2_000_000,
    "rerank-2.5": 2_000_000,
    "rerank-3-lite": 4_000_000,
    "rerank-2.5-lite": 4_000_000,
}
_RPM = 2000
_RERANK_TOKEN_BUDGET = 600_000
_POST_ATTEMPTS = 12
_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})

_limiters: dict[str, RateLimiter] = {}


class VoyageApiError(RuntimeError):
    pass


def is_voyage_model(model: str) -> bool:
    return model.startswith("voyage-") or model.startswith("rerank-")


def is_contextual_model(model: str) -> bool:
    return model.startswith("voyage-context")


def resolve_api_key(explicit: str | None = None) -> str:
    """The explicit key, else `~/.bc-rag/config`, else the environment."""
    from bc_rag.usersettings import config_setting

    key = (
        explicit
        or config_setting("voyage-api-key")
        or os.environ.get(VOYAGE_API_KEY_ENV)
        or os.environ.get("VOYAGE_API_KEY")
        or ""
    ).strip()
    if not key:
        raise VoyageApiError(
            "no Voyage key. Run `bc-rag config set voyage-api-key <key> --global` "
            f"or set {VOYAGE_API_KEY_ENV}."
        )
    return key


def limiter_for(model: str) -> RateLimiter:
    found = _limiters.get(model)
    if found is None:
        found = RateLimiter(rpm=_RPM, tpm=_TPM.get(model, 3_000_000))
        _limiters[model] = found
    return found


def rate_snapshot() -> dict[str, int | None]:
    """One view of every Voyage model limiter. The index panel reads this, not Jina."""
    snaps = [item.snapshot() for item in _limiters.values()]
    if not snaps:
        return {
            "rpm": 0,
            "tpm": 0,
            "remaining_requests": None,
            "remaining_tokens": None,
            "retries": 0,
            "retries_min": 0,
            "cool_s": 0,
            "last_ok_ms": 0,
        }

    def total(key: str) -> int:
        return sum(int(item.get(key) or 0) for item in snaps)

    return {
        "rpm": total("rpm"),
        "tpm": total("tpm"),
        "remaining_requests": None,
        "remaining_tokens": None,
        "retries": total("retries"),
        "retries_min": total("retries_min"),
        "cool_s": max(int(item.get("cool_s") or 0) for item in snaps),
        "last_ok_ms": max(int(item.get("last_ok_ms") or 0) for item in snaps),
    }


def split_token_spans(
    token_counts: list[int],
    *,
    max_tokens: int = VOYAGE_PRECHUNK_TOKENS,
    max_items: int = VOYAGE_MAX_CHUNKS,
) -> list[tuple[int, int]]:
    """Half-open index ranges. One oversized item stays in a range by itself."""
    spans: list[tuple[int, int]] = []
    start = 0
    total = 0
    count = 0
    for index, cost in enumerate(token_counts):
        if count and (count >= max_items or total + cost > max_tokens):
            spans.append((start, index))
            start = index
            total = 0
            count = 0
        total += cost
        count += 1
    if count:
        spans.append((start, start + count))
    return spans


def embed_texts(
    *,
    model: str,
    texts: list[str],
    query: bool = False,
    dimensions: int | None = None,
    api_key: str | None = None,
    base_url: str = VOYAGE_API_URL,
) -> tuple[list[list[float]], int]:
    """Flat `/v1/embeddings`. One string in, one vector out. Files may share a request."""
    if not texts:
        return [], 0
    payload: dict[str, Any] = {
        "input": texts,
        "model": model,
        "input_type": "query" if query else "document",
    }
    if dimensions is not None:
        payload["output_dimension"] = dimensions
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/embeddings",
        payload,
        api_key=api_key,
        model=model,
        tokens=estimate_tokens(texts),
    )
    return _flat_vectors(body, expected=len(texts)), _usage_tokens(body)


def embed_contextual_chunks(
    *,
    model: str,
    chunk_texts: list[str],
    dimensions: int | None = None,
    api_key: str | None = None,
    base_url: str = VOYAGE_API_URL,
) -> tuple[list[list[float]], int]:
    """One document. `chunk_texts` is that document's chunks, in order."""
    if not chunk_texts:
        return [], 0
    payload: dict[str, Any] = {
        "inputs": [chunk_texts],
        "model": model,
        "input_type": "document",
    }
    if dimensions is not None:
        payload["output_dimension"] = dimensions
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/contextualizedembeddings",
        payload,
        api_key=api_key,
        model=model,
        tokens=estimate_tokens(chunk_texts),
    )
    groups = _contextual_groups(body)
    if len(groups) != 1 or len(groups[0]) != len(chunk_texts):
        raise VoyageApiError("voyage contextual response does not match the chunk list")
    return groups[0], _usage_tokens(body)


def embed_contextual_query(
    *,
    model: str,
    text: str,
    dimensions: int | None = None,
    api_key: str | None = None,
    base_url: str = VOYAGE_API_URL,
) -> tuple[list[float], int]:
    """A query has no siblings. It still has to use the contextual endpoint."""
    payload: dict[str, Any] = {
        "inputs": [text],
        "model": model,
        "input_type": "query",
    }
    if dimensions is not None:
        payload["output_dimension"] = dimensions
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/contextualizedembeddings",
        payload,
        api_key=api_key,
        model=model,
        tokens=estimate_tokens([text]),
    )
    groups = _contextual_groups(body)
    if not groups or not groups[0]:
        raise VoyageApiError("voyage contextual query returned no vector")
    return groups[0][0], _usage_tokens(body)


def embed_contextual_auto(
    *,
    model: str,
    document: str,
    dimensions: int | None = None,
    api_key: str | None = None,
    base_url: str = VOYAGE_API_URL,
) -> tuple[list[tuple[str, list[float]]], int]:
    """Voyage splits the file. Returns `(chunk text, vector)` in chunk order."""
    payload: dict[str, Any] = {
        "inputs": [document],
        "model": model,
        "input_type": "document",
        "enable_auto_chunking": True,
    }
    if dimensions is not None:
        payload["output_dimension"] = dimensions
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/contextualizedembeddings",
        payload,
        api_key=api_key,
        model=model,
        tokens=min(estimate_tokens([document]), VOYAGE_AUTO_TOKENS),
    )
    pieces = _contextual_pieces(body)
    if not pieces:
        raise VoyageApiError("voyage auto-chunk returned no chunks")
    return pieces, _usage_tokens(body)


def rerank_texts(
    *,
    model: str,
    query: str,
    documents: list[str],
    api_key: str | None = None,
    base_url: str = VOYAGE_API_URL,
) -> tuple[list[float], int]:
    if not documents:
        return [], 0
    documents = _trim_rerank_documents(query, documents)
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    repeated_query_tokens = estimate_tokens([query]) * (len(documents) - 1)
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/rerank",
        payload,
        api_key=api_key,
        model=model,
        tokens=estimate_tokens([query, *documents]) + repeated_query_tokens,
    )
    # Voyage answers {"object": "list", "data": [{"index", "relevance_score"}], ...}.
    # Jina and Cohere name the same list "results"; Voyage does not.
    results = body.get("data")
    if not isinstance(results, list):
        raise VoyageApiError("voyage rerank response is missing data")
    scores = [0.0] * len(documents)
    for row in results:
        if not isinstance(row, dict):
            continue
        index = int(row.get("index", -1))
        if 0 <= index < len(scores):
            scores[index] = float(row.get("relevance_score", 0.0))
    return scores, _usage_tokens(body)


def _trim_rerank_documents(query: str, documents: list[str]) -> list[str]:
    """Stay under the 600k token total published for the rerank-2.5 family."""
    query_tokens = estimate_tokens([query])
    kept: list[str] = []
    total = 0
    for document in documents:
        cost = query_tokens + estimate_tokens([document])
        if kept and total + cost > _RERANK_TOKEN_BUDGET:
            break
        kept.append(document)
        total += cost
    return kept or documents[:1]


def _flat_vectors(body: dict[str, Any], *, expected: int) -> list[list[float]]:
    rows = body.get("data")
    if not isinstance(rows, list) or len(rows) != expected:
        raise VoyageApiError("voyage embeddings response length does not match input")
    ordered = sorted(rows, key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0)
    vectors: list[list[float]] = []
    for row in ordered:
        if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
            raise VoyageApiError("voyage embeddings response is missing float vectors")
        vectors.append([float(value) for value in row["embedding"]])
    return vectors


def _contextual_groups(body: dict[str, Any]) -> list[list[list[float]]]:
    rows = body.get("data")
    if not isinstance(rows, list):
        raise VoyageApiError("voyage contextual response is missing data")
    ordered = sorted(rows, key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0)
    groups: list[list[list[float]]] = []
    for row in ordered:
        if not isinstance(row, dict):
            raise VoyageApiError("voyage contextual response row is not an object")
        nested = row.get("data")
        if isinstance(row.get("embedding"), list):
            groups.append([[float(value) for value in row["embedding"]]])
            continue
        if not isinstance(nested, list):
            raise VoyageApiError("voyage contextual response is missing chunk vectors")
        nested_ordered = sorted(
            nested, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0
        )
        vectors: list[list[float]] = []
        for item in nested_ordered:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise VoyageApiError("voyage contextual chunk is missing a vector")
            vectors.append([float(value) for value in item["embedding"]])
        groups.append(vectors)
    return groups


def _contextual_pieces(body: dict[str, Any]) -> list[tuple[str, list[float]]]:
    rows = body.get("data")
    if not isinstance(rows, list) or not rows:
        return []
    first = rows[0]
    if not isinstance(first, dict):
        return []
    nested = first.get("data")
    if not isinstance(nested, list):
        return []
    ordered = sorted(
        nested,
        key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
    )
    pieces: list[tuple[str, list[float]]] = []
    for item in ordered:
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            continue
        text = item.get("text")
        vector = [float(value) for value in item["embedding"]]
        pieces.append((text if isinstance(text, str) else "", vector))
    return pieces


def _usage_tokens(body: dict[str, Any]) -> int:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get("total_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    model: str,
    tokens: int,
) -> dict[str, Any]:
    key = resolve_api_key(api_key)
    limiter = limiter_for(model)
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(_POST_ATTEMPTS):
        limiter.acquire(tokens)
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise VoyageApiError("voyage response is not a JSON object")
            return body
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:800]
            retryable = error.code in _RETRYABLE_HTTP and attempt < _POST_ATTEMPTS - 1
            if retryable:
                last_error = VoyageApiError(f"voyage {error.code}: {detail}")
                if error.code == 429:
                    limiter.note_429(_retry_after_seconds(error))
                    continue
                time.sleep(min(2 ** attempt, 30))
                continue
            raise VoyageApiError(f"voyage {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            reason = getattr(error, "reason", error)
            last_error = VoyageApiError(f"voyage connection failed: {reason}")
            if attempt < _POST_ATTEMPTS - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise last_error from error
    raise last_error or VoyageApiError("voyage request failed")
