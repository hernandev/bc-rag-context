"""Jina Search Foundation HTTP client. Dense embed and rerank only."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

from bc_rag.defaults import JINA_API_KEY_ENV, JINA_API_URL, JINA_RPM, JINA_TPM


class JinaApiError(RuntimeError):
    pass


class RateLimiter:
    """Sliding 60s window for RPM and TPM."""

    def __init__(self, rpm: int = JINA_RPM, tpm: int = JINA_TPM) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self.remaining_requests: int | None = None
        self.remaining_tokens: int | None = None
        self.limit_requests: int | None = None
        self.limit_tokens: int | None = None
        self.last_header_names: list[str] = []
        self._cool_until = 0.0
        self._backoff = 1.0
        self.retries = 0
        self.last_ok_ms = 0.0
        self._rate_limits: deque[float] = deque()

    def acquire(self, tokens: int = 0) -> None:
        """Wait out a shared 429 cooldown, then record the send for the HUD."""
        self.wait_if_cooling()
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._requests and self._requests[0] < cutoff:
                self._requests.popleft()
            while self._tokens and self._tokens[0][0] < cutoff:
                self._tokens.popleft()
            self._requests.append(now)
            if tokens:
                self._tokens.append((now, tokens))

    def wait_if_cooling(self) -> None:
        while True:
            with self._lock:
                remain = self._cool_until - time.monotonic()
            if remain <= 0:
                return
            time.sleep(min(remain, 0.25))

    def note_429(self, retry_after: float | None = None, *, token_limit: bool = False) -> float:
        with self._lock:
            now = time.monotonic()
            if token_limit:
                delay = 60.0
            elif retry_after is not None:
                delay = retry_after
            else:
                delay = self._backoff
            delay = min(max(delay, 1.0), 60.0)
            self._cool_until = max(self._cool_until, now + delay)
            self._backoff = min(self._backoff * 2.0, 60.0)
            self.retries += 1
            self._rate_limits.append(now)
            return delay

    def note_success(self, *, ms: float) -> None:
        with self._lock:
            self.last_ok_ms = ms

    def snapshot(self) -> dict[str, int | None]:
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._requests and self._requests[0] < cutoff:
                self._requests.popleft()
            while self._tokens and self._tokens[0][0] < cutoff:
                self._tokens.popleft()
            while self._rate_limits and self._rate_limits[0] < cutoff:
                self._rate_limits.popleft()
            return {
                "rpm": len(self._requests),
                "rpm_cap": self.rpm,
                "tpm": sum(count for _, count in self._tokens),
                "tpm_cap": self.tpm,
                "remaining_requests": self.remaining_requests,
                "remaining_tokens": self.remaining_tokens,
                "limit_requests": self.limit_requests,
                "limit_tokens": self.limit_tokens,
                "header_names": list(self.last_header_names),
                "retries": self.retries,
                "retries_min": len(self._rate_limits),
                "cool_s": max(0, int(self._cool_until - now)),
                "last_ok_ms": int(self.last_ok_ms),
            }

    def note_headers(self, headers: Any) -> None:
        parsed = parse_rate_headers(headers)
        names = header_names(headers)
        with self._lock:
            self.last_header_names = names
            for key, value in parsed.items():
                if value is not None:
                    setattr(self, key, value)


_embed_limiter = RateLimiter()
_rerank_limiter = RateLimiter()


def embed_rate_snapshot() -> dict[str, int | None]:
    return _embed_limiter.snapshot()


def parse_rate_headers(headers: Any) -> dict[str, int | None]:
    """Read remaining/limit ints from whatever names the server actually sent."""
    out: dict[str, int | None] = {
        "remaining_requests": None,
        "remaining_tokens": None,
        "limit_requests": None,
        "limit_tokens": None,
    }
    if headers is None:
        return out
    try:
        items = list(headers.items())
    except Exception:
        return out
    for key, raw in items:
        low = str(key).lower().replace("_", "-")
        try:
            number = int(str(raw).split(",")[0].strip())
        except (TypeError, ValueError):
            continue
        if "remaining" in low and "request" in low:
            out["remaining_requests"] = number
        elif "remaining" in low and "token" in low:
            out["remaining_tokens"] = number
        elif low.endswith("remaining") and "rate" in low:
            out["remaining_requests"] = out["remaining_requests"] or number
        elif "limit" in low and "request" in low and "remaining" not in low:
            out["limit_requests"] = number
        elif "limit" in low and "token" in low and "remaining" not in low:
            out["limit_tokens"] = number
    return out


def header_names(headers: Any) -> list[str]:
    if headers is None:
        return []
    try:
        return [str(key) for key, _ in headers.items()]
    except Exception:
        return []


def estimate_tokens(texts: list[str]) -> int:
    return max(1, sum(len(text) for text in texts) // 4)


def usage_tokens(body: dict[str, Any]) -> int:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    if value is None:
        value = usage.get("prompt_tokens")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def api_model_name(model: str) -> str:
    if model.startswith("jinaai/"):
        return model.removeprefix("jinaai/")
    return model


def resolve_api_key(explicit: str | None = None) -> str:
    key = (explicit or os.environ.get(JINA_API_KEY_ENV) or "").strip()
    if not key:
        raise JinaApiError(
            f"{JINA_API_KEY_ENV} is not set. Get a key at https://jina.ai/api-dashboard/key-manager"
        )
    return key


def embed_texts(
    *,
    model: str,
    texts: list[str],
    query: bool = False,
    api_key: str | None = None,
    base_url: str = JINA_API_URL,
) -> tuple[list[list[float]], int]:
    if not texts:
        return [], 0
    api_model = api_model_name(model)
    payload: dict[str, Any] = {
        "model": api_model,
        "input": texts,
        "normalized": True,
        "truncate": True,
        "embedding_type": "float",
    }
    if api_model.startswith("jina-embeddings-v5") or api_model.startswith("jina-embeddings-v4"):
        payload["task"] = "retrieval.query" if query else "retrieval.passage"
        if api_model.endswith("text-small") or api_model.endswith("omni-small"):
            payload["dimensions"] = 1024
    elif query and not api_model.startswith("jina-embeddings-v2"):
        payload["task"] = "retrieval.query"
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/embeddings",
        payload,
        api_key=api_key,
        limiter=_embed_limiter,
        tokens=estimate_tokens(texts),
    )
    rows = body.get("data")
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise JinaApiError("jina embeddings response length does not match input")
    ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
    vectors: list[list[float]] = []
    for row in ordered:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            raise JinaApiError("jina embeddings response is missing float vectors")
        vectors.append([float(value) for value in embedding])
    return vectors, usage_tokens(body)


def rerank_texts(
    *,
    model: str,
    query: str,
    documents: list[str],
    api_key: str | None = None,
    base_url: str = JINA_API_URL,
) -> tuple[list[float], int]:
    if not documents:
        return [], 0
    payload = {
        "model": api_model_name(model),
        "query": query,
        "documents": documents,
        "return_documents": False,
    }
    body = _post_json(
        f"{base_url.rstrip('/')}/v1/rerank",
        payload,
        api_key=api_key,
        limiter=_rerank_limiter,
        tokens=estimate_tokens([query, *documents]),
    )
    results = body.get("results")
    if not isinstance(results, list):
        raise JinaApiError("jina rerank response is missing results")
    scores = [0.0] * len(documents)
    for row in results:
        if not isinstance(row, dict):
            continue
        index = int(row.get("index", -1))
        if 0 <= index < len(scores):
            scores[index] = float(row.get("relevance_score", 0.0))
    return scores, usage_tokens(body)


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    limiter: RateLimiter | None = None,
    tokens: int = 0,
) -> dict[str, Any]:
    key = resolve_api_key(api_key)
    if limiter is not None:
        limiter.acquire(tokens)
    encoded = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    attempts = 12
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=encoded, method="POST")
        request.add_header("Authorization", f"Bearer {key}")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        try:
            sent = time.perf_counter()
            with urllib.request.urlopen(request, timeout=120) as response:
                if limiter is not None:
                    limiter.note_headers(response.headers)
                raw = response.read().decode("utf-8")
            if limiter is not None:
                limiter.note_success(ms=(time.perf_counter() - sent) * 1000)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise JinaApiError("jina response is not a JSON object")
            return parsed
        except urllib.error.HTTPError as error:
            if limiter is not None:
                limiter.note_headers(error.headers)
            detail = error.read().decode("utf-8", errors="replace")[:800]
            retryable = error.code in {408, 429, 500, 502, 503, 504}
            if retryable and attempt < attempts - 1:
                last_error = JinaApiError(f"jina {error.code}: {detail}")
                if error.code == 429 and limiter is not None:
                    token_limit = "TOKEN" in detail.upper() or "tpm" in detail.lower()
                    limiter.note_429(
                        _retry_after_seconds(error),
                        token_limit=token_limit,
                    )
                    continue
                time.sleep(_retry_delay(error, attempt))
                continue
            raise JinaApiError(f"jina {error.code}: {detail}") from error
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            last_error = JinaApiError(f"jina request failed: {error}")
            if attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise last_error from error
    raise last_error or JinaApiError("jina request failed")


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    header = error.headers.get("Retry-After") if error.headers else None
    if not header:
        return None
    try:
        return min(float(header), 120.0)
    except ValueError:
        return None


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    header = _retry_after_seconds(error)
    if header is not None:
        return header
    if error.code == 429:
        return min(1.0 * (2 ** attempt), 60.0)
    return min(2 ** attempt, 20)
