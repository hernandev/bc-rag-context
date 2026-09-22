"""Docker Compose helpers for the shared Qdrant HTTP server."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from bc_rag.defaults import QDRANT_COMPOSE_PROJECT, QDRANT_DOCKER_CONTEXT, QDRANT_HTTP_URL


def compose_file() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "compose.yml"


def compose_cmd(*args: str) -> list[str]:
    return [
        "docker",
        "--context",
        QDRANT_DOCKER_CONTEXT,
        "compose",
        "-f",
        str(compose_file()),
        "-p",
        QDRANT_COMPOSE_PROJECT,
        *args,
    ]


def run_compose(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        compose_cmd(*args),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"docker compose {' '.join(args)} failed")
    return result


def start() -> None:
    Path.home().joinpath(".bc-rag", "qdrant-storage").mkdir(parents=True, exist_ok=True)
    run_compose("up", "-d")
    wait_ready()


def stop() -> None:
    run_compose("stop")


def restart() -> None:
    run_compose("restart")
    wait_ready()


def import_embedded_collection(
    *,
    source_path: Path,
    url: str,
    collection: str,
    dense_dim: int,
    source_collection: str = "chunks",
    batch: int = 128,
) -> int:
    """Copy vectors from embedded-path Qdrant to Docker. Does not re-embed."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    from bc_rag.store import HybridStore

    src = QdrantClient(path=str(source_path))
    dest = HybridStore(url=url, collection=collection)
    try:
        if not src.collection_exists(source_collection):
            return 0
        dest.ensure_collection(dense_dim)
        copied = 0
        offset = None
        while True:
            records, offset = src.scroll(
                collection_name=source_collection,
                offset=offset,
                limit=batch,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            dest.client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(id=record.id, vector=record.vector, payload=record.payload)
                    for record in records
                ],
            )
            copied += len(records)
            if offset is None:
                break
        return copied
    finally:
        dest.close()
        close = getattr(src, "close", None)
        if callable(close):
            close()


def wait_ready(*, timeout_s: float = 30.0, url: str = QDRANT_HTTP_URL) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/readyz", timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = str(error)
        time.sleep(0.25)
    raise RuntimeError(f"qdrant not ready at {url}: {last}")
