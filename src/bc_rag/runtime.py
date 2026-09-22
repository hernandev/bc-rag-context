"""Shared CLI wiring: resolve root, open store, load embedder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from bc_rag.catalog import register_project, store_dir_for_root
from bc_rag.config import RagConfig, load_config, with_jina_api
from bc_rag.defaults import STORE_DIRNAME
from bc_rag.embeddings import Embedder, Reranker
from bc_rag.store import HybridStore


@dataclass
class Session:
    root: Path
    config: RagConfig
    config_path: Path | None
    store: HybridStore
    embedder: Embedder
    reranker: Reranker | None
    console: Console

    def close(self) -> None:
        self.store.close()


def resolve_root(root: Path | None) -> Path:
    return (root or Path.cwd()).expanduser().resolve()


def open_session(
    root: Path,
    *,
    need_reranker: bool = False,
    write: bool = True,
    console: Console | None = None,
    jina_api: bool | None = None,
) -> Session:
    console = console or Console(stderr=True)
    config, config_path = load_config(root)
    if jina_api is not None:
        config = with_jina_api(config, jina_api)
    if write:
        register_project(root)
    legacy = root / STORE_DIRNAME
    had_legacy = legacy.is_dir()
    dest = store_dir_for_root(root)
    if had_legacy and not legacy.is_dir():
        console.print(f"migrated {legacy} -> {dest}")
    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(root),
        read_only=not write,
    )
    embedder = Embedder(
        config.embed.active_dense(),
        config.embed.sparse,
        jina_api=config.embed.jina_api,
    )
    reranker = None
    rerank_model = config.embed.active_rerank()
    if need_reranker and config.retrieve.rerank and rerank_model:
        reranker = Reranker(rerank_model, jina_api=config.embed.jina_api)
    return Session(
        root=root,
        config=config,
        config_path=config_path,
        store=store,
        embedder=embedder,
        reranker=reranker,
        console=console,
    )
