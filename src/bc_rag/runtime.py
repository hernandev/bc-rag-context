"""Shared CLI wiring: resolve root, open store, load embedder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from bc_rag.catalog import register_project, store_dir_for_root
from bc_rag.config import RagConfig, SpaceConfig, load_config, with_jina_api
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
    space: str | None = None

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
    space: str | None = None,
) -> Session:
    console = console or Console(stderr=True)
    config, config_path = load_config(root)
    if jina_api is not None:
        config = with_jina_api(config, jina_api)
    spec: SpaceConfig | None = None
    name = space or config.default_space
    spec = config.space_named(name)
    space = name
    if write:
        register_project(root)
    legacy = root / STORE_DIRNAME
    had_legacy = legacy.is_dir()
    dest = store_dir_for_root(root)
    if had_legacy and not legacy.is_dir():
        console.print(f"migrated {legacy} -> {dest}")
    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(root, space),
        read_only=not write,
    )
    embedder = make_embedder(config, spec)
    reranker = None
    rerank_model = None if spec.rerank is None else spec.rerank.model
    rerank_is_jina = spec.rerank is not None and spec.rerank.provider == "jina"
    if need_reranker and spec.retrieve.rerank and rerank_model:
        reranker = Reranker(rerank_model, jina_api=rerank_is_jina)
    return Session(
        root=root,
        config=config,
        config_path=config_path,
        store=store,
        embedder=embedder,
        reranker=reranker,
        console=console,
        space=space,
    )


def make_embedder(config: RagConfig, spec: SpaceConfig | None) -> Embedder:
    """A voyage model uses HTTP. Anything else keeps the Jina or FastEmbed path."""
    from bc_rag.voyage_api import is_contextual_model, is_voyage_model

    if spec is not None and (spec.dense_provider() == "voyage" or is_voyage_model(spec.dense_id())):
        return Embedder(
            spec.dense_id(),
            spec.sparse.model,
            voyage_api=True,
            dimensions=spec.dimensions,
            contextual=is_contextual_model(spec.dense_id()),
            contextual_input="chunks",
        )
    dense = spec.dense_id()
    jina = spec.dense_provider() == "jina"
    return Embedder(dense, spec.sparse.model, jina_api=jina)


def open_space_sessions(
    root: Path,
    *,
    need_reranker: bool = False,
    write: bool = True,
    console: Console | None = None,
    jina_api: bool | None = None,
) -> list[Session]:
    """One session per configured space. No spaces means one session, today's store."""
    config, _path = load_config(root)
    if jina_api is not None:
        config = with_jina_api(config, jina_api)
    names: list[str] = list(config.spaces)
    return [
        open_session(
            root,
            need_reranker=need_reranker,
            write=write,
            console=console,
            jina_api=jina_api,
            space=name,
        )
        for name in names
    ]
