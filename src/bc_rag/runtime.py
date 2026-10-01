"""Shared wiring for one registered project: open the store, load the embedder, index."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from bc_rag.catalog import ProjectEntry
from bc_rag.config import RagConfig, SpaceConfig, load_config
from bc_rag.embeddings import Embedder, Reranker
from bc_rag.store import HybridStore


@dataclass
class Session:
    entry: ProjectEntry
    config: RagConfig
    config_path: Path | None
    store: HybridStore
    embedder: Embedder
    reranker: Reranker | None
    console: Console
    space: str | None = None

    @property
    def root(self) -> Path:
        return self.entry.root_path()

    def close(self) -> None:
        self.store.close()


def open_session(
    entry: ProjectEntry,
    *,
    need_reranker: bool = False,
    write: bool = True,
    console: Console | None = None,
    space: str | None = None,
    config: RagConfig | None = None,
    config_path: Path | None = None,
) -> Session:
    """One space of one project. Pass `config` to reuse a config that is already loaded."""
    console = console or Console(stderr=True)
    root = entry.root_path()
    if config is None:
        config, config_path = load_config(root)
    name = space or config.default_space
    spec = config.space_named(name)
    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(root, name),
        read_only=not write,
    )
    embedder = make_embedder(config, spec)
    reranker = None
    rerank_model = None if spec.rerank is None else spec.rerank.model
    rerank_is_jina = spec.rerank is not None and spec.rerank.provider == "jina"
    if need_reranker and spec.retrieve.rerank and rerank_model:
        reranker = Reranker(rerank_model, jina_api=rerank_is_jina)
    return Session(
        entry=entry,
        config=config,
        config_path=config_path,
        store=store,
        embedder=embedder,
        reranker=reranker,
        console=console,
        space=name,
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
    entry: ProjectEntry,
    *,
    need_reranker: bool = False,
    write: bool = True,
    console: Console | None = None,
) -> list[Session]:
    """One session per configured space. The config is loaded once and shared."""
    config, config_path = load_config(entry.root_path())
    return [
        open_session(
            entry,
            need_reranker=need_reranker,
            write=write,
            console=console,
            space=name,
            config=config,
            config_path=config_path,
        )
        for name in config.spaces
    ]


def index_project(
    entry: ProjectEntry,
    *,
    force: bool = False,
    only_paths: list[str] | None = None,
    console: Console | None = None,
    stop: threading.Event | None = None,
):
    """One index pass over every space of one project. Sessions open fresh, so edits apply.

    `stop` ends the pass after the files in flight. Raises IndexBusyError when another
    process is indexing this project.
    """
    from bc_rag.cache import cache_dir
    from bc_rag.indexer import Indexer, index_spaces

    console = console or Console(stderr=True)
    root = entry.root_path()
    sessions = open_space_sessions(entry, write=True, console=console)
    try:
        indexers = [
            Indexer(
                entry,
                session.config,
                session.embedder,
                session.store,
                console,
                space=session.space,
            )
            for session in sessions
        ]
        first = sessions[0]
        console.print(f"[dim]project[/dim] {entry.name}  {root}")
        console.print(f"[dim]config[/dim] {first.config_path}")
        console.print(f"[dim]cache[/dim] {cache_dir(first.config.project_dir(root))}")
        for session in sessions:
            spec = session.config.space_named(session.space or session.config.default_space)
            console.print(
                f"[dim]space[/dim] {session.space}  dense {spec.dense.provider} "
                f"{spec.dense.model}  sparse {spec.sparse.model}"
            )
            console.print(f"[dim]store[/dim] {session.config.store_dir(root, session.space)}")
        return index_spaces(indexers, force=force, only_paths=only_paths, stop=stop)
    finally:
        for session in sessions:
            session.close()
