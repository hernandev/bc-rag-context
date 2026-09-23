"""Dense retrieve, then optional cross-encoder rerank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bc_rag.catalog import ProjectEntry, find_project, living_projects
from bc_rag.config import RagConfig, load_config
from bc_rag.embeddings import Embedder, Reranker
from bc_rag.store import Hit, HybridStore


@dataclass(slots=True)
class QueryResult:
    query: str
    hits: list[Hit]


def search(
    *,
    query: str,
    config: RagConfig,
    embedder: Embedder,
    store: HybridStore,
    reranker: Reranker | None,
    limit: int | None = None,
    use_rerank: bool | None = None,
    groups: list[str] | None = None,
    tags: list[str] | None = None,
) -> QueryResult:
    retrieve = config.retrieve
    final_limit = limit if limit is not None else retrieve.limit
    do_rerank = retrieve.rerank if use_rerank is None else use_rerank
    dense, _sparse = embedder.embed_query(query)
    candidate_limit = max(retrieve.prefetch, final_limit)
    if do_rerank and reranker is not None:
        candidate_limit = max(candidate_limit, final_limit * 4)
    hits = store.query(
        dense=dense,
        prefetch=candidate_limit,
        limit=candidate_limit if (do_rerank and reranker is not None) else final_limit,
        groups=groups,
        tags=tags,
    )
    if do_rerank and reranker is not None and hits:
        scores = reranker.rerank(query, [hit.text for hit in hits])
        ranked = sorted(zip(scores, hits, strict=True), key=lambda row: row[0], reverse=True)
        hits = []
        for score, hit in ranked[:final_limit]:
            hit.score = float(score)
            hits.append(hit)
    else:
        hits = hits[:final_limit]
    return QueryResult(query=query, hits=hits)


def search_sparse(
    *,
    query: str,
    config: RagConfig,
    embedder: Embedder,
    store: HybridStore,
    limit: int | None = None,
    groups: list[str] | None = None,
    tags: list[str] | None = None,
) -> QueryResult:
    """BM25 only. No dense vector and no rerank."""
    final_limit = limit if limit is not None else config.retrieve.limit
    sparse = embedder.embed_sparse_query(query)
    hits = store.query_sparse(
        sparse=sparse,
        limit=final_limit,
        groups=groups,
        tags=tags,
    )
    return QueryResult(query=query, hits=hits)


_embedder_cache: dict[tuple[str, str, bool], Embedder] = {}
_reranker_cache: dict[tuple[str, bool], Reranker] = {}


def search_projects(
    query: str,
    *,
    project: str | None = None,
    limit: int | None = None,
    use_rerank: bool = True,
    groups: list[str] | None = None,
    tags: list[str] | None = None,
    sparse: bool = False,
) -> QueryResult:
    """Search one cataloged project, or all of them through a single MCP."""
    entries = living_projects()
    if project:
        found = find_project(project, entries)
        if found is None:
            raise ValueError(f"unknown project: {project}")
        entries = [found]
    if not entries:
        return QueryResult(query=query, hits=[])

    pooled: list[Hit] = []
    reranker: Reranker | None = None
    final_limit = 8
    for entry in entries:
        hits, cfg, ranker = _search_entry(
            entry,
            query,
            limit=limit,
            use_rerank=False,
            groups=groups,
            tags=tags,
            sparse=sparse,
        )
        pooled.extend(hits)
        if ranker is not None:
            reranker = ranker
        final_limit = limit if limit is not None else cfg.retrieve.limit

    if not sparse and use_rerank and reranker is not None and pooled:
        scores = reranker.rerank(query, [hit.text for hit in pooled])
        ranked = sorted(zip(scores, pooled, strict=True), key=lambda row: row[0], reverse=True)
        hits = []
        for score, hit in ranked[:final_limit]:
            hit.score = float(score)
            hits.append(hit)
        return QueryResult(query=query, hits=hits)

    pooled.sort(key=lambda hit: hit.score, reverse=True)
    return QueryResult(query=query, hits=pooled[:final_limit])


def _search_entry(
    entry: ProjectEntry,
    query: str,
    *,
    limit: int | None,
    use_rerank: bool,
    groups: list[str] | None = None,
    tags: list[str] | None = None,
    sparse: bool = False,
) -> tuple[list[Hit], RagConfig, Reranker | None]:
    root = Path(entry.root)
    config, _ = load_config(root)
    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(root),
        read_only=True,
    )
    try:
        try:
            empty = store.count() == 0
        except Exception:
            return [], config, None
        if empty:
            return [], config, None
        embedder = _embedder_for(config.embed)
        reranker = None
        if use_rerank and config.retrieve.rerank and config.embed.active_rerank():
            reranker = _reranker_for(
                config.embed.active_rerank(), jina_api=config.embed.jina_api
            )
        if sparse:
            result = search_sparse(
                query=query,
                config=config,
                embedder=embedder,
                store=store,
                limit=limit,
                groups=groups,
                tags=tags,
            )
        else:
            result = search(
                query=query,
                config=config,
                embedder=embedder,
                store=store,
                reranker=reranker,
                limit=limit,
                use_rerank=use_rerank,
                groups=groups,
                tags=tags,
            )
        for hit in result.hits:
            hit.project = entry.name
        return result.hits, config, reranker if config.retrieve.rerank else None
    finally:
        store.close()


def _embedder_for(embed) -> Embedder:
    key = (embed.active_dense(), embed.sparse, embed.jina_api)
    cached = _embedder_cache.get(key)
    if cached is None:
        cached = Embedder(embed.active_dense(), embed.sparse, jina_api=embed.jina_api)
        _embedder_cache[key] = cached
    return cached


def _reranker_for(model: str, *, jina_api: bool = False) -> Reranker:
    key = (model, jina_api)
    cached = _reranker_cache.get(key)
    if cached is None:
        cached = Reranker(model, jina_api=jina_api)
        _reranker_cache[key] = cached
    return cached
