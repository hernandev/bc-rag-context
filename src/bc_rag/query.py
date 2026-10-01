"""Search one space: dense retrieve then rerank, or BM25 only.

Rerank follows the space's config, the same rule everywhere: a space reranks when
`retrieve.rerank` is true and it names a `rerank` model. Dense search then asks the
store for a wider candidate pool (`max(prefetch, limit * 4)`) and keeps the best
`limit` after rerank.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from bc_rag.catalog import ProjectEntry, find_project, living_projects
from bc_rag.config import RagConfig, SpaceConfig, load_config
from bc_rag.embeddings import Embedder, Reranker
from bc_rag.facets import Facets
from bc_rag.store import Hit, HybridStore


@dataclass(slots=True)
class QueryResult:
    query: str
    hits: list[Hit]
    reranked: bool = False


def search(
    *,
    query: str,
    config: RagConfig,
    embedder: Embedder,
    store: HybridStore,
    reranker: Reranker | None,
    limit: int | None = None,
    use_rerank: bool | None = None,
    facets: Facets | None = None,
    exclude: Facets | None = None,
    space: str | None = None,
) -> QueryResult:
    """Dense search. Reranks when `use_rerank` (None = the space's retrieve.rerank) and a
    reranker is given."""
    retrieve = config.space_named(space or config.default_space).retrieve
    final_limit = limit if limit is not None else retrieve.limit
    do_rerank = (retrieve.rerank if use_rerank is None else use_rerank) and reranker is not None
    dense, _sparse = embedder.embed_query(query)
    candidates = max(retrieve.prefetch, final_limit * 4) if do_rerank else final_limit
    hits = store.query(dense=dense, limit=candidates, facets=facets, exclude=exclude)
    if do_rerank and hits:
        assert reranker is not None
        scores = reranker.rerank(query, [hit.text for hit in hits])
        ranked = sorted(zip(scores, hits, strict=True), key=lambda row: row[0], reverse=True)
        hits = []
        for score, hit in ranked[:final_limit]:
            hit.score = float(score)
            hits.append(hit)
        return QueryResult(query=query, hits=hits, reranked=True)
    return QueryResult(query=query, hits=hits[:final_limit])


def search_sparse(
    *,
    query: str,
    config: RagConfig,
    embedder: Embedder,
    store: HybridStore,
    limit: int | None = None,
    facets: Facets | None = None,
    exclude: Facets | None = None,
    space: str | None = None,
) -> QueryResult:
    """BM25 only. No dense vector and no rerank."""
    retrieve = config.space_named(space or config.default_space).retrieve
    final_limit = limit if limit is not None else retrieve.limit
    sparse = embedder.embed_sparse_query(query)
    hits = store.query_sparse(sparse=sparse, limit=final_limit, facets=facets, exclude=exclude)
    return QueryResult(query=query, hits=hits)


def search_projects(
    query: str,
    *,
    project: str,
    space: str,
    limit: int | None = None,
    use_rerank: bool | None = None,
    facets: Facets | None = None,
    exclude: Facets | None = None,
    sparse: bool = False,
) -> QueryResult:
    """Search one space of one registered project. `use_rerank` None follows the space."""
    if not project:
        raise ValueError("project is required")
    if not space:
        raise ValueError("space is required")
    entries = living_projects()
    found = find_project(project, entries)
    if found is None:
        known = ", ".join(entry.name for entry in entries) or "none"
        raise ValueError(f"unknown project: {project}. Registered projects: {known}")
    return _search_entry(
        found,
        query,
        space=space,
        limit=limit,
        use_rerank=use_rerank,
        facets=facets,
        exclude=exclude,
        sparse=sparse,
    )


def _search_entry(
    entry: ProjectEntry,
    query: str,
    *,
    space: str,
    limit: int | None,
    use_rerank: bool | None,
    facets: Facets | None = None,
    exclude: Facets | None = None,
    sparse: bool = False,
) -> QueryResult:
    root = entry.root_path()
    config, _path = load_config(root, groups=False)
    spec = config.space_named(space)
    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(root, space),
        read_only=True,
    )
    try:
        try:
            stored_keys = store.facet_keys()
            empty = store.count() == 0
        except Exception as error:
            raise RuntimeError(f"qdrant unreachable at {store.url}: {error}") from error
        for key in [*(facets or {}), *(exclude or {})]:
            if key not in stored_keys and not empty:
                known = ", ".join(stored_keys) or "none"
                raise ValueError(
                    f"facet key {key!r} is not stored in space {space!r}. Keys here: {known}"
                )
        if empty:
            return QueryResult(query=query, hits=[])
        embedder = embedder_for(config, spec)
        if sparse:
            result = search_sparse(
                query=query,
                config=config,
                embedder=embedder,
                store=store,
                limit=limit,
                facets=facets,
                exclude=exclude,
                space=space,
            )
        else:
            want = spec.retrieve.rerank if use_rerank is None else use_rerank
            reranker = None
            if want and spec.rerank is not None:
                reranker = _reranker_for(spec.rerank.model, jina_api=spec.rerank.provider == "jina")
            result = search(
                query=query,
                config=config,
                embedder=embedder,
                store=store,
                reranker=reranker,
                limit=limit,
                use_rerank=want,
                facets=facets,
                exclude=exclude,
                space=space,
            )
        for hit in result.hits:
            hit.project = entry.name
        return result
    finally:
        store.close()


# -- caches ------------------------------------------------------------------------
# One MCP process serves every chat. Building an embedder loads the BM25 ONNX model,
# and a local reranker loads a cross-encoder, so both are kept for the process.

_cache_lock = threading.Lock()
_embedders: dict[tuple[str, str, str, int | None], Embedder] = {}
_rerankers: dict[tuple[str, bool], Reranker] = {}


def embedder_for(config: RagConfig, spec: SpaceConfig) -> Embedder:
    from bc_rag.runtime import make_embedder

    key = (spec.dense.provider, spec.dense.model, spec.sparse.model, spec.dimensions)
    with _cache_lock:
        cached = _embedders.get(key)
        if cached is None:
            cached = make_embedder(config, spec)
            _embedders[key] = cached
        return cached


def _reranker_for(model: str, *, jina_api: bool = False) -> Reranker:
    key = (model, jina_api)
    with _cache_lock:
        cached = _rerankers.get(key)
        if cached is None:
            cached = Reranker(model, jina_api=jina_api)
            _rerankers[key] = cached
        return cached
