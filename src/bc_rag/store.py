"""Qdrant local (path) store. Named dense vector + BM25 sparse, fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from bc_rag.chunking import Chunk
from bc_rag.defaults import COLLECTION_NAME
from bc_rag.embeddings import SparseVec
from bc_rag.facets import SIDECAR_KEYS, parse_tag_clause


@dataclass(slots=True)
class Hit:
    score: float
    path: str
    language: str
    kind: str
    symbol: str | None
    heading_path: str | None
    start_line: int
    end_line: int
    text: str
    payload: dict[str, Any]
    project: str | None = None


class HybridStore:
    def __init__(
        self,
        path: Path | None = None,
        *,
        url: str | None = None,
        collection: str | None = None,
        read_only: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient

        self.path = path
        self.url = url
        self.read_only = read_only
        self.collection = collection or COLLECTION_NAME
        if url:
            self.client = QdrantClient(url=url, timeout=60, check_compatibility=False)
            return
        if path is None:
            raise ValueError("HybridStore needs path= for tests or url= for Docker Qdrant")
        if not read_only:
            path.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {"path": str(path)}
        try:
            self.client = QdrantClient(read_only=read_only, **kwargs)
        except TypeError:
            self.client = QdrantClient(**kwargs)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def ensure_collection(self, dense_dim: int) -> None:
        from qdrant_client.models import (
            Distance,
            Modifier,
            SparseVectorParams,
            VectorParams,
        )

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = _dense_size(info)
            if existing is not None and existing != dense_dim:
                self.client.delete_collection(self.collection)
            else:
                self._ensure_compact_storage(info)
                self.ensure_payload_indexes()
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE, on_disk=True),
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(modifier=Modifier.IDF),
            },
            quantization_config=_scalar_quantization(),
        )
        self.ensure_payload_indexes()

    def _ensure_compact_storage(self, info: Any) -> None:
        """Move full dense vectors to disk and keep int8 copies in RAM.

        Collections created before this setting get migrated in place. Qdrant
        rebuilds the segments in the background; no reindex is needed.
        """
        if self.read_only:
            return
        from qdrant_client.models import VectorParamsDiff

        params = info.config.params
        vectors = params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        on_disk = bool(getattr(dense, "on_disk", False))
        quantized = info.config.quantization_config is not None
        if on_disk and quantized:
            return
        self.client.update_collection(
            collection_name=self.collection,
            vectors_config=None if on_disk else {"dense": VectorParamsDiff(on_disk=True)},
            quantization_config=None if quantized else _scalar_quantization(),
        )

    def recreate_collection(self, dense_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection(dense_dim)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        result = self.client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    def delete_paths(self, paths: list[str]) -> None:
        if not paths or not self.client.collection_exists(self.collection):
            return
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="path", match=MatchAny(any=paths))])
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVec],
    ) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        if not chunks:
            return
        points = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense, sparse, strict=True):
            points.append(
                PointStruct(
                    id=_point_id(chunk),
                    vector={
                        "dense": dense_vec,
                        "bm25": SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    },
                    payload=_payload(chunk),
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def ensure_payload_indexes(self) -> None:
        from qdrant_client.models import PayloadSchemaType

        if not self.client.collection_exists(self.collection):
            return
        for key in ("group", "tags", *SIDECAR_KEYS):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=key,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def query(
        self,
        *,
        dense: list[float],
        sparse: SparseVec,
        prefetch: int,
        limit: int,
        groups: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[Hit]:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        self.ensure_payload_indexes()
        query_filter = _payload_filter(groups=groups, tags=tags)
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=dense, using="dense", limit=prefetch, filter=query_filter),
                Prefetch(
                    query=SparseVector(indices=sparse.indices, values=sparse.values),
                    using="bm25",
                    limit=prefetch,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[Hit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                Hit(
                    score=float(point.score),
                    path=str(payload.get("path", "")),
                    language=str(payload.get("language", "")),
                    kind=str(payload.get("kind", "")),
                    symbol=payload.get("symbol"),
                    heading_path=payload.get("heading_path"),
                    start_line=int(payload.get("start_line") or 0),
                    end_line=int(payload.get("end_line") or 0),
                    text=str(payload.get("text", "")),
                    payload=payload,
                )
            )
        return hits

    def facet_values(self, key: str, *, limit: int = 500) -> list[tuple[str, int]]:
        """Unique payload values and how many points have each one."""
        if not self.client.collection_exists(self.collection):
            return []
        self.ensure_payload_indexes()
        result = self.client.facet(
            collection_name=self.collection,
            key=key,
            limit=limit,
            exact=False,
        )
        rows: list[tuple[str, int]] = []
        for hit in getattr(result, "hits", []) or []:
            value = getattr(hit, "value", None)
            count = int(getattr(hit, "count", 0) or 0)
            if value is None:
                continue
            rows.append((str(value), count))
        return rows


def _payload_filter(
    *,
    groups: list[str] | None,
    tags: list[str] | None,
):
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must = []
    if groups:
        must.append(FieldCondition(key="group", match=MatchAny(any=list(groups))))
    if tags:
        for tag in tags:
            if not tag:
                continue
            clause = parse_tag_clause(tag)
            if clause is not None:
                key, value = clause
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
            else:
                must.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
    if not must:
        return None
    return Filter(must=must)


def _scalar_quantization():
    """int8 copies of the dense vectors, 4x smaller than float32, pinned in RAM."""
    from qdrant_client.models import (
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
    )

    return ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
    )


def _point_id(chunk: Chunk) -> str:
    return str(uuid5(NAMESPACE_URL, f"{chunk.path}:{chunk.start_byte}:{chunk.end_byte}"))


def _payload(chunk: Chunk) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": chunk.path,
        "language": chunk.language,
        "kind": chunk.kind,
        "symbol": chunk.symbol,
        "heading_path": chunk.heading_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "tags": list(chunk.tags),
        "group": chunk.group,
        "priority": chunk.priority,
    }
    for key, value in chunk.metadata.items():
        if key in SIDECAR_KEYS and value:
            payload[key] = value
    return payload


def _dense_size(info: Any) -> int | None:
    vectors = getattr(info.config, "params", None)
    if vectors is None:
        return None
    config = getattr(vectors, "vectors", None)
    if isinstance(config, dict) and "dense" in config:
        return int(config["dense"].size)
    dense = getattr(config, "dense", None) if config is not None else None
    if dense is not None:
        return int(dense.size)
    return None
