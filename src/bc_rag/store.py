"""Qdrant store: one collection per space, with a dense vector and a BM25 vector per chunk.

`query` searches the dense vector; `query_sparse` searches BM25. Each point carries
its facets as a nested `facets` object, and every facet key gets a keyword index
(`facets.<key>`) the first time a point with that key is written, so the key can
be filtered and listed from that pass on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from bc_rag.chunking import Chunk
from bc_rag.defaults import COLLECTION_NAME
from bc_rag.embeddings import SparseVec
from bc_rag.facets import Facets

FACETS_FIELD = "facets"
# bc-rag's own ceiling on the values one facet listing returns.
FACET_VALUES_LIMIT = 10_000


def qdrant_api_key() -> str | None:
    """The `qdrant-api-key` setting, sent as the api-key header. None for a local Qdrant."""
    from bc_rag.usersettings import get_setting

    return get_setting("qdrant-api-key")


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

    @property
    def facets(self) -> Facets:
        return dict(self.payload.get(FACETS_FIELD) or {})


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
        # payload fields known to have a keyword index in this collection.
        self._indexed: set[str] = set()
        if url:
            self.client = QdrantClient(
                url=url, api_key=qdrant_api_key(), timeout=60, check_compatibility=False
            )
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
                self._indexed.clear()
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
        """Keep full dense vectors on disk and int8 copies in RAM."""
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
        self._indexed.clear()
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

    # -- indexes -----------------------------------------------------------------

    def _payload_schema(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection)
        return dict(getattr(info, "payload_schema", None) or {})

    def _create_index(self, field: str) -> None:
        from qdrant_client.models import PayloadSchemaType

        self.client.create_payload_index(
            collection_name=self.collection,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._indexed.add(field)

    def ensure_payload_indexes(self) -> None:
        """The base index on `path`. Read-only sessions never write indexes."""
        if self.read_only or not self.client.collection_exists(self.collection):
            return
        if not self._indexed:
            self._indexed.update(self._payload_schema())
        if "path" not in self._indexed:
            self._create_index("path")

    def _ensure_facet_indexes(self, keys: set[str]) -> None:
        for key in sorted(keys):
            field = f"{FACETS_FIELD}.{key}"
            if field not in self._indexed:
                self._create_index(field)

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVec],
    ) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        if not chunks:
            return
        # a new key is indexed before its first point lands, so it is listable at once.
        self._ensure_facet_indexes({key for chunk in chunks for key in chunk.facets})
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

    # -- search ------------------------------------------------------------------

    def query(
        self,
        *,
        dense: list[float],
        limit: int,
        facets: Facets | None = None,
        exclude: Facets | None = None,
    ) -> list[Hit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=dense,
            using="dense",
            query_filter=facet_filter(facets, exclude),
            limit=limit,
            with_payload=True,
        )
        return [_hit(point) for point in response.points]

    def query_sparse(
        self,
        *,
        sparse: SparseVec,
        limit: int,
        facets: Facets | None = None,
        exclude: Facets | None = None,
    ) -> list[Hit]:
        from qdrant_client.models import SparseVector

        response = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(indices=sparse.indices, values=sparse.values),
            using="bm25",
            query_filter=facet_filter(facets, exclude),
            limit=limit,
            with_payload=True,
        )
        return [_hit(point) for point in response.points]

    # -- listing -----------------------------------------------------------------

    def facet_keys(self) -> dict[str, int]:
        """Facet key -> how many points carry it, from the collection's indexes."""
        if not self.client.collection_exists(self.collection):
            return {}
        prefix = f"{FACETS_FIELD}."
        keys: dict[str, int] = {}
        for field, schema in self._payload_schema().items():
            if not field.startswith(prefix):
                continue
            points = int(getattr(schema, "points", 0) or 0)
            if points > 0:
                keys[field[len(prefix) :]] = points
        return dict(sorted(keys.items()))

    def facet_values(self, key: str, *, limit: int = FACET_VALUES_LIMIT) -> list[tuple[str, int]]:
        """The distinct values of one facet key, with point counts, most points first."""
        if not self.client.collection_exists(self.collection):
            return []
        result = self.client.facet(
            collection_name=self.collection,
            key=f"{FACETS_FIELD}.{key}",
            limit=limit,
            exact=True,
        )
        rows: list[tuple[str, int]] = []
        for hit in getattr(result, "hits", []) or []:
            value = getattr(hit, "value", None)
            if value is None:
                continue
            rows.append((str(value), int(getattr(hit, "count", 0) or 0)))
        return rows


def facet_filter(facets: Facets | None, exclude: Facets | None = None):
    """Any value within one key, every key, and none of `exclude`. None when empty."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    def conditions(source: Facets | None) -> list[Any]:
        return [
            FieldCondition(key=f"{FACETS_FIELD}.{key}", match=MatchAny(any=list(values)))
            for key, values in (source or {}).items()
        ]

    must = conditions(facets)
    must_not = conditions(exclude)
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _hit(point: Any) -> Hit:
    payload = point.payload or {}
    return Hit(
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
    return {
        "path": chunk.path,
        "language": chunk.language,
        "kind": chunk.kind,
        "symbol": chunk.symbol,
        "heading_path": chunk.heading_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "priority": chunk.priority,
        FACETS_FIELD: {key: list(values) for key, values in chunk.facets.items()},
    }


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
