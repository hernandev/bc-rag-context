"""Dense + sparse embeddings. Dense/rerank may be local FastEmbed or Jina HTTP."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from bc_rag.defaults import DENSE_DIMS


@dataclass(slots=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder:
    def __init__(
        self,
        dense_model: str,
        sparse_model: str,
        *,
        jina_api: bool = False,
    ) -> None:
        import os

        from fastembed import SparseTextEmbedding

        from bc_rag.models import cache_directory

        thread_count = os.cpu_count() or 4
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.jina_api = jina_api
        self.jina_embed_tokens = 0
        self.jina_embed_calls = 0
        self.jina_rerank_tokens = 0
        self.jina_rerank_calls = 0
        self._sparse_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        cache = str(cache_directory())
        self._sparse = SparseTextEmbedding(
            model_name=sparse_model, threads=thread_count, cache_dir=cache
        )
        self._dense = None
        if jina_api:
            self.dense_dim = DENSE_DIMS.get(dense_model) or DENSE_DIMS.get(
                dense_model.removeprefix("jinaai/")
            )
            if self.dense_dim is None:
                probe = self._jina_embed(["dimension probe"], query=False)
                self.dense_dim = len(probe[0])
                self.jina_embed_tokens = 0
                self.jina_embed_calls = 0
        else:
            from fastembed import TextEmbedding

            self._dense = TextEmbedding(
                model_name=dense_model, threads=thread_count, cache_dir=cache
            )
            self.dense_dim = DENSE_DIMS.get(dense_model) or int(
                next(self._dense.embed(["dimension probe"])).shape[0]
            )

    def embed_docs(
        self, dense_texts: list[str], sparse_texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVec]]:
        if len(dense_texts) != len(sparse_texts):
            raise ValueError("dense_texts and sparse_texts must be the same length")
        if not dense_texts:
            return [], []
        if self.jina_api:
            dense = self._jina_embed(dense_texts, query=False)
        else:
            dense = [vector.tolist() for vector in self._dense.embed(dense_texts, batch_size=16)]
        with self._sparse_lock:
            sparse = [_to_sparse(item) for item in self._sparse.embed(sparse_texts, batch_size=16)]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        if self.jina_api:
            dense = self._jina_embed([text], query=True)[0]
        else:
            dense = next(self._dense.query_embed([text])).tolist()
        sparse = _to_sparse(next(self._sparse.query_embed([text])))
        return dense, sparse

    def _jina_embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        from bc_rag.jina_api import embed_texts

        vectors, tokens = embed_texts(model=self.dense_model, texts=texts, query=query)
        with self._usage_lock:
            self.jina_embed_tokens += tokens
            self.jina_embed_calls += 1
        return vectors


class Reranker:
    def __init__(self, model_name: str, *, jina_api: bool = False) -> None:
        self.model_name = model_name
        self.jina_api = jina_api
        self._encoder = None
        if not jina_api:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            from bc_rag.models import cache_directory

            self._encoder = TextCrossEncoder(
                model_name=model_name, cache_dir=str(cache_directory())
            )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self.jina_api:
            from bc_rag.jina_api import rerank_texts

            scores, tokens = rerank_texts(
                model=self.model_name, query=query, documents=documents
            )
            return scores
        return list(self._encoder.rerank(query, documents))


def _to_sparse(item) -> SparseVec:
    indices = item.indices.tolist() if hasattr(item.indices, "tolist") else list(item.indices)
    values = item.values.tolist() if hasattr(item.values, "tolist") else list(item.values)
    return SparseVec(indices=[int(i) for i in indices], values=[float(v) for v in values])
