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
        voyage_api: bool = False,
        dimensions: int | None = None,
        contextual: bool = False,
        contextual_input: str = "chunks",
    ) -> None:
        import os

        from fastembed import SparseTextEmbedding

        from bc_rag.models import cache_directory

        thread_count = os.cpu_count() or 4
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.jina_api = jina_api and not voyage_api
        self.voyage_api = voyage_api
        self.dimensions = dimensions
        self.contextual = contextual
        self.contextual_input = contextual_input
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
        if voyage_api:
            self.dense_dim = dimensions or 1024
        elif jina_api:
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
        if self.voyage_api and self.contextual and self.contextual_input != "auto":
            dense = self._voyage_contextual_chunks(dense_texts)
        elif self.voyage_api:
            dense = self._voyage_embed(dense_texts, query=False)
        elif self.jina_api:
            dense = self._jina_embed(dense_texts, query=False)
        else:
            dense = [vector.tolist() for vector in self._dense.embed(dense_texts, batch_size=16)]
        with self._sparse_lock:
            sparse = [_to_sparse(item) for item in self._sparse.embed(sparse_texts, batch_size=16)]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        if self.voyage_api and self.contextual:
            dense = self._voyage_contextual_query(text)
        elif self.voyage_api:
            dense = self._voyage_embed([text], query=True)[0]
        elif self.jina_api:
            dense = self._jina_embed([text], query=True)[0]
        else:
            dense = next(self._dense.query_embed([text])).tolist()
        return dense, self.embed_sparse_query(text)

    def embed_sparse_texts(self, texts: list[str]) -> list[SparseVec]:
        if not texts:
            return []
        with self._sparse_lock:
            return [_to_sparse(item) for item in self._sparse.embed(texts, batch_size=16)]

    def embed_sparse_query(self, text: str) -> SparseVec:
        return _to_sparse(next(self._sparse.query_embed([text])))

    def _jina_embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        from bc_rag.jina_api import embed_texts

        vectors, tokens = embed_texts(model=self.dense_model, texts=texts, query=query)
        with self._usage_lock:
            self.jina_embed_tokens += tokens
            self.jina_embed_calls += 1
        return vectors

    def _voyage_embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        from bc_rag.voyage_api import embed_texts

        vectors, tokens = embed_texts(
            model=self.dense_model, texts=texts, query=query, dimensions=self.dimensions
        )
        self._note_embed(tokens)
        return vectors

    def _voyage_contextual_chunks(self, texts: list[str]) -> list[list[float]]:
        from bc_rag.voyage_api import embed_contextual_chunks

        vectors, tokens = embed_contextual_chunks(
            model=self.dense_model, chunk_texts=texts, dimensions=self.dimensions
        )
        self._note_embed(tokens)
        return vectors

    def _voyage_contextual_query(self, text: str) -> list[float]:
        from bc_rag.voyage_api import embed_contextual_query

        vector, tokens = embed_contextual_query(
            model=self.dense_model, text=text, dimensions=self.dimensions
        )
        self._note_embed(tokens)
        return vector

    def _note_embed(self, tokens: int) -> None:
        with self._usage_lock:
            self.jina_embed_tokens += tokens
            self.jina_embed_calls += 1


class Reranker:
    def __init__(self, model_name: str, *, jina_api: bool = False) -> None:
        self.model_name = model_name
        self.voyage_api = model_name.startswith("rerank-")
        # `jina-reranker-*` is the HTTP model id. `jinaai/jina-reranker-*` stays local FastEmbed.
        self.jina_api = (
            not self.voyage_api
            and (jina_api or model_name.startswith("jina-reranker"))
        )
        self._encoder = None
        if self.voyage_api or self.jina_api:
            return
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from bc_rag.models import cache_directory

        self._encoder = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_directory())
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self.voyage_api:
            from bc_rag.voyage_api import rerank_texts as voyage_rerank

            scores, _tokens = voyage_rerank(model=self.model_name, query=query, documents=documents)
            return scores
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
