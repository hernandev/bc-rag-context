import json
from pathlib import Path

import pytest

from bc_rag import query
from bc_rag.catalog import register_project
from bc_rag.embeddings import SparseVec
from bc_rag.store import Hit


def _project(tmp_path: Path, *, rerank: bool) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "prose",
                "groupsCommand": ["/no/such/program"],
                "spaces": {
                    "prose": {
                        "dense": {
                            "provider": "local",
                            "model": "jinaai/jina-embeddings-v2-base-en",
                        },
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                        "retrieve": {"prefetch": 5, "limit": 3, "rerank": rerank},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    register_project(tmp_path, name="proj")


def _hit(index: int, facets: dict | None = None) -> Hit:
    return Hit(
        score=1.0 - index / 10,
        path=f"p{index}",
        language="markdown",
        kind="prose",
        symbol=None,
        heading_path=None,
        start_line=1,
        end_line=1,
        text=f"t{index}",
        payload={"facets": facets or {"group": ["docs"]}},
    )


class Recorder:
    def __init__(self) -> None:
        self.limits: list[int] = []
        self.filters: list[tuple] = []
        self.rerankers: list[str] = []
        self.keys = {"group": 10, "area": 4}
        self.down = False


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    class Store:
        def __init__(self, **kwargs) -> None:
            self.url = kwargs.get("url")

        def count(self) -> int:
            if recorder.down:
                raise ConnectionError("connection refused")
            return 5

        def facet_keys(self) -> dict[str, int]:
            if recorder.down:
                raise ConnectionError("connection refused")
            return recorder.keys

        def query(self, *, dense, limit, facets=None, exclude=None):
            recorder.limits.append(limit)
            recorder.filters.append((facets, exclude))
            return [_hit(index) for index in range(min(limit, 5))]

        def query_sparse(self, *, sparse, limit, facets=None, exclude=None):
            recorder.limits.append(limit)
            return [_hit(index) for index in range(min(limit, 5))]

        def close(self) -> None:
            pass

    class Embedder:
        def embed_query(self, text):
            return [0.0], SparseVec(indices=[], values=[])

        def embed_sparse_query(self, text):
            return SparseVec(indices=[], values=[])

    class Reranker:
        def __init__(self, model: str, jina_api: bool = False) -> None:
            recorder.rerankers.append(model)

        def rerank(self, text: str, documents: list[str]) -> list[float]:
            # reverses the dense order.
            return [float(index) for index in range(len(documents))]

    monkeypatch.setattr(query, "HybridStore", Store)
    monkeypatch.setattr(query, "Reranker", Reranker)
    monkeypatch.setattr("bc_rag.runtime.make_embedder", lambda config, spec: Embedder())
    monkeypatch.setattr(query, "_embedders", {})
    monkeypatch.setattr(query, "_rerankers", {})
    return recorder


def test_reranks_when_the_space_says_so(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=True)
    result = query.search_projects("q", project="proj", space="prose")
    assert fakes.limits == [12]
    assert [hit.path for hit in result.hits] == ["p4", "p3", "p2"]
    assert result.reranked is True
    assert fakes.rerankers == ["jina-reranker-v3.5"]
    assert all(hit.project == "proj" for hit in result.hits)


def test_skips_rerank_when_the_space_turns_it_off(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=False)
    result = query.search_projects("q", project="proj", space="prose")
    assert fakes.limits == [3]
    assert [hit.path for hit in result.hits] == ["p0", "p1", "p2"]
    assert result.reranked is False
    assert fakes.rerankers == []


def test_reranker_is_built_once(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=True)
    query.search_projects("q", project="proj", space="prose")
    query.search_projects("q", project="proj", space="prose")
    assert fakes.rerankers == ["jina-reranker-v3.5"]


def test_facets_reach_the_store(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=False)
    query.search_projects(
        "q", project="proj", space="prose", facets={"area": ["engine"]}, exclude={"group": ["x"]}
    )
    assert fakes.filters == [({"area": ["engine"]}, {"group": ["x"]})]


def test_unknown_facet_key_names_the_stored_ones(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=False)
    with pytest.raises(ValueError, match="facet key 'team' is not stored in space 'prose'"):
        query.search_projects("q", project="proj", space="prose", facets={"team": ["a"]})


def test_qdrant_down_raises(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=False)
    fakes.down = True
    with pytest.raises(RuntimeError, match="qdrant unreachable"):
        query.search_projects("q", project="proj", space="prose")


def test_unknown_project_lists_the_registered_ones(tmp_path: Path, fakes: Recorder) -> None:
    _project(tmp_path, rerank=False)
    with pytest.raises(ValueError, match="unknown project: nope. Registered projects: proj"):
        query.search_projects("q", project="nope", space="prose")
