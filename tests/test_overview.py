import json
from pathlib import Path

import pytest

from bc_rag import runtime
from bc_rag.catalog import register_project
from bc_rag.overview import facets_report, project_overview

SPACE = {
    "dense": {"provider": "local", "model": "jinaai/jina-embeddings-v2-base-en"},
    "sparse": {"provider": "local", "model": "Qdrant/bm25"},
    "rerank": {"provider": "voyage", "model": "rerank-2.5"},
    "chunk": {"max_chars": 2400, "min_chars": 40},
}


def _project(root: Path) -> None:
    code = {**SPACE, "retrieve": {"rerank": False}}
    raw = {
        "defaultSpace": "prose",
        "spaces": {"prose": SPACE, "code": code},
        "groups": [
            {
                "name": "docs",
                "space": "prose",
                "include": ["docs/**"],
                "facets": {"area": "engine"},
            },
            {"name": "off", "space": "prose", "include": ["x/**"], "enabled": False,
             "facets": {"area": "hidden"}},
        ],
    }
    (root / ".bc-rag.json").write_text(json.dumps(raw), encoding="utf-8")
    register_project(root, name="proj")


class Store:
    data = {
        "prose": {"group": {"docs": 4}, "area": {"engine": 3, "admin": 1}},
        "code": {"group": {"src": 2}},
    }
    broken: set[str] = set()

    def __init__(self, *, url, collection, read_only) -> None:
        self.space = "prose" if "_prose_" in collection else "code"

    def _check(self) -> None:
        if self.space in self.broken:
            raise ConnectionError("refused")

    def count(self) -> int:
        self._check()
        return 7 if self.space == "prose" else 2

    def facet_keys(self) -> dict[str, int]:
        self._check()
        return {key: sum(values.values()) for key, values in self.data[self.space].items()}

    def facet_values(self, key: str, *, limit: int = 10_000):
        return sorted(self.data[self.space][key].items())

    def close(self) -> None:
        pass


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> type[Store]:
    monkeypatch.setattr("bc_rag.store.HybridStore", Store)
    Store.broken = set()
    return Store


def test_overview_has_one_row_per_space(tmp_path: Path, store) -> None:
    _project(tmp_path)
    from bc_rag.catalog import resolve_project

    rows = {row.space: row for row in project_overview(resolve_project("proj"))}
    assert rows["prose"].rerank == "voyage rerank-2.5"
    assert rows["code"].rerank == "(off)"
    assert rows["prose"].points == 7
    assert rows["code"].points == 2
    assert rows["prose"].manifest_time is None
    store.broken = {"code"}
    rows = {row.space: row for row in project_overview(resolve_project("proj"))}
    assert rows["code"].points is None
    assert "refused" in rows["code"].error


def test_facets_report_joins_declared_and_stored(tmp_path: Path, store) -> None:
    _project(tmp_path)
    from bc_rag.catalog import resolve_project

    report = facets_report(resolve_project("proj"))
    rows = {(row.key, row.value): row for row in report.rows}
    assert rows[("area", "engine")].source == "both"
    assert rows[("area", "engine")].points == {"prose": 3, "code": None}
    assert rows[("area", "admin")].source == "stored"
    assert ("area", "hidden") not in rows
    assert rows[("group", "docs")].source == "both"
    assert rows[("group", "src")].points == {"prose": 0, "code": 2}
    assert report.errors == {}


def test_facets_report_marks_an_unreachable_space(tmp_path: Path, store) -> None:
    _project(tmp_path)
    from bc_rag.catalog import resolve_project

    store.broken = {"code"}
    report = facets_report(resolve_project("proj"))
    assert "code" in report.errors
    assert all("code" not in row.points for row in report.rows)


def test_open_space_sessions_loads_the_config_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path)
    from bc_rag.catalog import resolve_project

    loads: list[Path] = []
    real = runtime.load_config

    def counting(root, **kwargs):
        loads.append(root)
        return real(root, **kwargs)

    monkeypatch.setattr(runtime, "load_config", counting)
    monkeypatch.setattr(runtime, "HybridStore", lambda **kwargs: object())
    monkeypatch.setattr(runtime, "make_embedder", lambda config, spec: object())
    sessions = runtime.open_space_sessions(resolve_project("proj"))
    assert [session.space for session in sessions] == ["prose", "code"]
    assert len(loads) == 1
