import asyncio
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from bc_rag.catalog import register_project
from bc_rag.mcp_server import build_server

SPACE = {
    "dense": {"provider": "local", "model": "jinaai/jina-embeddings-v2-base-en"},
    "sparse": {"provider": "local", "model": "Qdrant/bm25"},
    "rerank": {"provider": "voyage", "model": "rerank-2.5"},
    "chunk": {"max_chars": 2400, "min_chars": 40},
}


def _project(root: Path, name: str, **extra: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw = {"defaultSpace": "prose", "spaces": {"prose": SPACE, "code": SPACE}, **extra}
    (root / ".bc-rag.json").write_text(json.dumps(raw), encoding="utf-8")
    register_project(root, name=name)
    return root


def _call(tool: str, **arguments: object):
    server = build_server()
    result = asyncio.run(server.call_tool(tool, arguments))
    return result.structured_content


def _error(tool: str, **arguments: object) -> str:
    with pytest.raises(ToolError) as info:
        _call(tool, **arguments)
    assert not isinstance(info.value, UnexpectedToolError)
    return str(info.value)


class FakeStore:
    keys = {"group": 4, "area": 3}
    values = {"group": [("docs", 4)], "area": [("engine", 2), ("admin", 1)]}

    def __init__(self, **kwargs) -> None:
        self.url = kwargs.get("url")

    def count(self) -> int:
        return 4

    def facet_keys(self) -> dict[str, int]:
        return dict(self.keys)

    def facet_values(self, key: str, *, limit: int = 10_000):
        return self.values.get(key, [])[:limit]

    def close(self) -> None:
        pass


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bc_rag.query.HybridStore", FakeStore)
    monkeypatch.setattr("bc_rag.store.HybridStore", FakeStore)


def test_every_tool_is_read_only() -> None:
    tools = asyncio.run(build_server().list_tools())
    assert sorted(tool.name for tool in tools) == [
        "list_facets",
        "list_projects",
        "search",
        "search_sparse",
    ]
    assert all(tool.annotations.read_only_hint for tool in tools)


def test_unknown_project_reaches_the_client(tmp_path: Path) -> None:
    _project(tmp_path / "a", "alpha")
    message = _error("search", query="q", project="nope", space="prose")
    assert message == (
        "Error executing tool search: ValueError: unknown project: nope. "
        "Registered projects: alpha"
    )


def test_missing_config_reaches_the_client(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    register_project(root, name="bare")
    message = _error("search", query="q", project="bare", space="prose")
    assert ".bc-rag.json is required" in message


def test_unknown_space_lists_the_configured_ones(tmp_path: Path) -> None:
    _project(tmp_path / "a", "alpha")
    message = _error("search_sparse", query="q", project="alpha", space="docs")
    assert "unknown space 'docs'. Configured spaces: code, prose" in message


def test_facet_key_not_stored_names_the_stored_keys(tmp_path: Path, fake_store: None) -> None:
    _project(tmp_path / "a", "alpha")
    message = _error("search", query="q", project="alpha", space="prose", facets={"team": "x"})
    assert "facet key 'team' is not stored in space 'prose'. Keys here: group, area" in message


def test_a_bad_facets_value_is_named(tmp_path: Path) -> None:
    _project(tmp_path / "a", "alpha")
    message = _error("search", query="q", project="alpha", space="prose", facets={"area": [""]})
    assert "facet 'area' has value ''" in message


def test_broken_project_is_an_error_row(tmp_path: Path) -> None:
    _project(tmp_path / "a", "alpha", facetKeys={"area": "the product area"}, searchHints=["hi"])
    broken = tmp_path / "b"
    broken.mkdir()
    (broken / ".bc-rag.json").write_text("{ nope", encoding="utf-8")
    register_project(broken, name="broken")
    rows = {row["name"]: row for row in _call("list_projects")["result"]}
    assert [space["name"] for space in rows["alpha"]["spaces"]] == ["prose", "code"]
    assert rows["alpha"]["spaces"][0]["rerank"] == "voyage rerank-2.5"
    assert rows["alpha"]["facetKeys"] == {"area": "the product area"}
    assert rows["alpha"]["hints"] == ["hi"]
    assert rows["broken"]["spaces"] == []
    assert rows["broken"]["error"]


def test_list_facets_overview_and_one_key(tmp_path: Path, fake_store: None) -> None:
    _project(tmp_path / "a", "alpha", facetKeys={"area": "the product area"})
    overview = _call("list_facets", project="alpha", space="prose")
    by_key = {row["key"]: row for row in overview["keys"]}
    assert by_key["area"] == {
        "key": "area",
        "meaning": "the product area",
        "points": 3,
        "distinct": 2,
        "values": ["engine", "admin"],
        "truncated": False,
    }
    assert by_key["group"]["meaning"] == "the config group that claimed the file"
    one = _call("list_facets", project="alpha", space="prose", key="area", limit=1)
    assert one["values"] == [{"value": "engine", "points": 2}]
    assert one["truncated"] is True
    message = _error("list_facets", project="alpha", space="prose", key="team")
    assert "facet key 'team' is not stored" in message
