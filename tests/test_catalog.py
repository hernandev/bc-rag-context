from pathlib import Path

import pytest

from bc_rag import catalog
from bc_rag.catalog import (
    CatalogError,
    ProjectNotFound,
    register_project,
    resolve_project,
    set_project_every,
    try_resolve_project,
)


def _folder(tmp_path: Path, *parts: str) -> Path:
    path = tmp_path.joinpath(*parts)
    path.mkdir(parents=True)
    return path


def test_register_refuses_a_second_add_of_one_root(tmp_path: Path) -> None:
    engine = _folder(tmp_path, "engine")
    first = register_project(engine)
    with pytest.raises(CatalogError, match=f"already registered as {first.name}"):
        register_project(engine)
    assert [entry.name for entry in catalog.load_catalog()] == ["engine"]


def test_register_refuses_a_taken_name(tmp_path: Path) -> None:
    register_project(_folder(tmp_path, "a", "engine"))
    with pytest.raises(CatalogError, match="name 'engine' is used by"):
        register_project(_folder(tmp_path, "b", "engine"), name="engine")


def test_default_name_is_qualified_when_taken(tmp_path: Path) -> None:
    first = register_project(_folder(tmp_path, "a", "engine"))
    second = register_project(_folder(tmp_path, "b", "engine"))
    assert first.name == "engine"
    assert second.name == "b-engine"


def test_forget_project_by_name(tmp_path: Path) -> None:
    register_project(_folder(tmp_path, "engine"), name="engine")
    removed = catalog.forget_project("engine")
    assert removed is not None
    assert catalog.load_catalog() == []


def test_resolve_by_name_from_anywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _folder(tmp_path, "engine")
    register_project(root, name="eng")
    monkeypatch.chdir(_folder(tmp_path, "elsewhere"))
    assert resolve_project("eng").root == str(root.resolve())


def test_resolve_walks_up_from_a_subfolder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _folder(tmp_path, "repo")
    deep = _folder(tmp_path, "repo", "libs", "x")
    register_project(root)
    assert resolve_project(root=deep).name == "repo"
    monkeypatch.chdir(deep)
    assert resolve_project().name == "repo"


def test_resolve_picks_the_nearest_registered_root(tmp_path: Path) -> None:
    register_project(_folder(tmp_path, "outer"))
    inner = _folder(tmp_path, "outer", "inner")
    register_project(inner)
    assert resolve_project(root=inner / ".").name == "inner"


def test_resolve_not_found(tmp_path: Path) -> None:
    folder = _folder(tmp_path, "nothing")
    with pytest.raises(ProjectNotFound) as info:
        resolve_project(root=folder)
    assert info.value.path == folder.resolve()
    assert "not a bc-rag project" in str(info.value)
    with pytest.raises(ProjectNotFound, match="unknown project: nope"):
        resolve_project("nope")
    assert try_resolve_project(root=folder) is None
    with pytest.raises(ProjectNotFound):
        try_resolve_project("nope")


def test_resolve_refuses_both_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="use one of --project or --root"):
        resolve_project("x", tmp_path)


def test_every_round_trip(tmp_path: Path) -> None:
    root = _folder(tmp_path, "engine")
    register_project(root)
    assert set_project_every("engine", " 5M ").every == "5m"
    assert catalog.load_catalog()[0].every == "5m"
    assert set_project_every("engine", "off").every == "off"
    assert set_project_every("engine", None).every is None
    raw = catalog.catalog_path().read_text(encoding="utf-8")
    assert '"every"' not in raw


@pytest.mark.parametrize("value", ["0", "soon", ""])
def test_every_rejects_bad_values(tmp_path: Path, value: str) -> None:
    register_project(_folder(tmp_path, "engine"))
    with pytest.raises(CatalogError):
        set_project_every("engine", value)


def test_every_on_an_unknown_project(tmp_path: Path) -> None:
    with pytest.raises(ProjectNotFound):
        set_project_every("nope", "5m")


def test_store_lives_under_user_dir(tmp_path: Path) -> None:
    root = _folder(tmp_path, "bigcolony-workspaces")
    dest = catalog.store_dir_for_root(root)
    assert dest == catalog.user_dir() / "bigcolony-workspaces"
    assert dest.is_relative_to(catalog.user_dir())
