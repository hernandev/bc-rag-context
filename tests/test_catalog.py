from pathlib import Path

from bc_rag import catalog


def test_register_project_is_unique_by_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(catalog, "catalog_path", lambda: tmp_path / "catalog.json")
    first = tmp_path / "engine"
    second = tmp_path / "portal"
    first.mkdir()
    second.mkdir()

    engine = catalog.register_project(first)
    again = catalog.register_project(first)
    portal = catalog.register_project(second)

    assert engine.name == again.name
    assert engine.root == again.root
    names = {entry.name for entry in catalog.load_catalog()}
    assert names == {engine.name, portal.name}


def test_forget_project_by_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(catalog, "catalog_path", lambda: tmp_path / "catalog.json")
    root = tmp_path / "engine"
    root.mkdir()
    catalog.register_project(root, name="engine")

    removed = catalog.forget_project("engine")

    assert removed is not None
    assert catalog.load_catalog() == []


def test_store_lives_under_user_dir(tmp_path: Path) -> None:
    root = tmp_path / "bigcolony-workspaces"
    root.mkdir()
    dest = catalog.store_dir_for_root(root)

    assert dest == catalog.user_dir() / "bigcolony-workspaces"
    assert dest.is_relative_to(catalog.user_dir())
    assert dest != root / ".bc-rag"


def test_legacy_project_store_is_moved(tmp_path: Path) -> None:
    root = tmp_path / "engine"
    root.mkdir()
    legacy = root / ".bc-rag"
    legacy.mkdir()
    (legacy / "manifest.json").write_text("{}\n", encoding="utf-8")
    (legacy / "qdrant").mkdir()

    dest = catalog.store_dir_for_root(root)

    assert (dest / "manifest.json").is_file()
    assert (dest / "qdrant").is_dir()
    assert not (legacy / "manifest.json").exists()
    assert not (legacy / "qdrant").exists()
