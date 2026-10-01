from pathlib import Path

from tests.support import default_files_config, space_config

from bc_rag.defaults import CACHE_DIRNAME, CORPUS_DIRNAME, OPENAPI_MD_DIRNAME
from bc_rag.reset import reset_collections, reset_project, reset_targets


def test_reset_targets_cover_store_and_cache(tmp_path: Path) -> None:
    configuration = default_files_config()
    targets = reset_targets(tmp_path, configuration)
    names = {path.name for path in targets}
    assert CACHE_DIRNAME in names
    assert any("prose--" in path.name for path in targets)
    # the old cache folders are named only when they are still there.
    assert CORPUS_DIRNAME not in names
    project_dir = configuration.project_dir(tmp_path)
    (project_dir / OPENAPI_MD_DIRNAME).mkdir(parents=True)
    assert OPENAPI_MD_DIRNAME in {path.name for path in reset_targets(tmp_path, configuration)}


def test_reset_names_every_space_collection(tmp_path: Path, monkeypatch) -> None:
    from bc_rag import reset

    deleted: list[str] = []

    def fake_delete(url: str, name: str) -> str:
        deleted.append(name)
        return "deleted"

    monkeypatch.setattr(reset, "_delete_http_collection", fake_delete)
    configuration = space_config()
    report = reset_project(tmp_path, configuration)
    assert len(deleted) == 2
    assert report.collections == [(name, "deleted") for name in deleted]
    assert reset_collections(tmp_path, configuration) == deleted


def test_reset_project_removes_ledger_and_temps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.reset._delete_http_collection", lambda url, name: "already gone")
    configuration = default_files_config()
    store = configuration.store_dir(tmp_path)
    store.mkdir(parents=True)
    (store / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "qdrant").mkdir()
    project_dir = configuration.project_dir(tmp_path)
    (project_dir / CORPUS_DIRNAME).mkdir(parents=True)
    (project_dir / CORPUS_DIRNAME / "note.md").write_text("x\n", encoding="utf-8")
    (project_dir / OPENAPI_MD_DIRNAME).mkdir(parents=True)
    (project_dir / OPENAPI_MD_DIRNAME / "spec.md").write_text("y\n", encoding="utf-8")

    report = reset_project(tmp_path, configuration)
    assert not store.exists()
    assert not (project_dir / CORPUS_DIRNAME).exists()
    assert not (project_dir / OPENAPI_MD_DIRNAME).exists()
    assert str(store) in report.removed
