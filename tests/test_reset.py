from pathlib import Path

from bc_rag.config import RagConfig
from bc_rag.defaults import CORPUS_DIRNAME, OPENAPI_MD_DIRNAME
from bc_rag.reset import reset_project, reset_targets


def test_reset_targets_cover_store_corpus_and_openapi_md(tmp_path: Path) -> None:
    configuration = RagConfig()
    targets = reset_targets(tmp_path, configuration)
    names = {path.name for path in targets}
    assert CORPUS_DIRNAME in names
    assert OPENAPI_MD_DIRNAME in names
    assert any(path.name.startswith("local--") or "jina--" in path.name for path in targets)


def test_reset_project_removes_ledger_and_temps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.catalog.register_project", lambda root: None)
    configuration = RagConfig(follow_gitignore=False)
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
