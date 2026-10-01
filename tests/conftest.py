from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_bc_rag_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "_bc_rag_home"
    home.mkdir()
    monkeypatch.setattr("bc_rag.catalog.user_dir", lambda: home)
    # a CLI test must never start a real background daemon.
    monkeypatch.setenv("BC_RAG_DAEMON", "false")
    return home
