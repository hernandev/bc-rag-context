from pathlib import Path

import pytest
from tests.support import default_files_config, entry_for
from tests.test_indexer_incremental import FakeEmbedder, FakeStore

from bc_rag.indexer import Indexer
from bc_rag.manifest import Manifest, ManifestError, load_manifest, save_manifest


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m" / "manifest.json"
    save_manifest(path, Manifest(dense_model="d", sparse_model="s"))
    loaded = load_manifest(path)
    assert loaded is not None
    assert loaded.dense_model == "d"
    assert [item.name for item in path.parent.iterdir()] == ["manifest.json"]


@pytest.mark.parametrize("text", ['{"dense_model": "d", "files": {', "[]", '{"files": {"a": {}}}'])
def test_corrupt_manifest_raises(tmp_path: Path, text: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError, match="--force"):
        load_manifest(path)


def test_corrupt_manifest_never_recreates_the_collection(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nbody\n", encoding="utf-8")
    config = default_files_config()
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), store)
    indexer.run()
    assert store.recreated == 1
    config.manifest_path(tmp_path).write_text("{ truncated", encoding="utf-8")
    with pytest.raises(ManifestError):
        indexer.run()
    assert store.recreated == 1
    assert "a.md" in store.points
