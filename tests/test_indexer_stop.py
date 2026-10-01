import signal
import threading
from pathlib import Path

import pytest
from tests.support import default_files_config, entry_for
from tests.test_indexer_incremental import FakeEmbedder, FakeStore

from bc_rag.indexer import IndexBusyError, Indexer, acquire_index_lock, index_spaces


def _files(root: Path, count: int) -> None:
    for index in range(count):
        (root / f"f{index}.md").write_text(f"# doc {index}\n\nbody {index}\n", encoding="utf-8")


def test_a_set_stop_event_ends_the_run_without_deleting(tmp_path: Path) -> None:
    _files(tmp_path, 3)
    config = default_files_config()
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), store)
    indexer.run()
    (tmp_path / "f0.md").unlink()
    stop = threading.Event()
    stop.set()
    stats = indexer.run(stop=stop)
    assert stats.stopped is True
    # a stopped run never treats unvisited files as deleted.
    assert stats.deleted_files == 0
    assert "f0.md" in store.points


def test_stop_event_set_mid_run_finishes_the_current_file(tmp_path: Path) -> None:
    _files(tmp_path, 40)
    stop = threading.Event()

    class StoppingEmbedder(FakeEmbedder):
        def embed_docs(self, dense_texts, sparse_texts):
            stop.set()
            return super().embed_docs(dense_texts, sparse_texts)

    from bc_rag.manifest import load_manifest

    config = default_files_config()
    store = FakeStore()
    indexer = Indexer(entry_for(tmp_path), config, StoppingEmbedder(), store)
    stats = indexer.run(stop=stop)
    assert stats.stopped is True
    # files already queued finish, and the manifest records every one of them.
    manifest = load_manifest(config.manifest_path(tmp_path))
    assert manifest is not None
    assert set(manifest.files) == set(store.points)


def test_no_sigint_handler_off_the_main_thread(tmp_path: Path) -> None:
    _files(tmp_path, 2)
    before = signal.getsignal(signal.SIGINT)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            Indexer(entry_for(tmp_path), default_files_config(), FakeEmbedder(), FakeStore()).run()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(timeout=30)
    assert errors == []
    assert signal.getsignal(signal.SIGINT) is before


def test_the_index_lock_refuses_a_second_writer(tmp_path: Path) -> None:
    _files(tmp_path, 1)
    entry = entry_for(tmp_path)
    held = acquire_index_lock(entry)
    try:
        indexer = Indexer(entry, default_files_config(), FakeEmbedder(), FakeStore())
        with pytest.raises(IndexBusyError, match="already being indexed"):
            index_spaces([indexer])
    finally:
        held.release()
    assert index_spaces([indexer]).indexed_files == 1
