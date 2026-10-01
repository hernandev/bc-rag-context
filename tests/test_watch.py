import threading
from pathlib import Path

from rich.console import Console
from tests.support import default_files_config, entry_for
from tests.test_indexer_incremental import FakeEmbedder, FakeStore

from bc_rag.indexer import Indexer
from bc_rag.watch import WatchSession


class RecordingIndexer:
    def __init__(self, inner: Indexer) -> None:
        self.inner = inner
        self.calls: list[tuple[list[str], bool]] = []

    def run(self, **kwargs):
        on_main = threading.current_thread() is threading.main_thread()
        self.calls.append((kwargs.get("only_paths"), on_main))
        return self.inner.run(**kwargs)


def _session(tmp_path: Path) -> tuple[WatchSession, RecordingIndexer]:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.ts").write_text("export const alpha = 1;\n", encoding="utf-8")
    config = default_files_config()
    indexer = Indexer(entry_for(tmp_path), config, FakeEmbedder(), FakeStore())
    indexer.run()
    recorder = RecordingIndexer(indexer)
    session = WatchSession(tmp_path, config, recorder, Console(quiet=True))
    return session, recorder


def test_reindex_runs_on_the_main_thread(tmp_path: Path) -> None:
    session, recorder = _session(tmp_path)
    (tmp_path / "src" / "alpha.ts").write_text("export const alpha = 9;\n", encoding="utf-8")
    session.note(tmp_path / "src" / "alpha.ts")
    assert session.process_pending() is False
    assert recorder.calls == [(["src/alpha.ts"], True)]


def test_unchanged_paths_are_skipped(tmp_path: Path) -> None:
    session, recorder = _session(tmp_path)
    session.note(tmp_path / "src" / "alpha.ts")
    session.process_pending()
    assert recorder.calls == []


def test_files_outside_every_group_are_ignored(tmp_path: Path) -> None:
    session, recorder = _session(tmp_path)
    (tmp_path / "notes.txt").write_text("x\n", encoding="utf-8")
    session.note(tmp_path / "notes.txt")
    assert session._pending == set()
