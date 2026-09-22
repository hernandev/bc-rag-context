"""Filesystem watch that reindexes by content hash, same as `bc-rag index`."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from bc_rag.config import RagConfig
from bc_rag.defaults import HARD_EXCLUDE_DIR_NAMES, STORE_DIRNAME
from bc_rag.discover import classify_path
from bc_rag.indexer import Indexer, changed_paths


class WatchSession:
    def __init__(
        self,
        root: Path,
        config: RagConfig,
        indexer: Indexer,
        console: Console | None = None,
        debounce_s: float = 0.0,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.indexer = indexer
        self.console = console or Console(stderr=True)
        self.debounce_s = debounce_s
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._observer = Observer()

    def run(self) -> None:
        handler = _Handler(self)
        self._observer.schedule(handler, str(self.root), recursive=True)
        self._observer.start()
        worker = threading.Thread(target=self._loop, name="bc-rag-watch", daemon=True)
        worker.start()
        self.console.print(f"[green]watching[/green] {self.root}  (hash skip, same as index)")
        try:
            while not self._stop.is_set():
                time.sleep(0.4)
        except KeyboardInterrupt:
            self.console.print("\n[dim]stopping watch[/dim]")
        finally:
            self._stop.set()
            self._wakeup.set()
            self._observer.stop()
            self._observer.join(timeout=2)
            worker.join(timeout=5)

    def note(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(self.root).as_posix()
        except (OSError, ValueError):
            return
        if any(part in HARD_EXCLUDE_DIR_NAMES or part == STORE_DIRNAME for part in resolved.parts):
            return
        if classify_path(rel, resolved, self.config, peek=False) is None:
            return
        with self._lock:
            self._pending.add(rel)
        self._wakeup.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop.is_set():
                return
            if self.debounce_s > 0:
                time.sleep(self.debounce_s)
            with self._lock:
                paths = sorted(self._pending)
                self._pending.clear()
            if not paths:
                continue
            try:
                changed = changed_paths(self.root, self.config, paths)
            except Exception as exc:
                self.console.print(f"[red]watch hash failed[/red] {exc}")
                continue
            skipped = len(paths) - len(changed)
            if skipped:
                self.console.print(f"[dim]unchanged[/dim] {skipped} file(s)  (same content+sidecar hash)")
            if not changed:
                continue
            self.console.print(f"[cyan]reindex[/cyan] {len(changed)} file(s)")
            try:
                stats = self.indexer.run(only_paths=changed)
            except Exception as exc:
                self.console.print(f"[red]watch index failed[/red] {exc}")
                continue
            self.console.print(
                f"[green]ok[/green] indexed={stats.indexed_files} chunks={stats.chunks} "
                f"deleted={stats.deleted_files} unchanged={stats.skipped_unchanged}"
            )


class _Handler(FileSystemEventHandler):
    def __init__(self, session: WatchSession) -> None:
        self.session = session

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = getattr(event, "src_path", None)
        dest = getattr(event, "dest_path", None)
        if src:
            self.session.note(Path(str(src)))
        if dest:
            self.session.note(Path(str(dest)))
