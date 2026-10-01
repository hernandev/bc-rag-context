"""Filesystem watch that reindexes by content hash, same as `bc-rag index`.

The watchdog observer thread only records changed paths. Hashing and indexing run on
the main thread, where Ctrl+C reaches the indexer and Python allows signal handlers.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from bc_rag.config import RagConfig
from bc_rag.defaults import HARD_EXCLUDE_DIR_NAMES
from bc_rag.discover import classify_path
from bc_rag.indexer import changed_paths


class WatchSession:
    def __init__(
        self,
        root: Path,
        config: RagConfig,
        indexer,
        console: Console | None = None,
        debounce_s: float = 0.0,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        # anything with run(only_paths=...) -> IndexStats, such as one Indexer or every space.
        self.indexer = indexer
        self.console = console or Console(stderr=True)
        self.debounce_s = debounce_s
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._observer = Observer()

    def run(self) -> None:
        self._observer.schedule(_Handler(self), str(self.root), recursive=True)
        self._observer.start()
        self.console.print(f"[green]watching[/green] {self.root}  (hash skip, same as index)")
        try:
            while not self._stop.is_set():
                if not self._wakeup.wait(0.5):
                    continue
                self._wakeup.clear()
                if self.process_pending():
                    break
        except KeyboardInterrupt:
            self.console.print("\n[dim]stopping watch[/dim]")
        finally:
            self._stop.set()
            self._observer.stop()
            self._observer.join(timeout=2)

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()

    def note(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(self.root).as_posix()
        except (OSError, ValueError):
            return
        if any(part in HARD_EXCLUDE_DIR_NAMES for part in resolved.parts):
            return
        if classify_path(rel, resolved, self.config, peek=False) is None:
            return
        with self._lock:
            self._pending.add(rel)
        self._wakeup.set()

    def process_pending(self) -> bool:
        """Reindex the paths noted so far. True when the index run was stopped (Ctrl+C)."""
        if self.debounce_s > 0:
            time.sleep(self.debounce_s)
        with self._lock:
            paths = sorted(self._pending)
            self._pending.clear()
        if not paths:
            return False
        try:
            changed = changed_paths(self.root, self.config, paths)
        except Exception as exc:
            self.console.print(f"[red]watch hash failed[/red] {exc}")
            return False
        skipped = len(paths) - len(changed)
        if skipped:
            self.console.print(f"[dim]unchanged[/dim] {skipped} file(s)  (same content+sidecar)")
        if not changed:
            return False
        self.console.print(f"[cyan]reindex[/cyan] {len(changed)} file(s)")
        try:
            stats = self.indexer.run(only_paths=changed)
        except Exception as exc:
            self.console.print(f"[red]watch index failed[/red] {exc}")
            return False
        self.console.print(
            f"[green]ok[/green] indexed={stats.indexed_files} chunks={stats.chunks} "
            f"deleted={stats.deleted_files} unchanged={stats.skipped_unchanged}"
        )
        return bool(stats.stopped)


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
