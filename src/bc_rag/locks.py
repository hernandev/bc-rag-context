"""Advisory file locks held for a process's lifetime.

The operating system drops the lock when the process exits, even on a crash or
SIGKILL, so a lock never outlives its holder. The file itself stays; only the lock
on it matters. Used for the one supervisor per user and one indexer per project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO

IS_WINDOWS = sys.platform == "win32"


def try_lock(handle: IO[str]) -> bool:
    try:
        if IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def unlock(handle: IO[str]) -> None:
    try:
        if IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def open_lock(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a+", encoding="utf-8")


def is_locked(path: Path) -> bool:
    """True while some process holds the lock on `path`."""
    with open_lock(path) as handle:
        if try_lock(handle):
            unlock(handle)
            return False
        return True


class HeldLock:
    """A lock taken by `acquire` and given back by `release` (or `with`)."""

    def __init__(self, path: Path, handle: IO[str]) -> None:
        self.path = path
        self._handle: IO[str] | None = handle

    def release(self) -> None:
        if self._handle is None:
            return
        unlock(self._handle)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> HeldLock:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def acquire(path: Path, *, note: str = "") -> HeldLock | None:
    """Take the lock, or None when another process holds it. `note` is written into the file."""
    handle = open_lock(path)
    if not try_lock(handle):
        handle.close()
        return None
    if note:
        handle.seek(0)
        handle.truncate()
        handle.write(note)
        handle.flush()
        os.fsync(handle.fileno())
    return HeldLock(path, handle)


def read_note(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
