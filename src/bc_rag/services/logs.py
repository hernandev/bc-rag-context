"""Service logs that rotate themselves, so no log grows without bound.

Each service process points its own stdout and stderr at `~/.bc-rag/logs/{name}.log`
through `redirect_process_output`. When the file passes `MAX_BYTES`, it becomes
`{name}.log.1` (older copies shift up to `BACKUPS`) and a fresh file starts.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import IO

MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 2


def _rotate_files(path: Path, backups: int) -> None:
    for index in range(backups, 0, -1):
        older = path.with_name(f"{path.name}.{index}")
        newer = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
        if newer.exists():
            os.replace(newer, older)


class RotatingStream:
    """A text stream over one log file that rotates when the file passes `max_bytes`.

    With `redirect_fds`, the process's file descriptors 1 and 2 follow the current
    file too, so output from C code and child processes lands in the same log.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = MAX_BYTES,
        backups: int = BACKUPS,
        redirect_fds: bool = False,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self.redirect_fds = redirect_fds
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self._open()
        self._size = self._file.tell()

    def _open(self) -> IO[str]:
        handle = open(self.path, "a", encoding="utf-8", buffering=1)
        if self.redirect_fds:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
        return handle

    def write(self, text: str) -> int:
        with self._lock:
            if self._size + len(text) > self.max_bytes and self._size > 0:
                self._file.close()
                _rotate_files(self.path, self.backups)
                self._file = self._open()
                self._size = 0
            written = self._file.write(text)
            self._size += len(text.encode("utf-8", errors="replace"))
            return written

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._file.fileno()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def close(self) -> None:
        with self._lock:
            self._file.close()


def redirect_process_output(path: Path) -> RotatingStream:
    """Send this process's stdout and stderr into a rotating log at `path`."""
    stream = RotatingStream(path, redirect_fds=True)
    sys.stdout = stream  # type: ignore[assignment]
    sys.stderr = stream  # type: ignore[assignment]
    return stream


def log_size(path: Path) -> int:
    """Bytes in the log and its rotated copies."""
    total = 0
    for candidate in (path, *(path.with_name(f"{path.name}.{i}") for i in range(1, BACKUPS + 1))):
        try:
            total += candidate.stat().st_size
        except OSError:
            continue
    return total
