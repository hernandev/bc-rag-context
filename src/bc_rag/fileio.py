"""Atomic file writes: a reader sees the old file or the new one, never half of one.

The text goes to a temporary file in the same folder, is flushed to disk, and then
replaces the target with `os.replace`, which is atomic on one filesystem.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, text: str, *, mode: int | None = None) -> None:
    """Replace `path` with `text`. `mode` (such as 0o600) is set before the file appears."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: Any, *, mode: int | None = None) -> None:
    """Replace `path` with `data` as indented JSON and a trailing newline."""
    write_text_atomic(path, json.dumps(data, indent=2) + "\n", mode=mode)
