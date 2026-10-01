import json
import stat
from pathlib import Path

import pytest

from bc_rag import fileio
from bc_rag.fileio import write_json_atomic, write_text_atomic


def test_write_text_creates_folders_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    write_text_atomic(target, "one")
    write_text_atomic(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert [item.name for item in target.parent.iterdir()] == ["file.txt"]


def test_mode_is_applied(tmp_path: Path) -> None:
    target = tmp_path / "secret.json"
    write_json_atomic(target, {"key": "value"}, mode=0o600)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text(encoding="utf-8")) == {"key": "value"}


def test_failed_write_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "state" / "state.json"
    write_json_atomic(target, {"version": 1})

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(fileio.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"version": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert [item.name for item in target.parent.iterdir()] == ["state.json"]
