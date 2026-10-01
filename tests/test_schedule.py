import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from bc_rag.catalog import register_project, set_project_every
from bc_rag.config import dump_default_config
from bc_rag.schedule import (
    ScheduleState,
    load_state,
    mark_finished,
    read_schedule,
    resolve_every,
    save_state,
)
from bc_rag.services import indexer_service


def _project(tmp_path: Path, name: str, every: str | None = None) -> Path:
    root = tmp_path / name
    root.mkdir()
    raw = json.loads(dump_default_config())
    if every is not None:
        raw["schedule"] = {"every": every}
    (root / ".bc-rag.json").write_text(json.dumps(raw), encoding="utf-8")
    register_project(root)
    return root


def test_resolve_every() -> None:
    assert resolve_every("off", "15m") == (None, "override")
    assert resolve_every("5m", "15m") == (300.0, "override")
    assert resolve_every(None, "15m") == (900.0, "project")
    assert resolve_every(None, None) == (None, "none")
    # yours for every project comes last, after the team file.
    assert resolve_every(None, "15m", "30m") == (900.0, "project")
    assert resolve_every(None, None, "30m") == (1800.0, "global")
    assert resolve_every(None, None, "off") == (None, "global")


def test_a_global_every_schedules_projects_without_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qdrant_up: None
) -> None:
    from bc_rag.usersettings import set_setting

    _project(tmp_path, "a")
    _project(tmp_path, "b", every="15m")
    set_setting("every", "30m")
    calls: list[str] = []
    monkeypatch.setattr(
        "bc_rag.runtime.index_project",
        lambda entry, **kwargs: calls.append(entry.name) or Stats(),
    )
    indexer_service.run_indexer_service(
        threading.Event(), Console(quiet=True), say=lambda m: None, once=True
    )
    assert sorted(calls) == ["a", "b"]
    assert (load_state("a").every, load_state("a").source) == ("30m", "global")
    assert (load_state("b").every, load_state("b").source) == ("15m", "project")


def test_first_run_is_due_now_and_then_one_interval_after_it_finished() -> None:
    state = ScheduleState(project="p")
    assert state.is_due(300)
    mark_finished(state, result="ok", every_seconds=300)
    finished = datetime.strptime(state.last_finish, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert state.due_at(300) == finished + timedelta(seconds=300)
    assert not state.is_due(300, finished + timedelta(seconds=299))
    assert state.is_due(300, finished + timedelta(seconds=300))
    assert state.next_due is not None


def test_state_round_trip() -> None:
    state = ScheduleState(project="p", every="5m", source="override", result="ok")
    save_state(state)
    assert load_state("p") == state
    assert load_state("never-ran") == ScheduleState(project="never-ran")


def test_read_schedule_never_runs_groups_command(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    raw = json.loads(dump_default_config())
    raw["schedule"] = {"every": "15m"}
    raw["groupsCommand"] = ["/no/such/program"]
    (root / ".bc-rag.json").write_text(json.dumps(raw), encoding="utf-8")
    assert read_schedule(root) == "15m"


def test_read_schedule_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        read_schedule(tmp_path)
    (tmp_path / ".bc-rag.json").write_text('{"schedule": {"every": "soon"}}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_schedule(tmp_path)


@pytest.fixture
def qdrant_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bc_rag.services.qdrant.reachable", lambda *a, **k: True)


class Stats:
    stopped = False
    errors: list[str] = []
    scanned = indexed_files = skipped_unchanged = deleted_files = chunks = 0


def test_service_indexes_due_projects_in_turn_on_the_main_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qdrant_up: None
) -> None:
    _project(tmp_path, "a", every="15m")
    _project(tmp_path, "b")
    set_project_every("b", "5m")
    _project(tmp_path, "c")
    calls: list[tuple[str, bool]] = []

    def fake_index(entry, *, console, stop):
        calls.append((entry.name, threading.current_thread() is threading.main_thread()))
        return Stats()

    monkeypatch.setattr("bc_rag.runtime.index_project", fake_index)
    indexer_service.run_indexer_service(
        threading.Event(), Console(quiet=True), say=lambda m: None, once=True
    )
    assert sorted(calls) == [("a", True), ("b", True)]
    assert load_state("a").result == "ok"
    assert load_state("b").source == "override"
    # both ran just now, so neither is due on the next pass.
    calls.clear()
    indexer_service.run_indexer_service(
        threading.Event(), Console(quiet=True), say=lambda m: None, once=True
    )
    assert calls == []


def test_service_stops_after_the_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qdrant_up: None
) -> None:
    _project(tmp_path, "a", every="15m")
    _project(tmp_path, "b", every="15m")
    stop = threading.Event()
    calls: list[str] = []

    def fake_index(entry, *, console, stop):
        calls.append(entry.name)
        stop.set()
        return Stats()

    monkeypatch.setattr("bc_rag.runtime.index_project", fake_index)
    indexer_service.run_indexer_service(stop, Console(quiet=True), say=lambda m: None)
    assert len(calls) == 1


def test_service_waits_while_qdrant_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, "a", every="15m")
    monkeypatch.setattr("bc_rag.services.qdrant.reachable", lambda *a, **k: False)
    calls: list[str] = []
    monkeypatch.setattr(
        "bc_rag.runtime.index_project", lambda entry, **kwargs: calls.append(entry.name)
    )
    said: list[str] = []
    indexer_service.run_indexer_service(
        threading.Event(), Console(quiet=True), say=said.append, once=True
    )
    assert calls == []
    assert "qdrant unreachable" in said[0]


def test_service_survives_a_broken_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qdrant_up: None
) -> None:
    _project(tmp_path, "a", every="15m")
    _project(tmp_path, "b", every="15m")
    broken = tmp_path / "c"
    broken.mkdir()
    (broken / ".bc-rag.json").write_text("{ not json", encoding="utf-8")
    register_project(broken)
    calls: list[str] = []

    def fake_index(entry, *, console, stop):
        calls.append(entry.name)
        if entry.name == "a":
            raise RuntimeError("voyage down")
        return Stats()

    monkeypatch.setattr("bc_rag.runtime.index_project", fake_index)
    said: list[str] = []
    indexer_service.run_indexer_service(
        threading.Event(), Console(quiet=True), say=said.append, once=True
    )
    assert sorted(calls) == ["a", "b"]
    assert load_state("a").result == "error"
    assert load_state("a").error == "voyage down"
    assert load_state("b").result == "ok"
    assert any("c: schedule skipped" in line for line in said)
