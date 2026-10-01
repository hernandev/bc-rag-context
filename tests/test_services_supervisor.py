import json
from pathlib import Path

import pytest

from bc_rag import locks
from bc_rag.services import state as service_state
from bc_rag.services import supervisor as sup
from bc_rag.services.qdrant import ContainerStatus
from bc_rag.services.state import (
    RUNNING,
    STOPPED,
    bump_restart,
    load_desired,
    lock_path,
    read_runtime,
    set_desired,
    supervisor_pid,
    write_runtime,
)


class FakeProcess:
    next_pid = 1000

    def __init__(self, argv: list[str], **kwargs) -> None:
        FakeProcess.next_pid += 1
        self.pid = FakeProcess.next_pid
        self.argv = argv
        self.kwargs = kwargs
        self.returncode: int | None = None
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("TERM")

    def kill(self) -> None:
        self.signals.append("KILL")
        self.returncode = -9


class FakeOps:
    def __init__(self) -> None:
        self.up = True
        self.calls: list[str] = []
        self.status = ContainerStatus(exists=True, state="running", labels={"bc-rag": "qdrant"})

    def mode(self) -> str:
        return "docker"

    def url(self) -> str:
        return "http://127.0.0.1:32321"

    def reachable(self) -> bool:
        return self.up

    def ensure_running(self) -> str:
        self.calls.append("ensure_running")
        self.up = True
        return "started"

    def wait_ready(self) -> bool:
        return self.up

    def container_status(self) -> ContainerStatus:
        return self.status

    def stop(self) -> None:
        self.calls.append("stop")
        self.status = ContainerStatus(exists=True, state="exited", labels={"bc-rag": "qdrant"})


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def harness():
    processes: list[FakeProcess] = []

    def popen(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        processes.append(process)
        return process

    clock = Clock()
    ops = FakeOps()

    def make() -> sup.Supervisor:
        keeper = sup.QdrantKeeper(ops=ops, threaded=False, say=lambda message: None)
        return sup.Supervisor(popen=popen, clock=clock, env={}, qdrant=keeper, say=lambda m: None)

    return make, processes, clock, ops


def _named(processes: list[FakeProcess], name: str) -> list[FakeProcess]:
    return [process for process in processes if name in process.argv]


def test_lock_says_whether_a_supervisor_runs() -> None:
    assert sup.supervisor_running() is False
    held = locks.acquire(lock_path(), note="1")
    assert held is not None
    try:
        assert sup.supervisor_running() is True
        assert locks.acquire(lock_path()) is None
    finally:
        held.release()
    assert sup.supervisor_running() is False


def test_stop_without_a_supervisor() -> None:
    assert sup.stop_supervisor() is False


def test_pid_falls_back_to_the_old_daemon_file() -> None:
    assert supervisor_pid() is None
    service_state.legacy_state_path().write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    assert supervisor_pid() == 4242
    write_runtime({"pid": 5151})
    assert supervisor_pid() == 5151


def test_lock_pid_comes_first() -> None:
    held = locks.acquire(lock_path(), note="777")
    try:
        write_runtime({"pid": 5151})
        assert supervisor_pid() == 777
    finally:
        held.release()


def test_a_starting_supervisor_is_not_stale() -> None:
    # the lock carries the new supervisor's pid; supervisor.json is not written yet.
    held = locks.acquire(lock_path(), note="777")
    try:
        assert sup.is_stale() is False
        write_runtime({"pid": 777, "fingerprint": service_state.source_fingerprint()})
        assert sup.is_stale() is False
        write_runtime({"pid": 777, "fingerprint": "old-code"})
        assert sup.is_stale() is True
    finally:
        held.release()


def test_the_old_daemon_is_stale() -> None:
    # the old daemon held the lock without writing a pid into it.
    held = locks.acquire(lock_path())
    try:
        assert sup.is_stale() is True
    finally:
        held.release()


def test_fingerprint_covers_subpackages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "pkg"
    (package / "services").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    nested = package / "services" / "state.py"
    nested.write_text("a = 1\n", encoding="utf-8")
    import bc_rag

    monkeypatch.setattr(bc_rag, "__file__", str(package / "__init__.py"))
    before = service_state.source_fingerprint()
    nested.write_text("a = 22\n", encoding="utf-8")
    assert service_state.source_fingerprint() != before


def test_children_argv_and_env() -> None:
    argv = sup.child_argv("indexer")
    assert argv[1:5] == ["-m", "bc_rag", "services", "run"]
    assert argv[5:] == ["indexer", "--log-file", str(service_state.log_path("indexer"))]
    env = sup.child_env({"PATH": "/bin"})
    assert env == {"PATH": "/bin", "BC_RAG_AUTOSTART": "false", "BC_RAG_SUPERVISED": "1"}


def test_desired_state_defaults_and_round_trip() -> None:
    desired = load_desired()
    assert all(desired.wants(name) for name in ("qdrant", "indexer", "mcp"))
    set_desired(["indexer"], STOPPED, by="test")
    again = load_desired()
    assert again.services["indexer"].state == STOPPED
    assert again.services["mcp"].state == RUNNING
    assert again.generation == 1
    assert again.updated_by == "test"


def test_runtime_state_round_trip() -> None:
    write_runtime({"pid": 7, "children": {}})
    assert read_runtime() == {"pid": 7, "children": {}}


def test_the_fingerprint_is_the_code_loaded_at_start(
    harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    make, _processes, _clock, _ops = harness
    monkeypatch.setattr(sup, "source_fingerprint", lambda: "loaded")
    supervisor = make()
    monkeypatch.setattr(sup, "source_fingerprint", lambda: "edited-on-disk")
    supervisor.tick()
    assert read_runtime()["fingerprint"] == "loaded"


def test_starts_only_desired_services(harness) -> None:
    make, processes, _clock, _ops = harness
    set_desired(["mcp"], STOPPED, by="test")
    supervisor = make()
    assert supervisor.tick() is False
    assert len(_named(processes, "indexer")) == 1
    assert _named(processes, "mcp") == []
    runtime = read_runtime()
    assert runtime["children"]["indexer"]["state"] == RUNNING
    assert runtime["children"]["mcp"]["state"] == STOPPED


@pytest.mark.parametrize(("name", "deadline"), [("indexer", 120.0), ("mcp", 10.0)])
def test_stop_escalates_to_kill_after_the_deadline(harness, name: str, deadline: float) -> None:
    make, processes, clock, _ops = harness
    supervisor = make()
    supervisor.tick()
    process = _named(processes, name)[0]
    set_desired([name], STOPPED, by="test")
    supervisor.tick()
    assert process.signals == ["TERM"]
    assert supervisor.children[name].state == "stopping"
    clock.now += deadline - 1
    supervisor.tick()
    assert process.signals == ["TERM"]
    clock.now += 1
    supervisor.tick()
    assert process.signals == ["TERM", "KILL"]
    supervisor.tick()
    assert supervisor.children[name].state == STOPPED
    assert len(_named(processes, name)) == 1


def test_restart_counter_restarts_one_child(harness) -> None:
    make, processes, _clock, _ops = harness
    supervisor = make()
    supervisor.tick()
    first_mcp = _named(processes, "mcp")[0]
    indexer = _named(processes, "indexer")[0]
    bump_restart("mcp", by="test")
    supervisor.tick()
    assert first_mcp.signals == ["TERM"]
    first_mcp.returncode = 0
    supervisor.tick()
    assert len(_named(processes, "mcp")) == 2
    assert indexer.signals == []
    assert supervisor.children["mcp"].state == RUNNING


def test_port_in_use_waits_60s(harness) -> None:
    make, processes, clock, _ops = harness
    supervisor = make()
    supervisor.tick()
    _named(processes, "mcp")[0].returncode = 3
    supervisor.tick()
    assert supervisor.children["mcp"].state == "failed"
    assert supervisor.children["mcp"].reason == "port in use"
    clock.now += 2
    supervisor.tick()
    assert len(_named(processes, "mcp")) == 1
    clock.now += 58
    supervisor.tick()
    assert len(_named(processes, "mcp")) == 2


def test_backoff_doubles_and_caps(harness) -> None:
    make, processes, clock, _ops = harness
    supervisor = make()
    supervisor.tick()
    delays = []
    for _ in range(7):
        _named(processes, "indexer")[-1].returncode = 1
        supervisor.tick()
        child = supervisor.children["indexer"]
        assert child.state == "backoff"
        delays.append(child.next_start - clock.now)
        clock.now = child.next_start
        supervisor.tick()
    assert delays == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


def test_exits_when_every_service_is_stopped(harness) -> None:
    make, processes, _clock, ops = harness
    supervisor = make()
    supervisor.tick()
    set_desired(["qdrant", "indexer", "mcp"], STOPPED, by="test")
    assert supervisor.tick() is False
    for process in processes:
        process.returncode = 0
    assert supervisor.tick() is True
    assert "stop" in ops.calls


def test_qdrant_is_started_when_unreachable(harness) -> None:
    make, _processes, _clock, ops = harness
    ops.up = False
    supervisor = make()
    supervisor.tick()
    assert ops.calls == ["ensure_running"]
    assert supervisor.qdrant.state == RUNNING


def test_ensure_stack_respects_the_env_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sup, "spawn", lambda env=None: calls.append("spawn") or 1)
    monkeypatch.setenv("BC_RAG_DAEMON", "false")
    status = sup.ensure_stack()
    assert status.supervisor == "off"
    assert calls == []


def test_ensure_stack_respects_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from bc_rag.usersettings import set_setting

    monkeypatch.delenv("BC_RAG_DAEMON", raising=False)
    monkeypatch.delenv("BC_RAG_AUTOSTART", raising=False)
    set_setting("autostart", "false")
    notes: list[str] = []
    status = sup.ensure_stack(report=notes.append)
    assert status.supervisor == "off"
    assert "autostart is off" in notes[0]


def test_ensure_stack_spawns_without_touching_desired_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BC_RAG_DAEMON", raising=False)
    monkeypatch.delenv("BC_RAG_AUTOSTART", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(sup, "spawn", lambda env=None: calls.append("spawn") or 99)
    monkeypatch.setattr(sup.qdrant_ops, "wait_ready", lambda timeout=30: True)
    status = sup.ensure_stack()
    assert status.supervisor == "started"
    assert calls == ["spawn"]
    assert not service_state.desired_path().exists()


def test_ensure_stack_leaves_stopped_services_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BC_RAG_DAEMON", raising=False)
    monkeypatch.delenv("BC_RAG_AUTOSTART", raising=False)
    set_desired(["qdrant", "indexer", "mcp"], STOPPED, by="test")
    monkeypatch.setattr(sup.qdrant_ops, "reachable", lambda *a, **k: False)
    notes: list[str] = []
    status = sup.ensure_stack(report=notes.append)
    assert status.supervisor == STOPPED
    assert any("services start" in note for note in notes)


def test_ensure_stack_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BC_RAG_DAEMON", raising=False)
    monkeypatch.delenv("BC_RAG_AUTOSTART", raising=False)

    def boom(env=None):
        raise OSError("no fork")

    monkeypatch.setattr(sup, "spawn", boom)
    status = sup.ensure_stack()
    assert status.supervisor == "error"
    assert "no fork" in status.notes[0]
