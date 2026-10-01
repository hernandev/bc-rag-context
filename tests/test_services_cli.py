import os
import socket

import pytest
from typer.testing import CliRunner

from bc_rag.cli import app
from bc_rag.services import qdrant
from bc_rag.services import supervisor as sup
from bc_rag.services.mcp_service import run_mcp_service
from bc_rag.services.state import load_desired, write_runtime

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    from bc_rag.cli import _app

    monkeypatch.setattr(_app.console, "width", 400)
    monkeypatch.setattr(_app.err, "width", 400)
    monkeypatch.setattr(qdrant, "reachable", lambda *a, **k: False)
    monkeypatch.setattr(
        qdrant, "container_status", lambda: qdrant.ContainerStatus(exists=False)
    )


def test_list_without_a_supervisor() -> None:
    result = runner.invoke(app, ["services", "list"])
    assert result.exit_code == 0, result.output
    assert "supervisor not running" in result.output
    for name in ("qdrant", "indexer", "mcp"):
        assert name in result.output
    assert "no project has a schedule" in result.output


def test_list_shows_a_starting_supervisor_as_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    from bc_rag.services.state import RUNNING, STOPPED, set_desired

    monkeypatch.setattr(sup, "supervisor_running", lambda: True)
    monkeypatch.setattr(sup, "lock_pid", lambda: 4242)
    # supervisor.json is still the previous supervisor's.
    write_runtime({"pid": 1111, "fingerprint": "old", "children": {"mcp": {"state": "running"}}})
    set_desired(["mcp"], RUNNING, by="test")
    set_desired(["indexer"], STOPPED, by="test")

    result = runner.invoke(app, ["services", "list"])

    assert result.exit_code == 0, result.output
    assert "supervisor pid 4242  starting" in result.output
    assert "code old" not in result.output
    lines = {line.split("│")[1].strip(): line for line in result.output.splitlines() if "│" in line}
    assert "starting" in lines["mcp"]
    assert "stopped (stopped by you)" in lines["indexer"]


def test_list_shows_the_code_once_the_supervisor_has_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bc_rag.services.state import source_fingerprint

    monkeypatch.setattr(sup, "supervisor_running", lambda: True)
    monkeypatch.setattr(sup, "lock_pid", lambda: 4242)
    write_runtime({"pid": 4242, "fingerprint": source_fingerprint(), "children": {}})

    result = runner.invoke(app, ["services", "list"])

    assert result.exit_code == 0, result.output
    assert "supervisor pid 4242" in result.output
    assert "code current" in result.output
    assert "starting" not in result.output


def test_unknown_name_exits_2() -> None:
    result = runner.invoke(app, ["services", "stop", "redis"])
    assert result.exit_code == 2
    assert "unknown service 'redis'. Services: qdrant, indexer, mcp" in result.output


def test_stop_indexer_writes_desired_state_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(["stopping", "stopping", "stopped"])
    monkeypatch.setattr(sup, "supervisor_running", lambda: True)
    monkeypatch.setattr(
        "bc_rag.services.state.read_runtime",
        lambda: {"children": {"indexer": {"state": next(states, "stopped")}}},
    )
    result = runner.invoke(app, ["services", "stop", "indexer"])
    assert result.exit_code == 0, result.output
    assert "indexer  stopped" in result.output
    desired = load_desired()
    assert desired.services["indexer"].state == "stopped"
    assert desired.services["mcp"].state == "running"


def test_run_refuses_while_a_supervised_child_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BC_RAG_SUPERVISED", raising=False)
    monkeypatch.setattr(sup, "supervisor_running", lambda: True)
    write_runtime({"children": {"indexer": {"state": "running", "pid": os.getpid()}}})
    result = runner.invoke(app, ["services", "run", "indexer"])
    assert result.exit_code == 1
    assert "already runs under the supervisor" in result.output


def test_run_mcp_reports_a_taken_port(capsys: pytest.CaptureFixture[str]) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        assert run_mcp_service(port=port) == 3
    finally:
        holder.close()
    out = capsys.readouterr().out
    assert (
        f"mcp service already runs on 127.0.0.1:{port}; run `bc-rag services stop mcp` first"
        in out
    )


def test_indexer_service_stops_cleanly_on_sigterm(tmp_path) -> None:
    import signal
    import subprocess
    import sys
    import time

    home = tmp_path / "home"
    env = dict(
        os.environ,
        BC_RAG_HOME=str(home),
        BC_RAG_AUTOSTART="false",
        # nothing listens here, so the service waits for Qdrant inside stop.wait().
        BC_RAG_QDRANT_URL="http://127.0.0.1:9",
    )
    log = home / "logs" / "indexer.log"
    process = subprocess.Popen(
        [sys.executable, "-m", "bc_rag", "services", "run", "indexer", "--log-file", str(log)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while "qdrant unreachable" not in (log.read_text() if log.exists() else ""):
            assert time.monotonic() < deadline, "the service never reached its loop"
            time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
    text = log.read_text()
    assert "stopping: finishing the current file" in text
    assert text.rstrip().endswith("stopped")


def test_index_build_help_never_calls_ensure_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("bc_rag.services.ensure_stack", lambda **kwargs: calls.append("x"))
    assert runner.invoke(app, ["index", "--help"]).exit_code == 0
    assert calls == []
