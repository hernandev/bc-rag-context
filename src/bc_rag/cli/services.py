"""`bc-rag services ...`: the background services qdrant, indexer, and mcp."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from bc_rag.cli._app import command_group, console, err, fail

services_app = command_group("Background services: qdrant, indexer, and mcp.")

NameArg = Annotated[
    str | None,
    typer.Argument(help="qdrant, indexer, or mcp. Omit for all three."),
]


def _names(name: str | None) -> list[str]:
    from bc_rag.services.state import SERVICE_NAMES

    if name is None:
        return list(SERVICE_NAMES)
    if name not in SERVICE_NAMES:
        raise fail(
            "services",
            f"unknown service '{name}'. Services: {', '.join(SERVICE_NAMES)}",
            code=2,
        )
    return [name]


def _ago(iso: str) -> str:
    try:
        stamp = time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S%z"))
    except (TypeError, ValueError):
        return iso or ""
    seconds = max(0, int(time.time() - stamp))
    if seconds < 120:
        return f"{seconds}s ago"
    if seconds < 7200:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def _indexer_detail() -> str:
    from bc_rag.catalog import living_projects
    from bc_rag.schedule import effective_every, load_state

    upcoming: list[str] = []
    for entry in living_projects():
        state = load_state(entry.name)
        if state.running:
            return f"indexing {entry.name} since {state.last_start}"
        try:
            every, seconds, _source = effective_every(entry)
        except ValueError:
            continue
        if seconds is None:
            continue
        upcoming.append(f"next {entry.name} at {state.next_due or 'now'} (every {every})")
    if not upcoming:
        return "no project has a schedule"
    return "idle; " + "; ".join(upcoming)


def _mcp_detail(row: dict) -> str:
    from bc_rag.defaults import MCP_HTTP_PATH, MCP_HTTP_PORT, MCP_HTTPS_PORT, MCP_TLS_HOST
    from bc_rag.mcp_server import mcp_tls_files

    if row.get("state") == "failed" and row.get("reason") == "port in use":
        wait = row.get("next_start_in")
        return f"failed: port 127.0.0.1:{MCP_HTTP_PORT} in use (retry in {wait}s)"
    urls = [f"http://127.0.0.1:{MCP_HTTP_PORT}{MCP_HTTP_PATH}"]
    if mcp_tls_files() is not None:
        urls.append(f"https://{MCP_TLS_HOST}:{MCP_HTTPS_PORT}{MCP_HTTP_PATH}")
    return "  ".join(urls)


def _qdrant_detail() -> tuple[str, str]:
    from bc_rag.services import qdrant

    target = qdrant.url()
    up = qdrant.reachable()
    if qdrant.mode() == "external":
        state = "running" if up else "unreachable"
        return state, f"external {target} (reachable {'yes' if up else 'no'})"
    try:
        status = qdrant.container_status()
    except qdrant.QdrantError as error:
        return "unknown", str(error)
    if status.exists and not status.ours:
        return "foreign", qdrant.foreign_message(status)
    if not status.exists:
        return "stopped", f"docker {target}  (no container yet)"
    state = "running" if status.running and up else (status.state or "unknown")
    detail = f"docker {target}  volume {qdrant.QDRANT_VOLUME}"
    if status.drifted:
        detail += f"  image {status.image} (expected {qdrant.QDRANT_IMAGE}; start --recreate)"
    return state, detail


def _rows() -> list[dict]:
    from bc_rag.services.logs import log_size
    from bc_rag.services.state import load_desired, log_path, read_runtime
    from bc_rag.services.supervisor import supervisor_running

    running = supervisor_running()
    runtime = read_runtime() if running else {}
    children = runtime.get("children") or {}
    desired = load_desired()
    rows = []
    q_state, q_detail = _qdrant_detail()
    rows.append(
        {
            "service": "qdrant",
            "desired": desired.services["qdrant"].state,
            "state": q_state,
            "pid": None,
            "since": "",
            "detail": q_detail,
            "log": "docker logs bc-rag-qdrant",
        }
    )
    for name in ("indexer", "mcp"):
        row = children.get(name) or {}
        state = row.get("state") if running else "stopped"
        detail = _indexer_detail() if name == "indexer" else _mcp_detail(row)
        if row.get("reason") and name == "indexer":
            detail = f"{row['reason']}; {detail}"
        path = log_path(name)
        rows.append(
            {
                "service": name,
                "desired": desired.services[name].state,
                "state": state or "stopped",
                "pid": row.get("pid") if running else None,
                "since": row.get("since", "") if running else "",
                "detail": detail,
                "log": f"{path}  ({log_size(path) / 1e6:.1f} MB)",
            }
        )
    return rows


@services_app.command("list")
def services_list(
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show each service, its state, pid, and log."""
    from bc_rag.services.state import read_runtime, source_fingerprint
    from bc_rag.services.supervisor import supervisor_running
    from bc_rag.usersettings import autostart_enabled

    running = supervisor_running()
    runtime = read_runtime() if running else {}
    rows = _rows()
    if json_out:
        console.print_json(
            json.dumps(
                {
                    "supervisor": {
                        "running": running,
                        "pid": runtime.get("pid"),
                        "started_at": runtime.get("started_at"),
                        "code": (
                            "current"
                            if runtime.get("fingerprint") == source_fingerprint()
                            else "old"
                        )
                        if running
                        else None,
                    },
                    "autostart": autostart_enabled(),
                    "services": rows,
                }
            )
        )
        return
    if running:
        code = "current" if runtime.get("fingerprint") == source_fingerprint() else "old"
        title = (
            f"supervisor pid {runtime.get('pid')}  started {_ago(runtime.get('started_at', ''))}"
            f"  code {code}  autostart {'on' if autostart_enabled() else 'off'}"
        )
    else:
        title = f"supervisor not running  autostart {'on' if autostart_enabled() else 'off'}"
    table = Table(title=title)
    for column in ("service", "state", "pid", "since", "detail", "log"):
        table.add_column(column)
    for row in rows:
        state = row["state"]
        if row["desired"] == "stopped":
            state += " (stopped by you)" if state == "stopped" else " (stopping)"
        table.add_row(
            row["service"],
            state,
            str(row["pid"] or ""),
            _ago(row["since"]) if row["since"] else "",
            row["detail"],
            row["log"],
        )
    console.print(table)


def _start_qdrant(recreate: bool) -> None:
    from bc_rag.services import qdrant

    if qdrant.mode() == "external":
        state = "reachable" if qdrant.reachable() else "not reachable"
        console.print(f"qdrant  external {qdrant.url()}  {state}  (not started by bc-rag)")
        return
    try:
        done = qdrant.ensure_running(recreate=recreate)
    except qdrant.QdrantError as error:
        raise fail("qdrant", error) from error
    if not qdrant.wait_ready():
        raise fail("qdrant", f"{done}, but not ready at {qdrant.url()} after 30s")
    console.print(f"qdrant  {done}  {qdrant.url()}")


def _wait_children(names: list[str], timeout: float = 15.0) -> int:
    """Poll supervisor.json until each child runs or fails. Exit code 0 or 1."""
    from bc_rag.services.state import read_runtime

    deadline = time.monotonic() + timeout
    states: dict[str, dict] = {}
    while time.monotonic() < deadline:
        children = read_runtime().get("children") or {}
        states = {name: children.get(name) or {} for name in names}
        if all(row.get("state") in ("running", "failed", "backoff") for row in states.values()):
            break
        time.sleep(0.5)
    code = 0
    for name in names:
        row = states.get(name) or {}
        state = row.get("state") or "unknown"
        detail = f"pid {row.get('pid')}" if state == "running" else row.get("reason", "")
        console.print(f"{name}  {state}  {detail}".rstrip())
        if state != "running":
            code = 1
    return code


@services_app.command("start")
def services_start(
    name: NameArg = None,
    recreate: Annotated[
        bool,
        typer.Option("--recreate", help="Recreate the Qdrant container from the pinned image."),
    ] = False,
) -> None:
    """Start one or all: qdrant first, then indexer and mcp."""
    from bc_rag.services.state import RUNNING, set_desired
    from bc_rag.services.supervisor import replace_if_stale, spawn, supervisor_running

    names = _names(name)
    set_desired(names, RUNNING, by="services start")
    if "qdrant" in names:
        _start_qdrant(recreate)
    if not supervisor_running():
        pid = spawn()
        err.print(f"[dim]supervisor started  pid {pid}[/dim]")
    else:
        replace_if_stale(lambda message: err.print(f"[dim]{message}[/dim]"))
    children = [item for item in names if item != "qdrant"]
    if children:
        raise typer.Exit(code=_wait_children(children))


def _stop_progress(names: list[str], cap: float) -> int:
    """Poll until each named child has stopped. Exit code 0 or 1."""
    from bc_rag.services.state import read_runtime
    from bc_rag.services.supervisor import supervisor_running

    started = time.monotonic()
    last_note = 0.0
    while True:
        waited = time.monotonic() - started
        children = read_runtime().get("children") or {} if supervisor_running() else {}
        pending = [
            n for n in names if (children.get(n) or {}).get("state") not in (None, "stopped")
        ]
        if not pending:
            return 0
        if waited >= cap:
            err.print(f"[red]still stopping[/red] {', '.join(pending)} after {cap:g}s")
            return 1
        if waited - last_note >= 5:
            last_note = waited
            hint = "finishing the current file" if "indexer" in pending else "shutting down"
            err.print(f"stopping {', '.join(pending)}  {hint}  {waited:.0f}s")
        time.sleep(0.5)


@services_app.command("stop")
def services_stop(name: NameArg = None) -> None:
    """Stop one or all. A stopped service stays stopped."""
    from bc_rag.services import qdrant
    from bc_rag.services.state import STOPPED, set_desired
    from bc_rag.services.supervisor import STOP_TIMEOUT, stop_supervisor, supervisor_running

    names = _names(name)
    set_desired(names, STOPPED, by="services stop")
    code = 0
    children = [item for item in names if item != "qdrant"]
    if name is None:
        if supervisor_running():
            cap = sum(STOP_TIMEOUT.values()) + 10
            try:
                stop_supervisor(timeout=cap, progress=lambda m: err.print(m))
            except RuntimeError as error:
                raise fail("services stop", error) from error
        console.print("indexer  stopped")
        console.print("mcp  stopped")
    elif children:
        cap = sum(STOP_TIMEOUT[item] for item in children) + 10
        code = _stop_progress(children, cap)
        for item in children:
            console.print(f"{item}  {'stopped' if code == 0 else 'still stopping'}")
    if "qdrant" in names:
        if qdrant.mode() == "external":
            console.print("qdrant  external, not stopped by bc-rag")
        else:
            try:
                status = qdrant.container_status()
                if status.exists and not status.ours:
                    err.print(f"[yellow]qdrant[/yellow] {qdrant.foreign_message(status)}")
                elif status.running:
                    qdrant.stop()
                console.print("qdrant  stopped")
            except qdrant.QdrantError as error:
                raise fail("qdrant", error) from error
    raise typer.Exit(code=code)


@services_app.command("restart")
def services_restart(name: NameArg = None) -> None:
    """Restart one or all, with this shell's environment."""
    from bc_rag.services import qdrant
    from bc_rag.services.state import RUNNING, bump_restart, set_desired
    from bc_rag.services.supervisor import spawn, stop_supervisor, supervisor_running

    names = _names(name)
    if name is None:
        if supervisor_running():
            try:
                stop_supervisor(progress=lambda m: err.print(m))
            except RuntimeError as error:
                raise fail("services restart", error) from error
        set_desired(names, RUNNING, by="services restart")
        _start_qdrant(recreate=False)
        pid = spawn()
        err.print(f"[dim]supervisor started  pid {pid}  with this shell's environment[/dim]")
        raise typer.Exit(code=_wait_children(["indexer", "mcp"]))
    if name == "qdrant":
        if qdrant.mode() == "external":
            raise fail("qdrant", "external Qdrant is not restarted by bc-rag")
        set_desired(["qdrant"], RUNNING, by="services restart")
        try:
            qdrant.stop()
        except qdrant.QdrantError:
            pass
        _start_qdrant(recreate=False)
        return
    from bc_rag.services.supervisor import STOP_TIMEOUT

    bump_restart(name, by="services restart")
    if not supervisor_running():
        spawn()
    console.print(
        f"{name}  restarting under the supervisor's environment. "
        "Run `bc-rag services restart` with no name to pick up this shell's PATH or keys."
    )
    # the old process gets its stop timeout before the new one starts.
    time.sleep(2.5)
    raise typer.Exit(code=_wait_children([name], timeout=STOP_TIMEOUT[name] + 15))


def _refuse_if_supervised(name: str) -> None:
    from bc_rag.services.state import SUPERVISED_ENV, read_runtime
    from bc_rag.services.supervisor import supervisor_running

    if os.environ.get(SUPERVISED_ENV):
        return
    if not supervisor_running():
        return
    row = (read_runtime().get("children") or {}).get(name) or {}
    from bc_rag.services.state import pid_alive

    if pid_alive(row.get("pid")):
        raise fail(
            "services run",
            f"{name} already runs under the supervisor (pid {row.get('pid')}). "
            f"Run `bc-rag services stop {name}` first.",
        )


def _redirect(log_file: Path | None) -> None:
    if log_file is not None:
        from bc_rag.services.logs import redirect_process_output

        redirect_process_output(log_file)


@services_app.command("run")
def services_run(
    name: Annotated[str, typer.Argument(help="qdrant, indexer, or mcp.")],
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Write output to this rotating log instead of the screen."),
    ] = None,
) -> None:
    """Run one service here, log on screen. Ctrl+C stops it."""
    names = _names(name)
    service = names[0]
    if service == "qdrant":
        from bc_rag.services import qdrant

        if qdrant.mode() == "external":
            raise fail("services run", "qdrant is external; bc-rag does not run it")
        import subprocess

        try:
            status = qdrant.container_status()
            if status.exists and not status.ours:
                raise qdrant.QdrantError(qdrant.foreign_message(status))
            if not status.exists:
                qdrant.create_and_start()
        except qdrant.QdrantError as error:
            raise fail("qdrant", error) from error
        # attached: the log streams here, and Ctrl+C stops the container.
        argv = [*qdrant.docker_base(), "start", "-a", qdrant.QDRANT_CONTAINER]
        raise typer.Exit(code=subprocess.call(argv))
    _refuse_if_supervised(service)
    _redirect(log_file)
    if service == "mcp":
        from bc_rag.services.mcp_service import run_mcp_service

        raise typer.Exit(code=run_mcp_service())
    _run_indexer()


def _run_indexer() -> None:
    from rich.console import Console

    from bc_rag.services.indexer_service import run_indexer_service

    stop = threading.Event()
    hits = 0

    def say(message: str) -> None:
        print(f"[indexer {time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal hits
        hits += 1
        if signum == signal.SIGINT and hits > 1:
            raise KeyboardInterrupt
        stop.set()
        say("stopping: finishing the current file")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    # the index log panel only belongs on a terminal; the service logs one line per run.
    quiet = Console(file=sys.stdout, force_terminal=False, quiet=True)
    try:
        run_indexer_service(stop, quiet, say=say)
    except KeyboardInterrupt:
        say("aborted")
        raise typer.Exit(code=130) from None
    say("stopped")


@services_app.command("supervise", hidden=True)
def services_supervise() -> None:
    """The supervisor process itself. `services start` runs it in the background."""
    from bc_rag.services.supervisor import run_supervisor

    raise typer.Exit(code=run_supervisor())
