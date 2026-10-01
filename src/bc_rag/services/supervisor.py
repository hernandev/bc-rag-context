"""The supervisor: one background process per user that keeps each service as you asked.

It reads `services.json` (desired) and writes `supervisor.json` (runtime). It runs the
`indexer` and `mcp` services as child processes and keeps the Qdrant container up
when `qdrant=docker`. A stopped service stays stopped until `bc-rag services start`.

Children inherit the supervisor's environment, which is the environment of the
shell that started it. `bc-rag services restart` with no name starts a new
supervisor from the current shell.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bc_rag import __version__, locks
from bc_rag.services import qdrant as qdrant_ops
from bc_rag.services.state import (
    RUNNING,
    STOPPED,
    SUPERVISED_ENV,
    DesiredState,
    desired_mtime,
    home,
    legacy_state_path,
    load_desired,
    lock_path,
    lock_pid,
    log_path,
    now_iso,
    read_runtime,
    source_fingerprint,
    supervisor_pid,
    write_runtime,
)

CHILD_NAMES = ("indexer", "mcp")
# How long a child gets after SIGTERM before SIGKILL. The indexer finishes its
# current file first: a Voyage POST may retry up to 12 times.
STOP_TIMEOUT = {"indexer": 120.0, "mcp": 10.0}
POLL_SECONDS = 2.0
MAX_BACKOFF = 60.0
# `services run mcp` exits 3 when its port is taken. Retrying every 2s would spin.
EXIT_PORT_IN_USE = 3
PORT_RETRY = 60.0
QDRANT_RETRY = 30.0

Report = Callable[[str], None]


def child_argv(name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bc_rag",
        "services",
        "run",
        name,
        "--log-file",
        str(log_path(name)),
    ]


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    # a child must never start or replace the services itself.
    env["BC_RAG_AUTOSTART"] = "false"
    env[SUPERVISED_ENV] = "1"
    return env


def _say(message: str) -> None:
    print(f"[supervisor {time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


# -- children ---------------------------------------------------------------------


@dataclass
class Child:
    name: str
    process: Any = None
    # stopped, starting, running, stopping, backoff, or failed.
    state: str = STOPPED
    since: str = ""
    failures: int = 0
    next_start: float = 0.0
    started_at: float = 0.0
    stop_deadline: float | None = None
    exit_code: int | None = None
    reason: str = ""
    restart_seen: int = 0
    restart_pending: bool = False

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def row(self, now: float) -> dict[str, Any]:
        next_in = max(0.0, self.next_start - now) if self.state in ("backoff", "failed") else None
        return {
            "state": self.state,
            "pid": self.process.pid if self.alive() else None,
            "since": self.since,
            "argv": child_argv(self.name),
            "exit_code": self.exit_code,
            "failures": self.failures,
            "next_start_in": None if next_in is None else round(next_in),
            "stop_timeout": STOP_TIMEOUT[self.name],
            "reason": self.reason,
            "log": str(log_path(self.name)),
        }


class QdrantKeeper:
    """Keeps the container in the desired state. Docker calls run off the poll loop."""

    def __init__(self, *, ops: Any = qdrant_ops, threaded: bool = True, say: Report = _say):
        self.ops = ops
        self.threaded = threaded
        self.say = say
        self.state = "unknown"
        self.reason = ""
        self.next_try = 0.0
        self._thread: threading.Thread | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self, job: Callable[[], None]) -> None:
        if not self.threaded:
            job()
            return
        self._thread = threading.Thread(target=job, name="bc-rag-qdrant", daemon=True)
        self._thread.start()

    def tick(self, want: bool, now: float) -> None:
        if self.busy():
            return
        if self.ops.mode() == "external":
            self.state = RUNNING if self.ops.reachable() else "unreachable"
            self.reason = "" if self.state == RUNNING else f"no answer at {self.ops.url()}"
            return
        if want:
            if self.ops.reachable():
                self.state, self.reason = RUNNING, ""
                return
            if now < self.next_try:
                return
            self.state = "starting"
            self._run(lambda: self._start(now))
            return
        if self.state != STOPPED:
            self.state = "stopping"
            self._run(self._stop)

    def _start(self, now: float) -> None:
        try:
            done = self.ops.ensure_running()
            if not self.ops.wait_ready():
                raise qdrant_ops.QdrantError(f"not ready at {self.ops.url()} after 30s")
            self.state, self.reason = RUNNING, ""
            self.say(f"qdrant {done}")
        except Exception as error:
            self.state, self.reason = "failed", str(error)
            self.next_try = now + QDRANT_RETRY
            self.say(f"qdrant failed: {error} (retry in {QDRANT_RETRY:g}s)")

    def _stop(self) -> None:
        try:
            status = self.ops.container_status()
            if status.exists and status.ours and status.running:
                self.ops.stop()
                self.say("qdrant stopped")
            self.state, self.reason = STOPPED, ""
        except Exception as error:
            self.state, self.reason = "failed", str(error)
            self.say(f"qdrant stop failed: {error}")

    def row(self) -> dict[str, Any]:
        return {"mode": self.ops.mode(), "url": self.ops.url(), "state": self.state,
                "reason": self.reason}


class Supervisor:
    """The poll loop's state. `tick` is one pass; tests drive it with a fake clock."""

    def __init__(
        self,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        env: dict[str, str] | None = None,
        qdrant: QdrantKeeper | None = None,
        say: Report = _say,
    ) -> None:
        self.popen = popen
        self.clock = clock
        self.env = child_env(env)
        self.say = say
        self.qdrant = qdrant or QdrantKeeper(say=say)
        self.children = {name: Child(name) for name in CHILD_NAMES}
        self.desired: DesiredState = load_desired()
        self._desired_mtime = desired_mtime()
        for name, child in self.children.items():
            # a restart asked for before this supervisor started is already satisfied.
            child.restart_seen = self.desired.services[name].restart
        self.started_at = now_iso()
        # the code this process loaded. Taken once: the files on disk may change later,
        # and a later reading would hide that this process still runs the old code.
        self.fingerprint = source_fingerprint()
        self._last_written: dict[str, Any] | None = None

    def reload_desired(self) -> None:
        mtime = desired_mtime()
        if mtime != self._desired_mtime:
            self._desired_mtime = mtime
            self.desired = load_desired()

    def _start(self, child: Child, now: float) -> None:
        path = log_path(child.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as log:
            child.process = self.popen(
                child_argv(child.name),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=self.env,
                cwd=str(home()),
            )
        child.state = RUNNING
        child.since = now_iso()
        child.started_at = now
        child.stop_deadline = None
        child.reason = ""
        self.say(f"started {child.name} pid {child.process.pid}")

    def _stop(self, child: Child, now: float) -> None:
        if child.state != "stopping":
            child.process.terminate()
            child.state = "stopping"
            child.stop_deadline = now + STOP_TIMEOUT[child.name]
            self.say(f"stopping {child.name} pid {child.process.pid}")
        elif child.stop_deadline is not None and now >= child.stop_deadline:
            child.process.kill()
            child.reason = f"killed after {STOP_TIMEOUT[child.name]:g}s"
            child.stop_deadline = None
            self.say(f"killed {child.name}: no exit {STOP_TIMEOUT[child.name]:g}s after SIGTERM")

    def _reap(self, child: Child, want: bool, now: float) -> None:
        code = child.process.returncode
        child.process = None
        child.exit_code = code
        if child.state == "stopping":
            child.state = STOPPED
            child.stop_deadline = None
            if child.restart_pending and want:
                child.restart_pending = False
                child.next_start = now
            self.say(f"{child.name} stopped (exit {code})")
            return
        if code == EXIT_PORT_IN_USE:
            child.state = "failed"
            child.reason = "port in use"
            child.next_start = now + PORT_RETRY
            self.say(f"{child.name} exited 3: port in use (retry in {PORT_RETRY:g}s)")
            return
        if now - child.started_at > MAX_BACKOFF:
            child.failures = 0
        child.failures += 1
        delay = min(MAX_BACKOFF, 2.0**child.failures)
        child.state = "backoff"
        child.reason = f"exited with {code}"
        child.next_start = now + delay
        self.say(f"{child.name} exited with {code}; restart in {delay:g}s")

    def _tick_child(self, child: Child, stopping: bool, now: float) -> None:
        want = not stopping and self.desired.wants(child.name)
        counter = self.desired.services[child.name].restart
        if counter > child.restart_seen:
            child.restart_seen = counter
            if want:
                child.restart_pending = True
                child.next_start = now
        if child.alive():
            if not want or child.restart_pending:
                self._stop(child, now)
            return
        if child.process is not None:
            self._reap(child, want, now)
        if want and now >= child.next_start:
            child.restart_pending = False
            self._start(child, now)
        elif not want and child.state != STOPPED:
            child.state = STOPPED
            child.reason = ""

    def tick(self, stopping: bool = False) -> bool:
        """One pass. True when the supervisor should exit."""
        now = self.clock()
        self.reload_desired()
        if not stopping:
            self.qdrant.tick(self.desired.wants("qdrant"), now)
        for child in self.children.values():
            self._tick_child(child, stopping, now)
        self.write_state(now)
        alive = any(child.alive() for child in self.children.values())
        if stopping:
            return not alive
        return not self.desired.any_running() and not alive and not self.qdrant.busy()

    def write_state(self, now: float, *, running: bool = True) -> None:
        from bc_rag.usersettings import autostart_enabled

        values = {
            "pid": os.getpid() if running else None,
            "fingerprint": self.fingerprint if running else None,
            "version": __version__,
            "python": sys.executable,
            "started_at": self.started_at,
            "autostart": autostart_enabled(),
            "qdrant": self.qdrant.row(),
            "children": {name: child.row(now) for name, child in self.children.items()},
        }
        if values != self._last_written:
            self._last_written = values
            write_runtime({**values, "updated_at": now_iso()})


def run_supervisor() -> int:
    """The supervisor process. Exits 0 at once when another supervisor holds the lock."""
    from bc_rag.services.logs import redirect_process_output

    held = locks.acquire(lock_path(), note=str(os.getpid()))
    if held is None:
        _say("another supervisor already runs; exiting")
        return 0
    redirect_process_output(log_path("supervisor"))
    stopping = False

    def on_signal(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    legacy_state_path().unlink(missing_ok=True)
    supervisor = Supervisor()
    _say(f"supervisor pid {os.getpid()}  code {supervisor.fingerprint}")
    try:
        while not supervisor.tick(stopping):
            time.sleep(POLL_SECONDS)
    finally:
        # a second pass with stopping set: SIGTERM, then SIGKILL past each deadline.
        while not supervisor.tick(stopping=True):
            time.sleep(0.5)
        supervisor.write_state(supervisor.clock(), running=False)
        _say("supervisor stopped")
        held.release()
    return 0


# -- client side --------------------------------------------------------------------


def supervisor_running() -> bool:
    return locks.is_locked(lock_path())


def spawn(env: dict[str, str] | None = None) -> int:
    """Start a supervisor in the background with `env` (default: this process's)."""
    path = log_path("supervisor")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "bc_rag", "services", "supervise"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=str(home()),
            env=env,
            close_fds=True,
            start_new_session=True,
        )
    return process.pid


def stop_supervisor(
    *,
    timeout: float | None = None,
    progress: Report | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """SIGTERM the supervisor and wait for its lock. False when none was running.

    Raises RuntimeError naming the pid when it outlives `timeout`.
    """
    if not supervisor_running():
        return False
    pid = supervisor_pid()
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    limit = timeout if timeout is not None else sum(STOP_TIMEOUT.values()) + 10
    waited = 0.0
    while supervisor_running():
        if pid is None:
            # a supervisor that was still starting may have written its pid by now.
            pid = supervisor_pid()
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if waited >= limit:
            raise RuntimeError(
                f"supervisor pid {pid} did not stop within {limit:g}s. Run `kill -9 {pid}`."
            )
        if progress is not None and waited and waited % 5 == 0:
            progress(f"stopping  waiting for the services to finish  {waited:g}s")
        sleep(0.5)
        waited += 0.5
    return True


def is_stale() -> bool:
    """True when the running supervisor runs other code, or is the old daemon.

    A supervisor writes its pid into daemon.lock first and supervisor.json on its first
    pass, so a lock pid without a matching supervisor.json is one that is starting up.
    """
    holder = lock_pid()
    if holder is None:
        # the lock is held, but not by a supervisor: the old daemon.
        return True
    runtime = read_runtime()
    if runtime.get("pid") != holder:
        return False
    return runtime.get("fingerprint") != source_fingerprint()


def replace_if_stale(report: Report | None = None) -> bool:
    if not supervisor_running() or not is_stale():
        return False
    if report:
        report("services run old code; replacing them (the indexer finishes its current file)")
    stop_supervisor()
    pid = spawn()
    if report:
        report(
            f"services restarted on new code (supervisor pid {pid}, log {log_path('supervisor')})"
        )
    return True


@dataclass
class StackStatus:
    # off, stopped, running, started, replaced, or error.
    supervisor: str = "off"
    qdrant_ready: bool = False
    notes: list[str] = field(default_factory=list)


def ensure_stack(
    report: Report | None = None, *, wait_qdrant: bool = True, timeout: float = 30.0
) -> StackStatus:
    """Bring up the services a command needs. Never raises; never changes desired state."""
    from bc_rag.usersettings import autostart_enabled, setting_source

    status = StackStatus()

    def note(message: str) -> None:
        status.notes.append(message)
        if report:
            report(message)

    try:
        if not autostart_enabled():
            if setting_source("autostart")[1] == "config":
                note("autostart is off (bc-rag config set autostart true --global turns it on)")
            return status
        desired = load_desired()
        if not desired.any_running():
            status.supervisor = STOPPED
            note("services are stopped. `bc-rag services start` brings them up")
        elif supervisor_running():
            status.supervisor = "replaced" if replace_if_stale(note) else RUNNING
        else:
            pid = spawn()
            status.supervisor = "started"
            note(f"services started (supervisor pid {pid}, log {log_path('supervisor')})")
        if desired.wants("qdrant") and wait_qdrant:
            status.qdrant_ready = qdrant_ops.wait_ready(timeout)
            if not status.qdrant_ready:
                note(f"qdrant is not reachable at {qdrant_ops.url()}. See `bc-rag services list`")
        else:
            status.qdrant_ready = qdrant_ops.reachable()
            if not status.qdrant_ready:
                note("qdrant is stopped. `bc-rag services start qdrant` starts it")
    except Exception as error:
        status.supervisor = "error"
        note(f"services: {error}")
    return status
