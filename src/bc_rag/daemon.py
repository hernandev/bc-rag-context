"""The bc-rag daemon: started on demand by any bc-rag command, like the Nx daemon.

No launchd, no systemd, no Task Scheduler. The first bc-rag command that runs
starts one detached supervisor process. That process inherits the caller's
environment, so `node`, `pnpm` and `docker` resolve from the user's own PATH.
API keys come from `~/.bc-rag/config` (see `bc_rag.usersettings`).

The supervisor runs two children and restarts either one when it exits:

- `bc-rag index --all --every <daemon-index-every>` re-indexes every cataloged project.
- `bc-rag mcp --http` is the one MCP listener every chat shares.

One daemon per user. A lock file held for the daemon's lifetime says whether it
runs. When the bc-rag source changes, the next command restarts the daemon so
it never serves old code.

Set `BC_RAG_DAEMON=false` to stop commands from starting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from bc_rag import __version__, catalog

DAEMON_ENV = "BC_RAG_DAEMON"
DEFAULT_INDEX_EVERY = "5m"
LOG_ROTATE_BYTES = 10 * 1024 * 1024
POLL_SECONDS = 2.0
STOP_WAIT_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 60.0

IS_WINDOWS = sys.platform == "win32"


def daemon_dir() -> Path:
    return catalog.user_dir()


def lock_path() -> Path:
    return daemon_dir() / "daemon.lock"


def state_path() -> Path:
    return daemon_dir() / "daemon.json"


def log_path() -> Path:
    return daemon_dir() / "logs" / "daemon.log"


def disabled() -> bool:
    return os.environ.get(DAEMON_ENV, "").strip().lower() in {"0", "false", "off", "no"}


# -- lock ---------------------------------------------------------------------


def _try_lock(handle: IO[str]) -> bool:
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


def _unlock(handle: IO[str]) -> None:
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


def _open_lock() -> IO[str]:
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a+", encoding="utf-8")


def is_running() -> bool:
    """True while some process holds the daemon lock."""
    with _open_lock() as handle:
        if _try_lock(handle):
            _unlock(handle)
            return False
        return True


# -- state --------------------------------------------------------------------


def source_fingerprint() -> str:
    """Changes whenever any bc-rag source file changes, so a stale daemon is replaced."""
    digest = hashlib.sha256(__version__.encode("utf-8"))
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        stat = path.stat()
        digest.update(f"{path.relative_to(package)}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()[:16]


def read_state() -> dict[str, object]:
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(values: dict[str, object]) -> None:
    state_path().write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


# -- client side --------------------------------------------------------------


def _python_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "bc_rag", *args]


def _rotate_log() -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > LOG_ROTATE_BYTES:
        path.replace(path.with_suffix(".log.1"))


def _detached_kwargs() -> dict[str, object]:
    if IS_WINDOWS:
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
        return {"creationflags": flags}
    return {"start_new_session": True}


def spawn() -> int:
    """Start the supervisor in the background with this process's environment."""
    _rotate_log()
    with open(log_path(), "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            _python_command("daemon", "run"),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=str(daemon_dir()),
            close_fds=True,
            **_detached_kwargs(),
        )
    return process.pid


def _signal_pid(pid: int) -> None:
    if IS_WINDOWS:
        # taskkill /T also ends the two children.
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def stop(timeout: float = STOP_WAIT_SECONDS) -> bool:
    """Stop the daemon and wait for its lock. Returns False when none was running."""
    if not is_running():
        return False
    pid = read_state().get("pid")
    if isinstance(pid, int):
        _signal_pid(pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_running():
            return True
        time.sleep(0.2)
    raise RuntimeError(f"daemon pid {pid} did not stop within {timeout:g}s")


def ensure(report: Callable[[str], None] | None = None) -> None:
    """Start the daemon when none runs, and replace one that runs old code."""
    if disabled():
        return
    if is_running():
        if read_state().get("fingerprint") == source_fingerprint():
            return
        stop()
        pid = spawn()
        if report:
            report(f"bc-rag daemon restarted on new code (pid {pid}, log {log_path()})")
        return
    pid = spawn()
    if report:
        report(f"bc-rag daemon started (pid {pid}, log {log_path()})")


# -- supervisor ---------------------------------------------------------------


@dataclass
class Child:
    name: str
    args: list[str]
    process: subprocess.Popen[bytes] | None = None
    failures: int = 0
    next_start: float = 0.0
    started_at: float = 0.0

    def start(self, env: dict[str, str]) -> None:
        self.process = subprocess.Popen(
            _python_command(*self.args),
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            cwd=str(daemon_dir()),
        )
        self.started_at = time.monotonic()
        _say(f"started {self.name} pid {self.process.pid}: bc-rag {' '.join(self.args)}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def _say(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[daemon {stamp}] {message}", flush=True)


def children() -> list[Child]:
    from bc_rag.usersettings import get_setting

    every = get_setting("daemon-index-every") or DEFAULT_INDEX_EVERY
    return [
        Child("index", ["index", "--all", "--every", every]),
        Child("mcp", ["mcp", "--http"]),
    ]


def run() -> int:
    """The supervisor loop. Exits at once when another daemon holds the lock."""
    handle = _open_lock()
    if not _try_lock(handle):
        handle.close()
        _say("another daemon already runs; exiting")
        return 0

    stopping = False

    def on_signal(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    # children must not try to start or replace the daemon themselves.
    env = dict(os.environ)
    env[DAEMON_ENV] = "false"

    workers = children()
    _write_state(
        {
            "pid": os.getpid(),
            "fingerprint": source_fingerprint(),
            "version": __version__,
            "python": sys.executable,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "children": {worker.name: worker.args for worker in workers},
        }
    )
    _say(f"supervisor pid {os.getpid()}")
    try:
        while not stopping:
            now = time.monotonic()
            for worker in workers:
                exited = worker.process is not None and worker.process.poll() is not None
                if exited:
                    code = worker.process.returncode if worker.process else None
                    # a child that ran for a while before exiting resets its backoff.
                    if now - worker.started_at > MAX_BACKOFF_SECONDS:
                        worker.failures = 0
                    worker.failures += 1
                    delay = min(MAX_BACKOFF_SECONDS, 2.0 ** worker.failures)
                    worker.next_start = now + delay
                    worker.process = None
                    _say(f"{worker.name} exited with {code}; restart in {delay:g}s")
                if worker.process is None and now >= worker.next_start:
                    worker.start(env)
            time.sleep(POLL_SECONDS)
    finally:
        _say("stopping")
        for worker in workers:
            worker.stop()
        try:
            state_path().unlink()
        except OSError:
            pass
        _unlock(handle)
        handle.close()
    return 0
