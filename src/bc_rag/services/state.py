"""Files the services share under `~/.bc-rag/`.

- `services.json`: what you asked for (desired state). Written only by
  `bc-rag services start/stop/restart`. A missing file means every service runs.
- `supervisor.json`: what is happening (runtime state). Written only by the supervisor.
- `daemon.lock`: held by the one supervisor for its whole life.
- `logs/`: one rotating log per service, plus the supervisor's.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import bc_rag
from bc_rag import __version__, catalog
from bc_rag.fileio import write_json_atomic

SERVICE_NAMES = ("qdrant", "indexer", "mcp")
RUNNING = "running"
STOPPED = "stopped"

# The env var a supervised child carries, so `services run` can tell it apart.
SUPERVISED_ENV = "BC_RAG_SUPERVISED"


def home() -> Path:
    return catalog.user_dir()


def lock_path() -> Path:
    # the old daemon used this same file, so new code also sees an old supervisor.
    return home() / "daemon.lock"


def desired_path() -> Path:
    return home() / "services.json"


def runtime_path() -> Path:
    return home() / "supervisor.json"


def legacy_state_path() -> Path:
    return home() / "daemon.json"


def logs_dir() -> Path:
    return home() / "logs"


def log_path(name: str) -> Path:
    return logs_dir() / f"{name}.log"


def source_fingerprint() -> str:
    """Changes whenever any bc-rag source file changes, so a stale supervisor is replaced."""
    digest = hashlib.sha256(__version__.encode("utf-8"))
    package = Path(bc_rag.__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        stat = path.stat()
        digest.update(f"{path.relative_to(package)}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()[:16]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# -- desired state ------------------------------------------------------------------


@dataclass
class ServiceDesire:
    state: str = RUNNING
    # Bumped by `services restart NAME`. The supervisor restarts the child when it grows.
    restart: int = 0


@dataclass
class DesiredState:
    generation: int = 0
    services: dict[str, ServiceDesire] = field(
        default_factory=lambda: {name: ServiceDesire() for name in SERVICE_NAMES}
    )
    updated_at: str = ""
    updated_by: str = ""

    def wants(self, name: str) -> bool:
        return self.services[name].state == RUNNING

    def any_running(self) -> bool:
        return any(self.wants(name) for name in SERVICE_NAMES)


def load_desired() -> DesiredState:
    """The saved desired state. Missing file, bad JSON, or missing names = running."""
    desired = DesiredState()
    try:
        raw = json.loads(desired_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return desired
    if not isinstance(raw, dict):
        return desired
    desired.generation = int(raw.get("generation") or 0)
    desired.updated_at = str(raw.get("updated_at") or "")
    desired.updated_by = str(raw.get("updated_by") or "")
    services = raw.get("services") if isinstance(raw.get("services"), dict) else {}
    for name in SERVICE_NAMES:
        row = services.get(name)
        if not isinstance(row, dict):
            continue
        state = RUNNING if row.get("state") != STOPPED else STOPPED
        desired.services[name] = ServiceDesire(state=state, restart=int(row.get("restart") or 0))
    return desired


def save_desired(desired: DesiredState, *, by: str) -> None:
    desired.generation += 1
    desired.updated_at = now_iso()
    desired.updated_by = by
    write_json_atomic(desired_path(), asdict(desired))


def set_desired(names: list[str], state: str, *, by: str) -> DesiredState:
    desired = load_desired()
    for name in names:
        desired.services[name].state = state
    save_desired(desired, by=by)
    return desired


def bump_restart(name: str, *, by: str) -> DesiredState:
    desired = load_desired()
    desired.services[name].state = RUNNING
    desired.services[name].restart += 1
    save_desired(desired, by=by)
    return desired


def desired_mtime() -> int:
    """Nanosecond mtime of services.json, so the supervisor notices every write."""
    try:
        return desired_path().stat().st_mtime_ns
    except OSError:
        return 0


# -- runtime state ------------------------------------------------------------------


def read_runtime() -> dict[str, Any]:
    try:
        data = json.loads(runtime_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_runtime(values: dict[str, Any]) -> None:
    write_json_atomic(runtime_path(), values)


def lock_pid() -> int | None:
    """The pid a supervisor wrote into daemon.lock when it took the lock.

    The old daemon wrote nothing there, so None while the lock is held means the old daemon.
    """
    from bc_rag import locks

    text = locks.read_note(lock_path())
    return int(text) if text.isdigit() and int(text) > 0 else None


def supervisor_pid() -> int | None:
    """The pid in daemon.lock, else supervisor.json, else the old daemon's daemon.json."""
    found = lock_pid()
    if found is not None:
        return found
    for path in (runtime_path(), legacy_state_path()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = data.get("pid") if isinstance(data, dict) else None
        if isinstance(pid, int) and pid > 0:
            return pid
    return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
