"""Qdrant: a Docker container bc-rag runs, or one you run yourself.

Setting `qdrant`:

- `docker`: bc-rag creates and starts the container `bc-rag-qdrant` from
  `qdrant/qdrant` with its data on the Docker volume `bc-rag-qdrant-data`. The
  ports are published on 127.0.0.1 only.
- `external`: bc-rag never calls docker. It only checks that `qdrant-url` answers.

A container named `bc-rag-qdrant` without the label `bc-rag=qdrant` was not made by
this code (for example by the old compose file). It is reported, never touched.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from bc_rag.defaults import QDRANT_GRPC_PORT, QDRANT_HTTP_PORT, QDRANT_IMAGE

QDRANT_CONTAINER = "bc-rag-qdrant"
QDRANT_VOLUME = "bc-rag-qdrant-data"
QDRANT_LABEL = "bc-rag=qdrant"
STOP_SECONDS = 30
READY_SECONDS = 30.0


class QdrantError(RuntimeError):
    pass


@dataclass
class ContainerStatus:
    exists: bool
    # Docker's state: running, exited, created, paused, ...
    state: str = ""
    image: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def ours(self) -> bool:
        key, _, value = QDRANT_LABEL.partition("=")
        return self.labels.get(key) == value

    @property
    def drifted(self) -> bool:
        return bool(self.image) and self.image != QDRANT_IMAGE


def mode() -> str:
    from bc_rag.usersettings import get_setting

    return get_setting("qdrant") or "docker"


def url() -> str:
    from bc_rag.usersettings import qdrant_url

    return qdrant_url()


def api_key() -> str | None:
    from bc_rag.usersettings import get_setting

    return get_setting("qdrant-api-key")


def docker_base() -> list[str]:
    from bc_rag.usersettings import get_setting

    context = (get_setting("docker-context") or "").strip()
    return ["docker", *(["--context", context] if context else [])]


def run_argv() -> list[str]:
    """The `docker run` that creates the container, ports on loopback only."""
    return [
        *docker_base(),
        "run",
        "-d",
        "--name",
        QDRANT_CONTAINER,
        "--label",
        QDRANT_LABEL,
        "--restart",
        "unless-stopped",
        "-p",
        f"127.0.0.1:{QDRANT_HTTP_PORT}:6333",
        "-p",
        f"127.0.0.1:{QDRANT_GRPC_PORT}:6334",
        "-v",
        f"{QDRANT_VOLUME}:/qdrant/storage",
        QDRANT_IMAGE,
    ]


def _docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [*docker_base(), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise QdrantError("docker is not on PATH. Install Docker or set qdrant external.") from None
    except subprocess.TimeoutExpired:
        raise QdrantError(f"docker {' '.join(args)} timed out after {timeout:g}s") from None


def _checked(*args: str, timeout: float = 60) -> str:
    result = _docker(*args, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise QdrantError(detail or f"docker {' '.join(args)} failed")
    return result.stdout


def container_status() -> ContainerStatus:
    result = _docker("container", "inspect", QDRANT_CONTAINER, "--format", "{{json .}}")
    if result.returncode != 0:
        detail = (result.stderr or "").lower()
        if "no such" in detail:
            return ContainerStatus(exists=False)
        raise QdrantError((result.stderr or result.stdout or "docker inspect failed").strip())
    data = json.loads(result.stdout)
    config = data.get("Config") or {}
    mounts = data.get("Mounts") or []
    return ContainerStatus(
        exists=True,
        state=str((data.get("State") or {}).get("Status") or ""),
        image=str(config.get("Image") or ""),
        labels=dict(config.get("Labels") or {}),
        volumes=[str(mount.get("Name") or mount.get("Source") or "") for mount in mounts],
    )


def foreign_message(status: ContainerStatus) -> str:
    return (
        f"a container named {QDRANT_CONTAINER} exists that bc-rag did not create "
        f"(no {QDRANT_LABEL} label, image {status.image or '?'}). bc-rag leaves it alone. "
        f"Remove it with `docker rm -f {QDRANT_CONTAINER}` to let bc-rag create its own."
    )


def create_and_start() -> None:
    _checked("volume", "create", "--label", QDRANT_LABEL, QDRANT_VOLUME)
    _checked(*run_argv()[len(docker_base()) :], timeout=300)


def start() -> None:
    _checked("start", QDRANT_CONTAINER)


def stop() -> None:
    _checked("stop", "-t", str(STOP_SECONDS), QDRANT_CONTAINER, timeout=STOP_SECONDS + 30)


def remove() -> None:
    _checked("rm", "-f", QDRANT_CONTAINER)


def ensure_running(*, recreate: bool = False) -> str:
    """Create or start the container. Returns what was done. Raises QdrantError."""
    status = container_status()
    if status.exists and not status.ours:
        raise QdrantError(foreign_message(status))
    if status.exists and recreate:
        remove()
        status = ContainerStatus(exists=False)
    if not status.exists:
        create_and_start()
        return "created"
    if status.running:
        return "running"
    start()
    return "started"


def reachable(target: str | None = None, key: str | None = None, *, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(f"{(target or url()).rstrip('/')}/readyz")
    key = key if key is not None else api_key()
    if key:
        request.add_header("api-key", key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_ready(timeout: float = READY_SECONDS, *, sleep=time.sleep, clock=time.monotonic) -> bool:
    deadline = clock() + timeout
    while True:
        if reachable():
            return True
        if clock() >= deadline:
            return False
        sleep(0.25)
