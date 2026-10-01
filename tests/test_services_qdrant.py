import json
import subprocess

import pytest

from bc_rag.services import qdrant
from bc_rag.services.supervisor import QdrantKeeper
from bc_rag.usersettings import set_setting


class DockerRecorder:
    def __init__(self, inspect: dict | None = None) -> None:
        self.calls: list[list[str]] = []
        self.inspect = inspect

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if "inspect" in argv:
            if self.inspect is None:
                return subprocess.CompletedProcess(argv, 1, "", "Error: No such container")
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.inspect), "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _container(labels: dict | None, state: str = "running") -> dict:
    return {
        "State": {"Status": state},
        "Config": {"Image": "qdrant/qdrant:v1.19.1", "Labels": labels},
        "Mounts": [{"Name": "bc-rag-qdrant-data"}],
    }


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> DockerRecorder:
    recorder = DockerRecorder()
    monkeypatch.setattr(qdrant.subprocess, "run", recorder)
    return recorder


def test_context_argv() -> None:
    assert qdrant.docker_base() == ["docker"]
    set_setting("docker-context", "orbstack")
    assert qdrant.docker_base() == ["docker", "--context", "orbstack"]


def test_exact_docker_run() -> None:
    assert qdrant.run_argv() == [
        "docker",
        "run",
        "-d",
        "--name",
        "bc-rag-qdrant",
        "--label",
        "bc-rag=qdrant",
        "--restart",
        "unless-stopped",
        "-p",
        "127.0.0.1:32321:6333",
        "-p",
        "127.0.0.1:32322:6334",
        "-v",
        "bc-rag-qdrant-data:/qdrant/storage",
        "qdrant/qdrant:v1.19.1",
    ]


def test_missing_container_is_created_on_the_volume(docker: DockerRecorder) -> None:
    assert qdrant.ensure_running() == "created"
    assert docker.calls[1] == [
        "docker", "volume", "create", "--label", "bc-rag=qdrant", "bc-rag-qdrant-data"
    ]
    assert docker.calls[2] == qdrant.run_argv()


def test_stopped_container_is_started(docker: DockerRecorder) -> None:
    docker.inspect = _container({"bc-rag": "qdrant"}, state="exited")
    assert qdrant.ensure_running() == "started"
    assert docker.calls[-1] == ["docker", "start", "bc-rag-qdrant"]


def test_foreign_container_is_reported_and_left_alone(docker: DockerRecorder) -> None:
    docker.inspect = _container({"com.docker.compose.project": "bc-rag"})
    with pytest.raises(qdrant.QdrantError, match="did not create"):
        qdrant.ensure_running(recreate=True)
    assert [call[1] for call in docker.calls] == ["container"]


def test_external_never_calls_docker(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_setting("qdrant", "external")
    monkeypatch.setattr(qdrant, "reachable", lambda *a, **k: True)
    keeper = QdrantKeeper(threaded=False, say=lambda message: None)
    keeper.tick(want=True, now=0.0)
    keeper.tick(want=False, now=1.0)
    assert keeper.state == "running"
    assert docker.calls == []


def test_reachable_sends_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        seen.append(request)
        return Response()

    monkeypatch.setattr(qdrant.urllib.request, "urlopen", fake_urlopen)
    set_setting("qdrant-api-key", "secret-key")
    assert qdrant.reachable() is True
    assert seen[0].full_url == "http://127.0.0.1:32321/readyz"
    assert seen[0].get_header("Api-key") == "secret-key"


def test_store_passes_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import qdrant_client

    from bc_rag.store import HybridStore

    seen: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeClient)
    set_setting("qdrant-api-key", "secret-key")
    HybridStore(url="http://q:6333", collection="c")
    assert seen[0]["api_key"] == "secret-key"
    assert seen[0]["url"] == "http://q:6333"
