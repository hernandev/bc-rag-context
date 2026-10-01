import pytest

from bc_rag import daemon


def test_lock_says_whether_a_daemon_runs() -> None:
    assert daemon.is_running() is False
    handle = daemon._open_lock()
    try:
        assert daemon._try_lock(handle) is True
        assert daemon.is_running() is True
    finally:
        daemon._unlock(handle)
        handle.close()
    assert daemon.is_running() is False


def test_stop_without_daemon() -> None:
    assert daemon.stop() is False


def test_fingerprint_is_stable() -> None:
    assert daemon.source_fingerprint() == daemon.source_fingerprint()


def test_ensure_respects_the_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(daemon, "spawn", lambda: calls.append("spawn") or 1)
    daemon.ensure()
    assert calls == []
    monkeypatch.setenv("BC_RAG_DAEMON", "true")
    daemon.ensure()
    assert calls == ["spawn"]


def test_children_use_the_configured_interval() -> None:
    from bc_rag.usersettings import set_setting

    set_setting("daemon-index-every", "10m")
    names = {child.name: child.args for child in daemon.children()}
    assert names == {
        "index": ["index", "--all", "--every", "10m"],
        "mcp": ["mcp", "--http"],
    }
