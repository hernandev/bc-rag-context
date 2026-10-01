import stat

import pytest

from bc_rag.usersettings import (
    SettingsError,
    autostart_enabled,
    config_setting,
    ensure_defaults,
    get_setting,
    load_settings,
    masked,
    qdrant_url,
    set_setting,
    setting_source,
    settings_path,
    unset_setting,
)
from bc_rag.voyage_api import VoyageApiError, resolve_api_key


def test_set_writes_owner_only_file() -> None:
    set_setting("voyage-api-key", "  pa-secret-key  ")
    assert load_settings() == {"voyage-api-key": "pa-secret-key"}
    assert stat.S_IMODE(settings_path().stat().st_mode) == 0o600


def test_config_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "from-env")
    assert get_setting("voyage-api-key") == "from-env"
    set_setting("voyage-api-key", "from-config")
    assert get_setting("voyage-api-key") == "from-config"
    assert resolve_api_key() == "from-config"


def test_environment_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_AI_API_KEY", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "legacy-name")
    assert config_setting("voyage-api-key") is None
    assert resolve_api_key() == "legacy-name"


def test_missing_key_names_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_AI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(VoyageApiError, match="bc-rag config set voyage-api-key"):
        resolve_api_key()


def test_unset_and_unknown_keys() -> None:
    set_setting("qdrant", "external")
    assert unset_setting("qdrant") is True
    assert unset_setting("qdrant") is False
    with pytest.raises(SettingsError):
        set_setting("no-such-key", "x")


def test_masked_hides_only_secrets() -> None:
    assert masked("voyage-api-key", "pa-1234567890") == "pa-1...7890"
    assert masked("qdrant-api-key", "short") == "*****"
    assert masked("autostart", "true") == "true"


def test_ensure_defaults_writes_non_secrets_once() -> None:
    added = ensure_defaults()
    assert added == ["autostart", "qdrant", "qdrant-url", "docker-context"]
    assert load_settings() == {
        "autostart": "true",
        "docker-context": "",
        "qdrant": "docker",
        "qdrant-url": "http://127.0.0.1:32321",
    }
    assert stat.S_IMODE(settings_path().stat().st_mode) == 0o600
    assert ensure_defaults() == []


def test_ensure_defaults_keeps_existing_values_and_secrets() -> None:
    set_setting("voyage-api-key", "pa-secret")
    set_setting("qdrant", "external")
    ensure_defaults()
    values = load_settings()
    assert values["voyage-api-key"] == "pa-secret"
    assert values["qdrant"] == "external"
    assert "jina-api-key" not in values
    assert "qdrant-api-key" not in values


@pytest.mark.parametrize(
    ("key", "value"),
    [("autostart", "maybe"), ("qdrant", "both"), ("qdrant-url", "127.0.0.1:6333")],
)
def test_validators_reject_bad_values(key: str, value: str) -> None:
    with pytest.raises(SettingsError):
        set_setting(key, value)
    assert key not in load_settings()


def test_validators_normalize() -> None:
    assert set_setting("autostart", "OFF") == "false"
    assert set_setting("qdrant", "External") == "external"
    assert set_setting("qdrant-url", "https://q.example:6333/") == "https://q.example:6333"


def test_non_secret_environment_wins_over_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_defaults()
    monkeypatch.delenv("BC_RAG_QDRANT_URL", raising=False)
    assert qdrant_url() == "http://127.0.0.1:32321"
    assert setting_source("qdrant-url") == ("http://127.0.0.1:32321", "config")
    monkeypatch.setenv("BC_RAG_QDRANT_URL", "http://10.0.0.5:6333/")
    assert qdrant_url() == "http://10.0.0.5:6333"
    assert setting_source("qdrant-url")[1] == "env"


def test_autostart_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BC_RAG_DAEMON", raising=False)
    monkeypatch.delenv("BC_RAG_AUTOSTART", raising=False)
    assert autostart_enabled() is True
    set_setting("autostart", "false")
    assert autostart_enabled() is False
    monkeypatch.setenv("BC_RAG_DAEMON", "true")
    assert autostart_enabled() is True
    unset_setting("autostart")
    monkeypatch.setenv("BC_RAG_DAEMON", "false")
    assert autostart_enabled() is False


def test_unset_default_reports_the_default() -> None:
    assert setting_source("qdrant") == ("docker", "default")
    assert setting_source("jina-api-key")[1] in {"unset", "env"}
