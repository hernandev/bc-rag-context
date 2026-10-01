import stat

import pytest

from bc_rag.usersettings import (
    SettingsError,
    config_setting,
    get_setting,
    load_settings,
    masked,
    set_setting,
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
    set_setting("daemon-index-every", "10m")
    assert unset_setting("daemon-index-every") is True
    assert unset_setting("daemon-index-every") is False
    with pytest.raises(SettingsError):
        set_setting("no-such-key", "x")


def test_masked_hides_only_secrets() -> None:
    assert masked("voyage-api-key", "pa-1234567890") == "pa-1...7890"
    assert masked("daemon-index-every", "5m") == "5m"
