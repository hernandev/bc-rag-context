"""User settings in `~/.bc-rag/config`, written by `bc-rag config set`.

One JSON object per user, shared by every project. API keys live here so the
background daemon needs no shell profile. A setting missing from the file falls
back to its environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from bc_rag import catalog

SETTINGS_FILENAME = "config"


@dataclass(frozen=True)
class Setting:
    key: str
    env: tuple[str, ...]
    secret: bool
    help: str


SETTINGS: dict[str, Setting] = {
    item.key: item
    for item in (
        Setting(
            "voyage-api-key",
            ("VOYAGE_AI_API_KEY", "VOYAGE_API_KEY"),
            True,
            "Voyage AI key for dense embeddings and rerank.",
        ),
        Setting(
            "jina-api-key",
            ("JINA_API_KEY",),
            True,
            "Jina AI key, only for spaces that use the Jina API.",
        ),
        Setting(
            "daemon-index-every",
            ("BC_RAG_DAEMON_INDEX_EVERY",),
            False,
            "How often the daemon re-indexes every cataloged project, such as 5m.",
        ),
    )
}


class SettingsError(ValueError):
    pass


def settings_path() -> Path:
    return catalog.user_dir() / SETTINGS_FILENAME


def _known(key: str) -> Setting:
    found = SETTINGS.get(key)
    if found is None:
        raise SettingsError(f"unknown setting {key!r}. Known: {', '.join(sorted(SETTINGS))}")
    return found


def load_settings() -> dict[str, str]:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SettingsError(f"{path} is not JSON: {error}") from error
    if not isinstance(data, dict):
        raise SettingsError(f"{path} must hold one JSON object")
    return {str(key): str(value) for key, value in data.items()}


def _save(values: dict[str, str]) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(values, indent=2, sort_keys=True) + "\n"
    # the file holds API keys, so only the owner may read it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def set_setting(key: str, value: str) -> None:
    _known(key)
    value = value.strip()
    if not value:
        raise SettingsError(f"{key} needs a value. Use `bc-rag config unset {key}` to remove it.")
    values = load_settings()
    values[key] = value
    _save(values)


def unset_setting(key: str) -> bool:
    _known(key)
    values = load_settings()
    if key not in values:
        return False
    del values[key]
    _save(values)
    return True


def config_setting(key: str) -> str | None:
    """The value in `~/.bc-rag/config` only. None when the file does not have it."""
    _known(key)
    value = load_settings().get(key, "").strip()
    return value or None


def get_setting(key: str) -> str | None:
    """`~/.bc-rag/config` first, then the environment. None when neither has it."""
    value = config_setting(key)
    if value is not None:
        return value
    for name in SETTINGS[key].env:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def masked(key: str, value: str) -> str:
    if not _known(key).secret:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
