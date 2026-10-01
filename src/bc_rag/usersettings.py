"""User settings in `~/.bc-rag/config`, written by `bc-rag config set --global`.

One JSON object per user, shared by every project. API keys live here so the
background services need no shell profile.

A key in PROJECT_KEYS may also be set for one project (`bc-rag config set KEY VALUE`
inside it). That value lives in the project's catalog row and wins over this file.
Every other key applies to every project, so it is set with `--global` only.

Every command fills in the missing non-secret settings with their defaults
(`ensure_defaults`), so the file always shows what bc-rag runs with.

Where a value comes from:

- A secret (an API key): the file first, then its environment variable.
- Any other setting: its environment variable first, then the file, then the default.
  The file always holds those keys, so an environment variable is the one-shot override.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bc_rag import catalog
from bc_rag.defaults import QDRANT_HTTP_URL
from bc_rag.fileio import write_json_atomic

SETTINGS_FILENAME = "config"

_TRUE = frozenset({"1", "true", "on", "yes"})
_FALSE = frozenset({"0", "false", "off", "no"})


class SettingsError(ValueError):
    pass


def _bool_value(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return "true"
    if lowered in _FALSE:
        return "false"
    raise SettingsError(f"{value!r} is not true or false")


def _qdrant_mode(value: str) -> str:
    lowered = value.strip().lower()
    if lowered not in {"docker", "external"}:
        raise SettingsError(
            f"{value!r} is not a Qdrant mode. Use docker (bc-rag runs it) or external"
        )
    return lowered


def _every_value(value: str) -> str:
    from bc_rag.catalog import EVERY_OFF
    from bc_rag.duration import parse_duration

    lowered = value.strip().lower()
    if lowered == EVERY_OFF:
        return lowered
    try:
        parse_duration(lowered)
    except ValueError as error:
        raise SettingsError(f"{error}. Use a duration like 10m, or off") from None
    return lowered


def _http_url(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith(("http://", "https://")):
        raise SettingsError(f"{value!r} must start with http:// or https://")
    return stripped.rstrip("/")


@dataclass(frozen=True)
class Setting:
    key: str
    env: tuple[str, ...]
    secret: bool
    help: str
    # None = no default. Secrets never have one and are never written by ensure_defaults.
    default: str | None = None
    # Returns the value to store, or raises SettingsError.
    validate: Callable[[str], str] | None = None


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
            "autostart",
            ("BC_RAG_AUTOSTART", "BC_RAG_DAEMON"),
            False,
            "Start the services when an index or search command needs them.",
            default="true",
            validate=_bool_value,
        ),
        Setting(
            "qdrant",
            ("BC_RAG_QDRANT",),
            False,
            "docker: bc-rag runs Qdrant in Docker. external: you run it at qdrant-url.",
            default="docker",
            validate=_qdrant_mode,
        ),
        Setting(
            "qdrant-url",
            ("BC_RAG_QDRANT_URL",),
            False,
            "Qdrant HTTP URL.",
            default=QDRANT_HTTP_URL,
            validate=_http_url,
        ),
        Setting(
            "qdrant-api-key",
            ("QDRANT_API_KEY",),
            True,
            "Qdrant API key, sent as the api-key header. Only for a Qdrant that asks for one.",
        ),
        Setting(
            "docker-context",
            ("BC_RAG_DOCKER_CONTEXT",),
            False,
            "Docker context for the Qdrant container. Empty = the docker CLI's current one.",
            default="",
        ),
        Setting(
            "every",
            (),
            False,
            "Background indexing interval, like 15m, or off. No value = no background indexing.",
            validate=_every_value,
        ),
    )
}

# keys that can also be set for one project. The rest apply to every project.
PROJECT_KEYS = frozenset({"every"})


def settings_path() -> Path:
    return catalog.user_dir() / SETTINGS_FILENAME


def validate_setting(key: str, value: str) -> str:
    """The value to store for `key`, normalized, or SettingsError."""
    setting = _known(key)
    value = value.strip()
    if not value:
        raise SettingsError(f"{key} needs a value. Use `bc-rag config unset {key}` to remove it.")
    return setting.validate(value) if setting.validate is not None else value


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
    # the file holds API keys, so only the owner may read it.
    write_json_atomic(settings_path(), dict(sorted(values.items())), mode=0o600)


def set_setting(key: str, value: str) -> str:
    """Store one setting and return the stored value (validators may normalize it)."""
    value = validate_setting(key, value)
    values = load_settings()
    values[key] = value
    _save(values)
    return value


def unset_setting(key: str) -> bool:
    _known(key)
    values = load_settings()
    if key not in values:
        return False
    del values[key]
    _save(values)
    return True


def ensure_defaults() -> list[str]:
    """Write every missing non-secret setting with its default. Returns the keys added."""
    values = load_settings()
    added = [
        key
        for key, setting in SETTINGS.items()
        if setting.default is not None and key not in values
    ]
    if not added:
        return []
    for key in added:
        values[key] = SETTINGS[key].default or ""
    _save(values)
    return added


def config_setting(key: str) -> str | None:
    """The value in `~/.bc-rag/config` only. None when the file does not have it."""
    _known(key)
    value = load_settings().get(key, "").strip()
    return value or None


def _env_value(setting: Setting) -> str | None:
    for name in setting.env:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def setting_source(key: str) -> tuple[str | None, str]:
    """(value, where it came from): "config", "env", "default", or "unset"."""
    setting = _known(key)

    def from_file() -> str | None:
        # a non-secret stored empty (docker-context "") is still a value from the file.
        values = load_settings()
        if not setting.secret and key in values:
            return values[key].strip()
        return config_setting(key)

    if setting.secret:
        order = (("config", from_file), ("env", lambda: _env_value(setting)))
    else:
        order = (("env", lambda: _env_value(setting)), ("config", from_file))
    for source, read in order:
        value = read()
        if value is not None:
            return value, source
    if setting.default:
        return setting.default, "default"
    return None, "unset"


def get_setting(key: str) -> str | None:
    """The effective value. See the module docstring for the order. None when unset."""
    return setting_source(key)[0]


def qdrant_url() -> str:
    """The Qdrant HTTP URL with no trailing slash."""
    return _http_url(get_setting("qdrant-url") or QDRANT_HTTP_URL)


def autostart_enabled() -> bool:
    """False when the environment or the file says so. True otherwise."""
    value = get_setting("autostart") or "true"
    try:
        return _bool_value(value) == "true"
    except SettingsError:
        return True


def masked(key: str, value: str) -> str:
    if not _known(key).secret:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
