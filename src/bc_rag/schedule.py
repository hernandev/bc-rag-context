"""Background indexing schedule: each project's interval and the state of its last run.

A project is indexed in the background only when it has an interval, the config key
`every`, looked up in this order:

- yours for this project, in the catalog (`bc-rag config set every 5m`, or `off`), else
- the team default in `.bc-rag.json` (`"schedule": {"every": "15m"}`), else
- yours for every project, in `~/.bc-rag/config` (`bc-rag config set every 30m --global`), else
- nothing: no background indexing. There is no built-in interval.

The indexer service writes `~/.bc-rag/{name}/schedule.json` after each run. The next
run is due one interval after the last one finished, so runs never overlap and a
service restart never triggers an extra run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bc_rag.catalog import EVERY_OFF, ProjectEntry, project_store_dir
from bc_rag.duration import parse_duration
from bc_rag.fileio import write_json_atomic

SCHEDULE_FILENAME = "schedule.json"

SOURCE_OVERRIDE = "override"
SOURCE_PROJECT = "project"
SOURCE_GLOBAL = "global"
SOURCE_NONE = "none"

# how `config list` and `status` name each source.
SOURCE_LABELS = {
    SOURCE_OVERRIDE: "you, this project",
    SOURCE_PROJECT: ".bc-rag.json",
    SOURCE_GLOBAL: "you, every project",
    SOURCE_NONE: "not set",
}


def _seconds(value: str) -> float | None:
    return None if value == EVERY_OFF else parse_duration(value)


def resolve_every(
    override: str | None, project_default: str | None, global_default: str | None = None
) -> tuple[float | None, str]:
    """(seconds or None, where it came from). None seconds = no background indexing."""
    if override is not None:
        return _seconds(override), SOURCE_OVERRIDE
    if project_default is not None:
        return parse_duration(project_default), SOURCE_PROJECT
    if global_default is not None:
        return _seconds(global_default), SOURCE_GLOBAL
    return None, SOURCE_NONE


def effective_every(entry: ProjectEntry) -> tuple[str | None, float | None, str]:
    """(value, seconds, source) for one project. Raises ValueError for a bad .bc-rag.json."""
    from bc_rag.usersettings import config_setting

    team = read_schedule(entry.root_path())
    everywhere = config_setting("every")
    seconds, source = resolve_every(entry.every, team, everywhere)
    value = {SOURCE_OVERRIDE: entry.every, SOURCE_PROJECT: team, SOURCE_GLOBAL: everywhere}
    return value.get(source), seconds, source


def read_schedule(root: Path) -> str | None:
    """`schedule.every` from `.bc-rag.json`. Reads only that block; never runs groupsCommand.

    Raises ValueError when the file is missing, is not JSON, or holds a bad duration.
    """
    from bc_rag.config import config_path
    from bc_rag.jsonc import strip_jsonc

    path = config_path(root)
    try:
        raw = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise ValueError(f"{path} is missing") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    block = raw.get("schedule") if isinstance(raw, dict) else None
    if block is None:
        return None
    every = block.get("every") if isinstance(block, dict) else None
    if every is None:
        return None
    if not isinstance(every, str):
        raise ValueError(f"{path}: schedule.every must be a duration like 15m")
    parse_duration(every)
    return every.strip()


def describe_every(entry: ProjectEntry) -> tuple[str, str]:
    """(value, from) for `status` and `project list`, such as ("5m", "you, this project").

    Raises ValueError for a bad .bc-rag.json.
    """
    value, _seconds_, source = effective_every(entry)
    if value is None:
        return "none", "not set; bc-rag config set every 10m"
    return value, SOURCE_LABELS[source]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


@dataclass
class ScheduleState:
    """`~/.bc-rag/{name}/schedule.json`. Written only by the indexer service."""

    project: str
    every: str | None = None
    source: str = SOURCE_NONE
    running: bool = False
    pid: int | None = None
    last_start: str | None = None
    last_finish: str | None = None
    # ok, error, stopped, or busy (another process held the project's index lock).
    result: str | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    next_due: str | None = None

    def due_at(self, every_seconds: float) -> datetime | None:
        """When the next run is due. None = due now (no finished run yet)."""
        finished = _parse_iso(self.last_finish)
        if finished is None:
            return None
        return datetime.fromtimestamp(finished.timestamp() + every_seconds, UTC)

    def is_due(self, every_seconds: float, now: datetime | None = None) -> bool:
        due = self.due_at(every_seconds)
        return due is None or (now or datetime.now(UTC)) >= due


def schedule_path(name: str) -> Path:
    return project_store_dir(name) / SCHEDULE_FILENAME


def load_state(name: str) -> ScheduleState:
    """The saved state, or a fresh one (due now) when there is none or it cannot be read."""
    path = schedule_path(name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return ScheduleState(project=name)
    if not isinstance(raw, dict):
        return ScheduleState(project=name)
    known = {key: raw[key] for key in ScheduleState.__dataclass_fields__ if key in raw}
    known["project"] = name
    return ScheduleState(**known)


def save_state(state: ScheduleState) -> None:
    write_json_atomic(schedule_path(state.project), asdict(state))


def mark_started(state: ScheduleState, *, every: str | None, source: str, pid: int) -> None:
    state.every = every
    state.source = source
    state.running = True
    state.pid = pid
    state.last_start = _now_iso()
    state.error = None


def mark_finished(
    state: ScheduleState,
    *,
    result: str,
    every_seconds: float | None,
    error: str | None = None,
    stats: dict[str, Any] | None = None,
) -> None:
    state.running = False
    state.pid = None
    state.last_finish = _now_iso()
    state.result = result
    state.error = error
    state.stats = stats or {}
    due = state.due_at(every_seconds) if every_seconds else None
    state.next_due = due.strftime("%Y-%m-%dT%H:%M:%SZ") if due else None
