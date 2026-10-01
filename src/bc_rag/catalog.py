"""The registry of projects in `~/.bc-rag/catalog.json`.

A folder is a bc-rag project only after `bc-rag project add` registers it here.
Every other command finds its project in this list, by name or by walking up
from the current folder. Nothing else writes a new row.

Generated data lives under `~/.bc-rag/{project-name}/` (manifests, corpus, index
log). Vectors live in Qdrant. `.bc-rag.json` stays in the project folder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bc_rag.defaults import CATALOG_DIRNAME, CATALOG_FILENAME
from bc_rag.fileio import write_json_atomic

# `config set every off` stores this to turn background indexing off for one project.
EVERY_OFF = "off"


class CatalogError(ValueError):
    """A registry change that cannot be made, such as registering a root twice."""


class ProjectNotFound(LookupError):
    """No registered project matches. `path` or `project` says what was looked up."""

    def __init__(self, *, path: Path | None = None, project: str | None = None) -> None:
        self.path = path
        self.project = project
        if project is not None:
            message = f"unknown project: {project}"
        else:
            message = f"not a bc-rag project: {path}"
        super().__init__(message)


@dataclass
class ProjectEntry:
    name: str
    root: str
    updated_at: str
    # Your background indexing interval for this project, or "off". None = follow .bc-rag.json.
    every: str | None = None

    def root_path(self) -> Path:
        return Path(self.root)


def user_dir() -> Path:
    override = os.environ.get("BC_RAG_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / CATALOG_DIRNAME


def catalog_path() -> Path:
    return user_dir() / CATALOG_FILENAME


def _safe_project_name(name: str) -> str:
    cleaned = name.strip().replace("/", "-").replace("\\", "-")
    if cleaned in {CATALOG_FILENAME, "", ".", ".."}:
        return "project"
    return cleaned


def project_store_dir(name: str) -> Path:
    return user_dir() / _safe_project_name(name)


def store_dir_for_root(root: Path) -> Path:
    """`~/.bc-rag/{name}/` for the project registered at `root` (its folder name if none)."""
    root = root.resolve()
    entry = next((item for item in load_catalog() if Path(item.root) == root), None)
    return project_store_dir(entry.name if entry is not None else (root.name or "project"))


def load_catalog() -> list[ProjectEntry]:
    path = catalog_path()
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("projects") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return []
    entries: list[ProjectEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        root = str(row.get("root") or "").strip()
        name = str(row.get("name") or "").strip()
        if not root or not name:
            continue
        every = str(row.get("every") or "").strip() or None
        entries.append(
            ProjectEntry(
                name=name,
                root=root,
                updated_at=str(row.get("updated_at") or ""),
                every=every,
            )
        )
    return entries


def save_catalog(entries: list[ProjectEntry]) -> None:
    rows = []
    for entry in sorted(entries, key=lambda item: item.name):
        row = {"name": entry.name, "root": entry.root, "updated_at": entry.updated_at}
        if entry.every is not None:
            row["every"] = entry.every
        rows.append(row)
    write_json_atomic(catalog_path(), {"projects": rows})


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def register_project(root: Path, name: str | None = None) -> ProjectEntry:
    """Add a new row. Refuses a root that is registered and a name another root uses."""
    root = root.resolve()
    entries = load_catalog()
    existing = next((e for e in entries if Path(e.root) == root), None)
    if existing is not None:
        raise CatalogError(f"{root} is already registered as {existing.name}")
    if name is not None:
        name = name.strip()
        if not name:
            raise CatalogError("a project name cannot be empty")
        taken = next((e for e in entries if e.name == name), None)
        if taken is not None:
            raise CatalogError(f"name {name!r} is used by {taken.root}")
    entry = ProjectEntry(name=name or unique_name(root, entries), root=str(root), updated_at=_now())
    entries.append(entry)
    save_catalog(entries)
    return entry


def set_project_every(name: str, value: str | None) -> ProjectEntry:
    """Store your interval for one project: a duration, "off", or None to drop it."""
    if value is not None:
        value = value.strip().lower()
        if value != EVERY_OFF:
            from bc_rag.duration import parse_duration

            try:
                parse_duration(value)
            except ValueError as error:
                raise CatalogError(f"{error}. Use a duration like 10m, or off") from None
    entries = load_catalog()
    entry = next((e for e in entries if e.name == name), None)
    if entry is None:
        raise ProjectNotFound(project=name)
    entry.every = value
    entry.updated_at = _now()
    save_catalog(entries)
    return entry


def resolve_project(project: str | None = None, root: Path | None = None) -> ProjectEntry:
    """The registered project named `project`, or the one whose root holds `root` (or cwd).

    Walks up from the folder through its parents, so any subfolder of a project works.
    Raises ProjectNotFound when nothing matches.
    """
    if project is not None and root is not None:
        raise ValueError("use one of --project or --root")
    entries = load_catalog()
    if project is not None:
        found = find_project(project, entries)
        if found is None:
            raise ProjectNotFound(project=project)
        return found
    start = (root or Path.cwd()).expanduser().resolve()
    by_root = {Path(entry.root): entry for entry in entries}
    for folder in (start, *start.parents):
        found = by_root.get(folder)
        if found is not None:
            return found
    raise ProjectNotFound(path=start)


def try_resolve_project(
    project: str | None = None, root: Path | None = None
) -> ProjectEntry | None:
    """Like resolve_project, but None outside a project. A bad --project still raises."""
    try:
        return resolve_project(project, root)
    except ProjectNotFound:
        if project is not None:
            raise
        return None


def forget_project(name_or_root: str) -> ProjectEntry | None:
    entries = load_catalog()
    resolved = Path(name_or_root).expanduser()
    kept: list[ProjectEntry] = []
    removed: ProjectEntry | None = None
    for entry in entries:
        if entry.name == name_or_root or Path(entry.root) == resolved.resolve():
            removed = entry
            continue
        kept.append(entry)
    if removed is not None:
        save_catalog(kept)
    return removed


def find_project(
    name_or_root: str, entries: list[ProjectEntry] | None = None
) -> ProjectEntry | None:
    entries = entries if entries is not None else load_catalog()
    resolved: Path | None
    try:
        resolved = Path(name_or_root).expanduser().resolve()
    except OSError:
        resolved = None
    for entry in entries:
        if entry.name == name_or_root:
            return entry
        if resolved is not None and Path(entry.root) == resolved:
            return entry
    return None


def living_projects(entries: list[ProjectEntry] | None = None) -> list[ProjectEntry]:
    """Catalog rows whose root still exists on disk."""
    entries = entries if entries is not None else load_catalog()
    return [e for e in entries if Path(e.root).is_dir()]


def unique_name(root: Path, entries: list[ProjectEntry]) -> str:
    """The folder name, or a parent-qualified one when another project already has it."""
    used = {e.name for e in entries}
    base = root.name or "project"
    if base not in used:
        return base
    parent = root.parent.name
    candidate = f"{parent}-{base}" if parent else f"{base}-2"
    if candidate not in used:
        return candidate
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"
