"""User-level catalog of indexed projects.

One MCP server reads this list. Two repos do not need two MCP entries.

Generated data lives under `~/.bc-rag/{project-name}/` (Qdrant, manifest, index log).
`.bc-rag.json` stays in the project folder. `{project}/.bc-rag/` is legacy and is
moved once if the new store is empty.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bc_rag.defaults import CATALOG_DIRNAME, CATALOG_FILENAME, STORE_DIRNAME


@dataclass
class ProjectEntry:
    name: str
    root: str
    updated_at: str

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
    root = root.resolve()
    entry = next((item for item in load_catalog() if Path(item.root) == root), None)
    name = entry.name if entry is not None else (root.name or "project")
    dest = project_store_dir(name)
    migrate_legacy_store(root, dest)
    return dest


def migrate_legacy_store(root: Path, dest: Path) -> bool:
    """Move `{root}/.bc-rag` into `dest` when dest has no manifest or Qdrant yet."""
    legacy = root / STORE_DIRNAME
    if not legacy.is_dir():
        return False
    dest_has = (dest / "manifest.json").is_file() or (dest / "qdrant").exists()
    if dest_has:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    moved = False
    for item in list(legacy.iterdir()):
        target = dest / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
        moved = True
    try:
        legacy.rmdir()
    except OSError:
        pass
    return moved


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
        entries.append(
            ProjectEntry(
                name=name,
                root=root,
                updated_at=str(row.get("updated_at") or ""),
            )
        )
    return entries


def save_catalog(entries: list[ProjectEntry]) -> None:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "projects": [
            {"name": e.name, "root": e.root, "updated_at": e.updated_at}
            for e in sorted(entries, key=lambda item: item.name)
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def register_project(root: Path, name: str | None = None) -> ProjectEntry:
    root = root.resolve()
    entries = load_catalog()
    existing = next((e for e in entries if Path(e.root) == root), None)
    chosen = name or (existing.name if existing else _unique_name(root, entries))
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = ProjectEntry(name=chosen, root=str(root), updated_at=now)
    entries = [e for e in entries if Path(e.root) != root]
    # Rename collision: another root already uses this name.
    if any(e.name == entry.name and Path(e.root) != root for e in entries):
        entry.name = _unique_name(root, entries)
    entries.append(entry)
    save_catalog(entries)
    return entry


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


def find_project(name_or_root: str, entries: list[ProjectEntry] | None = None) -> ProjectEntry | None:
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


def _unique_name(root: Path, entries: list[ProjectEntry]) -> str:
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
