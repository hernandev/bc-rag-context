"""Read the Nx `project.json` that owns a file.

The stored group is the package path with slashes turned into hyphens.
Tags are that slug, the Nx package name, and every `tags` entry in project.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NxProject:
    directory: Path
    canonical: str
    package_name: str | None
    tags: list[str]


class NxProjectIndex:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._directory_project: dict[Path, NxProject | None] = {}

    def project_for(self, file_path: Path) -> NxProject | None:
        start = file_path.parent.resolve()
        if start in self._directory_project:
            return self._directory_project[start]
        chain: list[Path] = []
        current = start
        found: NxProject | None = None
        while True:
            chain.append(current)
            project_file = current / "project.json"
            if project_file.is_file():
                found = _read_project(project_file, self.root)
                break
            if current == self.root or current.parent == current:
                break
            current = current.parent
        for directory in chain:
            self._directory_project[directory] = found
        return found

    def tags_for(self, file_path: Path) -> list[str]:
        project = self.project_for(file_path)
        if project is None:
            return []
        return list(project.tags)


DROPPED_NX_TAGS = frozenset(
    {
        "npm:public",
        "watch:true",
        "watch:yes",
        "type:lib",
        "target:customer",
    }
)

# Same map as bc-project-bundler scripts/build-workspace-corpus.ts AREA_BY_NX_GROUP.
AREA_BY_NX_GROUP = {
    "admin": "admin",
    "engine": "engine",
    "client": "client",
    "vendor": "vendor",
    "common": "common",
    "web": "portal",
}


def canonical_from_relative(relative_path: str) -> str:
    return relative_path.strip("/").replace("/", "-")


def _read_project(project_file: Path, root: Path) -> NxProject:
    relative = project_file.parent.resolve().relative_to(root).as_posix()
    canonical = canonical_from_relative(relative)
    try:
        data = json.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    package_name = data.get("name")
    if not isinstance(package_name, str) or not package_name.strip():
        package_name = None
    tags: list[str] = [canonical]
    if package_name:
        if package_name not in tags:
            tags.append(package_name)
        package_tag = f"package:{package_name}"
        if package_tag not in tags:
            tags.append(package_tag)
    raw = data.get("tags")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str) or not item or item in tags:
                continue
            if item in DROPPED_NX_TAGS:
                continue
            # vendor:<name> is spec-only in the bundler sidecars. Nx puts it on
            # vendor packages; do not copy it onto source-code points.
            if item.startswith("vendor:"):
                continue
            tags.append(item)
            if item.startswith("group:") and len(item) > 6:
                short = item.split(":", 1)[1]
                if short and short not in tags:
                    tags.append(short)
                area = AREA_BY_NX_GROUP.get(short, "_none")
                area_tag = f"area:{area}"
                if area_tag not in tags:
                    tags.append(area_tag)
    return NxProject(
        directory=project_file.parent.resolve(),
        canonical=canonical,
        package_name=package_name,
        tags=tags,
    )
