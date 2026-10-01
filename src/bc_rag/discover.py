"""Pick files from named source groups.

Include and exclude are expanded on the filesystem. Git is not consulted
for the candidate list.

Ingest order is highest `priority` first, then the group name. List order in
JSON does not matter for ingest. Within one space, the first group to match a
file claims it, and the file carries only that group's facets.
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from wcmatch.glob import BRACE, DOTGLOB, GLOBSTAR, NODIR, glob
from wcmatch.glob import compile as compile_globs

from bc_rag import catalog
from bc_rag.config import ChunkConfig, RagConfig, SourceGroup
from bc_rag.defaults import HARD_EXCLUDE_DIR_NAMES, LANGUAGE_BY_SUFFIX, OPENAPI_LANGUAGE
from bc_rag.facets import Facets

_GLOB_FLAGS = GLOBSTAR | BRACE | DOTGLOB
_EXPAND_FLAGS = GLOBSTAR | BRACE | DOTGLOB | NODIR


class SourceFile:
    __slots__ = (
        "path",
        "rel_path",
        "language",
        "size",
        "facets",
        "group",
        "priority",
        "chunk",
        "space",
        "mtime_ns",
    )

    def __init__(
        self,
        path: Path,
        rel_path: str,
        language: str,
        size: int,
        facets: Facets | None = None,
        group: str | None = None,
        priority: int = 0,
        chunk: ChunkConfig | None = None,
        space: str | None = None,
        mtime_ns: int = 0,
    ) -> None:
        self.path = path
        self.rel_path = rel_path
        self.language = language
        self.size = size
        # the modification time read with the size. 0 when unknown: index then hashes.
        self.mtime_ns = mtime_ns
        # the claiming group's facets. bc-rag adds `group` and its own keys per chunk.
        self.facets: Facets = dict(facets or {})
        # the name of the config group that claimed the file.
        self.group = group
        self.priority = priority
        self.chunk = chunk
        self.space = space


@dataclass(slots=True)
class GroupResolveStats:
    name: str
    enabled: bool
    priority: int
    glob_ms: float
    file_count: int
    include_count: int
    skipped: bool


def _require_groups(config: RagConfig) -> None:
    if config.is_partial():
        raise ValueError("this config was loaded without groupsCommand and cannot match files")


def debug_resolve_groups(root: Path, config: RagConfig) -> list[GroupResolveStats]:
    """Time include expansion per group. Disabled groups are not globbed."""
    _require_groups(config)
    root = root.resolve()
    stats: list[GroupResolveStats] = []
    configured = list(config.groups) if config.groups else list(config.resolved_groups())
    for group in configured:
        if not group.enabled:
            stats.append(
                GroupResolveStats(
                    name=group.name,
                    enabled=False,
                    priority=group.priority,
                    glob_ms=0.0,
                    file_count=0,
                    include_count=len(group.include),
                    skipped=True,
                )
            )
            continue
        started = time.perf_counter()
        paths = _relative_paths_for_group(root, config, group)
        elapsed_ms = (time.perf_counter() - started) * 1000
        stats.append(
            GroupResolveStats(
                name=group.name,
                enabled=True,
                priority=group.priority,
                glob_ms=elapsed_ms,
                file_count=len(paths),
                include_count=len(group.include),
                skipped=False,
            )
        )
    stats.sort(key=lambda row: (-row.glob_ms, -row.priority, row.name))
    return stats


def language_for(path: Path) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower())


def classify_path(
    rel: str,
    path: Path,
    config: RagConfig,
    *,
    peek: bool = True,
) -> str | None:
    source = source_for_path(rel, path, config)
    del peek
    if source is None:
        return None
    return source.language


def _chunk_for(config: RagConfig, primary: SourceGroup) -> ChunkConfig:
    spec = config.space_named(primary.space)
    if spec.chunk is None:
        raise ValueError(f"space {primary.space!r} requires chunk")
    return spec.chunk


def source_for_path(
    rel: str,
    path: Path,
    config: RagConfig,
    *,
    matched_group: SourceGroup | None = None,
    space: str | None = None,
    walked: bool = False,
) -> SourceFile | None:
    """The file as index sees it in one space.

    Without `matched_group`, the highest-priority enabled group that matches claims it,
    the same group a full pass would pick. `space` limits the candidates to that space.
    `walked` means the path came from the group's own glob, which already dropped
    hard-excluded folders and the user store.
    """
    _require_groups(config)
    if not walked and (_is_hard_excluded(rel) or _is_under_user_store(path)):
        return None
    if matched_group is not None:
        if not matched_group.enabled:
            return None
        if _matches_any(rel, config.exclude) or _matches_any(rel, matched_group.exclude):
            return None
        matches = [matched_group]
    else:
        matches = [
            group
            for group in config.resolved_groups()
            if (space is None or group.space == space) and _group_accepts(rel, path, group, config)
        ]
        if not matches:
            return None
        matches.sort(key=lambda group: (-group.priority, group.name))
    primary = matches[0]
    language = OPENAPI_LANGUAGE if primary.kind == "openapi" else language_for(path)
    if language is None:
        return None
    size = 0
    mtime_ns = 0
    try:
        info = path.stat()
    except OSError:
        info = None
    if info is not None and stat.S_ISREG(info.st_mode):
        size = info.st_size
        mtime_ns = info.st_mtime_ns
    return SourceFile(
        path=path,
        rel_path=rel,
        language=language,
        size=size,
        facets=primary.facets,
        group=primary.name,
        priority=primary.priority,
        chunk=_chunk_for(config, primary),
        space=primary.space,
        mtime_ns=mtime_ns,
    )


def iter_source_groups(
    root: Path,
    config: RagConfig,
    include_disabled: set[str] | None = None,
) -> Iterator[tuple[SourceGroup, list[SourceFile]]]:
    """Yield one group at a time, with its files already resolved.

    Disabled groups are skipped unless their name is in `include_disabled`.
    """
    _require_groups(config)
    root = root.resolve()
    seen: set[tuple[str, str]] = set()
    groups = sorted(
        config.groups or config.resolved_groups(),
        key=lambda group: (-group.priority, group.name),
    )
    named = include_disabled or set()
    for group in groups:
        if not group.enabled and group.name not in named:
            continue
        space_name = group.space or ""
        relative_paths = _relative_paths_for_group(root, config, group)
        relative_paths.sort()
        files: list[SourceFile] = []
        for relative_path in relative_paths:
            key = (space_name, relative_path)
            if key in seen:
                continue
            path = root / relative_path
            source = source_for_path(
                relative_path, path, config, matched_group=group, walked=True
            )
            # a modification time means source_for_path already saw a regular file.
            if source is None or (not source.mtime_ns and not path.is_file()):
                continue
            seen.add(key)
            files.append(source)
        if files:
            yield group, files


def iter_source_files(root: Path, config: RagConfig) -> Iterator[SourceFile]:
    for _group, files in iter_source_groups(root, config):
        yield from files


def _user_store() -> str:
    return _resolved_dir(str(catalog.user_dir()))


def _is_under_user_store(path: Path, store: str | None = None) -> bool:
    # resolving every file walks every folder above it. Folders resolve once, from a
    # cache; only a file that is itself a symlink is resolved in full.
    store = store or _user_store()
    if path.is_symlink():
        try:
            target = str(path.resolve())
        except OSError:
            return False
    else:
        target = os.path.join(_resolved_dir(str(path.parent)), path.name)
    return target == store or target.startswith(store + os.sep)


@lru_cache(maxsize=16384)
def _resolved_dir(directory: str) -> str:
    try:
        return str(Path(directory).resolve())
    except OSError:
        return directory


def _relative_paths_for_group(root: Path, config: RagConfig, group: SourceGroup) -> list[str]:
    exclude_patterns = _disk_exclude_patterns(config, group)
    store = _user_store()
    found: list[str] = []
    seen: set[str] = set()
    for pattern in group.include:
        matches = glob(
            pattern,
            root_dir=str(root),
            flags=_EXPAND_FLAGS,
            limit=0,
            exclude=exclude_patterns or None,
        )
        for match in matches:
            relative_path = str(match).replace("\\", "/")
            if relative_path in seen:
                continue
            if _is_hard_excluded(relative_path):
                continue
            if _is_under_user_store(root / relative_path, store):
                continue
            seen.add(relative_path)
            found.append(relative_path)
    return found


def _disk_exclude_patterns(config: RagConfig, group: SourceGroup) -> list[str]:
    # glob checks exclude patterns against each match, not while it walks folders, so
    # the hard-excluded folder names are left to the cheaper `_is_hard_excluded`.
    return [*config.exclude, *group.exclude]


def _group_accepts(rel: str, path: Path, group: SourceGroup, config: RagConfig) -> bool:
    if not group.include:
        return False
    if _matches_any(rel, config.exclude) or _matches_any(rel, group.exclude):
        return False
    if not _matches_any(rel, group.include):
        return False
    if group.kind == "openapi":
        return True
    return language_for(path) is not None


def _is_hard_excluded(rel: str) -> bool:
    return any(part in HARD_EXCLUDE_DIR_NAMES for part in rel.split("/") if part)


def _matches_any(rel: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return _matcher(tuple(patterns)).match(rel)


@lru_cache(maxsize=1024)
def _matcher(patterns: tuple[str, ...]):
    # compiling a pattern costs more than matching it, and the same lists repeat per file.
    return compile_globs(list(patterns), flags=_GLOB_FLAGS)
