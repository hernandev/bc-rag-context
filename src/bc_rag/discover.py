"""Pick files from named source groups.

Include and exclude are expanded on the filesystem. Git is not consulted
for the candidate list. A group name is the tag stored on the chunk.

Ingest order is highest `priority` first, then the group name. List order in
JSON does not matter for ingest.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from wcmatch.glob import BRACE, DOTGLOB, GLOBSTAR, NODIR, glob, globmatch

from bc_rag import catalog
from bc_rag.config import ChunkConfig, EmbedConfig, RagConfig, SourceGroup
from bc_rag.defaults import HARD_EXCLUDE_DIR_NAMES, LANGUAGE_BY_SUFFIX, OPENAPI_LANGUAGE
from bc_rag.facets import parse_tag_clause
from bc_rag.nx_tags import NxProject, NxProjectIndex

_GLOB_FLAGS = GLOBSTAR | BRACE | DOTGLOB
_EXPAND_FLAGS = GLOBSTAR | BRACE | DOTGLOB | NODIR


class SourceFile:
    __slots__ = (
        "path",
        "rel_path",
        "language",
        "size",
        "tags",
        "metadata",
        "group",
        "config_group",
        "priority",
        "chunk",
        "embed",
    )

    def __init__(
        self,
        path: Path,
        rel_path: str,
        language: str,
        size: int,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        group: str | None = None,
        config_group: str | None = None,
        priority: int = 0,
        chunk: ChunkConfig | None = None,
        embed: EmbedConfig | None = None,
    ) -> None:
        self.path = path
        self.rel_path = rel_path
        self.language = language
        self.size = size
        self.tags = tags or []
        self.metadata = dict(metadata or {})
        self.group = group
        self.config_group = config_group
        self.priority = priority
        self.chunk = chunk
        self.embed = embed


@dataclass(slots=True)
class GroupResolveStats:
    name: str
    enabled: bool
    priority: int
    glob_ms: float
    file_count: int
    include_count: int
    skipped: bool


def debug_resolve_groups(root: Path, config: RagConfig) -> list[GroupResolveStats]:
    """Time include expansion per group. Disabled groups are not globbed."""
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


def source_for_path(
    rel: str,
    path: Path,
    config: RagConfig,
    *,
    nx_index: NxProjectIndex | None = None,
    matched_group: SourceGroup | None = None,
) -> SourceFile | None:
    if _is_hard_excluded(rel):
        return None
    if _is_under_user_store(path):
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
            if _group_accepts(rel, path, group, config)
        ]
        if not matches:
            return None
        matches.sort(key=lambda group: (-group.priority, group.name))
    primary = matches[0]
    language = OPENAPI_LANGUAGE if primary.kind == "openapi" else language_for(path)
    if language is None:
        return None
    size = 0
    if path.is_file():
        try:
            size = path.stat().st_size
        except OSError:
            return None
    nx_project = nx_index.project_for(path) if nx_index is not None else None
    group_name = nx_project.canonical if nx_project is not None else primary.name
    return SourceFile(
        path=path,
        rel_path=rel,
        language=language,
        size=size,
        tags=_merge_tags(matches, nx_project),
        metadata=_merge_metadata(matches),
        group=group_name,
        config_group=primary.name,
        priority=primary.priority,
        chunk=primary.resolved_chunk(config.chunk),
        embed=primary.resolved_embed(config.embed) if primary.embed is not None else None,
    )


def iter_source_groups(
    root: Path, config: RagConfig
) -> Iterator[tuple[SourceGroup, list[SourceFile]]]:
    """Yield one enabled group at a time, with its files already resolved."""
    root = root.resolve()
    nx_index = NxProjectIndex(root)
    seen: set[str] = set()
    groups = sorted(config.resolved_groups(), key=lambda group: (-group.priority, group.name))
    for group in groups:
        relative_paths = _relative_paths_for_group(root, config, group)
        relative_paths.sort()
        files: list[SourceFile] = []
        for relative_path in relative_paths:
            if relative_path in seen:
                continue
            path = root / relative_path
            source = source_for_path(
                relative_path,
                path,
                config,
                nx_index=nx_index,
                matched_group=group,
            )
            if source is None or not path.is_file():
                continue
            seen.add(relative_path)
            files.append(source)
        if files:
            yield group, files


def iter_source_files(root: Path, config: RagConfig) -> Iterator[SourceFile]:
    for _group, files in iter_source_groups(root, config):
        yield from files


def _merge_tags(groups: list[SourceGroup], nx_project: NxProject | None) -> list[str]:
    del nx_project
    tags: list[str] = []
    for group in groups:
        for item in group.tags:
            if item and item not in tags:
                tags.append(item)
    return tags


def _is_under_user_store(path: Path) -> bool:
    try:
        path.resolve().relative_to(catalog.user_dir().resolve())
    except (ValueError, OSError):
        return False
    return True


def _merge_metadata(groups: list[SourceGroup]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for group in groups:
        for key, value in group.metadata.items():
            if key and value:
                merged[key] = value
        for item in group.tags:
            clause = parse_tag_clause(item)
            if clause is not None:
                merged[clause[0]] = clause[1]
    return merged


def _relative_paths_for_group(root: Path, config: RagConfig, group: SourceGroup) -> list[str]:
    exclude_patterns = _disk_exclude_patterns(config, group)
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
            if _is_under_user_store(root / relative_path):
                continue
            seen.add(relative_path)
            found.append(relative_path)
    return found


def _disk_exclude_patterns(config: RagConfig, group: SourceGroup) -> list[str]:
    patterns: list[str] = []
    for name in HARD_EXCLUDE_DIR_NAMES:
        patterns.append(f"**/{name}/**")
        patterns.append(f"**/{name}")
        patterns.append(name)
    patterns.extend(config.exclude)
    patterns.extend(group.exclude)
    return patterns


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
    return any(globmatch(rel, pattern, flags=_GLOB_FLAGS) for pattern in patterns)
