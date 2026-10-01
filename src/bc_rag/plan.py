"""What index would do with each file, without writing vectors."""

from __future__ import annotations

from pathlib import Path

from bc_rag.chunking import Chunk
from bc_rag.config import RagConfig, load_config
from bc_rag.defaults import CODE_LANGUAGES, JSON_LANGUAGE, MARKDOWN_LANGUAGES, OPENAPI_LANGUAGE
from bc_rag.discover import SourceFile, iter_source_groups, source_for_path
from bc_rag.indexer import chunks_for
from bc_rag.split_md import materialize_openapi_sources
from bc_rag.voyage_api import is_contextual_model


def iter_index_files(
    root: Path,
    config: RagConfig,
    include_disabled: set[str] | None = None,
):
    """Files index would chunk, after OpenAPI specs become spec.md.

    `include_disabled` names groups that stay in the listing even when enabled is false.
    """
    for _name, files in iter_source_groups(root, config, include_disabled=include_disabled):
        yield from materialize_openapi_sources(root, config, files)


def listed_group(source: SourceFile) -> str:
    """Group name printed by `bc-rag files`."""
    return source.config_group or source.group or ""


def listed_space(config: RagConfig, source: SourceFile) -> str:
    """Space name printed by `bc-rag files`."""
    return source.space or config.default_space


def file_selected(
    config: RagConfig,
    source: SourceFile,
    groups: set[str] | None,
    spaces: set[str] | None,
) -> bool:
    """A file must match every filter that was passed. Each filter is any-of."""
    if groups and listed_group(source) not in groups:
        return False
    if spaces and listed_space(config, source) not in spaces:
        return False
    return True


def unknown_file_filters(
    config: RagConfig,
    groups: list[str] | None,
    spaces: list[str] | None,
) -> str | None:
    """The first group or space name that is not in the loaded config."""
    known_groups = {group.name for group in config.groups}
    for name in groups or []:
        if name not in known_groups:
            return f"unknown group: {name}"
    known_spaces = set(config.spaces)
    for name in spaces or []:
        if name not in known_spaces:
            return f"unknown space: {name}"
    return None


def file_strategy(config: RagConfig, source: SourceFile) -> dict[str, str]:
    chunk = source.chunk
    if source.language in MARKDOWN_LANGUAGES:
        chunker = "markdown"
    elif source.language in CODE_LANGUAGES:
        chunker = "code"
    elif source.language == JSON_LANGUAGE:
        chunker = "json"
    elif source.language == OPENAPI_LANGUAGE:
        chunker = "openapi"
    else:
        chunker = "lines"
    model = _dense_model(config, source)
    if is_contextual_model(model):
        embed = "contextual"
    elif model.startswith("voyage-"):
        embed = "voyage"
    elif config.space_named(listed_space(config, source)).dense_provider() == "jina":
        embed = "jina"
    else:
        embed = "local"
    return {
        "path": source.rel_path,
        "space": listed_space(config, source),
        "group": listed_group(source),
        "chunker": chunker,
        "max_chars": str(chunk.max_chars),
        "embed": embed,
        "model": model,
    }


def chunks_for_user_path(root: Path, config: RagConfig, raw: str) -> tuple[SourceFile, list[Chunk]]:
    """Resolve one path the way index does, then return its chunks."""
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        rel = path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside the project: {raw}") from error
    source = source_for_path(rel, path, config)
    if source is None or not path.is_file():
        raise ValueError(f"config does not include {rel}")
    materialized = materialize_openapi_sources(root, config, [source])
    if not materialized:
        raise ValueError(f"config does not include {rel}")
    chosen = materialized[0]
    text = chosen.path.read_text(encoding="utf-8", errors="replace")
    return chosen, chunks_for(root, chosen, text, config)


def _dense_model(config: RagConfig, source: SourceFile) -> str:
    return config.space_named(listed_space(config, source)).dense_id()


def load_project(root: Path | None) -> tuple[Path, RagConfig]:
    from bc_rag.runtime import resolve_root

    project = resolve_root(root)
    config, _path = load_config(project)
    return project, config
