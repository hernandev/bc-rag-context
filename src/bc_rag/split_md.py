"""Write full-spec OpenAPI markdown under ~/.bc-rag/{project}/openapi-md/."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from bc_rag.defaults import OPENAPI_LANGUAGE, OPENAPI_MD_DIRNAME
from bc_rag.discover import SourceFile, iter_source_files
from bc_rag.openapi import markdown_from_spec_file


@dataclass
class SplitSpecResult:
    spec_rel: str
    dest: Path
    operations: int
    stubs: int


def openapi_md_root(store_dir: Path) -> Path:
    return store_dir / OPENAPI_MD_DIRNAME


def spec_output_dir(store_dir: Path, spec_rel: str) -> Path:
    return openapi_md_root(store_dir) / Path(spec_rel).with_suffix("")


def split_rel_for(spec_rel: str, filename: str) -> str:
    folder = Path(spec_rel).with_suffix("")
    return str(Path(OPENAPI_MD_DIRNAME) / folder / filename)


OPERATION_HEADING = re.compile(
    r"^## (GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\b",
    re.MULTILINE,
)


def split_spec(spec_path: Path, spec_rel: str, store_dir: Path) -> SplitSpecResult:
    dest = spec_output_dir(store_dir, spec_rel)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    text = markdown_from_spec_file(spec_path)
    (dest / "spec.md").write_text(text, encoding="utf-8")
    operations = len(OPERATION_HEADING.findall(text))
    stubs = 1 if _looks_like_stub(text) else 0
    (dest / "INDEX.md").write_text(
        "\n".join(
            [
                f"# {spec_rel}",
                "",
                f"operations: {operations}",
                "",
                "- [spec.md](spec.md)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return SplitSpecResult(
        spec_rel=spec_rel,
        dest=dest,
        operations=operations,
        stubs=stubs,
    )


def split_source_files(files: list[SourceFile], store_dir: Path) -> list[SplitSpecResult]:
    results: list[SplitSpecResult] = []
    for source in files:
        if source.language != OPENAPI_LANGUAGE:
            continue
        results.append(split_spec(source.path, source.rel_path, store_dir))
    return results


def listed_openapi_sources(root: Path, config) -> list[SourceFile]:
    return [source for source in iter_source_files(root, config) if source.language == OPENAPI_LANGUAGE]


def sources_from_paths(root: Path, config, paths: list[Path]) -> list[SourceFile]:
    from bc_rag.discover import source_for_path

    root = root.resolve()
    found: list[SourceFile] = []
    for raw in paths:
        path = raw if raw.is_absolute() else (root / raw)
        path = path.resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        source = source_for_path(rel, path, config)
        if source is None:
            source = SourceFile(
                path=path,
                rel_path=rel,
                language=OPENAPI_LANGUAGE,
                size=path.stat().st_size if path.is_file() else 0,
            )
        found.append(source)
    return found


def materialize_openapi_sources(root: Path, config, files: list[SourceFile]) -> list[SourceFile]:
    """Replace OpenAPI JSON sources with generated per-endpoint markdown files."""
    store_dir = config.project_dir(root)
    expanded: list[SourceFile] = []
    for source in files:
        if source.language != OPENAPI_LANGUAGE:
            expanded.append(source)
            continue
        dest = spec_output_dir(store_dir, source.rel_path)
        markdowns = _endpoint_markdowns(dest)
        if not markdowns:
            try:
                split_spec(source.path, source.rel_path, store_dir)
            except Exception as error:
                print(f"openapi skip {source.rel_path}: {error}", flush=True)
                continue
            markdowns = _endpoint_markdowns(dest)
        chunk = source.chunk
        if chunk is not None:
            chunk = chunk.model_copy(update={"max_chars": chunk.openapi_max_chars})
        for markdown in markdowns:
            expanded.append(
                SourceFile(
                    path=markdown,
                    rel_path=split_rel_for(source.rel_path, markdown.name),
                    language="markdown",
                    size=markdown.stat().st_size,
                    tags=list(source.tags),
                    metadata=dict(source.metadata),
                    group=source.group,
                    config_group=source.config_group,
                    priority=source.priority,
                    chunk=chunk,
                    embed=source.embed,
                )
            )
    return expanded


def _endpoint_markdowns(dest: Path) -> list[Path]:
    if not dest.is_dir():
        return []
    spec = dest / "spec.md"
    if spec.is_file():
        return [spec]
    return sorted(path for path in dest.glob("*.md") if path.name != "INDEX.md")


def _looks_like_stub(text: str) -> bool:
    stripped = text.strip()
    if "Base URL" in stripped:
        return False
    if "### Parameters" in stripped or "### Request Body" in stripped or "### Responses" in stripped:
        return False
    return stripped.startswith("# ") and len(stripped) < 400
