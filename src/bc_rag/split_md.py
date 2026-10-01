"""Render each OpenAPI spec to one spec.md under ~/.bc-rag/{project}/cache/openapi-md/.

`source.sha256` beside spec.md records the spec it came from. A spec whose hash
differs is rendered again, so an edited spec reaches the index.
"""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bc_rag.cache import cache_dir
from bc_rag.defaults import OPENAPI_LANGUAGE, OPENAPI_MD_DIRNAME
from bc_rag.discover import SourceFile
from bc_rag.manifest import file_sha256
from bc_rag.openapi import markdown_from_spec_file

SOURCE_HASH_FILENAME = "source.sha256"


@dataclass
class SplitSpecResult:
    spec_rel: str
    dest: Path
    operations: int
    stubs: int


def openapi_md_root(store_dir: Path) -> Path:
    return cache_dir(store_dir) / OPENAPI_MD_DIRNAME


def spec_output_dir(store_dir: Path, spec_rel: str) -> Path:
    return openapi_md_root(store_dir) / Path(spec_rel).with_suffix("")


def split_rel_for(spec_rel: str, filename: str) -> str:
    # a path prefix, not the folder on disk; see OPENAPI_MD_DIRNAME.
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
    (dest / SOURCE_HASH_FILENAME).write_text(file_sha256(spec_path) + "\n", encoding="utf-8")
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


def _print_error(rel_path: str, error: Exception) -> None:
    print(f"openapi skip {rel_path}: {error}", file=sys.stderr, flush=True)


def is_current(dest: Path, spec_path: Path) -> bool:
    """True when dest holds a spec.md rendered from the spec as it is now."""
    recorded = dest / SOURCE_HASH_FILENAME
    if not (dest / "spec.md").is_file() or not recorded.is_file():
        return False
    try:
        return recorded.read_text(encoding="utf-8").strip() == file_sha256(spec_path)
    except OSError:
        return False


def materialize_openapi_sources(
    root: Path,
    config,
    files: list[SourceFile],
    *,
    on_error: Callable[[str, Exception], None] = _print_error,
) -> list[SourceFile]:
    """Replace each OpenAPI spec with its rendered spec.md, rendering it when stale.

    A spec that cannot be rendered is reported through `on_error` and left out.
    """
    store_dir = config.project_dir(root)
    expanded: list[SourceFile] = []
    for source in files:
        if source.language != OPENAPI_LANGUAGE:
            expanded.append(source)
            continue
        dest = spec_output_dir(store_dir, source.rel_path)
        if not is_current(dest, source.path):
            try:
                split_spec(source.path, source.rel_path, store_dir)
            except Exception as error:
                on_error(source.rel_path, error)
                continue
        markdowns = _endpoint_markdowns(dest)
        chunk = source.chunk
        for markdown in markdowns:
            info = markdown.stat()
            expanded.append(
                SourceFile(
                    path=markdown,
                    rel_path=split_rel_for(source.rel_path, markdown.name),
                    language="markdown",
                    size=info.st_size,
                    facets=source.facets,
                    group=source.group,
                    priority=source.priority,
                    chunk=chunk,
                    space=source.space,
                    mtime_ns=info.st_mtime_ns,
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
    if (
        "### Parameters" in stripped
        or "### Request Body" in stripped
        or "### Responses" in stripped
    ):
        return False
    return stripped.startswith("# ") and len(stripped) < 400
