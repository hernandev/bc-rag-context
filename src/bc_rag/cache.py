"""`~/.bc-rag/{name}/cache/`: what indexing derives from the repo.

Two folders, both rebuilt by the next index run when they are missing, without an
embedding API call:

- `corpus/`: one document plus one Bedrock sidecar per indexed file.
- `openapi-md/`: each OpenAPI spec rendered to markdown.

Before this folder existed, both lived directly in `~/.bc-rag/{name}/`.
`move_legacy_cache` renames them into place once, so nothing is rebuilt.
"""

from __future__ import annotations

from pathlib import Path

from bc_rag.defaults import CACHE_DIRNAME, CORPUS_DIRNAME, OPENAPI_MD_DIRNAME


def cache_dir(project_dir: Path) -> Path:
    return project_dir / CACHE_DIRNAME


def move_legacy_cache(project_dir: Path) -> list[Path]:
    """Rename `corpus/` and `openapi-md/` from the project folder into `cache/`.

    A rename keeps every file and its modification time, so the next run still skips
    unchanged files without reading them. Returns the folders it moved.
    """
    moved: list[Path] = []
    for name in (CORPUS_DIRNAME, OPENAPI_MD_DIRNAME):
        old = project_dir / name
        new = cache_dir(project_dir) / name
        if not old.is_dir() or new.exists():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        moved.append(new)
    return moved
