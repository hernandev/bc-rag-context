"""Load `.bc-rag.json` or fall back to defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bc_rag.defaults import (
    CONFIG_FILENAME,
    DEFAULT_CHUNK_MAX_CHARS,
    QDRANT_HTTP_URL,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_DENSE_MODEL,
    DEFAULT_INCLUDE,
    DEFAULT_LIMIT,
    DEFAULT_OPENAPI_INCLUDE,
    DEFAULT_OPENAPI_MAX_CHARS,
    DEFAULT_PREFETCH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_SPARSE_MODEL,
)


class EmbedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense: str = DEFAULT_DENSE_MODEL
    sparse: str = DEFAULT_SPARSE_MODEL
    rerank: str | None = DEFAULT_RERANK_MODEL
    jina_api: bool = False
    jina_dense: str = "jina-embeddings-v5-text-small"
    jina_rerank: str | None = "jina-reranker-v3.5"

    def active_dense(self) -> str:
        return self.jina_dense if self.jina_api else self.dense

    def active_rerank(self) -> str | None:
        return self.jina_rerank if self.jina_api else self.rerank


class RetrieveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefetch: int = Field(default=DEFAULT_PREFETCH, ge=1, le=400)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=50)
    rerank: bool = True


class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=DEFAULT_CHUNK_MAX_CHARS, ge=200, le=80_000)
    openapi_max_chars: int = Field(default=DEFAULT_OPENAPI_MAX_CHARS, ge=500, le=80_000)
    min_chars: int = Field(default=DEFAULT_CHUNK_MIN_CHARS, ge=0, le=2000)


class ChunkOverride(BaseModel):
    """Partial chunk settings. Omitted keys inherit from the top-level chunk block."""

    model_config = ConfigDict(extra="forbid")

    max_chars: int | None = Field(default=None, ge=200, le=80_000)
    openapi_max_chars: int | None = Field(default=None, ge=500, le=80_000)
    min_chars: int | None = Field(default=None, ge=0, le=2000)


class EmbedOverride(BaseModel):
    """Partial embed settings. A different dense model is rejected at index time."""

    model_config = ConfigDict(extra="forbid")

    dense: str | None = None
    sparse: str | None = None
    rerank: str | None = None


class OpenApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    include: list[str] = Field(default_factory=lambda: list(DEFAULT_OPENAPI_INCLUDE))


class SourceGroup(BaseModel):
    """One named include/exclude set. `name` is stored as a tag on every chunk."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    priority: int = 0
    follow_gitignore: bool | None = None
    kind: Literal["auto", "openapi"] = "auto"
    enabled: bool = True
    chunk: ChunkOverride | None = None
    embed: EmbedOverride | None = None

    def resolved_chunk(self, base: ChunkConfig) -> ChunkConfig:
        override = self.chunk
        if override is None:
            return base
        return ChunkConfig(
            max_chars=override.max_chars if override.max_chars is not None else base.max_chars,
            openapi_max_chars=(
                override.openapi_max_chars
                if override.openapi_max_chars is not None
                else base.openapi_max_chars
            ),
            min_chars=override.min_chars if override.min_chars is not None else base.min_chars,
        )

    def resolved_embed(self, base: EmbedConfig) -> EmbedConfig:
        override = self.embed
        if override is None:
            return base
        return EmbedConfig(
            dense=override.dense if override.dense is not None else base.dense,
            sparse=override.sparse if override.sparse is not None else base.sparse,
            rerank=base.rerank if override.rerank is None else override.rerank,
            jina_api=base.jina_api,
            jina_dense=base.jina_dense,
            jina_rerank=base.jina_rerank,
        )


class RagConfig(BaseModel):
    """Project config. Missing file means these defaults."""

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude: list[str] = Field(default_factory=list)
    groups: list[SourceGroup] = Field(default_factory=list)
    follow_gitignore: bool = False
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    openapi: OpenApiConfig = Field(default_factory=OpenApiConfig)
    qdrant_url: str | None = None

    def resolved_groups(self) -> list[SourceGroup]:
        if self.groups:
            return [group for group in self.groups if group.enabled]
        groups = [
            SourceGroup(
                name="default",
                include=list(self.include),
                exclude=list(self.exclude),
                priority=0,
            )
        ]
        if self.openapi.enabled and self.openapi.include:
            groups.append(
                SourceGroup(
                    name="openapi",
                    include=list(self.openapi.include),
                    kind="openapi",
                    priority=100,
                )
            )
        return groups

    def project_dir(self, root: Path) -> Path:
        from bc_rag.catalog import store_dir_for_root

        return store_dir_for_root(root)

    def backend_id(self) -> str:
        """Folder name for one vector space. Mode + models that change embeddings."""
        mode = "jina" if self.embed.jina_api else "local"
        dense = _backend_slug(self.embed.active_dense())
        sparse = _backend_slug(self.embed.sparse)
        return (
            f"{mode}--{dense}--{sparse}"
            f"--c{self.chunk.max_chars}-o{self.chunk.openapi_max_chars}"
        )

    def store_dir(self, root: Path) -> Path:
        base = self.project_dir(root)
        dest = base / self.backend_id()
        _adopt_legacy_local_store(base, dest, self)
        return dest

    def qdrant_path(self, root: Path) -> Path:
        return self.store_dir(root) / "qdrant"

    def qdrant_http_url(self) -> str:
        import os

        return (os.environ.get("BC_RAG_QDRANT_URL") or self.qdrant_url or QDRANT_HTTP_URL).rstrip(
            "/"
        )

    def qdrant_collection(self, root: Path) -> str:
        import hashlib
        import re

        from bc_rag.catalog import load_catalog

        root = root.resolve()
        name = next(
            (entry.name for entry in load_catalog() if Path(entry.root) == root),
            root.name or "project",
        )
        digest = hashlib.sha1(self.backend_id().encode("utf-8")).hexdigest()[:10]
        mode = "jina" if self.embed.jina_api else "local"
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:48] or "project"
        return f"bcrag_{slug}_{mode}_{digest}"

    def manifest_path(self, root: Path) -> Path:
        return self.store_dir(root) / "manifest.json"


def _backend_slug(name: str) -> str:
    return name.replace("/", "-").replace(" ", "-")


def _adopt_legacy_local_store(base: Path, dest: Path, configuration: RagConfig) -> None:
    """Move unlabeled `qdrant/` into the keyed local folder when that folder is empty."""
    if configuration.embed.jina_api:
        return
    legacy = base / "qdrant"
    if not legacy.is_dir():
        return
    dest_qdrant = dest / "qdrant"
    if dest_qdrant.exists() or (dest / "manifest.json").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.move(str(legacy), str(dest_qdrant))
    for name in ("manifest.json", "index.jsonl"):
        src = base / name
        if src.exists() and not (dest / name).exists():
            shutil.move(str(src), str(dest / name))


def with_jina_api(configuration: RagConfig, enabled: bool) -> RagConfig:
    embed = configuration.embed.model_copy(update={"jina_api": enabled})
    return configuration.model_copy(update={"embed": embed})


def config_path(root: Path) -> Path:
    return root / CONFIG_FILENAME


def load_config(root: Path) -> tuple[RagConfig, Path | None]:
    """Return (config, path_or_none_if_defaults)."""
    path = config_path(root)
    if not path.is_file():
        return RagConfig(), None
    from bc_rag.jsonc import strip_jsonc

    raw = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return RagConfig.model_validate(_normalize(raw)), path


def dump_default_config() -> str:
    configuration = RagConfig()
    data = configuration.model_dump()
    data["groups"] = [group.model_dump() for group in configuration.resolved_groups()]
    data.pop("include", None)
    return json.dumps(data, indent=2) + "\n"


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept camelCase aliases so a later UI can write either shape."""
    aliases = {
        "followGitignore": "follow_gitignore",
    }
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[aliases.get(key, key)] = value
    top_jina = out.pop("useJinaApi", None)
    if top_jina is None:
        top_jina = out.pop("jinaApi", None)
    embed = out.get("embed")
    if not isinstance(embed, dict) and top_jina is not None:
        embed = {}
        out["embed"] = embed
    if isinstance(embed, dict):
        if "useJinaApi" in embed and "jina_api" not in embed:
            embed["jina_api"] = embed.pop("useJinaApi")
        if "jinaApi" in embed and "jina_api" not in embed:
            embed["jina_api"] = embed.pop("jinaApi")
        if "jinaDense" in embed and "jina_dense" not in embed:
            embed["jina_dense"] = embed.pop("jinaDense")
        if "jinaRerank" in embed and "jina_rerank" not in embed:
            embed["jina_rerank"] = embed.pop("jinaRerank")
        if top_jina is not None and "jina_api" not in embed:
            embed["jina_api"] = top_jina
    if isinstance(out.get("chunk"), dict):
        chunk = out["chunk"]
        if "maxChars" in chunk and "max_chars" not in chunk:
            chunk["max_chars"] = chunk.pop("maxChars")
        if "minChars" in chunk and "min_chars" not in chunk:
            chunk["min_chars"] = chunk.pop("minChars")
    groups = out.get("groups")
    if isinstance(groups, list):
        normalized_groups = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            item = dict(group)
            if "followGitignore" in item and "follow_gitignore" not in item:
                item["follow_gitignore"] = item.pop("followGitignore")
            normalized_groups.append(item)
        out["groups"] = normalized_groups
    return out
