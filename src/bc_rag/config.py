"""Load `.bc-rag.json` or fall back to defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bc_rag.defaults import (
    CONFIG_FILENAME,
    DEFAULT_CHUNK_MAX_CHARS,
    QDRANT_HTTP_URL,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_DENSE_MODEL,
    DEFAULT_INCLUDE,
    DEFAULT_LIMIT,
    DEFAULT_OPENAPI_INCLUDE,
    DEFAULT_PREFETCH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_SPARSE_MODEL,
    HARD_EXCLUDE_DIR_NAMES,
)


class ModelSpec(BaseModel):
    """Where a model runs, and the id that provider expects."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "jina", "voyage"]
    model: str

    def model_id(self) -> str:
        return self.model


class SpaceConfig(BaseModel):
    """One vector space. The dense model owns the collection."""

    model_config = ConfigDict(extra="forbid")

    dense: ModelSpec
    sparse: ModelSpec
    rerank: ModelSpec | None = None
    dimensions: int | None = Field(default=None, ge=1)
    chunk: "ChunkConfig | None" = None
    retrieve: "RetrieveConfig" = Field(default_factory=lambda: RetrieveConfig())

    def dense_id(self) -> str:
        return self.dense.model

    def dense_provider(self) -> str:
        return self.dense.provider


class EmbedConfig(BaseModel):
    """Kept so older call sites fail at import time if they still build an embed block."""

    model_config = ConfigDict(extra="forbid")


class RetrieveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefetch: int = Field(default=DEFAULT_PREFETCH, ge=1, le=400)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=50)
    rerank: bool = True


class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=DEFAULT_CHUNK_MAX_CHARS, ge=200, le=80_000)
    min_chars: int = Field(default=DEFAULT_CHUNK_MIN_CHARS, ge=0, le=2000)


SpaceConfig.model_rebuild()


class ChunkOverride(BaseModel):
    """Partial chunk settings. Omitted keys inherit from the top-level chunk block."""

    model_config = ConfigDict(extra="forbid")

    max_chars: int | None = Field(default=None, ge=200, le=80_000)
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
    space: str


class RagConfig(BaseModel):
    """Project config. `spaces` and `defaultSpace` are required."""

    model_config = ConfigDict(extra="forbid")

    exclude: list[str] = Field(default_factory=list)
    groups: list[SourceGroup] = Field(default_factory=list)
    # A program whose stdout is {"groups": [...]}. Those groups are appended to `groups`.
    groups_command: str | list[str] | None = None
    follow_gitignore: bool = False
    spaces: dict[str, SpaceConfig]
    default_space: str
    qdrant_url: str | None = None

    @model_validator(mode="after")
    def _check_spaces(self) -> RagConfig:
        if not self.spaces:
            raise ValueError("spaces is required")
        if self.default_space not in self.spaces:
            raise ValueError(
                f"defaultSpace {self.default_space!r} is not in spaces"
            )
        for name, spec in self.spaces.items():
            _check_model(f"space {name!r} dense", spec.dense)
            _check_model(f"space {name!r} sparse", spec.sparse)
            if spec.rerank is not None:
                _check_model(f"space {name!r} rerank", spec.rerank)
            if spec.chunk is None:
                raise ValueError(f"space {name!r} requires chunk")
        for group in self.groups:
            if group.space not in self.spaces:
                raise ValueError(
                    f"group {group.name!r} space {group.space!r} is not in spaces"
                )
        return self

    def space_named(self, space: str) -> SpaceConfig:
        found = self.spaces.get(space)
        if found is None:
            known = ", ".join(sorted(self.spaces))
            raise ValueError(f"unknown space {space!r}. Configured spaces: {known}")
        return found

    def resolved_groups(self) -> list[SourceGroup]:
        return [group for group in self.groups if group.enabled]

    def project_dir(self, root: Path) -> Path:
        from bc_rag.catalog import store_dir_for_root

        return store_dir_for_root(root)

    def backend_id(self, space: str | None = None) -> str:
        """Folder name for one vector space. Mode + models that change embeddings."""
        name = space or self.default_space
        spec = self.spaces[name]
        dense_id = spec.dense_id()
        chunk = spec.chunk
        if chunk is None:
            raise ValueError(f"space {name!r} requires chunk")
        if dense_id.startswith("voyage-") or spec.dense_provider() == "voyage":
            mode = "voyage"
            dims = spec.dimensions or 1024
        elif spec.dense_provider() == "jina":
            mode = "jina"
            dims = spec.dimensions or 0
        else:
            mode = "local"
            dims = spec.dimensions or 0
        return (
            f"{_backend_slug(name)}--{mode}--{_backend_slug(dense_id)}"
            f"--d{dims}--{_backend_slug(spec.sparse.model)}"
            f"--c{chunk.max_chars}"
        )

    def store_dir(self, root: Path, space: str | None = None) -> Path:
        base = self.project_dir(root)
        dest = base / self.backend_id(space)
        if (space or self.default_space) == self.default_space:
            _adopt_legacy_local_store(base, dest, self)
        return dest

    def qdrant_path(self, root: Path) -> Path:
        return self.store_dir(root) / "qdrant"

    def qdrant_http_url(self) -> str:
        import os

        return (os.environ.get("BC_RAG_QDRANT_URL") or self.qdrant_url or QDRANT_HTTP_URL).rstrip(
            "/"
        )

    def qdrant_collection(self, root: Path, space: str | None = None) -> str:
        import hashlib
        import re

        from bc_rag.catalog import load_catalog

        root = root.resolve()
        name = next(
            (entry.name for entry in load_catalog() if Path(entry.root) == root),
            root.name or "project",
        )
        digest = hashlib.sha1(self.backend_id(space).encode("utf-8")).hexdigest()[:10]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:48] or "project"
        name = space or self.default_space
        spec = self.spaces[name]
        if spec.dense_provider() == "voyage" or spec.dense_id().startswith("voyage-"):
            mode = "voyage"
        elif spec.dense_provider() == "jina":
            mode = "jina"
        else:
            mode = "local"
        space_slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:24] or "space"
        return f"bcrag_{slug}_{space_slug}_{mode}_{digest}"

    def manifest_path(self, root: Path, space: str | None = None) -> Path:
        return self.store_dir(root, space) / "manifest.json"


def _check_model(label: str, spec: ModelSpec) -> None:
    model = spec.model
    if spec.provider == "voyage" and not (
        model.startswith("voyage-") or model.startswith("rerank-")
    ):
        raise ValueError(f"{label}: voyage model {model!r} is not a Voyage id")
    if spec.provider == "jina" and not model.startswith("jina"):
        raise ValueError(f"{label}: jina model {model!r} is not a Jina API id")
    if spec.provider == "local" and (
        model.startswith("voyage-") or model.startswith("rerank-")
    ):
        raise ValueError(f"{label}: {model!r} is an API id and cannot use provider local")


def _backend_slug(name: str) -> str:
    return name.replace("/", "-").replace(" ", "-")


def _adopt_legacy_local_store(base: Path, dest: Path, configuration: RagConfig) -> None:
    """Move unlabeled `qdrant/` into the keyed local folder when that folder is empty."""
    if configuration.spaces[configuration.default_space].dense_provider() == "jina":
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
    """Unused. The dense model lives on each space."""
    del configuration, enabled
    raise ValueError("with_jina_api is not a config key. Set provider and model on a space.")


def config_path(root: Path) -> Path:
    return root / CONFIG_FILENAME


def load_config(root: Path) -> tuple[RagConfig, Path | None]:
    """Return (config, path_or_none_if_defaults)."""
    path = config_path(root)
    if not path.is_file():
        raise ValueError(f"{path} is required. spaces and defaultSpace are required.")
    from bc_rag.jsonc import strip_jsonc

    raw = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    configuration = RagConfig.model_validate(_normalize(raw))
    if configuration.groups_command:
        configuration = _append_command_groups(root, configuration)
    return configuration, path


def public_config(configuration: RagConfig) -> dict[str, Any]:
    """Resolved config. Groups are last. Keys that do not apply are omitted."""
    data = configuration.model_dump(mode="json")
    data["groupsCommand"] = data.pop("groups_command")
    data["defaultSpace"] = data.pop("default_space")
    if data.get("follow_gitignore") is False:
        data.pop("follow_gitignore", None)
    elif "follow_gitignore" in data:
        data["followGitignore"] = data.pop("follow_gitignore")
    if data.get("qdrant_url") is None:
        data.pop("qdrant_url", None)
    if data.get("groupsCommand") is None:
        data.pop("groupsCommand", None)
    groups = [_public_group(group) for group in data.pop("groups", [])]
    first = (
        "exclude",
        "groupsCommand",
        "followGitignore",
        "defaultSpace",
        "spaces",
        "qdrant_url",
    )
    ordered = {key: data[key] for key in first if key in data}
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    ordered["groups"] = groups
    return ordered


def _public_group(group: dict[str, Any]) -> dict[str, Any]:
    kept = {
        key: value
        for key, value in group.items()
        if value not in (None, [], {}) and not (key == "kind" and value == "auto")
    }
    return kept


def dump_default_config() -> str:
    """File written by `bc-rag init`. One space, dense model on that space."""
    configuration = RagConfig(
        default_space="default",
        exclude=[f"**/{name}/**" for name in sorted(HARD_EXCLUDE_DIR_NAMES)],
        spaces={
            "default": SpaceConfig(
                dense=ModelSpec(provider="jina", model="jina-embeddings-v5-text-small"),
                sparse=ModelSpec(provider="local", model=DEFAULT_SPARSE_MODEL),
                rerank=ModelSpec(provider="voyage", model="rerank-2.5"),
                chunk=ChunkConfig(),
            )
        },
    )
    return format_config(public_config(configuration))


_SPACE_KEYS = ("dimensions", "chunk", "retrieve", "dense", "sparse", "rerank")
_CHUNK_KEYS = ("min_chars", "max_chars")
_RETRIEVE_KEYS = ("prefetch", "limit", "rerank")
_MODEL_KEYS = ("provider", "model")


def format_config(data: dict[str, Any]) -> str:
    """Pretty JSON. Small objects stay on one line. Space keys follow one order."""
    lines = ["{"]
    items = list(data.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        if key == "spaces" and isinstance(value, dict):
            rendered = _format_spaces(value, 2)
        elif key == "groups" and isinstance(value, list):
            rendered = _format_groups(value, 2)
        elif isinstance(value, list) and value and all(not isinstance(item, (dict, list)) for item in value):
            rendered = _format_string_list(value, 2)
        else:
            rendered = _compact(value)
        lines.append(f"  {json.dumps(key)}: {rendered}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _format_spaces(spaces: dict[str, Any], indent: int) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    blocks: list[str] = []
    names = list(spaces)
    for index, name in enumerate(names):
        comma = "," if index < len(names) - 1 else ""
        blocks.append(f"{inner}{json.dumps(name)}: {_format_space(spaces[name], indent + 2)}{comma}")
    return "{\n" + "\n".join(blocks) + f"\n{pad}}}"


def _format_space(spec: dict[str, Any], indent: int) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    lines: list[str] = []
    present = [key for key in _SPACE_KEYS if key in spec and spec[key] is not None]
    for index, key in enumerate(present):
        comma = "," if index < len(present) - 1 else ""
        value = spec[key]
        if key == "chunk" and isinstance(value, dict):
            value = _pick(value, _CHUNK_KEYS)
        elif key == "retrieve" and isinstance(value, dict):
            value = _pick(value, _RETRIEVE_KEYS)
        elif key in ("dense", "sparse", "rerank") and isinstance(value, dict):
            value = _pick(value, _MODEL_KEYS)
        rendered = _compact(value) if isinstance(value, dict) else json.dumps(value)
        lines.append(f"{inner}{json.dumps(key)}: {rendered}{comma}")
    return "{\n" + "\n".join(lines) + f"\n{pad}}}"


def _format_groups(groups: list[Any], indent: int) -> str:
    if not groups:
        return "[]"
    pad = " " * indent
    inner = " " * (indent + 2)
    blocks: list[str] = []
    for index, group in enumerate(groups):
        comma = "," if index < len(groups) - 1 else ""
        body = json.dumps(group, indent=2)
        shifted = body.replace("\n", "\n" + inner)
        blocks.append(f"{inner}{shifted}{comma}")
    return "[\n" + "\n".join(blocks) + f"\n{pad}]"


def _format_string_list(values: list[Any], indent: int) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    lines = [f"{inner}{json.dumps(value)}" for value in values]
    return "[\n" + ",\n".join(lines) + f"\n{pad}]"


def _pick(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    ordered = {key: value[key] for key in keys if key in value}
    for key, item in value.items():
        if key not in ordered:
            ordered[key] = item
    return ordered


def _compact(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"{json.dumps(key)}: {_compact(item)}" for key, item in value.items()]
        return "{ " + ", ".join(parts) + " }"
    return json.dumps(value)


def _append_command_groups(root: Path, configuration: RagConfig) -> RagConfig:
    from bc_rag.groups_command import read_groups

    command = configuration.groups_command
    if command is None:
        return configuration
    incoming = read_groups(root, command)
    names = {group.name for group in configuration.groups}
    generated: list[SourceGroup] = []
    for item in incoming:
        group = SourceGroup.model_validate(item)
        if group.name in names:
            raise ValueError(f"groupsCommand repeated the group name {group.name!r}")
        names.add(group.name)
        generated.append(group)
    merged = configuration.model_copy(update={"groups": [*configuration.groups, *generated]})
    return RagConfig.model_validate(merged.model_dump())


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept camelCase aliases so a later UI can write either shape."""
    aliases = {
        "followGitignore": "follow_gitignore",
        "groupsCommand": "groups_command",
        "defaultSpace": "default_space",
    }
    banned = {
        "embed": "embed is not a config key. spaces, defaultSpace, sparse, and rerank are top level.",
        "chunk": "chunk is not a top-level key. Set chunk on each space.",
        "openapi": "openapi is not a config key. Use a group with kind openapi.",
        "include": "include is not a top-level key. Set include on each group.",
        "tags": "tags is not a top-level key. Set tags on each group.",
        "dense": "dense is not a top-level key. Set dense on each space.",
        "sparse": "sparse is not a top-level key. Set sparse on each space.",
        "rerank": "rerank is not a top-level key. Set rerank on each space.",
        "retrieve": "retrieve is not a top-level key. Set retrieve on each space.",
        "useJinaApi": "useJinaApi is not a config key. Set provider and model on each space.",
        "jinaApi": "jinaApi is not a config key. Set provider and model on each space.",
        "jinaDense": "jinaDense is not a config key. Set provider and model on each space.",
        "jinaRerank": "jinaRerank is not a config key. Set provider and model on rerank.",
    }
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in banned:
            raise ValueError(banned[key])
        out[aliases.get(key, key)] = value
    spaces = out.get("spaces")
    if isinstance(spaces, dict):
        for spec in spaces.values():
            if not isinstance(spec, dict):
                continue
            if "outputDimension" in spec and "dimensions" not in spec:
                spec["dimensions"] = spec.pop("outputDimension")
            if "input" in spec or "autoChunk" in spec:
                raise ValueError("space.input is not a config key. Chunking stays in bc-rag.")
            chunk = spec.get("chunk")
            if isinstance(chunk, dict):
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
