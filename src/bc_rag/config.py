"""Load `.bc-rag.json`, the project file shared with the team. It is required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from bc_rag.defaults import (
    CONFIG_FILENAME,
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_LIMIT,
    DEFAULT_PREFETCH,
    DEFAULT_SPARSE_MODEL,
    HARD_EXCLUDE_DIR_NAMES,
)
from bc_rag.facets import FACET_KEY, RESERVED_KEYS, check_not_reserved, normalize_facets


class ModelSpec(BaseModel):
    """Where a model runs, and the id that provider expects."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "jina", "voyage"]
    model: str

    def model_id(self) -> str:
        return self.model


class RetrieveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefetch: int = Field(default=DEFAULT_PREFETCH, ge=1, le=400)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=50)
    rerank: bool = True


class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=DEFAULT_CHUNK_MAX_CHARS, ge=200, le=80_000)
    min_chars: int = Field(default=DEFAULT_CHUNK_MIN_CHARS, ge=0, le=2000)


class SpaceConfig(BaseModel):
    """One vector space. The dense model owns the collection."""

    model_config = ConfigDict(extra="forbid")

    dense: ModelSpec
    sparse: ModelSpec
    rerank: ModelSpec | None = None
    dimensions: int | None = Field(default=None, ge=1)
    chunk: ChunkConfig | None = None
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)

    def dense_id(self) -> str:
        return self.dense.model

    def dense_provider(self) -> str:
        return self.dense.provider


class SourceGroup(BaseModel):
    """One named include/exclude set. Every file it claims carries its facets and its name."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # key -> value or list of values. Stored as key -> list.
    facets: dict[str, list[str]] = Field(default_factory=dict)
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    priority: int = 0
    kind: Literal["auto", "openapi"] = "auto"
    enabled: bool = True
    space: str

    @field_validator("facets", mode="before")
    @classmethod
    def _check_facets(cls, value: Any, info: Any) -> dict[str, list[str]]:
        name = (info.data or {}).get("name", "?")
        facets = normalize_facets(value, where=f"group {name!r} facets")
        check_not_reserved(facets, where=f"group {name!r} facets")
        return facets


class ScheduleConfig(BaseModel):
    """`schedule` in `.bc-rag.json`. `every` is a duration such as 15m."""

    model_config = ConfigDict(extra="forbid")

    every: str | None = None

    @field_validator("every")
    @classmethod
    def _check_every(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from bc_rag.duration import parse_duration

        parse_duration(value)
        return value.strip()


class RagConfig(BaseModel):
    """Project config. `spaces` and `defaultSpace` are required."""

    model_config = ConfigDict(extra="forbid")

    exclude: list[str] = Field(default_factory=list)
    groups: list[SourceGroup] = Field(default_factory=list)
    # A program whose stdout is {"groups": [...]}. Those groups are appended to `groups`.
    groups_command: str | list[str] | None = None
    spaces: dict[str, SpaceConfig]
    default_space: str
    # The team's background indexing interval. None = no background indexing.
    schedule: ScheduleConfig | None = None
    # One line per project facet key, telling a search client what the key means.
    facet_keys: dict[str, str] = Field(default_factory=dict)
    # Filter examples for this project, shown verbatim by the MCP list_projects tool.
    search_hints: list[str] = Field(default_factory=list)
    # True when groupsCommand was skipped (load_config groups=False).
    _partial: bool = PrivateAttr(default=False)

    def is_partial(self) -> bool:
        return self._partial

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
        names: set[str] = set()
        for group in self.groups:
            if group.name in names:
                raise ValueError(f"group name {group.name!r} is repeated")
            names.add(group.name)
            if group.space not in self.spaces:
                raise ValueError(
                    f"group {group.name!r} space {group.space!r} is not in spaces"
                )
        for key in self.facet_keys:
            if not FACET_KEY.fullmatch(key):
                raise ValueError(f"facetKeys: {key!r} is not a valid facet key")
            if key in RESERVED_KEYS:
                raise ValueError(f"facetKeys: {key!r} is set and described by bc-rag")
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
        return self.project_dir(root) / self.backend_id(space)

    def qdrant_http_url(self) -> str:
        """The user's `qdrant-url` setting. Qdrant is per user, not per project."""
        from bc_rag.usersettings import qdrant_url

        return qdrant_url()

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


def config_path(root: Path) -> Path:
    return root / CONFIG_FILENAME


def load_config(root: Path, *, groups: bool = True) -> tuple[RagConfig, Path]:
    """Return (config, path). `.bc-rag.json` is required.

    `groups=False` skips `groupsCommand`, for paths that only need spaces and models
    (search, show, delete). Such a config is partial: it can never list or index files.
    """
    path = config_path(root)
    if not path.is_file():
        raise ValueError(f"{path} is required. spaces and defaultSpace are required.")
    from bc_rag.jsonc import strip_jsonc

    raw = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    configuration = RagConfig.model_validate(_normalize(raw))
    if configuration.groups_command:
        if groups:
            configuration = _append_command_groups(root, configuration)
        else:
            configuration._partial = True
    return configuration, path


def public_config(configuration: RagConfig) -> dict[str, Any]:
    """Resolved config. Groups are last. Keys that do not apply are omitted."""
    data = configuration.model_dump(mode="json")
    data["groupsCommand"] = data.pop("groups_command")
    data["defaultSpace"] = data.pop("default_space")
    data["facetKeys"] = data.pop("facet_keys")
    data["searchHints"] = data.pop("search_hints")
    for key in ("schedule", "groupsCommand", "facetKeys", "searchHints"):
        if data.get(key) in (None, {}, []):
            data.pop(key, None)
    groups = [_public_group(group) for group in data.pop("groups", [])]
    first = (
        "exclude",
        "groupsCommand",
        "defaultSpace",
        "schedule",
        "facetKeys",
        "searchHints",
        "spaces",
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
    """File written by `bc-rag project add`. One space, dense model on that space."""
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
        elif (
            isinstance(value, list)
            and value
            and all(not isinstance(item, (dict, list)) for item in value)
        ):
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
        rendered = _format_space(spaces[name], indent + 2)
        blocks.append(f"{inner}{json.dumps(name)}: {rendered}{comma}")
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


_QDRANT_URL_MOVED = (
    "qdrant_url is not a project key. Run `bc-rag config set qdrant-url URL --global`."
)


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept camelCase aliases so a later UI can write either shape."""
    aliases = {
        "groupsCommand": "groups_command",
        "defaultSpace": "default_space",
        "facetKeys": "facet_keys",
        "searchHints": "search_hints",
    }
    banned = {
        "embed": (
            "embed is not a config key. "
            "spaces, defaultSpace, sparse, and rerank are top level."
        ),
        "chunk": "chunk is not a top-level key. Set chunk on each space.",
        "openapi": "openapi is not a config key. Use a group with kind openapi.",
        "include": "include is not a top-level key. Set include on each group.",
        "tags": "tags is not a config key. Set facets on each group.",
        "facets": "facets is not a top-level key. Set facets on each group.",
        "dense": "dense is not a top-level key. Set dense on each space.",
        "sparse": "sparse is not a top-level key. Set sparse on each space.",
        "rerank": "rerank is not a top-level key. Set rerank on each space.",
        "retrieve": "retrieve is not a top-level key. Set retrieve on each space.",
        "useJinaApi": "useJinaApi is not a config key. Set provider and model on each space.",
        "jinaApi": "jinaApi is not a config key. Set provider and model on each space.",
        "jinaDense": "jinaDense is not a config key. Set provider and model on each space.",
        "jinaRerank": "jinaRerank is not a config key. Set provider and model on rerank.",
        "qdrant_url": _QDRANT_URL_MOVED,
        "qdrantUrl": _QDRANT_URL_MOVED,
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
    return out
