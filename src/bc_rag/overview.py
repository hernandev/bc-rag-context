"""What one project has stored, per space: models, collection, points, files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from bc_rag.catalog import ProjectEntry
from bc_rag.config import load_config
from bc_rag.manifest import load_manifest


@dataclass
class SpaceOverview:
    space: str
    dense: str
    rerank: str
    collection: str
    # None when Qdrant could not be reached.
    points: int | None
    files: int
    # When the space's manifest was last written. None = never indexed.
    manifest_time: str | None
    error: str | None = None


def project_overview(entry: ProjectEntry) -> list[SpaceOverview]:
    """One row per configured space. Reads the config without running groupsCommand."""
    from bc_rag.store import HybridStore

    root = entry.root_path()
    config, _path = load_config(root, groups=False)
    rows: list[SpaceOverview] = []
    for name, spec in config.spaces.items():
        collection = config.qdrant_collection(root, name)
        manifest_path = config.manifest_path(root, name)
        manifest = load_manifest(manifest_path)
        manifest_time = None
        if manifest_path.is_file():
            stamp = datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)
            manifest_time = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        points: int | None = None
        error: str | None = None
        store = HybridStore(url=config.qdrant_http_url(), collection=collection, read_only=True)
        try:
            points = store.count()
        except Exception as exc:
            error = str(exc)
        finally:
            store.close()
        rerank = "(off)"
        if spec.rerank is not None and spec.retrieve.rerank:
            rerank = f"{spec.rerank.provider} {spec.rerank.model}"
        rows.append(
            SpaceOverview(
                space=name,
                dense=f"{spec.dense.provider} {spec.dense.model}",
                rerank=rerank,
                collection=collection,
                points=points,
                files=len(manifest.files) if manifest is not None else 0,
                manifest_time=manifest_time,
                error=error,
            )
        )
    return rows


COLLAPSE_VALUES = 50


@dataclass
class FacetRow:
    key: str
    # None on a collapsed row: the key has more than COLLAPSE_VALUES values.
    value: str | None
    # declared (in .bc-rag.json), stored (in Qdrant), or both.
    source: str
    # space -> points. None = the space has no index for this key ("-"); a missing
    # space = Qdrant could not be read ("?").
    points: dict[str, int | None]
    distinct: int = 1


@dataclass
class FacetsReport:
    spaces: list[str]
    rows: list[FacetRow]
    # space -> error text for spaces Qdrant could not answer for.
    errors: dict[str, str]


def facets_report(entry: ProjectEntry, *, key: str | None = None) -> FacetsReport:
    """Facets declared by the enabled groups, joined with the facets stored per space."""
    from bc_rag.facets import distinct_facet_values
    from bc_rag.store import HybridStore

    root = entry.root_path()
    config, _path = load_config(root)
    declared = distinct_facet_values(config.groups)
    spaces = list(config.spaces)
    stored: dict[str, dict[str, dict[str, int]]] = {}
    errors: dict[str, str] = {}
    for space in spaces:
        store = HybridStore(
            url=config.qdrant_http_url(),
            collection=config.qdrant_collection(root, space),
            read_only=True,
        )
        try:
            keys = store.facet_keys()
            stored[space] = {
                name: dict(store.facet_values(name))
                for name in keys
                if key is None or name == key
            }
        except Exception as exc:
            errors[space] = str(exc)
        finally:
            store.close()
    names = sorted({*declared, *(name for per_space in stored.values() for name in per_space)})
    if key is not None:
        names = [name for name in names if name == key]
    rows: list[FacetRow] = []
    for name in names:
        values = set(declared.get(name, []))
        for per_space in stored.values():
            values.update(per_space.get(name, {}))
        ordered = sorted(values)
        if key is None and len(ordered) > COLLAPSE_VALUES:
            points = {
                space: (sum(per_space[name].values()) if name in per_space else None)
                for space, per_space in stored.items()
            }
            source = _source(name in declared, any(name in s for s in stored.values()))
            rows.append(FacetRow(name, None, source, points, distinct=len(ordered)))
            continue
        for value in ordered:
            points = {
                space: (per_space[name].get(value, 0) if name in per_space else None)
                for space, per_space in stored.items()
            }
            in_store = any(
                value in per_space.get(name, {}) for per_space in stored.values()
            )
            source = _source(value in declared.get(name, []), in_store)
            rows.append(FacetRow(name, value, source, points))
    return FacetsReport(spaces=spaces, rows=rows, errors=errors)


def _source(declared: bool, stored: bool) -> str:
    if declared and stored:
        return "both"
    return "declared" if declared else "stored"
