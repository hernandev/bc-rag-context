import json
from pathlib import Path

from tests.support import entry_for, space_config
from tests.test_indexer_incremental import FakeStore

from bc_rag.config import SourceGroup
from bc_rag.corpus import (
    attributes_from_source,
    corpus_dir,
    corpus_key,
    prepare_document,
    prune_corpus,
    sidecar_path_for,
)
from bc_rag.discover import SourceFile
from bc_rag.indexer import Indexer, index_spaces
from bc_rag.store import HybridStore
from test_indexer import TS, FakeEmbedder


def test_sidecar_attributes_are_the_facets(tmp_path: Path) -> None:
    src = tmp_path / "docs" / "note.md"
    src.parent.mkdir(parents=True)
    src.write_text("# hello\n", encoding="utf-8")
    source = SourceFile(
        path=src,
        rel_path="docs/note.md",
        language="markdown",
        size=src.stat().st_size,
        facets={
            "scope": ["internal"],
            "system": ["bigcolony-workspaces"],
            "lifecycle": ["current"],
            "provider": ["olo", "toast"],
        },
        group="docs",
    )
    prepared = prepare_document(tmp_path / "store", source)
    assert prepared.key == "internal/current/bigcolony-workspaces/docs/note.md"
    sidecar = json.loads(prepared.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == {
        "metadataAttributes": {
            "group": ["docs"],
            "lifecycle": ["current"],
            "provider": ["olo", "toast"],
            "scope": ["internal"],
            "system": ["bigcolony-workspaces"],
        }
    }
    assert sidecar_path_for(prepared.path) == prepared.sidecar_path


def test_openapi_corpus_key_uses_external_vendor_slug() -> None:
    source = SourceFile(
        path=Path("/tmp/spec.md"),
        rel_path="openapi-md/docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi/spec.md",
        language="markdown",
        size=1,
        facets={"scope": ["external"], "vendor": ["olo"], "provider": ["olo"]},
    )
    assert corpus_key(source) == "external/olo/ordering-api-1.1.bundle/spec.md"


def test_index_prunes_a_disabled_group_from_the_cache(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "keep.md").write_text("# keep\n\nkeep body\n", encoding="utf-8")
    (tmp_path / "docs" / "drop.md").write_text("# drop\n\ndrop body\n", encoding="utf-8")
    keep_group = SourceGroup(space="prose", name="keep", include=["docs/keep.md"])
    drop_group = SourceGroup(space="prose", name="drop", include=["docs/drop.md"])
    first = space_config(groups=[keep_group, drop_group])
    entry = entry_for(tmp_path)
    store = FakeStore()
    Indexer(entry, first, FakeEmbedder(), store, space="prose").run()
    corpus = corpus_dir(first.project_dir(tmp_path))
    assert corpus.parent.name == "cache"
    assert {"keep.md", "drop.md"} <= {path.name for path in corpus.rglob("*.md")}
    second = space_config(groups=[keep_group, drop_group.model_copy(update={"enabled": False})])
    Indexer(entry, second, FakeEmbedder(), store, space="prose").run()
    names = {path.name for path in corpus.rglob("*.md")}
    assert "keep.md" in names
    assert "drop.md" not in names


def test_the_old_cache_folders_move_into_cache(tmp_path: Path) -> None:
    from bc_rag.cache import move_legacy_cache

    project_dir = tmp_path / "store"
    (project_dir / "corpus" / "internal").mkdir(parents=True)
    (project_dir / "corpus" / "internal" / "a.md").write_text("a", encoding="utf-8")
    (project_dir / "openapi-md" / "spec").mkdir(parents=True)
    (project_dir / "openapi-md" / "spec" / "spec.md").write_text("s", encoding="utf-8")

    moved = move_legacy_cache(project_dir)

    assert sorted(path.name for path in moved) == ["corpus", "openapi-md"]
    assert (project_dir / "cache" / "corpus" / "internal" / "a.md").is_file()
    assert (project_dir / "cache" / "openapi-md" / "spec" / "spec.md").is_file()
    assert not (project_dir / "corpus").exists()
    assert move_legacy_cache(project_dir) == []


def test_index_spaces_prunes_once_with_every_space(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nprose body text here\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.ts").write_text("export const b = 1;\n", encoding="utf-8")
    config = space_config(
        groups=[
            SourceGroup(name="docs", space="prose", include=["docs/**/*.md"]),
            SourceGroup(name="code", space="code", include=["src/**/*.ts"]),
        ]
    )
    entry = entry_for(tmp_path)
    indexers = [
        Indexer(entry, config, FakeEmbedder(), FakeStore(), space="prose"),
        Indexer(entry, config, FakeEmbedder(), FakeStore(), space="code"),
    ]
    index_spaces(indexers)
    root = corpus_dir(config.project_dir(tmp_path))
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert any(name.endswith("a.md") for name in files)
    assert any(name.endswith("b.ts.md") for name in files)


def test_facet_change_rewrites_sidecar_and_reindexes(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Location.ts").write_text(TS, encoding="utf-8")

    def config_with(facets: dict):
        return space_config(
            groups=[SourceGroup(space="prose", name="src", include=["src/**/*.ts"], facets=facets)]
        )

    first = config_with({"scope": "internal", "area": "engine"})
    store = HybridStore(tmp_path / "qdrant-local")
    embedder = FakeEmbedder()
    embedder.calls = 0

    def counting_embed_docs(dense_texts, sparse_texts):
        embedder.calls += 1
        return FakeEmbedder.embed_docs(embedder, dense_texts, sparse_texts)

    embedder.embed_docs = counting_embed_docs  # type: ignore[method-assign]
    try:
        Indexer(entry_for(tmp_path), first, embedder, store).run()
        after_first = embedder.calls
        same = Indexer(entry_for(tmp_path), first, embedder, store).run()
        assert same.skipped_unchanged == 1
        assert embedder.calls == after_first

        second = config_with({"scope": "internal", "area": "engine", "layer": "engine-core"})
        changed = Indexer(entry_for(tmp_path), second, embedder, store).run()
        assert changed.skipped_unchanged == 0
        assert changed.indexed_files == 1
        assert embedder.calls > after_first
        attrs = attributes_from_source(
            SourceFile(
                path=tmp_path / "src" / "Location.ts",
                rel_path="src/Location.ts",
                language="typescript",
                size=1,
                facets=second.groups[0].facets,
            )
        )
        assert attrs["layer"] == ["engine-core"]
    finally:
        store.close()


def test_prune_keeps_kept_keys_and_drops_emptied_folders(tmp_path: Path) -> None:
    root = corpus_dir(tmp_path)
    for rel in ("keep/a.md", "gone/deep/b.md", "mixed/c.md", "mixed/d.md"):
        document = root / rel
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text("x", encoding="utf-8")
        sidecar_path_for(document).write_text("{}", encoding="utf-8")

    pruned = prune_corpus(tmp_path, {"keep/a.md", "mixed/c.md"})

    assert sorted(pruned) == [
        "gone/deep/b.md",
        "gone/deep/b.md.metadata.json",
        "mixed/d.md",
        "mixed/d.md.metadata.json",
    ]
    assert (root / "keep" / "a.md").is_file()
    assert (root / "mixed" / "c.md.metadata.json").is_file()
    assert not (root / "gone").exists()
    assert root.is_dir()
