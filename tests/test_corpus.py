import json
from pathlib import Path

from bc_rag.config import SourceGroup
from tests.support import space_config
from bc_rag.corpus import (
    attributes_from_source,
    bundle_sources,
    corpus_dir,
    corpus_key,
    prepare_document,
    sidecar_path_for,
)
from bc_rag.discover import SourceFile
from bc_rag.indexer import Indexer
from bc_rag.store import HybridStore

from test_indexer import FakeEmbedder, TS


def test_prepare_writes_bedrock_sidecar(tmp_path: Path) -> None:
    src = tmp_path / "docs" / "note.md"
    src.parent.mkdir(parents=True)
    src.write_text("# hello\n", encoding="utf-8")
    source = SourceFile(
        path=src,
        rel_path="docs/note.md",
        language="markdown",
        size=src.stat().st_size,
        tags=["scope:internal", "system:bigcolony-workspaces", "lifecycle:current", "area:engine"],
        metadata={"content_type": "documentation"},
    )
    prepared = prepare_document(tmp_path / "store", source)
    assert prepared.key == "internal/current/bigcolony-workspaces/docs/note.md"
    sidecar = json.loads(prepared.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == {
        "metadataAttributes": {
            "scope": "internal",
            "system": "bigcolony-workspaces",
            "lifecycle": "current",
            "area": "engine",
            "content_type": "documentation",
        }
    }
    assert sidecar_path_for(prepared.path) == prepared.sidecar_path


def test_openapi_corpus_key_uses_external_vendor_slug() -> None:
    source = SourceFile(
        path=Path("/tmp/spec.md"),
        rel_path="openapi-md/docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi/spec.md",
        language="markdown",
        size=1,
        tags=["scope:external", "vendor:olo", "provider:olo"],
    )
    assert corpus_key(source) == "external/olo/ordering-api-1.1.bundle/spec.md"


def test_bundle_prunes_disabled_group(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.catalog.register_project", lambda root: None)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "keep.md").write_text("# keep\n", encoding="utf-8")
    (tmp_path / "docs" / "drop.md").write_text("# drop\n", encoding="utf-8")
    keep_group = SourceGroup(space="prose", 
        name="keep",
        include=["docs/keep.md"],
        tags=["scope:internal"],
        enabled=True,
    )
    drop_group = SourceGroup(space="prose", 
        name="drop",
        include=["docs/drop.md"],
        tags=["scope:internal"],
        enabled=True,
    )
    first = space_config(
        follow_gitignore=False,
        groups=[keep_group, drop_group],
    )
    bundle_sources(tmp_path, first)
    names = {path.name for path in corpus_dir(first.project_dir(tmp_path)).rglob("*.md")}
    assert "keep.md" in names
    assert "drop.md" in names
    second = space_config(
        follow_gitignore=False,
        groups=[keep_group, drop_group.model_copy(update={"enabled": False})],
    )
    _written, pruned = bundle_sources(tmp_path, second)
    names = {path.name for path in corpus_dir(second.project_dir(tmp_path)).rglob("*.md")}
    assert "keep.md" in names
    assert "drop.md" not in names
    assert any(name.endswith("drop.md") or name.endswith("drop.md.metadata.json") for name in pruned)


def test_tag_change_rewrites_sidecar_and_reindexes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bc_rag.indexer.register_project", lambda root: None)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Location.ts").write_text(TS, encoding="utf-8")
    first = space_config(
        follow_gitignore=False,
        groups=[
            SourceGroup(space="prose", 
                name="src",
                include=["src/**/*.ts"],
                tags=["scope:internal", "area:engine"],
            )
        ],
    )
    store = HybridStore(first.qdrant_path(tmp_path))
    embedder = FakeEmbedder()
    embedder.calls = 0

    def counting_embed_docs(dense_texts, sparse_texts):
        embedder.calls += 1
        return FakeEmbedder.embed_docs(embedder, dense_texts, sparse_texts)

    embedder.embed_docs = counting_embed_docs  # type: ignore[method-assign]
    try:
        Indexer(tmp_path, first, embedder, store).run()
        after_first = embedder.calls
        same = Indexer(tmp_path, first, embedder, store).run()
        assert same.skipped_unchanged == 1
        assert embedder.calls == after_first

        second = space_config(
            follow_gitignore=False,
            groups=[
                SourceGroup(space="prose", 
                    name="src",
                    include=["src/**/*.ts"],
                    tags=["scope:internal", "area:engine", "layer:engine-core"],
                )
            ],
        )
        changed = Indexer(tmp_path, second, embedder, store).run()
        assert changed.skipped_unchanged == 0
        assert changed.indexed_files == 1
        assert embedder.calls > after_first
        attrs = attributes_from_source(
            SourceFile(
                path=tmp_path / "src" / "Location.ts",
                rel_path="src/Location.ts",
                language="typescript",
                size=1,
                tags=["scope:internal", "area:engine", "layer:engine-core"],
            )
        )
        assert attrs["layer"] == "engine-core"
    finally:
        store.close()
