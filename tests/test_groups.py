from pathlib import Path

import pytest
from tests.support import space_config

from bc_rag.config import ChunkConfig, ModelSpec, SourceGroup, SpaceConfig, load_config
from bc_rag.discover import debug_resolve_groups, iter_source_files


def test_disabled_groups_are_not_indexed(tmp_path: Path) -> None:
    (tmp_path / "libs" / "api" / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "libs" / "api" / "pkg" / "src" / "a.ts").write_text(
        "export const a = 1;\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n", encoding="utf-8")
    configuration = space_config(
        groups=[
            SourceGroup(
                space="prose",
                name="docs-bigcolony-reference",
                include=["docs/**/*.md"],
                priority=80,
            ),
            SourceGroup(space="prose", 
                name="libs-api-pkg",
                include=["libs/api/**/src/**/*.ts"],
                priority=55,
                enabled=False,
            ),
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert [source.rel_path for source in files] == ["docs/a.md"]
    debug = debug_resolve_groups(tmp_path, configuration)
    by_name = {row.name: row for row in debug}
    assert by_name["libs-api-pkg"].skipped is True
    assert by_name["libs-api-pkg"].file_count == 0
    assert by_name["docs-bigcolony-reference"].skipped is False


def test_group_chunk_override_is_attached_to_source_files(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n" + ("x" * 100), encoding="utf-8")
    configuration = space_config(
        spaces={
            "prose": SpaceConfig(
                dense=ModelSpec(provider="local", model="jinaai/jina-embeddings-v2-base-en"),
                sparse=ModelSpec(provider="local", model="Qdrant/bm25"),
                rerank=ModelSpec(provider="jina", model="jina-reranker-v3.5"),
                chunk=ChunkConfig(max_chars=400, min_chars=1),
            )
        },
        groups=[
            SourceGroup(space="prose", 
                name="docs-bigcolony-reference",
                include=["docs/**/*.md"],
                priority=80,
            )
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert files[0].chunk is not None
    assert files[0].chunk.max_chars == 400
    assert files[0].chunk.min_chars == 1


def test_groups_load_from_json(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        """
{
  "defaultSpace": "prose",
  "spaces": {
    "prose": {
      "dense": {"provider": "local", "model": "jinaai/jina-embeddings-v2-base-en"},
      "sparse": {"provider": "local", "model": "Qdrant/bm25"},
      "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
      "chunk": {"max_chars": 2400, "min_chars": 40}
    }
  },
  "groups": [
    {"name": "source", "space": "prose", "include": ["libs/**/src/**/*.ts"], "priority": 10},
    {"name": "reference", "space": "prose", "include": ["docs/**/*.md"], "priority": 80}
  ]
}
""",
        encoding="utf-8",
    )
    configuration, _ = load_config(tmp_path)
    names = [group.name for group in configuration.resolved_groups()]
    assert names == ["source", "reference"]


def test_static_groups_reject_a_repeated_name() -> None:
    with pytest.raises(ValueError, match="group name 'docs' is repeated"):
        space_config(
            groups=[
                SourceGroup(name="docs", space="prose", include=["a/**"]),
                SourceGroup(name="docs", space="prose", include=["b/**"]),
            ]
        )


def test_reserved_facet_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="facet key 'group' is set by bc-rag"):
        SourceGroup(name="docs", space="prose", facets={"group": "x"})


def test_openapi_group_carries_only_its_own_facets(tmp_path: Path) -> None:
    spec_dir = tmp_path / "docs" / "providers" / "_openapi"
    spec_dir.mkdir(parents=True)
    (spec_dir / "olo--mini.json").write_text(
        '{"openapi":"3.0.3","info":{"title":"t","version":"1"},"paths":{}}\n',
        encoding="utf-8",
    )
    (spec_dir / "klaviyo--mini.json").write_text(
        '{"openapi":"3.0.3","info":{"title":"k","version":"1"},"paths":{}}\n',
        encoding="utf-8",
    )
    configuration = space_config(
        groups=[
            SourceGroup(space="prose", 
                name="docs-providers-_openapi-olo",
                kind="openapi",
                facets={"scope": "external", "provider": "olo", "vendor": "olo"},
                include=["docs/providers/_openapi/olo--*.json"],
                priority=100,
            ),
            SourceGroup(space="prose", 
                name="docs-providers-_openapi-klaviyo",
                kind="openapi",
                facets={"scope": "external", "provider": "klaviyo", "vendor": "klaviyo"},
                include=["docs/providers/_openapi/klaviyo--*.json"],
                priority=100,
            ),
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    by_name = {source.path.name: source.facets for source in files}
    assert by_name["olo--mini.json"] == {
        "scope": ["external"],
        "provider": ["olo"],
        "vendor": ["olo"],
    }
    assert by_name["klaviyo--mini.json"]["vendor"] == ["klaviyo"]


def test_ingest_order_follows_priority_not_list_order(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n", encoding="utf-8")
    (tmp_path / "libs" / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "libs" / "pkg" / "src" / "a.ts").write_text(
        "export const a = 1;\n",
        encoding="utf-8",
    )
    configuration = space_config(
        groups=[
            SourceGroup(space="prose", name="source", include=["libs/**/src/**/*.ts"], priority=10),
            SourceGroup(space="prose", name="reference", include=["docs/**/*.md"], priority=80),
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert [source.group for source in files] == ["reference", "source"]
    assert files[0].priority == 80
    assert files[0].group == "reference"
    assert files[0].facets == {}


def test_disk_include_does_not_care_about_gitignore(tmp_path: Path) -> None:
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "notes.md").write_text("# secret\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("secret/\n", encoding="utf-8")
    configuration = space_config(
        groups=[
            SourceGroup(space="prose", 
                name="ignored-notes",
                include=["secret/**/*.md"],
                priority=1,
            )
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert [source.rel_path for source in files] == ["secret/notes.md"]
