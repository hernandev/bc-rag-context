from pathlib import Path

from bc_rag.config import RagConfig, SourceGroup, load_config
from bc_rag.discover import debug_resolve_groups, iter_source_files


def test_disabled_groups_are_not_indexed(tmp_path: Path) -> None:
    (tmp_path / "libs" / "api" / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "libs" / "api" / "pkg" / "src" / "a.ts").write_text("export const a = 1;\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n", encoding="utf-8")
    configuration = RagConfig(
        groups=[
            SourceGroup(name="docs-bigcolony-reference", include=["docs/**/*.md"], priority=80),
            SourceGroup(
                name="libs-api-pkg",
                include=["libs/api/**/src/**/*.ts"],
                priority=55,
                enabled=False,
            ),
        ],
        openapi={"enabled": False, "include": []},
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
    configuration = RagConfig(
        groups=[
            SourceGroup(
                name="docs-bigcolony-reference",
                include=["docs/**/*.md"],
                priority=80,
                chunk={"max_chars": 400, "min_chars": 1},
            )
        ],
        openapi={"enabled": False, "include": []},
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert files[0].chunk is not None
    assert files[0].chunk.max_chars == 400
    assert files[0].chunk.min_chars == 1
    assert files[0].chunk.openapi_max_chars == configuration.chunk.openapi_max_chars


def test_groups_load_from_json(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        """
{
  "groups": [
    {"name": "source", "include": ["libs/**/src/**/*.ts"], "priority": 10},
    {"name": "reference", "include": ["docs/**/*.md"], "priority": 80}
  ]
}
""",
        encoding="utf-8",
    )
    configuration, _ = load_config(tmp_path)
    names = [group.name for group in configuration.resolved_groups()]
    assert names == ["source", "reference"]


def test_openapi_group_stamps_shared_and_vendor_tags(tmp_path: Path) -> None:
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
    configuration = RagConfig(
        groups=[
            SourceGroup(
                name="docs-providers-_openapi-olo",
                kind="openapi",
                tags=[
                    "scope:external",
                    "system:vendor-provider",
                    "lifecycle:current",
                    "provider:olo",
                    "vendor:olo",
                ],
                include=["docs/providers/_openapi/olo--*.json"],
                priority=100,
            ),
            SourceGroup(
                name="docs-providers-_openapi-klaviyo",
                kind="openapi",
                tags=[
                    "scope:external",
                    "system:vendor-provider",
                    "lifecycle:current",
                    "provider:klaviyo",
                    "vendor:klaviyo",
                ],
                include=["docs/providers/_openapi/klaviyo--*.json"],
                priority=100,
            ),
        ],
        openapi={"enabled": False, "include": []},
    )
    files = list(iter_source_files(tmp_path, configuration))
    by_name = {source.path.name: source.tags for source in files}
    assert "scope:external" in by_name["olo--mini.json"]
    assert "system:vendor-provider" in by_name["olo--mini.json"]
    assert "provider:olo" in by_name["olo--mini.json"]
    assert "vendor:olo" in by_name["olo--mini.json"]
    assert "vendor:klaviyo" not in by_name["olo--mini.json"]
    assert "scope:external" in by_name["klaviyo--mini.json"]
    assert "provider:klaviyo" in by_name["klaviyo--mini.json"]
    assert "vendor:klaviyo" in by_name["klaviyo--mini.json"]


def test_ingest_order_follows_priority_not_list_order(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# a\n", encoding="utf-8")
    (tmp_path / "libs" / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "libs" / "pkg" / "src" / "a.ts").write_text("export const a = 1;\n", encoding="utf-8")
    configuration = RagConfig(
        groups=[
            SourceGroup(name="source", include=["libs/**/src/**/*.ts"], priority=10),
            SourceGroup(name="reference", include=["docs/**/*.md"], priority=80),
        ],
        openapi={"enabled": False, "include": []},
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert [source.group for source in files] == ["reference", "source"]
    assert files[0].priority == 80
    assert files[0].group == "reference"
    assert files[0].tags == []


def test_disk_include_does_not_care_about_gitignore(tmp_path: Path) -> None:
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "notes.md").write_text("# secret\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("secret/\n", encoding="utf-8")
    configuration = RagConfig(
        groups=[
            SourceGroup(
                name="ignored-notes",
                include=["secret/**/*.md"],
                priority=1,
            )
        ],
        openapi={"enabled": False, "include": []},
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert [source.rel_path for source in files] == ["secret/notes.md"]
