from pathlib import Path

from tests.support import default_files_config, space_config

from bc_rag.config import SourceGroup
from bc_rag.discover import classify_path, iter_source_files


def test_brace_globs_match_src_and_extensions() -> None:
    from bc_rag.discover import _matches_any

    pattern = "libs/**/src/**/*.{ts,tsx,mts}"
    assert _matches_any(
        "libs/engine/flows/engine-flows-locations/src/services/LocationProjectorService.ts",
        [pattern],
    )
    assert _matches_any("libs/web/portal/src/Foo.tsx", [pattern])
    assert not _matches_any(
        "libs/engine/flows/engine-flows-locations/tests/unit/auth.test.ts",
        [pattern],
    )
    assert not _matches_any("libs/web/portal/src/Foo.vue", [pattern])


def test_discovers_typescript_and_markdown(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text("export const a = 1;\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.ts").write_text("export const skip = 1;\n", encoding="utf-8")

    sources = iter_source_files(tmp_path, default_files_config())
    found = {source.rel_path: source.language for source in sources}

    assert found["src/a.ts"] == "typescript"
    assert found["README.md"] == "markdown"
    assert "node_modules/skip.ts" not in found


def test_package_json_is_json_when_included(tmp_path: Path) -> None:
    configuration = space_config(
        groups=[SourceGroup(name="pkg", space="prose", include=["**/{package,project}.json"])]
    )
    relative = "libs/engine/flows/engine-flows-locations/package.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"name":"@bigcolony/engine-flows-locations"}\n', encoding="utf-8")

    language = classify_path(relative, path, configuration)

    assert language == "json"


def test_source_for_path_uses_only_the_claiming_group(tmp_path: Path) -> None:
    from bc_rag.indexer import sources_for_paths

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nbody\n", encoding="utf-8")
    configuration = space_config(
        groups=[
            SourceGroup(
                name="high", space="prose", priority=10, include=["docs/**/*.md"],
                facets={"area": "engine"},
            ),
            SourceGroup(
                name="low", space="prose", priority=1, include=["docs/**/*.md"],
                facets={"area": "admin", "team": "x"},
            ),
        ]
    )
    full = next(iter_source_files(tmp_path, configuration))
    by_path = sources_for_paths(tmp_path, configuration, {"docs/a.md"}, space="prose")[0]
    assert full.group == by_path.group == "high"
    assert full.facets == by_path.facets == {"area": ["engine"]}


def test_path_lookup_finds_the_file_in_each_space(tmp_path: Path) -> None:
    from bc_rag.indexer import sources_for_paths

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
    configuration = space_config(
        groups=[
            SourceGroup(name="prose-docs", space="prose", priority=5, include=["docs/**"]),
            SourceGroup(name="code-docs", space="code", priority=1, include=["docs/**"]),
        ]
    )
    prose = sources_for_paths(tmp_path, configuration, {"docs/a.md"}, space="prose")
    code = sources_for_paths(tmp_path, configuration, {"docs/a.md"}, space="code")
    assert [source.group for source in prose] == ["prose-docs"]
    assert [source.group for source in code] == ["code-docs"]


def test_openapi_glob_loads_json_that_is_not_an_openapi_document(tmp_path: Path) -> None:
    configuration = space_config(
        groups=[
            SourceGroup(
                name="specs",
                space="prose",
                kind="openapi",
                include=["docs/providers/_openapi/*.json"],
            )
        ]
    )
    relative = "docs/providers/_openapi/specs.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"providers": ["olo"], "sources": []}\n', encoding="utf-8")

    language = classify_path(relative, path, configuration, peek=True)

    assert language == "openapi"
