from pathlib import Path

from bc_rag.config import OpenApiConfig, RagConfig
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

    found = {source.rel_path: source.language for source in iter_source_files(tmp_path, RagConfig())}

    assert found["src/a.ts"] == "typescript"
    assert found["README.md"] == "markdown"
    assert "node_modules/skip.ts" not in found


def test_package_json_is_json_when_included(tmp_path: Path) -> None:
    configuration = RagConfig(include=["**/{package,project}.json"], openapi=OpenApiConfig(enabled=False))
    relative = "libs/engine/flows/engine-flows-locations/package.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"name":"@bigcolony/engine-flows-locations"}\n', encoding="utf-8")

    language = classify_path(relative, path, configuration)

    assert language == "json"


def test_openapi_glob_loads_json_that_is_not_an_openapi_document(tmp_path: Path) -> None:
    configuration = RagConfig(
        openapi=OpenApiConfig(enabled=True, include=["docs/providers/_openapi/*.json"])
    )
    relative = "docs/providers/_openapi/specs.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"providers": ["olo"], "sources": []}\n', encoding="utf-8")

    language = classify_path(relative, path, configuration, peek=True)

    assert language == "openapi"
