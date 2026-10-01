from pathlib import Path

from bc_rag.config import SourceGroup
from tests.support import space_config
from bc_rag.discover import iter_source_files
from bc_rag.nx_tags import NxProjectIndex


def test_nx_project_json_tags_are_copied_onto_source_files(tmp_path: Path) -> None:
    package = tmp_path / "libs" / "engine" / "flows" / "engine-flows-locations"
    source_dir = package / "src"
    source_dir.mkdir(parents=True)
    (package / "project.json").write_text(
        """
{
  "name": "@bigcolony/engine-flows-locations",
  "sourceRoot": "libs/engine/flows/engine-flows-locations/src",
  "tags": ["group:engine", "layer:engine-flow", "type:lib"]
}
""",
        encoding="utf-8",
    )
    (source_dir / "index.ts").write_text("export const ok = 1;\n", encoding="utf-8")
    configuration = space_config(
        groups=[
            SourceGroup(space="prose", 
                name="libs-engine",
                tags=["libs"],
                include=["libs/engine/**/src/**/*.ts"],
                priority=58,
            )
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert len(files) == 1
    assert files[0].group == "libs-engine-flows-engine-flows-locations"
    assert files[0].group == "libs-engine-flows-engine-flows-locations"
    assert files[0].tags == ["group:libs-engine", "libs"]


def test_nx_index_walks_up_to_project_json(tmp_path: Path) -> None:
    package = tmp_path / "libs" / "common" / "types" / "pkg"
    nested = package / "src" / "services"
    nested.mkdir(parents=True)
    (package / "project.json").write_text(
        '{"name": "@bigcolony/common-types-pkg", "tags": ["group:common"]}\n',
        encoding="utf-8",
    )
    file_path = nested / "Location.ts"
    file_path.write_text("export {}\n", encoding="utf-8")
    index = NxProjectIndex(tmp_path)
    project = index.project_for(file_path)
    assert project is not None
    assert project.canonical == "libs-common-types-pkg"
    assert project.package_name == "@bigcolony/common-types-pkg"
    assert index.tags_for(file_path) == [
        "libs-common-types-pkg",
        "@bigcolony/common-types-pkg",
        "package:@bigcolony/common-types-pkg",
        "group:common",
        "common",
        "area:common",
    ]


def test_nx_vendor_colon_tags_are_not_copied(tmp_path: Path) -> None:
    package = tmp_path / "libs" / "vendor" / "providers" / "olo" / "vendor-providers-olo-types"
    source_dir = package / "src"
    source_dir.mkdir(parents=True)
    (package / "project.json").write_text(
        """
{
  "name": "@bigcolony/vendor-providers-olo-types",
  "tags": ["group:vendor", "layer:vendor-types", "vendor:olo"]
}
""",
        encoding="utf-8",
    )
    (source_dir / "index.ts").write_text("export const ok = 1;\n", encoding="utf-8")
    configuration = space_config(
        groups=[
            SourceGroup(space="prose", 
                name="libs-vendor",
                include=["libs/vendor/**/src/**/*.ts"],
                priority=56,
            )
        ],
    )
    files = list(iter_source_files(tmp_path, configuration))
    assert len(files) == 1
    assert "vendor:olo" not in files[0].tags
