from pathlib import Path

from bc_rag.catalog import register_project, store_dir_for_root
from bc_rag.config import SourceGroup
from tests.support import space_config
from bc_rag.split_md import (
    materialize_openapi_sources,
    split_rel_for,
    split_spec,
)


def _mini_spec(path: Path) -> None:
    path.write_text(
        """
{
  "openapi": "3.0.3",
  "info": {"title": "Sites", "version": "1.0"},
  "paths": {
    "/sites/{id}": {
      "get": {
        "operationId": "getSite",
        "tags": ["Sites"],
        "summary": "Read one site",
        "responses": {"200": {"description": "The site"}}
      },
      "put": {
        "operationId": "writeSite",
        "summary": "Replace site",
        "responses": {"204": {"description": "Written"}}
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )


def test_split_spec_writes_under_store(tmp_path: Path) -> None:
    spec = tmp_path / "docs" / "providers" / "_openapi" / "olo--mini.json"
    spec.parent.mkdir(parents=True)
    _mini_spec(spec)
    store = store_dir_for_root(tmp_path)
    result = split_spec(spec, "docs/providers/_openapi/olo--mini.json", store)

    assert result.operations == 2
    assert result.dest.is_relative_to(store / "openapi-md")
    spec_md = result.dest / "spec.md"
    assert spec_md.is_file()
    body = spec_md.read_text(encoding="utf-8")
    assert "GET /sites/{id}" in body
    assert "PUT /sites/{id}" in body


def test_materialize_indexes_generated_markdown(tmp_path: Path) -> None:
    spec = tmp_path / "olo.json"
    _mini_spec(spec)
    register_project(tmp_path)
    config = space_config(
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
                space="prose",
                include=["olo.json"],
            )
        ]
    )
    source = config  # satisfy linters if unused below
    del source
    from bc_rag.discover import source_for_path

    json_source = source_for_path("olo.json", spec, config)
    assert json_source is not None
    expanded = materialize_openapi_sources(tmp_path, config, [json_source])
    assert len(expanded) == 1
    assert expanded[0].language == "markdown"
    assert expanded[0].path.name == "spec.md"
    assert expanded[0].rel_path.startswith("openapi-md/")
    assert split_rel_for("olo.json", "spec.md") == expanded[0].rel_path
    assert "scope:external" in expanded[0].tags
    assert "provider:olo" in expanded[0].tags
    assert "vendor:olo" in expanded[0].tags
    # the indexer keeps only files whose space matches, so the space must survive.
    assert expanded[0].space == "prose"
    assert expanded[0].space == json_source.space
