from pathlib import Path

from tests.support import entry_for, space_config

from bc_rag.catalog import store_dir_for_root
from bc_rag.config import SourceGroup
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
    assert result.dest.is_relative_to(store / "cache" / "openapi-md")
    spec_md = result.dest / "spec.md"
    assert spec_md.is_file()
    body = spec_md.read_text(encoding="utf-8")
    assert "GET /sites/{id}" in body
    assert "PUT /sites/{id}" in body


def test_materialize_indexes_generated_markdown(tmp_path: Path) -> None:
    spec = tmp_path / "olo.json"
    _mini_spec(spec)
    entry_for(tmp_path)
    config = space_config(
        groups=[
            SourceGroup(
                name="docs-providers-_openapi-olo",
                kind="openapi",
                facets={"scope": "external", "provider": "olo", "vendor": "olo"},
                space="prose",
                include=["olo.json"],
            )
        ]
    )
    from bc_rag.discover import source_for_path

    json_source = source_for_path("olo.json", spec, config)
    assert json_source is not None
    expanded = materialize_openapi_sources(tmp_path, config, [json_source])
    assert len(expanded) == 1
    assert expanded[0].language == "markdown"
    assert expanded[0].path.name == "spec.md"
    assert expanded[0].rel_path.startswith("openapi-md/")
    assert split_rel_for("olo.json", "spec.md") == expanded[0].rel_path
    assert expanded[0].facets == {"scope": ["external"], "provider": ["olo"], "vendor": ["olo"]}
    assert expanded[0].group == "docs-providers-_openapi-olo"
    # the indexer keeps only files whose space matches, so the space must survive.
    assert expanded[0].space == "prose"
    assert expanded[0].space == json_source.space


def _openapi_config():
    return space_config(
        groups=[SourceGroup(name="api", space="prose", kind="openapi", include=["olo.json"])]
    )


def test_an_edited_spec_is_rendered_again(tmp_path: Path) -> None:
    from bc_rag.discover import source_for_path

    spec = tmp_path / "olo.json"
    _mini_spec(spec)
    config = _openapi_config()
    source = source_for_path("olo.json", spec, config)
    first = materialize_openapi_sources(tmp_path, config, [source])[0]
    assert "/orders" not in first.path.read_text(encoding="utf-8")
    spec.write_text(
        spec.read_text(encoding="utf-8").replace("/sites/{id}", "/orders"), encoding="utf-8"
    )
    second = materialize_openapi_sources(tmp_path, config, [source])[0]
    assert "/orders" in second.path.read_text(encoding="utf-8")


def test_an_unchanged_spec_is_not_rendered_again(tmp_path: Path, monkeypatch) -> None:
    from bc_rag import split_md
    from bc_rag.discover import source_for_path

    spec = tmp_path / "olo.json"
    _mini_spec(spec)
    config = _openapi_config()
    source = source_for_path("olo.json", spec, config)
    materialize_openapi_sources(tmp_path, config, [source])
    calls: list[str] = []
    monkeypatch.setattr(split_md, "split_spec", lambda *args: calls.append("split"))
    materialize_openapi_sources(tmp_path, config, [source])
    assert calls == []


def test_a_broken_spec_is_reported_not_printed(tmp_path: Path, capsys) -> None:
    from bc_rag.discover import source_for_path

    spec = tmp_path / "olo.json"
    spec.write_text("{ not json", encoding="utf-8")
    config = _openapi_config()
    source = source_for_path("olo.json", spec, config)
    errors: list[str] = []
    expanded = materialize_openapi_sources(
        tmp_path, config, [source], on_error=lambda rel, error: errors.append(rel)
    )
    assert expanded == []
    assert errors == ["olo.json"]
    assert capsys.readouterr().out == ""
