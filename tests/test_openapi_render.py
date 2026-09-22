import json
from pathlib import Path

from bc_rag.discover import SourceFile
from bc_rag.openapi import (
    expand_openapi_file,
    is_spec_catalog,
    render_operations,
    render_spec_catalog,
)


def test_render_operations_one_document_per_method() -> None:
    document = {
        "info": {"title": "Sites", "version": "1.0"},
        "paths": {
            "/sites/{id}": {
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "get": {
                    "operationId": "getSite",
                    "summary": "Read one site",
                    "description": "Return the site overlay for a location.",
                    "responses": {"200": {"description": "The site"}},
                },
                "put": {
                    "operationId": "writeSite",
                    "summary": "Replace site overlay keys",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                }
                            }
                        },
                    },
                    "responses": {"204": {"description": "Written"}},
                },
            }
        },
    }

    operations = render_operations(document, "vendors/acme/openapi.yaml")

    assert len(operations) == 2
    symbols = {row["symbol"] for row in operations}
    assert symbols == {"GET /sites/{id}", "PUT /sites/{id}"}
    put = next(row for row in operations if row["symbol"].startswith("PUT"))
    assert "Replace site overlay keys" in put["markdown"]
    assert "operationId" in put["markdown"] or "writeSite" in put["markdown"]
    assert "```yaml" in put["markdown"]


def test_specs_json_catalog_is_loaded_without_redocly() -> None:
    catalog = {
        "providers": ["olo", "toast"],
        "totalSpecs": 2,
        "sources": [
            {"title": "Olo Ordering", "slug": "olo-ordering", "url": "./olo--ordering.json"},
            {"title": "Toast Orders", "slug": "toast-orders", "url": "./toast--toast-api-orders.json"},
        ],
    }
    assert is_spec_catalog(catalog) is True
    markdown = render_spec_catalog(catalog, "docs/providers/_openapi/specs.json")
    assert "Olo Ordering" in markdown
    assert "olo--ordering.json" in markdown
    source = SourceFile(
        path=Path("docs/providers/_openapi/specs.json"),
        rel_path="docs/providers/_openapi/specs.json",
        language="openapi",
        size=100,
    )

    chunks = expand_openapi_file(
        root=Path("."),
        source=source,
        text=json.dumps(catalog),
        max_chars=16000,
        min_chars=10,
    )
    assert chunks
    assert any("Olo Ordering" in chunk.text for chunk in chunks)
    assert chunks[0].kind == "openapi-catalog"


def test_expand_openapi_spec_one_document_per_method(tmp_path: Path) -> None:
    spec_path = tmp_path / "sites.json"
    spec_path.write_text(
        json.dumps(
            {
                "openapi": "3.0.3",
                "info": {"title": "Sites", "version": "1.0"},
                "paths": {
                    "/sites/{id}": {
                        "get": {
                            "operationId": "getSite",
                            "summary": "Read one site",
                            "responses": {"200": {"description": "The site"}},
                        },
                        "put": {
                            "operationId": "writeSite",
                            "summary": "Replace site overlay keys",
                            "responses": {"204": {"description": "Written"}},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    source = SourceFile(
        path=spec_path,
        rel_path="sites.json",
        language="openapi",
        size=spec_path.stat().st_size,
    )

    chunks = expand_openapi_file(
        root=tmp_path,
        source=source,
        text=spec_path.read_text(encoding="utf-8"),
        max_chars=16000,
        min_chars=10,
    )

    assert chunks
    assert all(chunk.kind == "openapi-operation" for chunk in chunks)
    symbols = {chunk.symbol for chunk in chunks}
    assert "GET /sites/{id}" in symbols
    assert "PUT /sites/{id}" in symbols


def test_expand_inlines_jsonref_component(tmp_path: Path) -> None:
    spec_path = tmp_path / "pets.json"
    spec_path.write_text(
        json.dumps(
            {
                "openapi": "3.0.3",
                "info": {"title": "Pets", "version": "1.0"},
                "paths": {
                    "/pets": {
                        "get": {
                            "operationId": "listPets",
                            "responses": {
                                "200": {
                                    "description": "A pet",
                                    "content": {
                                        "application/json": {
                                            "schema": {"$ref": "#/components/schemas/Pet"}
                                        }
                                    },
                                }
                            },
                        }
                    }
                },
                "components": {
                    "schemas": {
                        "Pet": {
                            "type": "object",
                            "properties": {
                                "nickname": {"type": "string"},
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    source = SourceFile(
        path=spec_path,
        rel_path="pets.json",
        language="openapi",
        size=spec_path.stat().st_size,
    )

    chunks = expand_openapi_file(
        root=tmp_path,
        source=source,
        text=spec_path.read_text(encoding="utf-8"),
        max_chars=16000,
        min_chars=10,
    )

    joined = "\n".join(chunk.text for chunk in chunks)
    assert "nickname" in joined
    assert "$ref" not in joined or "Pet" in joined
