from types import SimpleNamespace

from bc_rag.facets import (
    enrich_openapi_chunks,
    operation_from_heading_path,
    spec_slug_from_rel,
)


def test_spec_slug_from_generated_markdown_path() -> None:
    assert (
        spec_slug_from_rel(
            "openapi-md/docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi/spec.md"
        )
        == "ordering-api-1.1.bundle"
    )
    assert spec_slug_from_rel("docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi.json") == (
        "ordering-api-1.1.bundle"
    )


def test_operation_from_heading_path() -> None:
    assert operation_from_heading_path("APIs > POST /baskets/create > Request Body") == (
        "POST",
        "/baskets/create",
    )


def test_enrich_openapi_chunks_stamps_sidecar_operation_keys() -> None:
    chunks = [
        SimpleNamespace(
            heading_path="APIs > POST /baskets/create",
            text="**Operation ID**: `CreateBasket`\n**Tags**: creationRetrievalBasket\n\nCreates a basket.\n",
            tags=["scope:external", "vendor:olo"],
            metadata={"scope": "external", "vendor": "olo"},
        ),
        SimpleNamespace(
            heading_path="APIs > POST /baskets/create > Request Body",
            text="### Request Body\n",
            tags=["scope:external", "vendor:olo"],
            metadata={"scope": "external", "vendor": "olo"},
        ),
    ]
    enrich_openapi_chunks(
        "openapi-md/docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi/spec.md",
        chunks,
    )
    assert "specSlug:ordering-api-1.1.bundle" in chunks[0].tags
    assert "method:POST" in chunks[0].tags
    assert "path:/baskets/create" in chunks[0].tags
    assert "operationId:CreateBasket" in chunks[0].tags
    assert "tag:creationRetrievalBasket" in chunks[0].tags
    assert chunks[1].metadata["method"] == "POST"
    assert chunks[1].metadata["operationId"] == "CreateBasket"
    assert chunks[1].metadata["specSlug"] == "ordering-api-1.1.bundle"
