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
    assert "apiPath:/baskets/create" in chunks[0].tags
    assert chunks[0].metadata["apiPath"] == "/baskets/create"
    assert "path" not in chunks[0].metadata
    assert "operationId:CreateBasket" in chunks[0].tags
    assert "tag:creationRetrievalBasket" in chunks[0].tags
    assert chunks[1].metadata["method"] == "POST"
    assert chunks[1].metadata["operationId"] == "CreateBasket"
    assert chunks[1].metadata["specSlug"] == "ordering-api-1.1.bundle"
    assert chunks[1].metadata["apiPath"] == "/baskets/create"


def test_distinct_tag_values_skip_disabled_groups() -> None:
    from types import SimpleNamespace

    from bc_rag.facets import distinct_tag_values

    groups = [
        SimpleNamespace(enabled=True, tags=["scope:internal", "area:engine", "system:bigcolony-reference"]),
        SimpleNamespace(enabled=True, tags=["scope:external", "vendor:olo", "system:vendor-provider"]),
        SimpleNamespace(enabled=False, tags=["scope:internal", "area:hidden"]),
    ]
    assert distinct_tag_values(groups) == [
        ("area", ["engine"]),
        ("scope", ["external", "internal"]),
        ("system", ["bigcolony-reference", "vendor-provider"]),
        ("vendor", ["olo"]),
    ]


def test_tag_filter_is_any_of_within_a_key() -> None:
    from bc_rag.facets import tags_match

    have = ["scope:internal", "system:bigcolony-reference", "area:engine"]
    assert tags_match(have, ["scope:internal"])
    assert tags_match(have, ["area:engine", "area:admin"])
    assert not tags_match(have, ["scope:external"])
    assert tags_match(have, ["a:b", ["scope:internal", "area:engine"]])
    assert not tags_match(have, [["scope:internal", "area:admin"]])
    assert not tags_match(have, [["scope:internal", "-area:engine"]])
    assert tags_match(have, [["scope:internal", "-vendor:olo"]])


def test_api_route_does_not_replace_the_file_path() -> None:
    from bc_rag.chunking import Chunk
    from bc_rag.store import _payload

    chunk = Chunk(
        path="docs/providers/_openapi/yext/accounts.md",
        language="markdown",
        kind="markdown",
        symbol=None,
        heading_path="APIs > POST /accounts/{accountId}/newlocationaddrequests",
        start_line=10,
        end_line=40,
        start_byte=0,
        end_byte=20,
        text="creates a location",
        tags=["apiPath:/accounts/{accountId}/newlocationaddrequests"],
        metadata={"path": "/accounts/{accountId}/newlocationaddrequests", "vendor": "yext"},
    )
    payload = _payload(chunk)
    assert payload["path"] == "docs/providers/_openapi/yext/accounts.md"
    assert payload["apiPath"] == "/accounts/{accountId}/newlocationaddrequests"
    assert payload["vendor"] == "yext"
