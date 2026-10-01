from types import SimpleNamespace

import pytest

from bc_rag.facets import (
    distinct_facet_values,
    enrich_openapi_chunks,
    facets_line,
    facets_match,
    normalize_facets,
    operation_from_heading_path,
    parse_facet_options,
    spec_slug_from_rel,
)


def test_spec_slug_from_generated_markdown_path() -> None:
    assert (
        spec_slug_from_rel(
            "openapi-md/docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi/spec.md"
        )
        == "ordering-api-1.1.bundle"
    )
    assert spec_slug_from_rel(
        "docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi.json"
    ) == ("ordering-api-1.1.bundle")


def test_operation_from_heading_path() -> None:
    assert operation_from_heading_path("APIs > POST /baskets/create > Request Body") == (
        "POST",
        "/baskets/create",
    )


def test_enrich_sets_every_operation_facet_and_keeps_the_group_facets() -> None:
    chunks = [
        SimpleNamespace(
            heading_path="APIs > POST /baskets/create",
            text=(
                "**Operation ID**: `CreateBasket`\n"
                "**Tags**: creationRetrievalBasket, Baskets\n\nCreates a basket.\n"
            ),
            facets={"vendor": ["olo"], "group": ["olo-spec"]},
        ),
        SimpleNamespace(
            heading_path="APIs > POST /baskets/create > Request Body",
            text="### Request Body\n",
            facets={"vendor": ["olo"], "group": ["olo-spec"]},
        ),
        SimpleNamespace(heading_path="APIs > Models", text="x", facets={"vendor": ["olo"]}),
    ]
    enrich_openapi_chunks(
        "openapi-md/docs/providers/_openapi/olo--ordering-api-1.1.bundle.openapi/spec.md",
        chunks,
    )
    operation = {
        "vendor": ["olo"],
        "group": ["olo-spec"],
        "specSlug": ["ordering-api-1.1.bundle"],
        "method": ["POST"],
        "apiPath": ["/baskets/create"],
        "operationId": ["CreateBasket"],
        "apiTag": ["creationRetrievalBasket", "Baskets"],
    }
    assert chunks[0].facets == operation
    assert chunks[1].facets == operation
    assert chunks[2].facets == {"vendor": ["olo"], "specSlug": ["ordering-api-1.1.bundle"]}


def test_normalize_accepts_strings_and_lists() -> None:
    assert normalize_facets({"area": " engine ", "provider": ["olo", "toast", "olo"]}) == {
        "area": ["engine"],
        "provider": ["olo", "toast"],
    }
    assert normalize_facets(None) == {}


@pytest.mark.parametrize(
    ("raw", "named"),
    [
        ({"1area": "x"}, "'1area'"),
        ({"area:x": "y"}, "'area:x'"),
        ({"area": 3}, "3"),
        ({"area": ["ok", ""]}, "''"),
        ({"area": []}, "no values"),
        (["area:engine"], "must be an object"),
    ],
)
def test_normalize_names_the_bad_key_or_value(raw, named: str) -> None:
    with pytest.raises(ValueError, match=named):
        normalize_facets(raw)


def test_match_is_any_of_within_a_key_and_all_across_keys() -> None:
    have = {"scope": ["internal"], "area": ["engine"], "provider": ["olo", "toast"]}
    assert facets_match(have, {"area": ["engine", "admin"]})
    assert facets_match(have, {"scope": ["internal"], "provider": ["toast"]})
    assert not facets_match(have, {"scope": ["internal"], "area": ["admin"]})
    assert not facets_match(have, None, {"provider": ["olo"]})
    assert facets_match(have, {"scope": ["internal"]}, {"vendor": ["olo"]})


def test_cli_flags_build_the_same_object() -> None:
    assert parse_facet_options(["area=engine", "area=admin", "vendor=olo"], option="--facet") == {
        "area": ["engine", "admin"],
        "vendor": ["olo"],
    }
    with pytest.raises(ValueError, match="key=value"):
        parse_facet_options(["area"], option="--facet")


def test_distinct_values_skip_disabled_groups_and_include_group_names() -> None:
    groups = [
        SimpleNamespace(
            name="eng",
            enabled=True,
            facets={"area": ["engine"], "scope": ["internal"]},
        ),
        SimpleNamespace(
            name="olo",
            enabled=True,
            facets={"scope": ["external"], "vendor": ["olo"]},
        ),
        SimpleNamespace(name="off", enabled=False, facets={"area": ["hidden"]}),
    ]
    assert distinct_facet_values(groups) == {
        "area": ["engine"],
        "group": ["eng", "olo"],
        "scope": ["external", "internal"],
        "vendor": ["olo"],
    }


def test_facets_line() -> None:
    assert facets_line({"area": ["engine"], "provider": ["olo", "toast"]}) == (
        "area=engine, provider=olo|toast"
    )
