from bc_rag.store import _payload_filter


def test_payload_filter_requires_every_sidecar_key() -> None:
    filt = _payload_filter(
        groups=None,
        tags=[["vendor:olo", "system:vendor-provider", "scope:external"]],
    )
    assert filt is not None
    assert [condition.key for condition in filt.must] == ["vendor", "system", "scope"]
    assert [list(condition.match.any) for condition in filt.must] == [
        ["olo"],
        ["vendor-provider"],
        ["external"],
    ]


def test_outer_list_is_or_and_nested_list_is_and() -> None:
    filt = _payload_filter(
        groups=None,
        tags=["vendor:olo", ["area:engine", "system:bigcolony-reference"]],
    )
    assert filt is not None
    assert len(filt.must) == 1
    alternatives = filt.must[0].should
    assert alternatives[0].must[0].key == "vendor"
    assert list(alternatives[0].must[0].match.any) == ["olo"]
    assert [condition.key for condition in alternatives[1].must] == ["area", "system"]


def test_payload_filter_groups_remain_any_of() -> None:
    filt = _payload_filter(groups=["docs-providers-_openapi-olo", "docs-providers-_openapi-klaviyo"], tags=None)
    assert filt is not None
    assert len(filt.must) == 1
    assert filt.must[0].key == "group"
    assert list(filt.must[0].match.any) == [
        "docs-providers-_openapi-olo",
        "docs-providers-_openapi-klaviyo",
    ]


def test_group_name_filters_as_a_tag() -> None:
    filt = _payload_filter(groups=None, tags=[["group:docs-reference-02-engine", "-group:docs-loose"]])
    assert filt is not None
    assert filt.must[0].key == "group"
    assert list(filt.must[0].match.any) == ["docs-reference-02-engine"]
    assert filt.must_not[0].key == "group"
    assert list(filt.must_not[0].match.any) == ["docs-loose"]


def test_leading_dash_excludes_a_tag_value() -> None:
    filt = _payload_filter(
        groups=None,
        tags=[["scope:internal", "-vendor:olo", "-vendor:yext"]],
    )
    assert filt is not None
    assert [condition.key for condition in filt.must] == ["scope"]
    assert list(filt.must[0].match.any) == ["internal"]
    assert [condition.key for condition in filt.must_not] == ["vendor"]
    assert list(filt.must_not[0].match.any) == ["olo", "yext"]


def test_payload_filter_none_when_empty() -> None:
    assert _payload_filter(groups=None, tags=None) is None
    assert _payload_filter(groups=[], tags=[]) is None
    assert _payload_filter(groups=None, tags=[""]) is None
