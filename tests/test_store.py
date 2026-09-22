from bc_rag.store import _payload_filter


def test_payload_filter_requires_every_sidecar_key() -> None:
    filt = _payload_filter(
        groups=None,
        tags=["vendor:olo", "system:vendor-provider", "scope:external"],
    )
    assert filt is not None
    assert [condition.key for condition in filt.must] == ["vendor", "system", "scope"]
    assert [condition.match.value for condition in filt.must] == [
        "olo",
        "vendor-provider",
        "external",
    ]


def test_payload_filter_groups_remain_any_of() -> None:
    filt = _payload_filter(groups=["docs-providers-_openapi-olo", "docs-providers-_openapi-klaviyo"], tags=None)
    assert filt is not None
    assert len(filt.must) == 1
    assert filt.must[0].key == "group"
    assert list(filt.must[0].match.any) == [
        "docs-providers-_openapi-olo",
        "docs-providers-_openapi-klaviyo",
    ]


def test_payload_filter_none_when_empty() -> None:
    assert _payload_filter(groups=None, tags=None) is None
    assert _payload_filter(groups=[], tags=[]) is None
    assert _payload_filter(groups=None, tags=[""]) is None
