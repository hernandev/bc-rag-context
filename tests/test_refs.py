from bc_rag.refs import json_pointer, resolve_refs


def test_json_pointer_walks_components() -> None:
    document = {"components": {"schemas": {"Site": {"type": "object"}}}}

    found = json_pointer(document, "#/components/schemas/Site")

    assert found == {"type": "object"}


def test_resolve_refs_inlines_internal_pointers() -> None:
    document = {
        "components": {"schemas": {"Site": {"type": "object", "properties": {"id": {"type": "string"}}}}},
        "schema": {"$ref": "#/components/schemas/Site"},
    }

    resolved = resolve_refs(document["schema"], document)

    assert resolved["type"] == "object"
    assert resolved["properties"]["id"]["type"] == "string"


def test_resolve_refs_leaves_http_refs_alone() -> None:
    node = {"$ref": "https://example.com/openapi.yaml#/components/schemas/Site"}

    resolved = resolve_refs(node, {})

    assert resolved["$ref"].startswith("https://")


def test_resolve_refs_marks_cycles() -> None:
    document = {"components": {"schemas": {"Node": {"$ref": "#/components/schemas/Node"}}}}

    resolved = resolve_refs({"$ref": "#/components/schemas/Node"}, document)

    assert resolved["circular"] is True
