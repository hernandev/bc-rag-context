from pathlib import Path

from bc_rag.openapi import markdown_from_spec_file


def test_webhook_only_spec_renders(tmp_path: Path) -> None:
    spec = tmp_path / "hooks.json"
    spec.write_text(
        """
{
  "openapi": "3.1.0",
  "info": {"title": "Webhooks", "version": "2.0"},
  "webhooks": {
    "entitiesWebhook": {
      "post": {
        "operationId": "entitiesWebhook",
        "summary": "Entities: Webhook",
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    body = markdown_from_spec_file(spec)
    assert "Webhooks" in body
    assert "POST entitiesWebhook" in body
    assert "entitiesWebhook" in body


def test_parameter_ref_is_inlined(tmp_path: Path) -> None:
    spec = tmp_path / "ref.json"
    spec.write_text(
        """
{
  "openapi": "3.0.3",
  "info": {"title": "Ref", "version": "1"},
  "paths": {
    "/sites/{id}": {
      "get": {
        "operationId": "getSite",
        "parameters": [{"$ref": "#/components/parameters/SiteId"}],
        "responses": {"200": {"description": "ok"}}
      }
    }
  },
  "components": {
    "parameters": {
      "SiteId": {
        "name": "id",
        "in": "path",
        "required": true,
        "schema": {"type": "string"},
        "description": "site id"
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    body = markdown_from_spec_file(spec)
    assert "GET /sites/{id}" in body
    assert "site id" in body
    assert "| id |" in body


def test_example_values_do_not_need_openapi_core(tmp_path: Path) -> None:
    spec = tmp_path / "ex.json"
    spec.write_text(
        """
{
  "openapi": "3.0.3",
  "info": {"title": "Ex", "version": "1"},
  "paths": {
    "/r": {
      "get": {
        "summary": "Recommend",
        "responses": {
          "200": {
            "description": "ok",
            "content": {
              "application/json": {
                "schema": {"type": "object"},
                "examples": {
                  "one": {"value": {"id": 1, "ok": true}}
                }
              }
            }
          }
        }
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    body = markdown_from_spec_file(spec)
    assert "GET /r" in body
    assert "Recommend" in body
