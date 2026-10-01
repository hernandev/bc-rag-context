"""Render one OpenAPI spec file to one markdown document.

`prance.ResolvingParser` inlines `$ref` into a plain dict. Cycles are caught on the
first loop through `recursion_limit=1` and `handle_circular_loops`. Validation
errors from vendor specs are ignored after resolving. When prance cannot read the
file at all, the raw JSON or YAML is rendered as it is.

The Jinja templates in `openapi_templates/` write one `## METHOD /path` section per
operation, which the markdown chunker splits on and `facets.enrich_openapi_chunks`
reads the operation facets from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class OpenApiError(RuntimeError):
    pass


def markdown_from_spec_file(spec_path: Path) -> str:
    """One markdown document for the whole spec. Raises OpenApiError when it is empty."""
    try:
        parsed = dereference_spec(spec_path)
    except Exception:
        parsed = _load_spec_dict(spec_path)
    text = _render_spec_markdown(parsed)
    if not text.strip():
        raise OpenApiError(f"empty markdown for {spec_path}")
    return text


def handle_circular_loops(limit, parsed_url, recursions=()):
    """Replace a cyclic `$ref` instead of raising. Invoked at recursion_limit=1."""
    from prance.util.resolver import permissive_object_on_recursion

    return permissive_object_on_recursion(limit, parsed_url, recursions)


def dereference_spec(spec_path: Path) -> dict[str, Any]:
    """Inline `$ref` with ResolvingParser. Keep the dict if validation fails."""
    from prance import ResolvingParser, ValidationError
    from prance.util import url as prance_url
    from prance.util.resolver import RESOLVE_ALL

    # prance keeps every fetched file in a process-wide cache (a mutable default
    # argument). A long-running indexer would render an edited spec from that stale
    # copy, so the cache is emptied before each spec is read.
    for fetch in (prance_url.fetch_url, prance_url.fetch_url_text):
        defaults = fetch.__defaults__ or ()
        if defaults and isinstance(defaults[0], dict):
            defaults[0].clear()
    parser = ResolvingParser(
        str(spec_path.resolve()),
        lazy=True,
        recursion_limit=1,
        recursion_limit_handler=handle_circular_loops,
        resolve_types=RESOLVE_ALL,
    )
    try:
        parser.parse()
    except ValidationError:
        pass
    document = parser.specification
    if not isinstance(document, dict):
        raise OpenApiError(f"prance did not return an object for {spec_path}")
    return document


def _load_spec_dict(spec_path: Path) -> dict[str, Any]:
    raw = spec_path.read_text(encoding="utf-8")
    if spec_path.suffix.lower() in {".yaml", ".yml"}:
        parsed = yaml.safe_load(raw)
    else:
        parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise OpenApiError(f"{spec_path} is not an OpenAPI object")
    return parsed


def _json_default(value: Any) -> Any:
    name = type(value).__name__
    if name in {"Undefined", "Void"}:
        return None
    return str(value)


def _to_json_safe(value: Any) -> str:
    return json.dumps(value, indent=2, default=_json_default, ensure_ascii=False)


def _package_templates_dir() -> Path:
    path = Path(__file__).resolve().parent / "openapi_templates"
    if not path.is_dir():
        raise OpenApiError(f"openapi templates missing at {path}")
    return path


def _render_spec_markdown(spec_data: dict[str, Any]) -> str:
    from jinja2 import Environment, FileSystemLoader
    from openapi_markdown.generator import ref_to_link, ref_to_param, ref_to_schema

    spec = dict(spec_data)
    spec.setdefault("info", {})
    spec.setdefault("servers", [])
    spec.setdefault("paths", {})
    spec.setdefault("webhooks", {})
    if spec["paths"] is None:
        spec["paths"] = {}
    if spec["webhooks"] is None:
        spec["webhooks"] = {}
    env = Environment(
        loader=FileSystemLoader(str(_package_templates_dir())),
        autoescape=False,
    )
    env.filters["to_json"] = _to_json_safe
    env.filters["ref_to_link"] = ref_to_link
    env.filters["ref_to_param"] = lambda ref: ref_to_param(ref, spec)
    env.filters["ref_to_schema"] = lambda schema: ref_to_schema(schema, spec)
    template = env.get_template("api_doc_template.md.j2")
    return template.render(
        spec=spec,
        ref_to_param=lambda ref: ref_to_param(ref, spec),
        ref_to_schema=lambda ref: ref_to_schema(ref, spec),
    )
