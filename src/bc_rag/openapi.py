"""Turn an OpenAPI entrypoint into retrieval markdown.

`prance.ResolvingParser` inlines `$ref` into a plain dict. Cycles are caught
on the first loop via `recursion_limit=1` and `handle_circular_loops`.
Validation errors from vendor specs are ignored after resolve. Each HTTP
operation is copied into a one-path OpenAPI document and handed to
`openapi-markdown`. That markdown is split only if it is longer than the
chunk limit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bc_rag.chunking import Chunk, chunk_file
from bc_rag.discover import SourceFile
from bc_rag.refs import resolve_refs

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options", "trace")
SCHEMA_DUMP_LIMIT = 4000
REDOCLY_TIMEOUT_S = 120


class OpenApiError(RuntimeError):
    pass


def looks_like_openapi(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:8000]
    except OSError:
        return False
    lowered = head.lstrip().lower()
    return (
        lowered.startswith("openapi:")
        or lowered.startswith("swagger:")
        or '"openapi"' in head[:2000]
        or '"swagger"' in head[:2000]
        or "\nopenapi:" in head
        or "\nswagger:" in head
    )


def is_openapi_document(document: Any) -> bool:
    if not isinstance(document, dict):
        return False
    has_version = "openapi" in document or "swagger" in document
    has_operations = "paths" in document or "webhooks" in document
    return has_version and has_operations


def is_spec_catalog(document: Any) -> bool:
    if not isinstance(document, dict):
        return False
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    first = sources[0]
    return isinstance(first, dict) and ("url" in first or "title" in first)


def expand_openapi_file(
    *,
    root: Path,
    source: SourceFile,
    text: str,
    max_chars: int,
    min_chars: int,
) -> list[Chunk]:
    parsed = _parse_document(text, source.path)
    if is_openapi_document(parsed):
        document = dereference_spec(source.path)
        rows = render_operation_documents(document)
        return _chunks_from_markdown_rows(
            source, rows, max_chars, min_chars, kind="openapi-operation"
        )
    if is_spec_catalog(parsed):
        rows = [
            {
                "markdown": render_spec_catalog(parsed, source.rel_path),
                "symbol": "spec-catalog",
                "heading": "OpenAPI spec catalog",
            }
        ]
        return _chunks_from_markdown_rows(
            source, rows, max_chars, min_chars, kind="openapi-catalog"
        )
    rows = [
        {
            "markdown": render_json_document(parsed, source.rel_path),
            "symbol": source.path.name,
            "heading": source.path.name,
        }
    ]
    return _chunks_from_markdown_rows(source, rows, max_chars, min_chars, kind="json")


def _chunks_from_markdown_rows(
    source: SourceFile,
    rows: list[dict[str, str]],
    max_chars: int,
    min_chars: int,
    *,
    kind: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for row in rows:
        parts = chunk_file(
            path=source.path,
            rel_path=source.rel_path,
            language="markdown",
            text=row["markdown"],
            max_chars=max_chars,
            min_chars=min_chars,
        )
        for part in parts:
            part.kind = kind
            part.symbol = row["symbol"]
            part.heading_path = row["heading"]
            part.language = "openapi"
            chunks.append(part)
    return chunks


def render_spec_catalog(document: dict[str, Any], rel_path: str) -> str:
    providers = document.get("providers") if isinstance(document.get("providers"), list) else []
    sources = document.get("sources") if isinstance(document.get("sources"), list) else []
    total = document.get("totalSpecs", len(sources))
    lines = [
        "# OpenAPI spec catalog",
        "",
        f"- File: `{rel_path}`",
        f"- Total specs: {total}",
    ]
    if providers:
        lines.append("- Providers: " + ", ".join(str(name) for name in providers))
    lines.extend(["", "## Specs", ""])
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = str(source.get("title") or source.get("slug") or source.get("url") or "spec")
        lines.append(f"### {title}")
        lines.append("")
        if source.get("slug"):
            lines.append(f"- Slug: `{source['slug']}`")
        if source.get("url"):
            lines.append(f"- File: `{source['url']}`")
        if source.get("default") is True:
            lines.append("- Default: true")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json_document(document: Any, rel_path: str) -> str:
    dumped = json.dumps(document, indent=2, ensure_ascii=False)
    if len(dumped) > SCHEMA_DUMP_LIMIT * 8:
        dumped = dumped[: SCHEMA_DUMP_LIMIT * 8].rstrip() + "\n…"
    return f"# `{rel_path}`\n\n```json\n{dumped}\n```\n"


def _parse_document(text: str, path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    loaded = yaml.safe_load(text)
    return loaded


@dataclass(frozen=True)
class OperationSlice:
    path: str
    method: str
    operation: dict[str, Any]
    shared_params: list[Any]
    symbol: str
    heading: str


def handle_circular_loops(limit, parsed_url, recursions=()):
    """Replace a cyclic `$ref` instead of raising. Invoked at recursion_limit=1."""
    from prance.util.resolver import permissive_object_on_recursion

    return permissive_object_on_recursion(limit, parsed_url, recursions)


def dereference_spec(spec_path: Path) -> dict[str, Any]:
    """Inline `$ref` with ResolvingParser. Keep the dict if validation fails."""
    from prance import ResolvingParser, ValidationError
    from prance.util.resolver import RESOLVE_ALL

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


def render_operation_documents(document: dict[str, Any]) -> list[dict[str, str]]:
    """One OpenAPI document per HTTP method, converted with openapi-markdown."""
    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="bc-rag-op-") as tmp:
        tmp_path = Path(tmp)
        for slice_ in iter_operation_slices(document):
            envelope = operation_envelope(document, slice_)
            try:
                markdown = markdown_from_envelope(envelope, tmp_path)
            except Exception:
                markdown = _stub_operation_markdown(slice_)
            if markdown.strip():
                rows.append(
                    {
                        "markdown": markdown,
                        "symbol": slice_.symbol,
                        "heading": slice_.heading,
                    }
                )
    return rows


def iter_operation_slices(document: dict[str, Any]) -> list[OperationSlice]:
    slices: list[OperationSlice] = []
    for collection in ("paths", "webhooks"):
        items = document.get(collection)
        if not isinstance(items, dict):
            continue
        for path, item in items.items():
            slices.extend(_slices_from_path_item(str(path), item))
    return slices


def _slices_from_path_item(path: str, item: Any) -> list[OperationSlice]:
    if not isinstance(item, dict):
        return []
    shared = item.get("parameters") if isinstance(item.get("parameters"), list) else []
    results: list[OperationSlice] = []
    for method in HTTP_METHODS:
        operation = item.get(method)
        if not isinstance(operation, dict):
            continue
        symbol = f"{method.upper()} {path}"
        heading = str(operation.get("operationId") or operation.get("summary") or symbol)
        results.append(
            OperationSlice(
                path=path,
                method=method,
                operation=operation,
                shared_params=shared,
                symbol=symbol,
                heading=heading,
            )
        )
    return results


def operation_envelope(document: dict[str, Any], slice_: OperationSlice) -> dict[str, Any]:
    info = document.get("info") if isinstance(document.get("info"), dict) else {}
    title = str(info.get("title") or "API")
    version = str(info.get("version") or "0")
    openapi_ver = str(document.get("openapi") or "3.0.3")
    if not openapi_ver.startswith("3"):
        openapi_ver = "3.0.3"
    path_item: dict[str, Any] = {slice_.method: _plain(slice_.operation)}
    if slice_.shared_params:
        path_item["parameters"] = _plain(slice_.shared_params)
    envelope: dict[str, Any] = {
        "openapi": openapi_ver,
        "info": {
            "title": f"{title} {slice_.symbol}",
            "version": version,
        },
        "paths": {slice_.path: path_item},
    }
    servers = document.get("servers")
    if isinstance(servers, list) and servers:
        envelope["servers"] = _plain(servers)
    if _contains_ref(envelope) and isinstance(document.get("components"), dict):
        envelope["components"] = document["components"]
    return envelope


def _contains_ref(node: Any) -> bool:
    if isinstance(node, dict):
        if "$ref" in node:
            return True
        return any(_contains_ref(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_ref(item) for item in node)
    return False


def markdown_from_spec_file(spec_path: Path) -> str:
    """One markdown document for the whole spec.

    `$ref` is inlined with prance (`dereference_spec`). Jinja then walks a
    plain dict. openapi-core `Spec.from_dict` is a lazy SchemaPath accessor,
    not the inliner: missing keys become Undefined and webhook-only specs
    have no `.paths`.
    """
    try:
        parsed = dereference_spec(spec_path)
    except Exception:
        parsed = _load_spec_dict(spec_path)
    text = _render_spec_markdown(parsed)
    if not text.strip():
        raise OpenApiError(f"empty markdown for {spec_path}")
    return text


def markdown_from_envelope(envelope: dict[str, Any], tmp: Path) -> str:
    del tmp
    text = _render_spec_markdown(envelope)
    if not text.strip():
        raise OpenApiError("empty markdown for an operation envelope")
    return text


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


def _plain(node: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()
    if isinstance(node, dict):
        ident = id(node)
        if ident in seen:
            return {"description": "circular reference"}
        seen.add(ident)
        return {str(key): _plain(value, seen) for key, value in node.items()}
    if isinstance(node, list):
        ident = id(node)
        if ident in seen:
            return []
        seen.add(ident)
        return [_plain(item, seen) for item in node]
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    return str(node)


def _stub_operation_markdown(slice_: OperationSlice) -> str:
    operation = slice_.operation
    lines = [f"# {slice_.symbol}", ""]
    operation_id = operation.get("operationId")
    if operation_id:
        lines.append(f"- Operation ID: `{operation_id}`")
    summary = operation.get("summary")
    if summary:
        lines.append(f"- Summary: {summary}")
    description = str(operation.get("description") or "").strip()
    if description:
        lines.extend(["", description])
    return "\n".join(lines).rstrip() + "\n"


def _package_templates_dir() -> Path:
    path = Path(__file__).resolve().parent / "openapi_templates"
    if not path.is_dir():
        raise OpenApiError(f"openapi templates missing at {path}")
    return path


def bundle_spec(root: Path, spec_path: Path) -> dict[str, Any]:
    """Redocly follows split-file $refs. JSON pointer only cleans leftovers."""
    command = _redocly_command()
    with tempfile.TemporaryDirectory(prefix="bc-rag-openapi-") as tmp:
        tmp_path = Path(tmp)
        json_out = tmp_path / "bundled.json"
        yaml_out = tmp_path / "bundled.yaml"
        try:
            _run_redocly(command, root, spec_path, json_out, ext="json")
            return _load_json(json_out)
        except OpenApiError:
            _run_redocly(command, root, spec_path, yaml_out, ext="yaml")
            return _load_yaml(yaml_out)


def render_operations(document: dict[str, Any], rel_path: str) -> list[dict[str, str]]:
    info = document.get("info") if isinstance(document.get("info"), dict) else {}
    api_title = str(info.get("title") or rel_path)
    api_version = str(info.get("version") or "")
    paths = document.get("paths") if isinstance(document.get("paths"), dict) else {}
    webhooks = document.get("webhooks") if isinstance(document.get("webhooks"), dict) else {}
    rows: list[dict[str, str]] = []
    for path, item in paths.items():
        rows.extend(
            _render_path_item(
                document,
                rel_path,
                api_title,
                api_version,
                path,
                item,
                kind="path",
            )
        )
    for name, item in webhooks.items():
        rows.extend(
            _render_path_item(
                document,
                rel_path,
                api_title,
                api_version,
                name,
                item,
                kind="webhook",
            )
        )
    return rows


def _render_path_item(
    document: dict[str, Any],
    rel_path: str,
    api_title: str,
    api_version: str,
    path: str,
    item: Any,
    *,
    kind: str,
) -> list[dict[str, str]]:
    if not isinstance(item, dict):
        return []
    item = resolve_refs(item, document)
    if not isinstance(item, dict):
        return []
    shared_params = item.get("parameters") if isinstance(item.get("parameters"), list) else []
    results: list[dict[str, str]] = []
    for method in HTTP_METHODS:
        operation = item.get(method)
        if not isinstance(operation, dict):
            continue
        operation = resolve_refs(operation, document)
        if not isinstance(operation, dict):
            continue
        markdown, symbol, heading = _operation_markdown(
            document,
            rel_path=rel_path,
            api_title=api_title,
            api_version=api_version,
            path=path,
            method=method.upper(),
            operation=operation,
            shared_params=shared_params,
            kind=kind,
        )
        results.append({"markdown": markdown, "symbol": symbol, "heading": heading})
    return results


def _operation_markdown(
    document: dict[str, Any],
    *,
    rel_path: str,
    api_title: str,
    api_version: str,
    path: str,
    method: str,
    operation: dict[str, Any],
    shared_params: list[Any],
    kind: str,
) -> tuple[str, str, str]:
    operation_id = str(operation.get("operationId") or "")
    summary = str(operation.get("summary") or "")
    symbol = f"{method} {path}"
    heading = operation_id or summary or symbol
    tags = operation.get("tags") if isinstance(operation.get("tags"), list) else []
    lines: list[str] = [
        f"# {symbol}",
        "",
        f"- Spec: `{rel_path}`",
        f"- API: {api_title}" + (f" {api_version}" if api_version else ""),
        f"- Kind: {kind}",
    ]
    if operation_id:
        lines.append(f"- Operation ID: `{operation_id}`")
    if summary:
        lines.append(f"- Summary: {summary}")
    if tags:
        lines.append("- Tags: " + ", ".join(str(tag) for tag in tags))
    if operation.get("deprecated") is True:
        lines.append("- Deprecated: true")
    description = str(operation.get("description") or "").strip()
    if description:
        lines.extend(["", "## Description", "", description])

    parameters = _merge_parameters(shared_params, operation.get("parameters"), document)
    if parameters:
        lines.extend(["", "## Parameters", ""])
        lines.append("| name | in | required | type | description |")
        lines.append("| --- | --- | --- | --- | --- |")
        for param in parameters:
            schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
            ptype = _schema_type(schema)
            required = "yes" if param.get("required") else "no"
            desc = _one_line(str(param.get("description") or ""))
            lines.append(
                f"| `{param.get('name')}` | {param.get('in')} | {required} | `{ptype}` | {desc} |"
            )

    body = operation.get("requestBody")
    if isinstance(body, dict):
        body = resolve_refs(body, document)
        lines.extend(["", "## Request body", ""])
        if body.get("required"):
            lines.append("Required: yes")
            lines.append("")
        desc = str(body.get("description") or "").strip()
        if desc:
            lines.append(desc)
            lines.append("")
        lines.extend(_content_blocks(body.get("content"), document))

    responses = operation.get("responses")
    if isinstance(responses, dict):
        lines.extend(["", "## Responses", ""])
        for status, response in responses.items():
            if not isinstance(response, dict):
                continue
            response = resolve_refs(response, document)
            if not isinstance(response, dict):
                continue
            title = str(response.get("description") or "").strip() or "Response"
            lines.append(f"### {status} {title}")
            lines.append("")
            lines.extend(_content_blocks(response.get("content"), document))

    security = operation.get("security")
    if security is None:
        security = document.get("security")
    if isinstance(security, list) and security:
        lines.extend(["", "## Security", ""])
        for item in security:
            if isinstance(item, dict):
                names = ", ".join(item.keys()) or "(empty)"
                lines.append(f"- {names}")

    servers = operation.get("servers") or document.get("servers")
    if isinstance(servers, list) and servers:
        lines.extend(["", "## Servers", ""])
        for server in servers:
            if isinstance(server, dict) and server.get("url"):
                extra = str(server.get("description") or "").strip()
                suffix = f" {extra}" if extra else ""
                lines.append(f"- `{server['url']}`{suffix}")

    markdown = "\n".join(lines).rstrip() + "\n"
    return markdown, symbol, heading


def _merge_parameters(shared: list[Any], local: Any, document: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for raw in list(shared) + (list(local) if isinstance(local, list) else []):
        if not isinstance(raw, dict):
            continue
        param = resolve_refs(raw, document)
        if not isinstance(param, dict):
            continue
        key = (param.get("name"), param.get("in"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(param)
    return merged


def _content_blocks(content: Any, document: dict[str, Any]) -> list[str]:
    if not isinstance(content, dict) or not content:
        return []
    lines: list[str] = []
    for mime, media in content.items():
        if not isinstance(media, dict):
            continue
        media = resolve_refs(media, document)
        if not isinstance(media, dict):
            continue
        lines.append(f"**{mime}**")
        lines.append("")
        schema = media.get("schema")
        if schema is not None:
            schema = resolve_refs(schema, document)
            dumped = _dump_schema(schema)
            lines.append("```yaml")
            lines.append(dumped)
            lines.append("```")
            lines.append("")
        example = media.get("example")
        if example is None and isinstance(media.get("examples"), dict):
            first = next(iter(media["examples"].values()), None)
            if isinstance(first, dict):
                example = first.get("value")
        if example is not None:
            lines.append("Example:")
            lines.append("")
            lines.append("```json")
            lines.append(_dump_json(example))
            lines.append("```")
            lines.append("")
    return lines


def _schema_type(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return str(schema["$ref"])
    if "type" in schema:
        typ = str(schema["type"])
        if schema.get("format"):
            return f"{typ}:{schema['format']}"
        return typ
    if "oneOf" in schema:
        return "oneOf"
    if "anyOf" in schema:
        return "anyOf"
    if "allOf" in schema:
        return "allOf"
    return "object"


def _dump_schema(schema: Any) -> str:
    text = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True).rstrip()
    if len(text) > SCHEMA_DUMP_LIMIT:
        return text[:SCHEMA_DUMP_LIMIT].rstrip() + "\n# … truncated"
    return text


def _dump_json(value: Any) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if len(text) > SCHEMA_DUMP_LIMIT:
        return text[:SCHEMA_DUMP_LIMIT].rstrip() + "\n…"
    return text


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _redocly_command() -> list[str]:
    if shutil.which("redocly"):
        return ["redocly"]
    if shutil.which("npx"):
        return ["npx", "--yes", "@redocly/cli"]
    raise OpenApiError(
        "Redocly CLI not found. Install @redocly/cli or put `redocly` on PATH. "
        "Split OpenAPI files need `redocly bundle --dereferenced`."
    )


def _run_redocly(
    command: list[str],
    root: Path,
    spec_path: Path,
    output: Path,
    *,
    ext: str,
) -> None:
    args = [
        *command,
        "bundle",
        str(spec_path),
        "--dereferenced",
        "--force",
        "--ext",
        ext,
        "-o",
        str(output),
    ]
    result = subprocess.run(
        args,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=REDOCLY_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise OpenApiError(f"redocly bundle failed for {spec_path}: {detail or 'no output'}")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OpenApiError(f"{path} is not an OpenAPI object")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OpenApiError(f"{path} is not an OpenAPI object")
    return data
