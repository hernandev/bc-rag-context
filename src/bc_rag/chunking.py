"""Split source files into retrieval units.

TypeScript and TSX go through tree-sitter so a function or class stays whole.
Markdown splits on headings and keeps a breadcrumb of the section path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from bc_rag.defaults import CODE_LANGUAGES, JSON_LANGUAGE, MARKDOWN_LANGUAGES
from bc_rag.facets import facets_line

HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*)$")
FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
THEMATIC_RE = re.compile(
    r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$"
)
# Pandoc `-t plain` prints a horizontal rule as a line of hyphens, often 72,
# and may indent it. A line that is only dashes, stars, or underscores splits.
PLAIN_RULE_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")
DETAILS_OPEN_RE = re.compile(r"^ {0,3}<details\b", re.IGNORECASE)
DETAILS_CLOSE_RE = re.compile(r"^ {0,3}</details\s*>", re.IGNORECASE)
SUMMARY_RE = re.compile(
    r"<summary>\s*(?P<at>\d{4}-\d{2}-\d{2}T[^<\s]+)\s+-\s+(?P<role>user|agent)\s*:\s*(?P<preview>.*?)</summary>",
    re.IGNORECASE | re.DOTALL,
)

STRUCTURAL_TYPES: frozenset[str] = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "function",
        "class_declaration",
        "abstract_class_declaration",
        "class",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
        "public_field_definition",
        "internal_module",
        "module",
        "ambient_declaration",
        "lexical_declaration",
        "export_statement",
        "function_definition",
        "class_definition",
        "decorated_definition",
    }
)

CONTAINER_TYPES: frozenset[str] = frozenset(
    {
        "program",
        "module",
        "class_body",
        "interface_body",
        "statement_block",
        "object_type",
        "enum_body",
        "declaration_list",
        "block",
    }
)

SKIP_CHILD_TYPES: frozenset[str] = frozenset(
    {
        "{",
        "}",
        "(",
        ")",
        "[",
        "]",
        ";",
        ",",
        "comment",
        "html_comment",
    }
)

KIND_BY_TYPE: dict[str, str] = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "function": "function",
    "function_definition": "function",
    "method_definition": "method",
    "method_signature": "method",
    "abstract_method_signature": "method",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "class": "class",
    "class_definition": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "internal_module": "namespace",
    "module": "namespace",
    "lexical_declaration": "binding",
    "export_statement": "export",
    "public_field_definition": "field",
    "ambient_declaration": "ambient",
    "decorated_definition": "definition",
}


@dataclass(slots=True)
class Chunk:
    path: str
    language: str
    kind: str
    symbol: str | None
    heading_path: str | None
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str
    # key -> values. The group's facets, then the keys bc-rag sets (group, apiPath, ...).
    facets: dict[str, list[str]] = field(default_factory=dict)
    group: str | None = None
    priority: int = 0

    def embed_text(self) -> str:
        """Dense embedding input. Path, symbol and facets ride along as context."""
        header = [f"File: {self.path}"]
        if self.symbol:
            header.append(f"Symbol: {self.symbol}")
        if self.kind:
            header.append(f"Kind: {self.kind}")
        if self.heading_path:
            header.append(f"Section: {self.heading_path}")
        if self.facets:
            header.append("Facets: " + facets_line(self.facets))
        return "\n".join(header) + "\n\n" + self.text

    def sparse_text(self) -> str:
        """BM25 input. Raw source only so File:/Kind: prefixes do not pollute IDF."""
        return self.text


def chunk_file(
    *,
    path: Path,
    rel_path: str,
    language: str,
    text: str,
    max_chars: int,
    min_chars: int,
) -> list[Chunk]:
    if language in MARKDOWN_LANGUAGES:
        chunks = _chunk_markdown(rel_path, language, text, max_chars)
    elif language in CODE_LANGUAGES:
        chunks = _chunk_code(rel_path, language, text, max_chars)
    elif language == JSON_LANGUAGE:
        chunks = _chunk_json(path, rel_path, text, max_chars)
    else:
        chunks = _chunk_lines(rel_path, language, "file", None, None, text, 0, max_chars)

    return [c for c in chunks if c.text.strip() and len(c.text) >= min_chars] or [
        c for c in chunks if c.text.strip()
    ]


def _chunk_json(path: Path, rel_path: str, text: str, max_chars: int) -> list[Chunk]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _chunk_lines(rel_path, JSON_LANGUAGE, "json", path.name, None, text, 0, max_chars)
    name = path.name
    if name == "package.json" and isinstance(data, dict):
        rendered = _render_package_json(rel_path, data)
        kind = "package-json"
        symbol = str(data.get("name") or rel_path)
    elif name == "project.json" and isinstance(data, dict):
        rendered = _render_nx_project_json(rel_path, data)
        kind = "nx-project"
        symbol = str(data.get("name") or rel_path)
    else:
        dumped = json.dumps(data, indent=2, ensure_ascii=False)
        rendered = f"# `{rel_path}`\n\n```json\n{dumped}\n```\n"
        kind = "json"
        symbol = name
    if len(rendered) <= max_chars:
        return [
            Chunk(
                path=rel_path,
                language=JSON_LANGUAGE,
                kind=kind,
                symbol=symbol,
                heading_path=symbol,
                start_line=1,
                end_line=rendered.count("\n") + 1,
                start_byte=0,
                end_byte=len(rendered.encode("utf-8")),
                text=rendered,
            )
        ]
    return _chunk_markdown(rel_path, JSON_LANGUAGE, rendered, max_chars)


def _render_package_json(rel_path: str, data: dict) -> str:
    lines = [
        f"# package.json `{data.get('name') or rel_path}`",
        "",
        f"- File: `{rel_path}`",
    ]
    if data.get("version"):
        lines.append(f"- Version: {data['version']}")
    if data.get("description"):
        lines.append(f"- Description: {data['description']}")
    if data.get("private") is True:
        lines.append("- Private: true")
    lines.extend(_named_list("Dependencies", data.get("dependencies")))
    lines.extend(_named_list("Dev dependencies", data.get("devDependencies")))
    lines.extend(_named_list("Peer dependencies", data.get("peerDependencies")))
    scripts = data.get("scripts")
    if isinstance(scripts, dict) and scripts:
        lines.extend(["", "## Scripts", ""])
        for script_name, command in scripts.items():
            lines.append(f"- `{script_name}`: {command}")
    return "\n".join(lines).rstrip() + "\n"


def _render_nx_project_json(rel_path: str, data: dict) -> str:
    lines = [
        f"# Nx project `{data.get('name') or rel_path}`",
        "",
        f"- File: `{rel_path}`",
    ]
    for key in ("sourceRoot", "projectType", "root"):
        if data.get(key):
            lines.append(f"- {key}: `{data[key]}`")
    tags = data.get("tags")
    if isinstance(tags, list) and tags:
        lines.append("- Tags: " + ", ".join(str(tag) for tag in tags))
    targets = data.get("targets")
    if isinstance(targets, dict) and targets:
        lines.extend(["", "## Targets", ""])
        for target_name, target in targets.items():
            executor = ""
            if isinstance(target, dict) and target.get("executor"):
                executor = f" (`{target['executor']}`)"
            lines.append(f"- `{target_name}`{executor}")
    return "\n".join(lines).rstrip() + "\n"


def _named_list(title: str, values: object) -> list[str]:
    if not isinstance(values, dict) or not values:
        return []
    lines = ["", f"## {title}", ""]
    for package_name, version in values.items():
        lines.append(f"- `{package_name}`: {version}")
    return lines


def _chunk_code(rel_path: str, language: str, text: str, max_chars: int) -> list[Chunk]:
    source = text.encode("utf-8")
    try:
        parser = _parser_for(language)
        tree = parser.parse(source)
        root = tree.root_node
    except Exception:
        return _chunk_lines(rel_path, language, "file", None, None, text, 0, max_chars)

    chunks = _walk_code(root, source, text, rel_path, language, max_chars)
    if not chunks:
        return _chunk_lines(rel_path, language, "file", None, None, text, 0, max_chars)
    return chunks


@lru_cache(maxsize=16)
def _parser_for(language: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(language)


def _walk_code(
    node, source: bytes, text: str, rel_path: str, language: str, max_chars: int
) -> list[Chunk]:
    size = node.end_byte - node.start_byte
    if node.type in CONTAINER_TYPES or (size > max_chars and node.children):
        return _bundle_children(node, source, text, rel_path, language, max_chars)
    if size > max_chars:
        return _chunk_lines(
            rel_path,
            language,
            _kind_for(node),
            _symbol_for(node, source),
            None,
            _node_text(node, source),
            node.start_point[0],
            max_chars,
            start_byte=node.start_byte,
        )
    return [_make_code_chunk(node, source, rel_path, language)]


def _bundle_children(
    node, source: bytes, text: str, rel_path: str, language: str, max_chars: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    batch: list = []
    batch_size = 0

    def flush() -> None:
        nonlocal batch, batch_size
        if not batch:
            return
        if len(batch) == 1 and (batch[0].end_byte - batch[0].start_byte) > max_chars:
            chunks.extend(_walk_code(batch[0], source, text, rel_path, language, max_chars))
        else:
            chunks.append(_make_span_chunk(batch, source, rel_path, language))
        batch = []
        batch_size = 0

    for child in node.children:
        if child.type in SKIP_CHILD_TYPES:
            continue
        csize = child.end_byte - child.start_byte
        if csize == 0:
            continue
        # Definitions stay their own retrieval units even when they would fit together.
        if child.type in STRUCTURAL_TYPES:
            flush()
            if csize > max_chars:
                chunks.extend(_walk_code(child, source, text, rel_path, language, max_chars))
            else:
                chunks.append(_make_code_chunk(child, source, rel_path, language))
            continue
        if csize > max_chars:
            flush()
            chunks.extend(_walk_code(child, source, text, rel_path, language, max_chars))
            continue
        if batch and batch_size + csize > max_chars:
            flush()
        batch.append(child)
        batch_size += csize

    flush()
    return chunks


def _leading_comment(node):
    """Comment siblings written above a declaration belong to that declaration."""
    start = node
    current = node
    while True:
        prev = current.prev_sibling
        if prev is None or prev.type not in ("comment", "html_comment"):
            break
        start = prev
        current = prev
    return start


def _make_code_chunk(node, source: bytes, rel_path: str, language: str) -> Chunk:
    start = _leading_comment(node)
    return Chunk(
        path=rel_path,
        language=language,
        kind=_kind_for(node),
        symbol=_symbol_for(node, source),
        heading_path=None,
        start_line=start.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        start_byte=start.start_byte,
        end_byte=node.end_byte,
        text=source[start.start_byte : node.end_byte].decode("utf-8", errors="replace"),
    )


def _make_span_chunk(nodes: list, source: bytes, rel_path: str, language: str) -> Chunk:
    first, last = nodes[0], nodes[-1]
    start = _leading_comment(first)
    symbol = _symbol_for(first, source) if len(nodes) == 1 else None
    kind = _kind_for(first) if len(nodes) == 1 else "block"
    return Chunk(
        path=rel_path,
        language=language,
        kind=kind,
        symbol=symbol,
        heading_path=None,
        start_line=start.start_point[0] + 1,
        end_line=last.end_point[0] + 1,
        start_byte=start.start_byte,
        end_byte=last.end_byte,
        text=source[start.start_byte : last.end_byte].decode("utf-8", errors="replace"),
    )


def _kind_for(node) -> str:
    return KIND_BY_TYPE.get(node.type, node.type)


def _symbol_for(node, source: bytes) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return _node_text(name, source)
    if node.type == "export_statement" and node.named_children:
        return _symbol_for(node.named_children[0], source)
    if node.type == "lexical_declaration":
        for child in node.named_children:
            if child.type == "variable_declarator":
                ident = child.child_by_field_name("name")
                if ident is not None:
                    return _node_text(ident, source)
    if node.type == "public_field_definition":
        ident = node.child_by_field_name("name")
        if ident is not None:
            return _node_text(ident, source)
    return None


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _opening_fence(line: str) -> tuple[str, int] | None:
    match = FENCE_OPEN_RE.match(line.rstrip("\r\n"))
    if match is None:
        return None
    marker = match.group(2)
    info = match.group(3)
    if marker[0] == "`" and "`" in info:
        return None
    return marker[0], len(marker)


def _closing_fence(line: str, char: str, length: int) -> bool:
    raw = line.rstrip("\r\n")
    return re.match(rf"^ {{0,3}}{re.escape(char)}{{{length},}}\s*$", raw) is not None


def _chunk_markdown(rel_path: str, language: str, text: str, max_chars: int) -> list[Chunk]:
    sections: list[tuple[str | None, int, list[str]]] = []
    breadcrumb: list[str] = []
    current_title: str | None = None
    current_start = 1
    current_lines: list[str] = []
    fence: tuple[str, int] | None = None
    details_depth = 0

    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    def flush(at_line: int) -> None:
        nonlocal current_title, current_start, current_lines
        if current_lines or current_title:
            sections.append((current_title, current_start, current_lines))
        current_title = None
        current_start = at_line
        current_lines = []

    for index, line in enumerate(lines, start=1):
        raw = line.rstrip("\r\n")
        if fence is not None:
            current_lines.append(line)
            if _closing_fence(raw, fence[0], fence[1]):
                fence = None
            continue
        opened = _opening_fence(line)
        if opened is not None:
            fence = opened
            current_lines.append(line)
            continue
        if THEMATIC_RE.match(raw) or PLAIN_RULE_RE.match(raw.strip()):
            flush(index + 1)
            continue
        if DETAILS_OPEN_RE.match(raw):
            if details_depth == 0:
                flush(index)
            details_depth += 1
            current_lines.append(line)
            continue
        if details_depth > 0 and DETAILS_CLOSE_RE.match(raw):
            current_lines.append(line)
            details_depth -= 1
            if details_depth == 0:
                flush(index + 1)
            continue
        heading = None if details_depth else HEADING_RE.match(raw)
        if heading is None:
            current_lines.append(line)
            continue
        if current_lines or current_title:
            sections.append((current_title, current_start, current_lines))
        level = len(heading.group(1))
        title = heading.group(2).strip()
        breadcrumb = breadcrumb[: level - 1] + [title]
        current_title = " > ".join(breadcrumb)
        current_start = index
        current_lines = [line]

    if current_lines or current_title:
        sections.append((current_title, current_start, current_lines))

    chunks: list[Chunk] = []
    byte_cursor = 0
    doc_title: str | None = None
    last_user_body: str | None = None
    last_user_summary: str | None = None
    # Recompute byte offsets from the original text by walking sections in order.
    for title, sec_start, sec_lines in sections:
        body = "".join(sec_lines)
        if not body.strip():
            byte_cursor += len(body.encode("utf-8"))
            continue
        if title and " > " not in title and body.lstrip().startswith("# "):
            doc_title = title
        summary = SUMMARY_RE.search(body)
        summary_line = ""
        role = ""
        at = ""
        preview = ""
        if summary is not None:
            at = summary.group("at")
            role = summary.group("role").lower()
            preview = " ".join(summary.group("preview").split())
            summary_line = f"<summary>{at} - {role}: {preview}</summary>"
        heading_path = title
        if summary is not None:
            turn = f"{at} - {role}: {preview}"
            heading_path = f"{doc_title} > {turn}" if doc_title else turn
        parts = _split_text(body, max_chars)
        local = 0
        for index, part in enumerate(parts):
            text_out = part
            if summary is not None and role == "agent" and last_user_body:
                if index == 0:
                    text_out = last_user_body.rstrip() + "\n\n" + part
                else:
                    lead = [summary_line]
                    if last_user_summary:
                        lead.append(f"User: {last_user_summary}")
                    text_out = "\n\n".join(lead) + "\n\n" + part
            elif index > 0 and summary_line and summary_line not in part:
                text_out = summary_line + "\n\n" + part
            part_start_line = sec_start + body[:local].count("\n")
            part_end_line = part_start_line + max(part.count("\n"), 0)
            start_byte = byte_cursor + len(body[:local].encode("utf-8"))
            end_byte = start_byte + len(part.encode("utf-8"))
            facets: dict[str, list[str]] = {}
            if at:
                facets["at"] = [at]
                facets["day"] = [at[:10]]
            if role:
                facets["role"] = [role]
            chunks.append(
                Chunk(
                    path=rel_path,
                    language=language,
                    kind="turn" if summary is not None else ("heading" if title else "prose"),
                    symbol=(preview or (title.split(" > ")[-1] if title else None)),
                    heading_path=heading_path,
                    start_line=part_start_line,
                    end_line=max(part_end_line, part_start_line),
                    start_byte=start_byte,
                    end_byte=end_byte,
                    text=text_out,
                    facets=facets,
                )
            )
            local += len(part)
        if role == "user":
            last_user_body = body
            last_user_summary = preview
        byte_cursor += len(body.encode("utf-8"))
    return chunks


def _chunk_lines(
    rel_path: str,
    language: str,
    kind: str,
    symbol: str | None,
    heading_path: str | None,
    text: str,
    start_line_offset: int,
    max_chars: int,
    start_byte: int = 0,
) -> list[Chunk]:
    parts = _split_text(text, max_chars)
    chunks: list[Chunk] = []
    local = 0
    for part in parts:
        line_shift = text[:local].count("\n")
        start_line = start_line_offset + line_shift + 1
        end_line = start_line + max(part.count("\n"), 0)
        byte_shift = len(text[:local].encode("utf-8"))
        chunks.append(
            Chunk(
                path=rel_path,
                language=language,
                kind=kind,
                symbol=symbol,
                heading_path=heading_path,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte + byte_shift,
                end_byte=start_byte + byte_shift + len(part.encode("utf-8")),
                text=part,
            )
        )
        local += len(part)
    return chunks


def _split_text(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines(keepends=True)
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        if buf and size + len(line) > max_chars:
            parts.append("".join(buf))
            overlap_buf: list[str] = []
            overlap_size = 0
            for prev in reversed(buf):
                if overlap_size + len(prev) > overlap:
                    break
                overlap_buf.append(prev)
                overlap_size += len(prev)
            buf = list(reversed(overlap_buf))
            size = overlap_size
        buf.append(line)
        size += len(line)
    if buf:
        parts.append("".join(buf))
    return parts or [text]
