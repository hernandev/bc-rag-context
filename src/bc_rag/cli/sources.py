"""`bc-rag sources ...`: the repo files that indexing reads, and the groups that pick them."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.table import Table

from bc_rag.cli._app import (
    ExcludeOpt,
    FacetOpt,
    ProjectOpt,
    RootOpt,
    command_group,
    console,
    err,
    fail,
    parse_facet_flags,
    print_json,
    require_project,
)
from bc_rag.facets import facets_line

sources_app = command_group("The repo files that indexing reads, and the groups that pick them.")

JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON.")]


def _load(project: str | None, root):
    from bc_rag.config import load_config

    entry = require_project(project, root)
    try:
        configuration, path = load_config(entry.root_path())
    except Exception as error:
        raise fail("config", error) from error
    return entry, configuration, path


@sources_app.command("files")
def sources_files(
    project: ProjectOpt = None,
    root: RootOpt = None,
    group: Annotated[
        list[str] | None,
        typer.Option("--group", "-g", help="Any of these groups. Repeatable."),
    ] = None,
    space: Annotated[
        list[str] | None,
        typer.Option("--space", "-s", help="Any of these spaces. Repeatable."),
    ] = None,
    raw: Annotated[bool, typer.Option("--raw", help="One path per line, nothing else.")] = False,
    jsonl: Annotated[bool, typer.Option("--jsonl", help="One JSON object per file.")] = False,
) -> None:
    """Repo files the groups select, with space, group, chunker, and model."""
    from bc_rag.plan import file_selected, file_strategy, iter_index_files, unknown_file_filters

    entry, configuration, _path = _load(project, root)
    problem = unknown_file_filters(configuration, group, space)
    if problem is not None:
        raise fail("sources", problem, code=2)
    wanted_groups = set(group) if group else None
    wanted_spaces = set(space) if space else None
    rows = []
    for source in iter_index_files(
        entry.root_path(), configuration, include_disabled=wanted_groups
    ):
        if not file_selected(configuration, source, wanted_groups, wanted_spaces):
            continue
        row = file_strategy(configuration, source)
        if jsonl:
            row = {**row, "language": source.language, "size": source.size}
            sys.stdout.write(json.dumps(row, separators=(",", ":")) + "\n")
            continue
        rows.append(row)
    if jsonl:
        return
    if raw:
        for row in rows:
            sys.stdout.write(row["path"] + "\n")
        return
    table = Table(title=f"files ({len(rows)})")
    for column in ("path", "space", "group", "chunker", "max_chars", "embed", "model"):
        table.add_column(column, justify="right" if column == "max_chars" else "left")
    for row in rows:
        table.add_row(
            row["path"],
            row["space"],
            row["group"],
            row["chunker"],
            row["max_chars"],
            row["embed"],
            row["model"],
        )
    console.print(table)


@sources_app.command("groups")
def sources_groups(
    project: ProjectOpt = None,
    root: RootOpt = None,
    facet: FacetOpt = None,
    exclude: ExcludeOpt = None,
    timing: Annotated[
        bool,
        typer.Option("--timing", help="Time each group's file search instead, slowest first."),
    ] = False,
    json_out: JsonOpt = False,
) -> None:
    """Groups in .bc-rag.json, highest priority first, disabled ones included."""
    if timing:
        if facet or exclude:
            raise fail("sources groups", "--timing does not take --facet or --exclude", code=2)
        _timing(project, root, json_out=json_out)
        return
    from bc_rag.facets import facets_match, with_facet

    wanted, unwanted = parse_facet_flags(facet, exclude)
    entry, configuration, path = _load(project, root)
    groups = sorted(
        (
            group
            for group in configuration.groups
            # a group's name answers `group` the way the stored facet does.
            if facets_match(with_facet(group.facets, "group", group.name), wanted, unwanted)
        ),
        key=lambda group: (-group.priority, group.name),
    )
    if json_out:
        print_json(
            {
                "project": entry.name,
                "count": len(groups),
                "groups": [
                    {
                        "name": group.name,
                        "space": group.space,
                        "priority": group.priority,
                        "enabled": group.enabled,
                        "kind": group.kind,
                        "facets": group.facets,
                        "include": group.include,
                        "exclude": group.exclude,
                    }
                    for group in groups
                ],
            }
        )
        return
    table = Table(title=f"groups ({len(groups)})")
    table.add_column("priority", justify="right")
    table.add_column("enabled")
    table.add_column("name")
    table.add_column("space")
    table.add_column("kind")
    table.add_column("facets")
    table.add_column("includes", justify="right")
    for group in groups:
        table.add_row(
            str(group.priority),
            "yes" if group.enabled else "no",
            group.name,
            group.space,
            group.kind,
            facets_line(group.facets),
            str(len(group.include)),
        )
    console.print(table)
    err.print(f"[dim]config[/dim] {path}")


def _timing(project: str | None, root, *, json_out: bool) -> None:
    from bc_rag.discover import debug_resolve_groups

    entry, configuration, path = _load(project, root)
    rows = debug_resolve_groups(entry.root_path(), configuration)
    total = sum(row.glob_ms for row in rows)
    if json_out:
        print_json(
            {
                "project": entry.name,
                "groups": [
                    {
                        "name": row.name,
                        "enabled": row.enabled,
                        "skipped": row.skipped,
                        "priority": row.priority,
                        "glob_ms": round(row.glob_ms, 2),
                        "file_count": row.file_count,
                        "include_count": row.include_count,
                    }
                    for row in rows
                ],
                "total_glob_ms": round(total, 2),
            }
        )
        return
    table = Table(title="file search per group (slowest first)")
    table.add_column("ms", justify="right")
    table.add_column("files", justify="right")
    table.add_column("enabled")
    table.add_column("priority", justify="right")
    table.add_column("includes", justify="right")
    table.add_column("name")
    for row in rows:
        table.add_row(
            "-" if row.skipped else f"{row.glob_ms:.1f}",
            "-" if row.skipped else str(row.file_count),
            "yes" if row.enabled else "no",
            str(row.priority),
            str(row.include_count),
            row.name,
        )
    console.print(table)
    err.print(f"[dim]total[/dim] {total:.1f} ms")
    err.print(f"[dim]skipped disabled[/dim] {sum(1 for row in rows if row.skipped)}")
    err.print(f"[dim]config[/dim] {path}")


@sources_app.command("chunks")
def sources_chunks(
    path: Annotated[str, typer.Argument(help="File to chunk, relative to the project root.")],
    project: ProjectOpt = None,
    root: RootOpt = None,
) -> None:
    """The chunks one repo file becomes, as indexing would store them."""
    from bc_rag.plan import chunks_for_user_path, file_strategy

    entry, configuration, _path = _load(project, root)
    try:
        source, chunks = chunks_for_user_path(entry.root_path(), configuration, path)
    except ValueError as error:
        raise fail("chunks", error) from error
    row = file_strategy(configuration, source)
    sys.stdout.write(
        f"# {row['path']}  space={row['space']}  group={row['group']}  "
        f"chunker={row['chunker']}  max_chars={row['max_chars']}  "
        f"embed={row['embed']}  model={row['model']}\n"
    )
    for index, chunk in enumerate(chunks, start=1):
        sys.stdout.write(
            f"\n--- {index}  lines {chunk.start_line}-{chunk.end_line}  "
            f"chars {len(chunk.text)}  kind {chunk.kind}  symbol {chunk.symbol or ''}  "
            f"{chunk.heading_path or ''}\n"
        )
        sys.stdout.write(chunk.text)
        if not chunk.text.endswith("\n"):
            sys.stdout.write("\n")
