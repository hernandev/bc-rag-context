"""bc-rag command line for the bc-rag-context project."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from bc_rag import __version__
from bc_rag.catalog import forget_project, living_projects, project_store_dir, register_project
from bc_rag.config import (
    config_path,
    dump_default_config,
    format_config,
    load_config,
    public_config,
    with_jina_api,
)
from bc_rag.defaults import MCP_HTTP_HOST, MCP_HTTP_PORT
from bc_rag.discover import debug_resolve_groups, iter_source_files
from bc_rag.facets import distinct_tag_values, tags_match
from bc_rag.plan import (
    chunks_for_user_path,
    file_selected,
    file_strategy,
    iter_index_files,
    load_project,
    unknown_file_filters,
)
from bc_rag.indexer import Indexer
from bc_rag.manifest import load_manifest
from bc_rag.query import search, search_sparse
from bc_rag.runtime import open_session, resolve_root
from bc_rag.watch import WatchSession

def _command_group(help_text: str) -> typer.Typer:
    """A command group that prints its help when no subcommand is given."""
    group = typer.Typer(help=help_text, invoke_without_command=True)

    @group.callback()
    def _missing_command(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is not None:
            return
        from typer import rich_utils
        import click

        rich_utils.rich_format_error(click.UsageError("Missing command.", ctx))
        err.print(ctx.get_help())
        raise typer.Exit(code=2)

    return group


projects_app = _command_group("Catalog of indexed projects used by the MCP server.")
list_app = _command_group("Inspect groups and files that index would open.")
models_app = _command_group("FastEmbed model cache. Called from Python, not a FastEmbed CLI.")

app = typer.Typer(
    name="bc-rag",
    no_args_is_help=True,
    add_completion=False,
    help="Local hybrid RAG over TypeScript, Markdown, and OpenAPI.",
)
qdrant_app = _command_group("Docker Qdrant used by index, query, and MCP.")
tags_app = _command_group("Tags stored on indexed points.")
groups_app = _command_group("Config groups from .bc-rag.json.")
files_app = _command_group("Files index would chunk.")
config_app = _command_group(
    "Project config (.bc-rag.json) and user settings (~/.bc-rag/config)."
)
daemon_app = _command_group("The background daemon: index every project and serve MCP.")
app.add_typer(projects_app, name="projects")
app.add_typer(list_app, name="list")
app.add_typer(models_app, name="models")
app.add_typer(qdrant_app, name="qdrant")
app.add_typer(tags_app, name="tags")
app.add_typer(groups_app, name="groups")
app.add_typer(files_app, name="files")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
console = Console()
err = Console(stderr=True)

RootArg = Annotated[
    Path | None,
    typer.Option("--root", "-r", help="Project folder. Defaults to the current directory."),
]


def _print_version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


# commands that never start the daemon: they manage it, its settings, or its storage.
NO_DAEMON_COMMANDS = frozenset({"daemon", "config", "reset", "qdrant", "models", "init"})


@app.callback()
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    del version
    if ctx.invoked_subcommand in NO_DAEMON_COMMANDS:
        return
    # like the Nx daemon: the first command starts it, with this shell's environment.
    from bc_rag import daemon

    try:
        daemon.ensure(report=lambda message: err.print(f"[dim]{message}[/dim]"))
    except Exception as error:
        # the daemon is a convenience; a command still runs without it.
        err.print(f"[yellow]daemon[/yellow] {error}")


@app.command()
def init(
    root: RootArg = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing config.")] = False,
) -> None:
    """Write a `.bc-rag.json` with one space. The dense model, reranker, and chunk size sit on that space."""
    project = resolve_root(root)
    path = config_path(project)
    if path.exists() and not force:
        err.print(f"[yellow]already exists[/yellow] {path}")
        raise typer.Exit(code=1)
    path.write_text(dump_default_config(), encoding="utf-8")
    console.print(f"[green]wrote[/green] {path}")


@config_app.command("dump")
def config_cmd(root: RootArg = None) -> None:
    """Print the config after groupsCommand has been merged."""
    project = resolve_root(root)
    configuration, path = load_config(project)
    if path is None:
        err.print("[dim]config[/dim] defaults")
    else:
        err.print(f"[dim]config[/dim] {path}")
    sys.stdout.write(format_config(public_config(configuration)))


@list_app.command("paths")
def list_paths_cmd(
    root: RootArg = None,
    jsonl: Annotated[
        bool,
        typer.Option("--jsonl", help="Print one JSON object per file. No array wrap."),
    ] = False,
) -> None:
    """Stream files as they resolve. Prints the group name, then its paths."""
    project = resolve_root(root)
    configuration, _cfg_path = load_config(project)
    current_group: str | None = None
    for source in iter_source_files(project, configuration):
        bucket = source.config_group or source.group or ""
        if jsonl:
            sys.stdout.write(
                json.dumps(
                    {
                        "path": source.rel_path,
                        "language": source.language,
                        "size": source.size,
                        "group": source.group,
                        "config_group": source.config_group,
                        "tags": source.tags,
                        "priority": source.priority,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            continue
        if bucket != current_group:
            current_group = bucket
            sys.stdout.write(f"# {bucket}\n")
        sys.stdout.write(f"{source.rel_path}\n")


@list_app.command("groups")
def list_groups_cmd(
    root: RootArg = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            "-t",
            help="One alternative. Commas inside one flag must all match. Repeat the flag to OR. -vendor:olo excludes.",
        ),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """List config groups, highest priority first. Includes enabled: false."""
    project = resolve_root(root)
    configuration, cfg_path = load_config(project)
    groups = [
        group
        for group in (configuration.groups or configuration.resolved_groups())
        if tags_match(group.tags, tag)
    ]
    groups.sort(key=lambda group: (-group.priority, group.name))
    if json_out:
        payload = [
            {
                "name": group.name,
                "priority": group.priority,
                "enabled": group.enabled,
                "kind": group.kind,
                "tags": group.tags,
                "metadata": group.metadata,
                "include": group.include,
                "exclude": group.exclude,
            }
            for group in groups
        ]
        console.print(json.dumps({"root": str(project), "count": len(payload), "groups": payload}, indent=2))
        return
    listing = Table(title=f"list groups ({len(groups)})")
    listing.add_column("priority", justify="right")
    listing.add_column("enabled")
    listing.add_column("name")
    listing.add_column("kind")
    listing.add_column("tags")
    listing.add_column("includes", justify="right")
    for group in groups:
        listing.add_row(
            str(group.priority),
            "yes" if group.enabled else "no",
            group.name,
            group.kind,
            ", ".join(group.tags or []),
            str(len(group.include)),
        )
    console.print(listing)
    err.print(f"[dim]root[/dim] {project}")
    err.print(f"[dim]config[/dim] {cfg_path or 'defaults'}")


@config_app.command("tags")
def config_tags_cmd(
    root: RootArg = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """List each tag key and the distinct values on enabled groups."""
    project = resolve_root(root)
    configuration, cfg_path = load_config(project)
    rows = distinct_tag_values(list(configuration.groups))
    if json_out:
        console.print(
            json.dumps(
                {
                    "root": str(project),
                    "tags": [{"tag": key, "values": values} for key, values in rows],
                },
                indent=2,
            )
        )
        return
    listing = Table(title=f"config tags ({len(rows)})")
    listing.add_column("tag")
    listing.add_column("values")
    for key, values in rows:
        listing.add_row(key, ", ".join(values))
    console.print(listing)
    err.print(f"[dim]root[/dim] {project}")
    err.print(f"[dim]config[/dim] {cfg_path or 'defaults'}")


@list_app.command("tags")
def list_tags_cmd(
    root: RootArg = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
    jina: Annotated[bool, typer.Option("--jina", help="Read the Jina API collection.")] = False,
    local: Annotated[bool, typer.Option("--local", help="Read the FastEmbed collection.")] = False,
) -> None:
    """List payload tags and groups actually stored in Qdrant, with point counts."""
    project = resolve_root(root)
    session = open_session(
        project,
        need_reranker=False,
        write=False,
        console=err,
        jina_api=_backend_override(jina=jina, local=local),
    )
    try:
        from bc_rag.facets import SIDECAR_KEYS

        tags = session.store.facet_values("tags")
        groups = session.store.facet_values("group")
        facets = {key: session.store.facet_values(key) for key in SIDECAR_KEYS}
    finally:
        session.close()
    if json_out:
        console.print(
            json.dumps(
                {
                    "root": str(project),
                    "tags": [{"tag": name, "points": count} for name, count in tags],
                    "groups": [{"group": name, "points": count} for name, count in groups],
                    "facets": {
                        key: [{"value": name, "points": count} for name, count in rows]
                        for key, rows in facets.items()
                        if rows
                    },
                },
                indent=2,
            )
        )
        return
    listing = Table(title="list tags")
    listing.add_column("kind")
    listing.add_column("value")
    listing.add_column("points", justify="right")
    for name, count in tags:
        listing.add_row("tag", name, str(count))
    for name, count in groups:
        listing.add_row("group", name, str(count))
    for key, rows in facets.items():
        for name, count in rows:
            listing.add_row(key, name, str(count))
    if not tags and not groups and not any(facets.values()):
        console.print("[yellow]no tags or groups in this collection[/yellow]")
        return
    console.print(listing)


tags_app.command("list")(list_tags_cmd)
groups_app.command("list")(list_groups_cmd)
config_app.command("groups")(list_groups_cmd)


@list_app.command("debug")
def list_debug_cmd(
    root: RootArg = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Time include expansion for each group. Disabled groups are not globbed."""
    project = resolve_root(root)
    configuration, cfg_path = load_config(project)
    rows = debug_resolve_groups(project, configuration)
    if json_out:
        console.print(
            json.dumps(
                {
                    "root": str(project),
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
                    "total_glob_ms": round(sum(row.glob_ms for row in rows), 2),
                },
                indent=2,
            )
        )
        return
    listing = Table(title="list debug (slowest first)")
    listing.add_column("ms", justify="right")
    listing.add_column("files", justify="right")
    listing.add_column("enabled")
    listing.add_column("priority", justify="right")
    listing.add_column("includes", justify="right")
    listing.add_column("name")
    for row in rows:
        listing.add_row(
            "-" if row.skipped else f"{row.glob_ms:.1f}",
            "-" if row.skipped else str(row.file_count),
            "yes" if row.enabled else "no",
            str(row.priority),
            str(row.include_count),
            row.name,
        )
    console.print(listing)
    total = sum(row.glob_ms for row in rows)
    skipped = sum(1 for row in rows if row.skipped)
    err.print(f"[dim]glob total[/dim] {total:.1f} ms")
    err.print(f"[dim]skipped disabled[/dim] {skipped}")
    err.print(f"[dim]root[/dim] {project}")
    err.print(f"[dim]config[/dim] {cfg_path or 'defaults'}")


def parse_every(text: str) -> float:
    raw = text.strip().lower()
    if not raw:
        raise ValueError("--every needs a duration like 5m, 90s, or 2h")
    suffix = raw[-1]
    if suffix in {"s", "m", "h"} and raw[:-1]:
        amount = float(raw[:-1])
        seconds = {"s": 1.0, "m": 60.0, "h": 3600.0}[suffix] * amount
    else:
        seconds = float(raw)
    if seconds <= 0:
        raise ValueError("--every must be greater than 0")
    return seconds


def _print_index_stats(stats, *, embed_label: str = "embed") -> None:
    table = Table(title="index" if not stats.stopped else "index (stopped)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("scanned", str(stats.scanned))
    table.add_row("indexed files", str(stats.indexed_files))
    table.add_row("unchanged", str(stats.skipped_unchanged))
    table.add_row("too large", str(stats.skipped_large))
    table.add_row("deleted files", str(stats.deleted_files))
    table.add_row("chunks upserted", str(stats.chunks))
    table.add_row("errors", str(len(stats.errors)))
    if stats.jina_embed_calls or stats.jina_embed_tokens:
        name = embed_label or "embed"
        table.add_row(f"{name} embed calls", str(stats.jina_embed_calls))
        table.add_row(f"{name} embed tokens", str(stats.jina_embed_tokens))
    console.print(table)
    for message in stats.errors:
        err.print(f"[red]error[/red] {message}")


def _backend_override(*, jina: bool, local: bool) -> bool | None:
    if jina and local:
        err.print("[red]use one of --jina or --local[/red]")
        raise typer.Exit(code=1)
    if jina:
        return True
    if local:
        return False
    return None


@files_app.command("list")
def files_cmd(
    root: RootArg = None,
    group: Annotated[
        list[str] | None,
        typer.Option("--group", "-g", help="Any of these config group names. Repeatable."),
    ] = None,
    space: Annotated[
        list[str] | None,
        typer.Option("--space", "-s", help="Any of these vector spaces. Repeatable."),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Print one path per line and no other columns."),
    ] = False,
) -> None:
    """List every file index would chunk, and the strategy for that file."""
    project, configuration = load_project(root)
    problem = unknown_file_filters(configuration, group, space)
    if problem is not None:
        err.print(f"[red]{problem}[/red]")
        raise typer.Exit(code=1)
    wanted_groups = set(group) if group else None
    wanted_spaces = set(space) if space else None
    rows = []
    for source in iter_index_files(
        project, configuration, include_disabled=wanted_groups
    ):
        if not file_selected(configuration, source, wanted_groups, wanted_spaces):
            continue
        rows.append(file_strategy(configuration, source))
    if raw:
        for row in rows:
            sys.stdout.write(row["path"] + "\n")
        return
    listing = Table(title=f"files ({len(rows)})")
    listing.add_column("path")
    listing.add_column("space")
    listing.add_column("group")
    listing.add_column("chunker")
    listing.add_column("max_chars", justify="right")
    listing.add_column("embed")
    listing.add_column("model")
    for row in rows:
        listing.add_row(
            row["path"],
            row["space"],
            row["group"],
            row["chunker"],
            row["max_chars"],
            row["embed"],
            row["model"],
        )
    console.print(listing)


@app.command("chunks")
def chunks_cmd(
    path: Annotated[
        str,
        typer.Option("--path", "-p", help="Project path to chunk the way index would."),
    ],
    root: RootArg = None,
) -> None:
    """Print the chunks index would store for one file. Writes no vectors."""
    project, configuration = load_project(root)
    try:
        source, chunks = chunks_for_user_path(project, configuration, path)
    except ValueError as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error
    row = file_strategy(configuration, source)
    sys.stdout.write(
        f"# {row['path']}  space={row['space']}  group={row['group']}  "
        f"chunker={row['chunker']}  max_chars={row['max_chars']}  "
        f"embed={row['embed']}  model={row['model']}\n"
    )
    for index, chunk in enumerate(chunks, start=1):
        symbol = chunk.symbol or ""
        heading = chunk.heading_path or ""
        sys.stdout.write(
            f"\n--- {index}  lines {chunk.start_line}-{chunk.end_line}  "
            f"chars {len(chunk.text)}  kind {chunk.kind}  symbol {symbol}  {heading}\n"
        )
        sys.stdout.write(chunk.text)
        if not chunk.text.endswith("\n"):
            sys.stdout.write("\n")


@app.command("index")
def index_cmd(
    root: RootArg = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild the whole index, ignoring file hashes.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List files that would be indexed, then exit."),
    ] = False,
    jina: Annotated[
        bool,
        typer.Option("--jina", help="Write the Jina API index under ~/.bc-rag/{name}/jina/."),
    ] = False,
    local: Annotated[
        bool,
        typer.Option("--local", help="Write the FastEmbed index (default store)."),
    ] = False,
    path: Annotated[
        list[str] | None,
        typer.Option(
            "--path",
            "-p",
            help="Only these project-relative paths. Repeatable. Smallest sync job.",
        ),
    ] = None,
    every: Annotated[
        str | None,
        typer.Option(
            "--every",
            help="Repeat on this interval (5m, 90s, 2h). Each pass is a normal hash skip, not a filesystem watcher.",
        ),
    ] = None,
    all_projects: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Index every project in the catalog (`bc-rag projects list`). No --root needed.",
        ),
    ] = False,
) -> None:
    """Index the project. Unchanged files are skipped unless --force."""
    if dry_run:
        list_paths_cmd(root=root)
        return
    interval_s: float | None = None
    if every is not None:
        if force:
            err.print("[red]index[/red] --every cannot be combined with --force")
            raise typer.Exit(code=2)
        try:
            interval_s = parse_every(every)
        except ValueError as error:
            err.print(f"[red]index[/red] {error}")
            raise typer.Exit(code=2) from error
    if all_projects and (root is not None or path):
        err.print("[red]index[/red] --all cannot be combined with --root or --path")
        raise typer.Exit(code=2)
    jina_api = _backend_override(jina=jina, local=local)
    if interval_s is not None:
        err.print(
            f"[dim]every[/dim] {every}  ({interval_s:g}s)  hash skip, not a filesystem watcher"
        )
    try:
        while True:
            if all_projects:
                projects = [entry.root_path() for entry in living_projects()]
                if not projects:
                    err.print(
                        "[yellow]no cataloged projects[/yellow] run `bc-rag index` in a repo once"
                    )
            else:
                projects = [resolve_root(root)]
            failed = False
            stopped = False
            for project in projects:
                try:
                    stats = _index_project(
                        project, force=force, only_paths=path or None, jina_api=jina_api
                    )
                except Exception as error:
                    # one broken project must not stop the others, or kill a daemon.
                    if not all_projects and interval_s is None:
                        raise
                    err.print(f"[red]index failed[/red] {project}: {error}")
                    failed = True
                    continue
                failed = failed or bool(stats.errors)
                if stats.stopped:
                    stopped = True
                    break
            if interval_s is None or stopped:
                if failed:
                    raise typer.Exit(code=1)
                break
            err.print(f"[dim]next index in {every}[/dim]")
            time.sleep(interval_s)
    except KeyboardInterrupt:
        err.print("stopped  aborted")
        raise typer.Exit(code=130) from None


def _index_project(
    project: Path,
    *,
    force: bool,
    only_paths: list[str] | None,
    jina_api: bool | None,
):
    """One index pass over one project. Sessions are opened fresh so config edits apply."""
    from bc_rag.runtime import open_space_sessions

    sessions = open_space_sessions(
        project,
        need_reranker=False,
        write=True,
        console=err,
        jina_api=jina_api,
    )
    try:
        indexers = [
            Indexer(
                project,
                session.config,
                session.embedder,
                session.store,
                err,
                space=session.space,
            )
            for session in sessions
        ]
        from bc_rag.indexer import _embed_label, index_spaces

        session = sessions[0]
        source = session.config_path or "defaults (no .bc-rag.json)"
        threads = os.cpu_count() or 4
        err.print(f"[dim]root[/dim] {project}")
        err.print(f"[dim]config[/dim] {source}")
        from bc_rag.corpus import corpus_dir

        err.print(f"[dim]corpus[/dim] {corpus_dir(session.config.project_dir(project))}")
        for item in sessions:
            label = item.space or "default"
            store_dir = item.config.store_dir(project, item.space)
            err.print(f"[dim]space[/dim] {label}  {item.embedder.dense_model}")
            err.print(f"[dim]store[/dim] {store_dir}")
        space_spec = session.config.spaces[session.space or session.config.default_space]
        err.print(f"[dim]sparse[/dim] {space_spec.sparse.provider} {space_spec.sparse.model}")
        if any(item.embedder.voyage_api for item in sessions):
            err.print("[dim]embed[/dim] voyage http  https://api.voyageai.com  (sparse stays local BM25)")
        elif session.config.spaces[session.config.default_space].dense_provider() == "jina":
            err.print("[dim]embed[/dim] jina api  https://api.jina.ai  (sparse stays local BM25)")
        else:
            err.print(f"[dim]threads[/dim] {threads}  FastEmbed ONNX uses every CPU")
        labels = {_embed_label(item.embedder) for item in indexers}
        labels.discard("")
        embed_label = next(iter(labels)) if len(labels) == 1 else "embed"
        stats = index_spaces(indexers, force=force, only_paths=only_paths)
        _print_index_stats(stats, embed_label=embed_label)
        return stats
    finally:
        for session in sessions:
            session.close()


@app.command("reset")
def reset_cmd(
    root: RootArg = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Do not ask. Wipe Qdrant, corpus, openapi-md, and hash ledger."),
    ] = False,
    jina: Annotated[
        bool,
        typer.Option("--jina", help="Reset the Jina API store."),
    ] = False,
    local: Annotated[
        bool,
        typer.Option("--local", help="Reset the FastEmbed store."),
    ] = False,
) -> None:
    """Wipe Qdrant, corpus, generated OpenAPI markdown, and the hash ledger."""
    from bc_rag.reset import reset_project, reset_targets

    project = resolve_root(root)
    configuration, cfg_path = load_config(project)
    override = _backend_override(jina=jina, local=local)
    if override is not None:
        configuration = with_jina_api(configuration, override)
    register_project(project)
    targets = reset_targets(project, configuration)
    collection = configuration.qdrant_collection(project)
    err.print(f"[dim]root[/dim] {project}")
    err.print(f"[dim]config[/dim] {cfg_path or 'defaults'}")
    err.print("[dim]will remove[/dim]")
    for path in targets:
        err.print(f"  {path}")
    err.print(f"  qdrant collection {collection}")
    if not force:
        ok = typer.confirm("Wipe index state and start from a first run?", default=False)
        if not ok:
            err.print("reset  cancelled")
            raise typer.Exit(code=1)
    report = reset_project(project, configuration)
    table = Table(title="reset")
    table.add_column("what")
    table.add_column("result")
    for path in report.removed:
        table.add_row(path, "removed")
    for path in report.missing:
        table.add_row(path, "already gone")
    if report.collection:
        if report.collection_deleted:
            table.add_row(f"collection {report.collection}", "deleted")
        elif report.collection_error:
            table.add_row(f"collection {report.collection}", report.collection_error)
        else:
            table.add_row(f"collection {report.collection}", "already gone")
    console.print(table)


@app.command("bundle")
def bundle_cmd(root: RootArg = None) -> None:
    """Write the prepared corpus (documents + Bedrock sidecars) under ~/.bc-rag/{name}/corpus/."""
    from bc_rag.corpus import bundle_sources, corpus_dir

    project = resolve_root(root)
    configuration, cfg_path = load_config(project)
    register_project(project)
    written, pruned = bundle_sources(project, configuration)
    dest = corpus_dir(configuration.project_dir(project))
    err.print(f"[dim]root[/dim] {project}")
    err.print(f"[dim]config[/dim] {cfg_path or 'defaults'}")
    err.print(f"[dim]corpus[/dim] {dest}")
    table = Table(title="bundle")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("documents", str(len(written)))
    table.add_row("sidecars", str(len(written)))
    table.add_row("pruned", str(len(pruned)))
    console.print(table)


@app.command()
def query(
    q: Annotated[str, typer.Option("--query", "-q", help="The search text.")],
    root: RootArg = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="How many hits to return."),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")] = False,
    no_rerank: Annotated[
        bool, typer.Option("--no-rerank", help="Skip the cross-encoder rerank pass.")
    ] = False,
    sparse: Annotated[
        bool,
        typer.Option("--sparse", help="BM25 only. No dense vector and no rerank."),
    ] = False,
    jina: Annotated[
        bool, typer.Option("--jina", help="Query the Jina API index.")
    ] = False,
    local: Annotated[
        bool, typer.Option("--local", help="Query the FastEmbed index.")
    ] = False,
    compare: Annotated[
        bool,
        typer.Option("--compare", help="Query local FastEmbed and Jina API stores side by side."),
    ] = False,
    group: Annotated[
        list[str] | None,
        typer.Option("--group", "-g", help="Any of these config group names. Repeatable."),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            "-t",
            help="One alternative. Commas inside one flag must all match. Repeat the flag to OR. -vendor:olo excludes.",
        ),
    ] = None,
    space: Annotated[
        str | None,
        typer.Option("--space", help="Vector space. Defaults to defaultSpace."),
    ] = None,
) -> None:
    """Dense search, then rerank. --sparse is BM25 only."""
    text = q.strip()
    if not text:
        err.print("[red]empty query[/red]")
        raise typer.Exit(code=1)
    project = resolve_root(root)
    use_rerank = not no_rerank
    if compare:
        _query_compare(
            project,
            text,
            limit=limit,
            use_rerank=use_rerank,
            json_out=json_out,
            groups=group,
            tags=tag,
        )
        return
    session = open_session(
        project,
        need_reranker=use_rerank,
        write=False,
        console=err,
        jina_api=_backend_override(jina=jina, local=local),
        space=space,
    )
    try:
        if session.store.count() == 0:
            err.print("[red]empty index[/red] run `bc-rag index` first")
            raise typer.Exit(code=1)
        if sparse:
            result = search_sparse(
                query=text,
                config=session.config,
                embedder=session.embedder,
                store=session.store,
                limit=limit,
                groups=group,
                tags=tag,
                space=session.space,
            )
        else:
            result = search(
                query=text,
                config=session.config,
                embedder=session.embedder,
                store=session.store,
                reranker=session.reranker,
                limit=limit,
                use_rerank=use_rerank,
                groups=group,
                tags=tag,
                space=session.space,
            )
    finally:
        session.close()
    _print_query_result(result, json_out=json_out)


def _query_compare(
    project: Path,
    text: str,
    *,
    limit: int | None,
    use_rerank: bool,
    json_out: bool,
    groups: list[str] | None = None,
    tags: list[str] | None = None,
) -> None:
    payload = {"query": text, "backends": {}}
    for name, flag in (("local", False), ("jina", True)):
        session = open_session(
            project,
            need_reranker=use_rerank,
            write=False,
            console=err,
            jina_api=flag,
        )
        qdrant = session.config.qdrant_path(project)
        try:
            empty = not qdrant.exists() or session.store.count() == 0
            result = None
            if not empty:
                result = search(
                    query=text,
                    config=session.config,
                    embedder=session.embedder,
                    store=session.store,
                    reranker=session.reranker,
                    limit=limit,
                    use_rerank=use_rerank,
                    groups=groups,
                    tags=tags,
                    space=session.space,
                )
        finally:
            session.close()
        if empty or result is None:
            if json_out:
                payload["backends"][name] = {"hits": [], "empty": True}
            else:
                console.print()
                console.print(f"[bold]{name}[/bold]  [yellow]empty index[/yellow]  {qdrant}")
            continue
        if json_out:
            payload["backends"][name] = {
                "hits": [_hit_payload(hit) for hit in result.hits],
                "empty": False,
            }
        else:
            console.print()
            console.print(f"[bold]{name}[/bold]  {qdrant}")
            _print_query_result(result, json_out=False)
    if json_out:
        console.print(json.dumps(payload, indent=2))


def _hit_payload(hit) -> dict:
    return {
        "score": hit.score,
        "path": hit.path,
        "language": hit.language,
        "kind": hit.kind,
        "symbol": hit.symbol,
        "heading_path": hit.heading_path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "group": (hit.payload or {}).get("group"),
        "tags": (hit.payload or {}).get("tags") or [],
        "text": hit.text,
    }


def _print_query_result(result, *, json_out: bool) -> None:
    if json_out:
        payload = {
            "query": result.query,
            "hits": [_hit_payload(hit) for hit in result.hits],
        }
        console.print(json.dumps(payload, indent=2))
        return
    if not result.hits:
        console.print("[yellow]no hits[/yellow]")
        return
    for index, hit in enumerate(result.hits, start=1):
        where = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        extras = []
        group_name = (hit.payload or {}).get("group")
        if group_name:
            extras.append(str(group_name))
        if hit.symbol:
            extras.append(hit.symbol)
        if hit.heading_path:
            extras.append(hit.heading_path)
        extra = f"  {' · '.join(extras)}" if extras else ""
        console.print(
            f"[bold]{index}.[/bold] {where}  [dim]{hit.kind} score={hit.score:.4f}{extra}[/dim]"
        )
        lexer = {
            "typescript": "ts",
            "tsx": "tsx",
            "javascript": "js",
            "python": "python",
            "markdown": "markdown",
            "openapi": "markdown",
            "json": "json",
        }.get(hit.language, "text")
        console.print(Syntax(hit.text.rstrip() + "\n", lexer, word_wrap=True, padding=1))


@app.command()
def watch(
    root: RootArg = None,
    debounce: Annotated[
        float,
        typer.Option(
            "--debounce",
            help="Optional quiet period before hashing. 0 (default) hashes as soon as events arrive.",
        ),
    ] = 0.0,
    skip_initial: Annotated[
        bool,
        typer.Option(
            "--skip-initial",
            help="Do not run a full index first. Use after a complete bc-rag index.",
        ),
    ] = False,
    jina: Annotated[
        bool,
        typer.Option("--jina", help="Watch the Jina API store."),
    ] = False,
    local: Annotated[
        bool,
        typer.Option("--local", help="Watch the FastEmbed store."),
    ] = False,
) -> None:
    """Watch the tree and reindex only the files that change."""
    project = resolve_root(root)
    from bc_rag.runtime import open_space_sessions

    sessions = open_space_sessions(
        project,
        need_reranker=False,
        write=True,
        console=err,
        jina_api=_backend_override(jina=jina, local=local),
    )
    try:
        indexers = [
            Indexer(
                project,
                session.config,
                session.embedder,
                session.store,
                err,
                space=session.space,
            )
            for session in sessions
        ]

        class _Fanout:
            def run(self, **kwargs):
                from bc_rag.indexer import index_spaces

                return index_spaces(indexers, **kwargs)

        indexer = _Fanout()
        session = sessions[0]
        if skip_initial:
            err.print("[dim]watch[/dim] skip initial index")
        else:
            err.print("[dim]initial index[/dim]")
            stats = indexer.run()
            err.print(
                f"[green]ready[/green] files={stats.indexed_files + stats.skipped_unchanged} "
                f"chunks={session.store.count()}"
            )
        WatchSession(project, session.config, indexer, err, debounce_s=debounce).run()
    finally:
        for session in sessions:
            session.close()


@qdrant_app.command("start")
def qdrant_start() -> None:
    """Start Docker Qdrant on http://127.0.0.1:32321 (OrbStack context)."""
    from bc_rag.qdrant_docker import start

    try:
        start()
    except Exception as error:
        err.print(f"[red]qdrant start failed[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print("qdrant  http://127.0.0.1:32321  running")


@qdrant_app.command("stop")
def qdrant_stop() -> None:
    """Stop Docker Qdrant. Storage on disk is kept."""
    from bc_rag.qdrant_docker import stop

    try:
        stop()
    except Exception as error:
        err.print(f"[red]qdrant stop failed[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print("qdrant  stopped")


@qdrant_app.command("import-local")
def qdrant_import_local(root: RootArg = None) -> None:
    """Copy an existing embedded-path collection into Docker. Does not call Jina."""
    from bc_rag.qdrant_docker import import_embedded_collection, wait_ready

    project = resolve_root(root)
    config, _ = load_config(project)
    source = config.qdrant_path(project)
    if not source.is_dir():
        err.print(f"[red]no local qdrant folder[/red] {source}")
        raise typer.Exit(code=1)
    try:
        wait_ready(url=config.qdrant_http_url())
        copied = import_embedded_collection(
            source_path=source,
            url=config.qdrant_http_url(),
            collection=config.qdrant_collection(project),
            dense_dim=1024 if config.spaces[config.default_space].dense_provider() == "jina" else 768,
        )
    except Exception as error:
        err.print(f"[red]qdrant import failed[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(f"imported {copied} points into {config.qdrant_collection(project)}")


@qdrant_app.command("restart")
def qdrant_restart() -> None:
    """Restart Docker Qdrant."""
    from bc_rag.qdrant_docker import restart

    try:
        restart()
    except Exception as error:
        err.print(f"[red]qdrant restart failed[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print("qdrant  http://127.0.0.1:32321  running")


@app.command()
def status(root: RootArg = None) -> None:
    """Show config, models, and how many points are stored."""
    project = resolve_root(root)
    config, cfg_path = load_config(project)
    manifest = load_manifest(config.manifest_path(project))
    count = 0
    qdrant_ok = False
    from bc_rag.store import HybridStore

    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(project),
        read_only=True,
    )
    try:
        count = store.count()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    finally:
        store.close()

    table = Table(title="bc-rag status")
    table.add_column("key")
    table.add_column("value")
    table.add_row("root", str(project))
    table.add_row("config", str(cfg_path) if cfg_path else "(defaults)")
    enabled = sum(1 for group in config.groups if group.enabled)
    table.add_row("groups", f"{enabled} enabled / {len(config.groups)} total")
    extra = " ..." if len(config.exclude) > 6 else ""
    table.add_row("exclude", ", ".join(config.exclude[:6]) + extra if config.exclude else "(none)")
    table.add_row("defaultSpace", config.default_space)
    shown = config.spaces[config.default_space]
    table.add_row("sparse", f"{shown.sparse.provider} {shown.sparse.model}")
    table.add_row("rerank", shown.rerank.model if shown.rerank else "(off)")
    table.add_row("indexed files", str(len(manifest.files) if manifest else 0))
    table.add_row("qdrant", config.qdrant_http_url() + ("" if qdrant_ok else "  (not running)"))
    table.add_row("collection", config.qdrant_collection(project))
    table.add_row("points", str(count) if qdrant_ok else "(unavailable)")
    table.add_row("backend", config.backend_id())
    console.print(table)


@app.command("register")
def register_cmd(root: RootArg = None) -> None:
    """Register this folder as its own catalog project. Store lives under ~/.bc-rag/{name}/."""
    project = resolve_root(root)
    entry = register_project(project)
    store_dir = project_store_dir(entry.name)
    console.print(f"registered [bold]{entry.name}[/bold]")
    console.print(f"root {entry.root}")
    console.print(f"store {store_dir}")
    console.print(f"log  {store_dir / 'index.jsonl'}")


@projects_app.command("list")
def projects_list() -> None:
    """Show every cataloged project the MCP server can search."""
    entries = living_projects()
    if not entries:
        console.print("[yellow]no cataloged projects[/yellow] run `bc-rag index` in a repo")
        return
    table = Table(title="catalog")
    table.add_column("name")
    table.add_column("root")
    table.add_column("updated")
    for entry in entries:
        table.add_row(entry.name, entry.root, entry.updated_at)
    console.print(table)


@projects_app.command("forget")
def projects_forget(
    name: Annotated[str, typer.Option("--name", help="Catalog name or absolute root path.")],
) -> None:
    """Remove a project from the MCP catalog. Does not delete the on-disk index."""
    removed = forget_project(name)
    if removed is None:
        err.print(f"[red]not in catalog[/red] {name}")
        raise typer.Exit(code=1)
    console.print(f"[green]forgot[/green] {removed.name} ({removed.root})")


@models_app.command("list")
def models_list_cmd(root: RootArg = None) -> None:
    """List configured and cached FastEmbed models."""
    from bc_rag.models import cache_directory, list_models

    project = resolve_root(root)
    err.print(f"[dim]cache[/dim] {cache_directory()}")
    listing = Table(title="models")
    listing.add_column("configured")
    listing.add_column("cached")
    listing.add_column("kind")
    listing.add_column("GB", justify="right")
    listing.add_column("name")
    for record in list_models(project):
        listing.add_row(
            "yes" if record.configured else "",
            "yes" if record.cached else "",
            record.kind,
            f"{record.size_gb:.2f}" if record.size_gb is not None else "",
            record.name,
        )
    console.print(listing)


@models_app.command("download")
def models_download_cmd(
    names: Annotated[
        list[str] | None,
        typer.Option(
            "--name",
            help="Model id. Repeatable. Defaults to dense, sparse, and rerank from config.",
        ),
    ] = None,
    root: RootArg = None,
) -> None:
    """Download models via the FastEmbed Python API into the bc-rag cache."""
    from bc_rag.models import cache_directory, configured_model_names, download_model

    project = resolve_root(root)
    targets = names or configured_model_names(project)
    err.print(f"[dim]cache[/dim] {cache_directory()}")
    for name in targets:
        err.print(f"download {name}")
        path = download_model(name)
        console.print(f"[green]ready[/green] {name}  {path}")


@models_app.command("rm")
def models_rm_cmd(
    names: Annotated[
        list[str],
        typer.Option("--name", help="Model id to delete from the cache. Repeatable."),
    ],
) -> None:
    """Delete cached FastEmbed model folders."""
    from bc_rag.models import remove_model

    for name in names:
        path = remove_model(name)
        console.print(f"[red]removed[/red] {name}  {path}")


@app.command("split-md")
def split_md_cmd(
    paths: Annotated[
        list[Path] | None,
        typer.Option("--path", "-p", help="OpenAPI file. Repeatable. Defaults to the openapi groups."),
    ] = None,
    root: RootArg = None,
) -> None:
    """Write per-endpoint markdown under ~/.bc-rag/{project}/openapi-md/."""
    from bc_rag.split_md import listed_openapi_sources, sources_from_paths, split_source_files

    project = resolve_root(root)
    config, _ = load_config(project)
    entry = register_project(project)
    store_dir = project_store_dir(entry.name)
    sources = (
        sources_from_paths(project, config, paths)
        if paths
        else listed_openapi_sources(project, config)
    )
    if not sources:
        err.print("no OpenAPI files")
        raise typer.Exit(code=1)
    err.print(f"[dim]root[/dim] {project}")
    err.print(f"[dim]store[/dim] {store_dir / 'openapi-md'}")
    results = split_source_files(sources, store_dir)
    table = Table(title="split-md")
    table.add_column("spec")
    table.add_column("ops", justify="right")
    table.add_column("stubs", justify="right")
    table.add_column("dest")
    for result in results:
        table.add_row(
            result.spec_rel,
            str(result.operations),
            str(result.stubs),
            str(result.dest),
        )
        console.print(f"{result.spec_rel}  {result.operations} ops  stubs={result.stubs}")
        console.print(f"  {result.dest}")
    console.print(table)


def split_md_entry() -> None:
    """Console-script entry for `bc-rag-split-md`."""
    app(args=["split-md", *sys.argv[1:]])


SettingKeyArg = Annotated[
    str,
    typer.Argument(help="Setting name. `bc-rag config list` shows them."),
]


def _settings_call(action: Callable[[], object]) -> object:
    from bc_rag.usersettings import SettingsError

    try:
        return action()
    except SettingsError as error:
        err.print(f"[red]config[/red] {error}")
        raise typer.Exit(code=1) from error


@config_app.command("set")
def config_set_cmd(
    key: SettingKeyArg,
    value: Annotated[str, typer.Argument(help="The value to store.")],
) -> None:
    """Store a user setting in ~/.bc-rag/config, such as voyage-api-key."""
    from bc_rag.usersettings import masked, set_setting, settings_path

    if key == "daemon-index-every":
        try:
            parse_every(value)
        except ValueError as error:
            err.print(f"[red]config[/red] {error}")
            raise typer.Exit(code=2) from error
    _settings_call(lambda: set_setting(key, value))
    console.print(f"set  {key} = {masked(key, value.strip())}  ({settings_path()})")


@config_app.command("get")
def config_get_cmd(
    key: SettingKeyArg,
    reveal: Annotated[bool, typer.Option("--reveal", help="Print a secret in full.")] = False,
) -> None:
    """Print a user setting: ~/.bc-rag/config first, then its environment variable."""
    from bc_rag.usersettings import get_setting, masked

    value = _settings_call(lambda: get_setting(key))
    if value is None:
        err.print(f"[yellow]not set[/yellow] {key}")
        raise typer.Exit(code=1)
    sys.stdout.write(f"{value if reveal else masked(key, str(value))}\n")


@config_app.command("unset")
def config_unset_cmd(key: SettingKeyArg) -> None:
    """Remove a user setting from ~/.bc-rag/config."""
    from bc_rag.usersettings import unset_setting

    removed = _settings_call(lambda: unset_setting(key))
    console.print(f"{'removed' if removed else 'was not set'}  {key}")


@config_app.command("list")
def config_list_cmd() -> None:
    """Show every user setting, where its value comes from, and secrets masked."""
    from bc_rag.usersettings import SETTINGS, config_setting, get_setting, masked, settings_path

    table = Table(title=f"settings  {settings_path()}")
    table.add_column("key")
    table.add_column("value")
    table.add_column("from")
    table.add_column("env fallback")
    table.add_column("meaning")
    for key, setting in SETTINGS.items():
        value = _settings_call(lambda key=key: get_setting(key))
        source = "config" if _settings_call(lambda key=key: config_setting(key)) else "env"
        table.add_row(
            key,
            masked(key, str(value)) if value else "",
            source if value else "",
            ", ".join(setting.env),
            setting.help,
        )
    console.print(table)


@daemon_app.command("start")
def daemon_start_cmd() -> None:
    """Start the daemon, or replace one that runs old code."""
    from bc_rag import daemon

    state = daemon.read_state()
    if daemon.is_running() and state.get("fingerprint") == daemon.source_fingerprint():
        console.print(f"running  pid {state.get('pid')}")
        return
    daemon.ensure(report=console.print)


@daemon_app.command("stop")
def daemon_stop_cmd() -> None:
    """Stop the daemon and both of its children."""
    from bc_rag import daemon

    try:
        stopped = daemon.stop()
    except RuntimeError as error:
        err.print(f"[red]daemon[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print("stopped" if stopped else "not running")


@daemon_app.command("restart")
def daemon_restart_cmd() -> None:
    """Stop the daemon, then start it again with this shell's environment."""
    from bc_rag import daemon

    daemon_stop_cmd()
    pid = daemon.spawn()
    console.print(f"started  pid {pid}  log {daemon.log_path()}")


@daemon_app.command("status")
def daemon_status_cmd() -> None:
    """Show whether the daemon runs, its pid, children, and log."""
    from bc_rag import daemon

    running = daemon.is_running()
    state = daemon.read_state() if running else {}
    current = state.get("fingerprint") == daemon.source_fingerprint()
    table = Table(title="daemon")
    table.add_column("key")
    table.add_column("value")
    table.add_row("running", "yes" if running else "no")
    table.add_row("pid", str(state.get("pid", "")))
    table.add_row("started", str(state.get("started_at", "")))
    table.add_row("code", ("current" if current else "old") if running else "")
    for name, args in dict(state.get("children") or {}).items():
        table.add_row(name, "bc-rag " + " ".join(args))
    table.add_row("log", str(daemon.log_path()))
    table.add_row("autostart", "off (BC_RAG_DAEMON)" if daemon.disabled() else "on")
    console.print(table)


@daemon_app.command("run", hidden=True)
def daemon_run_cmd() -> None:
    """The supervisor itself. `bc-rag daemon start` runs this in the background."""
    from bc_rag import daemon

    raise typer.Exit(code=daemon.run())


@app.command("mcp")
def mcp_cmd(
    http: Annotated[
        bool,
        typer.Option(
            "--http",
            help=(
                "Listen on HTTP so every chat shares one process. "
                "Also listens on https://bc-rag.localhost:32324/mcp when "
                "~/.bc-rag/mcp.pem and ~/.bc-rag/mcp.key exist."
            ),
        ),
    ] = False,
    host: Annotated[
        str,
        typer.Option("--host", help="Bind address for --http. Stays on this machine."),
    ] = MCP_HTTP_HOST,
    port: Annotated[
        int,
        typer.Option("--port", help="Port for --http."),
    ] = MCP_HTTP_PORT,
) -> None:
    """Run the MCP server. Default is stdio. --http is one shared listener."""
    from bc_rag.mcp_server import run_mcp

    run_mcp(http=http, host=host, port=port)
