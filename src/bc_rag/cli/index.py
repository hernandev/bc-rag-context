"""`bc-rag index | search | status | facets`: the verbs that work on one project's index."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from bc_rag.cli._app import (
    ExcludeOpt,
    FacetOpt,
    ProjectOpt,
    RootOpt,
    app,
    console,
    err,
    fail,
    parse_facet_flags,
    print_index_stats,
    print_json,
    print_query_result,
    require_project,
)

PANEL = "Index"


def stack() -> None:
    """Bring up the services an index command needs. Never raises."""
    from bc_rag import services

    services.ensure_stack(report=lambda message: err.print(f"[dim]{message}[/dim]"))


@app.command("index", rich_help_panel=PANEL)
def index(
    project: ProjectOpt = None,
    root: RootOpt = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild every space, ignoring file hashes.")
    ] = False,
    path: Annotated[
        list[str] | None,
        typer.Option("--path", help="Only these project-relative paths. Repeatable."),
    ] = None,
    watch: Annotated[
        bool, typer.Option("--watch", help="Keep running and reindex files as they change.")
    ] = False,
    debounce: Annotated[
        float | None,
        typer.Option(
            "--debounce", help="With --watch: seconds of quiet before hashing. Default 0."
        ),
    ] = None,
    skip_initial: Annotated[
        bool,
        typer.Option("--skip-initial", help="With --watch: skip the full pass before watching."),
    ] = False,
) -> None:
    """Index the project. Unchanged files are skipped unless --force."""
    if not watch and (debounce is not None or skip_initial):
        raise fail("index", "--debounce and --skip-initial need --watch", code=2)
    if watch and path:
        raise fail("index", "--path and --watch cannot be used together", code=2)
    entry = require_project(project, root)
    stack()
    if watch:
        _watch(entry, force=force, debounce=debounce or 0.0, skip_initial=skip_initial)
        return
    from bc_rag.runtime import index_project

    try:
        stats = index_project(entry, force=force, only_paths=path or None, console=err)
    except KeyboardInterrupt:
        err.print("stopped  aborted")
        raise typer.Exit(code=130) from None
    print_index_stats(stats)
    if stats.errors:
        raise typer.Exit(code=1)


def _watch(entry, *, force: bool, debounce: float, skip_initial: bool) -> None:
    from bc_rag.indexer import Indexer, index_spaces
    from bc_rag.runtime import open_space_sessions
    from bc_rag.watch import WatchSession

    sessions = open_space_sessions(entry, write=True, console=err)
    try:
        indexers = [
            Indexer(
                entry, session.config, session.embedder, session.store, err, space=session.space
            )
            for session in sessions
        ]

        class _Fanout:
            def run(self, **kwargs):
                return index_spaces(indexers, **kwargs)

        fanout = _Fanout()
        first = sessions[0]
        if skip_initial:
            err.print("[dim]watch[/dim] skip initial index")
        else:
            err.print("[dim]initial index[/dim]")
            stats = fanout.run(force=force)
            err.print(f"[green]ready[/green] files={stats.indexed_files + stats.skipped_unchanged}")
        WatchSession(entry.root_path(), first.config, fanout, err, debounce_s=debounce).run()
    finally:
        for session in sessions:
            session.close()


@app.command("search", rich_help_panel=PANEL)
def search(
    query: Annotated[str, typer.Argument(help="The search text.")],
    project: ProjectOpt = None,
    root: RootOpt = None,
    space: Annotated[
        str | None, typer.Option("--space", "-s", help="Space to search. Defaults to defaultSpace.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Hits to return. Defaults to the space's.")
    ] = None,
    sparse: Annotated[
        bool, typer.Option("--sparse", help="BM25 only. No dense vector and no rerank.")
    ] = False,
    no_rerank: Annotated[bool, typer.Option("--no-rerank", help="Skip the rerank pass.")] = False,
    facet: FacetOpt = None,
    exclude: ExcludeOpt = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Search the project. Dense, then rerank; --sparse is BM25 only."""
    from bc_rag.config import load_config
    from bc_rag.query import search as dense_search
    from bc_rag.query import search_sparse
    from bc_rag.runtime import open_session

    text = query.strip()
    if not text:
        raise fail("search", "empty query", code=2)
    wanted, unwanted = parse_facet_flags(facet, exclude)
    entry = require_project(project, root)
    stack()
    try:
        config, config_path = load_config(entry.root_path(), groups=False)
        session = open_session(
            entry,
            need_reranker=not (sparse or no_rerank),
            write=False,
            console=err,
            space=space,
            config=config,
            config_path=config_path,
        )
    except ValueError as error:
        raise fail("search", error) from error
    try:
        if session.store.count() == 0:
            raise fail("search", "empty index. Run `bc-rag index` first.")
        if sparse:
            result = search_sparse(
                query=text,
                config=session.config,
                embedder=session.embedder,
                store=session.store,
                limit=limit,
                facets=wanted,
                exclude=unwanted,
                space=session.space,
            )
        else:
            result = dense_search(
                query=text,
                config=session.config,
                embedder=session.embedder,
                store=session.store,
                reranker=session.reranker,
                limit=limit,
                use_rerank=False if no_rerank else None,
                facets=wanted,
                exclude=unwanted,
                space=session.space,
            )
    finally:
        session.close()
    print_query_result(result, json_out=json_out)


@app.command("status", rich_help_panel=PANEL)
def status(
    project: ProjectOpt = None,
    root: RootOpt = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Each space with its points and files, and the last and next background run."""
    from dataclasses import asdict

    from bc_rag.catalog import project_store_dir
    from bc_rag.config import config_path
    from bc_rag.overview import project_overview
    from bc_rag.schedule import describe_every, load_state

    entry = require_project(project, root)
    stack()
    problem = None
    try:
        every, every_from = describe_every(entry)
    except ValueError as error:
        every, every_from, problem = "?", "", str(error)
    state = load_state(entry.name)
    try:
        rows = project_overview(entry)
    except ValueError as error:
        raise fail("status", error) from error
    last = state.last_finish or "never"
    if state.result:
        last += f"  ({state.result})"
    next_run = state.next_due or ("due now" if every not in {"none", "off", "?"} else "-")
    if json_out:
        print_json(
            {
                "project": entry.name,
                "root": entry.root,
                "every": every,
                "every_from": every_from,
                "indexing_now": state.running,
                "last_index": state.last_finish,
                "last_result": state.result,
                "last_error": state.error,
                "next_index": state.next_due,
                "spaces": [asdict(row) for row in rows],
            }
        )
        return
    table = Table(title=f"project {entry.name}", show_header=False)
    table.add_column("key")
    table.add_column("value")
    table.add_row("root", entry.root)
    table.add_row("config", str(config_path(entry.root_path())))
    table.add_row("every", f"{every}  ({every_from})" if every_from else every)
    if problem is not None:
        table.add_row("config error", problem)
    if state.running:
        table.add_row("indexing now", f"since {state.last_start}  (pid {state.pid})")
    table.add_row("last index", last)
    if state.error:
        table.add_row("last error", state.error)
    table.add_row("next index", next_run)
    table.add_row("store", str(project_store_dir(entry.name)))
    console.print(table)
    spaces = Table(title="spaces")
    for column in ("space", "dense", "rerank", "collection", "points", "files", "indexed"):
        spaces.add_column(column, justify="right" if column in ("points", "files") else "left")
    for row in rows:
        spaces.add_row(
            row.space,
            row.dense,
            row.rerank,
            row.collection,
            "(unavailable)" if row.points is None else str(row.points),
            str(row.files),
            row.manifest_time or "never",
        )
    console.print(spaces)
    for row in rows:
        if row.error:
            err.print(f"[yellow]{row.space}[/yellow] qdrant: {row.error}")


@app.command("facets", rich_help_panel=PANEL)
def facets(
    project: ProjectOpt = None,
    root: RootOpt = None,
    key: Annotated[
        str | None, typer.Option("--key", "-k", help="One key, with every one of its values.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Facets declared in .bc-rag.json and stored in each space, with point counts."""
    from dataclasses import asdict

    from bc_rag.overview import facets_report

    entry = require_project(project, root)
    stack()
    try:
        report = facets_report(entry, key=key)
    except ValueError as error:
        raise fail("facets", error) from error
    for space, message in report.errors.items():
        err.print(f"[yellow]{space}[/yellow] qdrant unreadable: {message}")
    if json_out:
        print_json({"project": entry.name, **asdict(report)})
        return
    table = Table(title=f"facets {entry.name}")
    table.add_column("key")
    table.add_column("value")
    table.add_column("source")
    for space in report.spaces:
        table.add_column(space, justify="right")
    for row in report.rows:
        cells = []
        for space in report.spaces:
            if space not in row.points:
                cells.append("?")
            elif row.points[space] is None:
                cells.append("-")
            else:
                cells.append(str(row.points[space]))
        value = row.value if row.value is not None else f"({row.distinct} values; --key {row.key})"
        table.add_row(row.key, value, row.source, *cells)
    console.print(table)
    err.print("[dim]- = no index for that key in the space; ? = Qdrant could not be read[/dim]")
