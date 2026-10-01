"""`bc-rag project ...`: the registered projects, and what is stored for each."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from bc_rag.catalog import (
    CatalogError,
    ProjectEntry,
    ProjectNotFound,
    forget_project,
    living_projects,
    load_catalog,
    project_store_dir,
    register_project,
)
from bc_rag.cli._app import (
    ProjectOpt,
    RootOpt,
    command_group,
    console,
    err,
    fail,
    require_project,
)

project_app = command_group("Registered projects, and what is stored for each.")


@project_app.command("list")
def project_list() -> None:
    """Every registered project, its interval, and its last index."""
    from bc_rag.schedule import describe_every, load_state

    entries = load_catalog()
    if not entries:
        console.print("no projects. run `bc-rag project add` in a repo")
        return
    living = {entry.name for entry in living_projects(entries)}
    table = Table(title=f"projects ({len(entries)})")
    table.add_column("name")
    table.add_column("root")
    table.add_column("every")
    table.add_column("from")
    table.add_column("last index")
    for entry in entries:
        try:
            every, every_from = describe_every(entry)
        except ValueError:
            every, every_from = "?", ".bc-rag.json unreadable"
        state = load_state(entry.name)
        last = state.last_finish or ""
        if state.result and state.result != "ok":
            last = f"{last} ({state.result})".strip()
        root = entry.root if entry.name in living else f"{entry.root}  (missing)"
        table.add_row(entry.name, root, every, every_from, last)
    console.print(table)


@project_app.command("add")
def project_add(
    root: Annotated[
        Path | None,
        typer.Option("--root", "-r", help="The project folder. Defaults to the current folder."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", "-p", help="The name to register. Defaults to the folder name."),
    ] = None,
) -> None:
    """Register a folder. Writes .bc-rag.json if missing."""
    from bc_rag.config import config_path, dump_default_config, load_config

    folder = (root or Path.cwd()).expanduser().resolve()
    if not folder.is_dir():
        raise fail("project add", f"not a folder: {folder}", code=2)
    path = config_path(folder)
    if path.exists():
        console.print(f"config  {path}  (kept)")
    else:
        path.write_text(dump_default_config(), encoding="utf-8")
        console.print(f"[green]wrote[/green]  {path}")
    try:
        # a broken file fails here, once, instead of on every background run.
        load_config(folder)
    except Exception as error:
        raise fail("project add", f"{path}: {error}") from error
    try:
        entry = register_project(folder, project)
    except CatalogError as error:
        raise fail("project add", error) from error
    console.print(f"[green]registered[/green]  {entry.name}  {entry.root}")
    console.print(f"store  {project_store_dir(entry.name)}")


@project_app.command("clear")
def project_clear(
    project: ProjectOpt = None,
    root: RootOpt = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask first.")] = False,
) -> None:
    """Delete the stored index: collections and cache. The project stays registered."""
    from bc_rag.cli.index import stack
    from bc_rag.config import load_config
    from bc_rag.reset import reset_collections, reset_project, reset_targets

    entry = require_project(project, root)
    stack()
    project_root = entry.root_path()
    try:
        configuration, _path = load_config(project_root, groups=False)
    except ValueError as error:
        raise fail("project clear", error) from error
    err.print(f"[dim]project[/dim] {entry.name}  {entry.root}")
    err.print("[dim]will remove[/dim]")
    for target in reset_targets(project_root, configuration):
        err.print(f"  {target}")
    for collection in reset_collections(project_root, configuration):
        err.print(f"  qdrant collection {collection}")
    if not yes and not typer.confirm("Delete this project's stored index?", default=False):
        err.print("clear  cancelled")
        raise typer.Exit(code=1)
    report = reset_project(project_root, configuration)
    _print_clear(entry, report)


def _print_clear(entry: ProjectEntry, report) -> None:
    table = Table(title=f"clear {entry.name}")
    table.add_column("what")
    table.add_column("result")
    for path in report.removed:
        table.add_row(path, "removed")
    for path in report.missing:
        table.add_row(path, "already gone")
    for collection, status in report.collections:
        table.add_row(f"collection {collection}", status)
    console.print(table)


@project_app.command("remove")
def project_remove(project: ProjectOpt = None, root: RootOpt = None) -> None:
    """Unregister the project. Its stored index stays on disk and in Qdrant."""
    entry = require_project(project, root)
    removed = forget_project(entry.name)
    if removed is None:
        raise fail("project remove", ProjectNotFound(project=entry.name))
    console.print(f"[green]removed[/green]  {removed.name}  ({removed.root})")
    console.print(
        f"its index stays in {project_store_dir(removed.name)} and in Qdrant. "
        "Run `bc-rag project clear` before removing to drop it."
    )
