"""`bc-rag models ...`: the FastEmbed model cache."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from bc_rag.cli._app import ProjectOpt, RootOpt, command_group, console, err, fail, soft_project

models_app = command_group("The FastEmbed model cache in ~/.cache/bc-rag/fastembed.")


@models_app.command("list")
def models_list(project: ProjectOpt = None, root: RootOpt = None) -> None:
    """List cached models, and which ones this project uses."""
    from bc_rag.models import cache_directory, list_models

    entry = soft_project(project, root)
    err.print(f"[dim]cache[/dim] {cache_directory()}")
    if entry is not None:
        err.print(f"[dim]project[/dim] {entry.name}")
    table = Table(title="models")
    for column in ("used here", "cached", "kind", "GB", "name"):
        table.add_column(column, justify="right" if column == "GB" else "left")
    for record in list_models(entry.root_path() if entry is not None else None):
        table.add_row(
            "yes" if record.configured else "",
            "yes" if record.cached else "",
            record.kind,
            f"{record.size_gb:.2f}" if record.size_gb is not None else "",
            record.name,
        )
    console.print(table)


@models_app.command("add")
def models_add(
    names: Annotated[
        list[str] | None,
        typer.Option("--name", help="Model id. Repeatable. Defaults to this project's local ones."),
    ] = None,
    project: ProjectOpt = None,
    root: RootOpt = None,
) -> None:
    """Download models into the cache. Defaults to this project's."""
    from bc_rag.models import (
        cache_directory,
        configured_model_names,
        download_model,
        is_api_model,
    )

    targets = list(names or [])
    if not targets:
        entry = soft_project(project, root)
        if entry is None:
            raise fail("models add", "outside a project, name the models with --name", code=2)
        every = configured_model_names(entry.root_path())
        for name in every:
            if is_api_model(name):
                err.print(f"[dim]skip {name}  (API model, runs remotely)[/dim]")
            else:
                targets.append(name)
    err.print(f"[dim]cache[/dim] {cache_directory()}")
    for name in targets:
        err.print(f"download {name}")
        try:
            path = download_model(name)
        except ValueError as error:
            raise fail("models add", error) from error
        console.print(f"[green]ready[/green] {name}  {path}")


@models_app.command("remove")
def models_remove(
    names: Annotated[
        list[str],
        typer.Option("--name", help="Model id to delete from the cache. Repeatable."),
    ],
) -> None:
    """Delete cached model folders by name."""
    from bc_rag.models import remove_model

    for name in names:
        try:
            path = remove_model(name)
        except FileNotFoundError as error:
            raise fail("models remove", error) from error
        console.print(f"[red]removed[/red] {name}  {path}")
