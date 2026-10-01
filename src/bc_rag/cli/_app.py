"""Shared pieces of the bc-rag command line: consoles, the root app, and project lookup."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer._click.exceptions import UsageError
from typer.core import TyperGroup

from bc_rag import __version__
from bc_rag.catalog import ProjectEntry, ProjectNotFound, resolve_project, try_resolve_project

if TYPE_CHECKING:
    # Typer bundles its own Click. Only the annotations below name it.
    from typer import _click as click

console = Console()
err = Console(stderr=True)


def command_group(help_text: str) -> typer.Typer:
    """A group that only names its subcommands. Run bare, it exits 2 and lists them."""
    group = typer.Typer(help=help_text, invoke_without_command=True)

    @group.callback()
    def _missing_command(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is not None:
            return
        from typer import rich_utils

        visible = [
            name
            for name in ctx.command.list_commands(ctx)
            if not ctx.command.get_command(ctx, name).hidden
        ]
        message = f"Please specify a subcommand: {', '.join(visible)}."
        rich_utils.rich_format_error(UsageError(message, ctx))
        err.print(ctx.get_help())
        raise typer.Exit(code=2)

    return group


class ExpandedHelpGroup(TyperGroup):
    """The root group. Its --help lists every command under its group, as `group sub`.

    Each group sits under its own heading (its rich_help_panel), and each level of
    nesting is indented further. The extra rows exist only while help is printed, so
    `bc-rag "project list"` still fails.
    """

    _expanding = False

    # left padding added per level of nesting.
    _INDENT = "   "

    def list_commands(self, ctx: typer.Context) -> list[str]:
        names = super().list_commands(ctx)
        if not self._expanding:
            return names
        expanded: list[str] = []
        for name in names:
            command = super().get_command(ctx, name)
            if command is None or command.hidden:
                continue
            # the command, then everything under it.
            expanded.extend([name, *self._descendants(ctx, command, name)])
        return expanded

    def _descendants(self, ctx: typer.Context, command: click.Command, path: str) -> list[str]:
        if not isinstance(command, TyperGroup):
            return []
        rows: list[str] = []
        for sub in command.list_commands(ctx):
            child = command.get_command(ctx, sub)
            if child.hidden:
                continue
            rows.append(f"{path} {sub}")
            rows.extend(self._descendants(ctx, child, f"{path} {sub}"))
        return rows

    def _resolve(self, ctx: typer.Context, words: list[str]) -> click.Command | None:
        command: click.Command | None = super().get_command(ctx, words[0])
        for word in words[1:]:
            if not isinstance(command, TyperGroup):
                return None
            command = command.get_command(ctx, word)
        return command

    def get_command(self, ctx: typer.Context, cmd_name: str) -> click.Command | None:
        if not self._expanding:
            return super().get_command(ctx, cmd_name)
        words = cmd_name.split(" ")
        found = self._resolve(ctx, words)
        if found is None or len(words) == 1:
            return found
        # a copy carries the indented path, so the real command keeps its own name.
        shown = copy.copy(found)
        shown.name = f"{self._INDENT * (len(words) - 1)}{cmd_name}"
        # and the heading of its top-level group, so `sources files` sits under Sources.
        top = super().get_command(ctx, words[0])
        shown.rich_help_panel = getattr(top, "rich_help_panel", None)
        return shown

    def format_help(self, ctx: typer.Context, formatter: click.HelpFormatter) -> None:
        self._expanding = True
        try:
            super().format_help(ctx, formatter)
        finally:
            self._expanding = False


app = typer.Typer(
    name="bc-rag",
    cls=ExpandedHelpGroup,
    no_args_is_help=True,
    add_completion=False,
    help="Local hybrid search over TypeScript, Markdown, and OpenAPI, served to Claude over MCP.",
)


def _print_version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


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
    if ctx.resilient_parsing:
        return
    from bc_rag.usersettings import SettingsError, ensure_defaults, settings_path

    try:
        added = ensure_defaults()
    except SettingsError as error:
        # a broken config file must never block --help or `config set --global`.
        err.print(f"[yellow]config[/yellow] {error}")
        return
    if added:
        keys = ", ".join(added)
        err.print(f"[dim]config  wrote defaults for {keys} to {settings_path()}[/dim]")


# -- project lookup -------------------------------------------------------------

ProjectOpt = Annotated[
    str | None,
    typer.Option(
        "--project",
        "-p",
        help="Registered project name. Defaults to the project that holds the current folder.",
    ),
]
RootOpt = Annotated[
    Path | None,
    typer.Option(
        "--root",
        "-r",
        help="A folder inside the project. Defaults to the current folder.",
    ),
]


FacetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--facet",
        "-f",
        help="Keep matches of key=value. Repeat a key for any-of; different keys all match.",
    ),
]
ExcludeOpt = Annotated[
    list[str] | None,
    typer.Option("--exclude", "-x", help="Drop matches of key=value. Repeatable."),
]


def parse_facet_flags(facet: list[str] | None, exclude: list[str] | None):
    """(facets, exclude) objects from the flags, or exit 2 naming the bad flag."""
    from bc_rag.facets import parse_facet_options

    try:
        return (
            parse_facet_options(facet, option="--facet"),
            parse_facet_options(exclude, option="--exclude"),
        )
    except ValueError as error:
        raise fail("facets", error, code=2) from error


def require_project(project: str | None, root: Path | None) -> ProjectEntry:
    """The registered project, or exit 2 naming the registered ones."""
    from bc_rag.catalog import load_catalog

    try:
        return resolve_project(project, root)
    except ProjectNotFound as error:
        names = ", ".join(entry.name for entry in load_catalog())
        if error.project is not None:
            err.print(f"[red]unknown project: {error.project}[/red]")
        else:
            err.print(f"[red]not a bc-rag project: {error.path}[/red]")
        if names:
            err.print(f"pass --project NAME. Registered: {names}")
        if error.project is None:
            err.print(
                "or run `bc-rag project add` here "
                "(it writes a starter .bc-rag.json when the folder has none)"
            )
        raise typer.Exit(code=2) from None
    except ValueError as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from None


def soft_project(project: str | None, root: Path | None) -> ProjectEntry | None:
    """The project when there is one. A wrong --project still exits 2."""
    try:
        return try_resolve_project(project, root)
    except (ProjectNotFound, ValueError):
        return require_project(project, root)


def fail(label: str, error: object, *, code: int = 1) -> typer.Exit:
    """Print `label error` in red. Returns the Exit to raise."""
    err.print(f"[red]{label}[/red] {error}")
    return typer.Exit(code=code)


def settings_call[T](action: Callable[[], T]) -> T:
    from bc_rag.usersettings import SettingsError

    try:
        return action()
    except SettingsError as error:
        raise fail("config", error) from error


def print_json(data: object) -> None:
    console.print_json(json.dumps(data))


# -- shared output ----------------------------------------------------------------


def hit_payload(hit) -> dict:
    return {
        "score": hit.score,
        "path": hit.path,
        "language": hit.language,
        "kind": hit.kind,
        "symbol": hit.symbol,
        "heading_path": hit.heading_path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "facets": hit.facets,
        "text": hit.text,
    }


_LEXERS = {
    "typescript": "ts",
    "tsx": "tsx",
    "javascript": "js",
    "python": "python",
    "markdown": "markdown",
    "openapi": "markdown",
    "json": "json",
}


def print_query_result(result, *, json_out: bool) -> None:
    if json_out:
        print_json(
            {
                "query": result.query,
                "reranked": result.reranked,
                "hits": [hit_payload(hit) for hit in result.hits],
            }
        )
        return
    if not result.hits:
        console.print("[yellow]no hits[/yellow]")
        return
    for index, hit in enumerate(result.hits, start=1):
        where = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        extras = []
        group_name = (hit.facets.get("group") or [None])[0]
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
        lexer = _LEXERS.get(hit.language, "text")
        console.print(Syntax(hit.text.rstrip() + "\n", lexer, word_wrap=True, padding=1))


def print_index_stats(stats) -> None:
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
        table.add_row("embed calls", str(stats.jina_embed_calls))
        table.add_row("embed tokens", str(stats.jina_embed_tokens))
    console.print(table)
    for message in stats.errors:
        err.print(f"[red]error[/red] {message}")
