"""`bc-rag config ...`: settings, like git config and npm config.

Without a flag, a command works on this project's settings (the project named by
--project or --root, else the one that holds the current folder). --global works on
~/.bc-rag/config, shared by every project.

Only a key in PROJECT_KEYS (today `every`) can be set for one project. The others
apply to every project, so they need --global.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from bc_rag.catalog import ProjectEntry
from bc_rag.cli._app import (
    ProjectOpt,
    RootOpt,
    command_group,
    console,
    err,
    fail,
    settings_call,
    soft_project,
)

config_app = command_group("Settings: this project by default, --global for every project.")

KeyArg = Annotated[str, typer.Argument(help="Setting name. `bc-rag config list` shows them.")]
GlobalOpt = Annotated[
    bool, typer.Option("--global", help="~/.bc-rag/config, shared by every project.")
]

# how setting_source names a source, as `config list` shows it.
_FROM = {
    "config": "you, every project",
    "env": "environment",
    "default": "default",
    "unset": "not set",
}


def _check_flags(global_: bool, project: str | None, root: Path | None) -> None:
    if global_ and (project is not None or root is not None):
        raise fail("config", "--global cannot be used with --project or --root", code=2)


def _known(key: str) -> None:
    from bc_rag.usersettings import SETTINGS

    if key not in SETTINGS:
        known = ", ".join(sorted(SETTINGS))
        raise fail("config", f"unknown setting {key!r}. Known: {known}", code=2)


def _project_for_write(key: str, project: str | None, root: Path | None) -> ProjectEntry:
    from bc_rag.usersettings import PROJECT_KEYS

    if key not in PROJECT_KEYS:
        raise fail(
            "config",
            f"{key} applies to every project. Add --global to set it in ~/.bc-rag/config.",
            code=2,
        )
    entry = soft_project(project, root)
    if entry is None:
        from bc_rag.catalog import load_catalog

        names = ", ".join(item.name for item in load_catalog()) or "none registered"
        raise fail(
            "config",
            f"not inside a project. Pass --project NAME ({names}), or --global.",
            code=2,
        )
    return entry


def _value(key: str, entry: ProjectEntry | None) -> tuple[str | None, str]:
    """(value in effect, where it comes from) for one key."""
    from bc_rag.usersettings import PROJECT_KEYS, setting_source

    if key in PROJECT_KEYS and entry is not None:
        from bc_rag.schedule import SOURCE_LABELS, effective_every

        try:
            value, _seconds, source = effective_every(entry)
        except ValueError as error:
            return None, f".bc-rag.json unreadable: {error}"
        return value, SOURCE_LABELS[source]
    value, source = settings_call(lambda: setting_source(key))
    return value, _FROM.get(source, source)


@config_app.command("list")
def config_list(
    project: ProjectOpt = None,
    root: RootOpt = None,
    global_: GlobalOpt = False,
) -> None:
    """Every key, its value, and where it comes from."""
    from bc_rag.usersettings import PROJECT_KEYS, SETTINGS, load_settings, masked, settings_path

    _check_flags(global_, project, root)
    entry = None if global_ else soft_project(project, root)
    title = f"config  {settings_path()}"
    if entry is not None:
        title += f"  and project {entry.name}"
    table = Table(title=title)
    for column in ("key", "value", "from", "scope", "env", "meaning"):
        table.add_column(column)
    stored = settings_call(load_settings) if global_ else {}
    for key, setting in SETTINGS.items():
        if global_:
            value = stored.get(key)
            where = "you, every project" if value is not None else "not set"
        else:
            value, where = _value(key, entry)
        table.add_row(
            key,
            masked(key, value) if value else "",
            where,
            "project or global" if key in PROJECT_KEYS else "global",
            ", ".join(setting.env),
            setting.help,
        )
    console.print(table)


@config_app.command("get")
def config_get(
    key: KeyArg,
    project: ProjectOpt = None,
    root: RootOpt = None,
    global_: GlobalOpt = False,
    reveal: Annotated[bool, typer.Option("--reveal", help="Print a secret in full.")] = False,
) -> None:
    """The value in effect, and where it comes from (on stderr)."""
    from bc_rag.usersettings import config_setting, masked

    _check_flags(global_, project, root)
    _known(key)
    if global_:
        value = settings_call(lambda: config_setting(key))
        where = "you, every project"
    else:
        value, where = _value(key, soft_project(project, root))
    if value is None:
        err.print(f"[yellow]not set[/yellow] {key}")
        raise typer.Exit(code=1)
    err.print(f"[dim]from {where}[/dim]")
    sys.stdout.write(f"{value if reveal else masked(key, value)}\n")


@config_app.command("set")
def config_set(
    key: KeyArg,
    value: Annotated[str, typer.Argument(help="The value to store.")],
    project: ProjectOpt = None,
    root: RootOpt = None,
    global_: Annotated[
        bool,
        typer.Option("--global", help="Write ~/.bc-rag/config instead, for every project."),
    ] = False,
) -> None:
    """Your setting for this project, like `every 10m`. --global: for every project."""
    from bc_rag.catalog import set_project_every
    from bc_rag.usersettings import masked, set_setting, settings_path, validate_setting

    _check_flags(global_, project, root)
    _known(key)
    if global_:
        stored = settings_call(lambda: set_setting(key, value))
        console.print(f"set  {key} = {masked(key, stored)}  (every project, {settings_path()})")
        return
    entry = _project_for_write(key, project, root)
    stored = settings_call(lambda: validate_setting(key, value))
    set_project_every(entry.name, stored)
    console.print(f"set  {key} = {stored}  (project {entry.name})")


@config_app.command("unset")
def config_unset(
    key: KeyArg,
    project: ProjectOpt = None,
    root: RootOpt = None,
    global_: Annotated[
        bool, typer.Option("--global", help="Drop it from ~/.bc-rag/config instead.")
    ] = False,
) -> None:
    """Drop your setting for this project, so the next source applies again."""
    from bc_rag.catalog import set_project_every
    from bc_rag.usersettings import SETTINGS, unset_setting

    _check_flags(global_, project, root)
    _known(key)
    if global_:
        removed = settings_call(lambda: unset_setting(key))
        console.print(f"{'removed' if removed else 'was not set'}  {key}  (every project)")
        default = SETTINGS[key].default
        if removed and default is not None:
            shown = default or "(empty)"
            console.print(f"the next bc-rag command writes the default back: {shown}")
        return
    entry = _project_for_write(key, project, root)
    updated = set_project_every(entry.name, None)
    value, where = _value(key, updated)
    console.print(f"unset  {key}  (project {entry.name}); now {value or 'not set'}  ({where})")
