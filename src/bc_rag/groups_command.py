"""Run the project command that prints extra groups as JSON."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# A generator that walks an Nx graph can take a couple of minutes.
_TIMEOUT_SECONDS = 180


class GroupsCommandError(ValueError):
    pass


def argv_for(root: Path, command: str | list[str]) -> list[str]:
    """A path runs as that program. A list is argv, with no shell."""
    if isinstance(command, str):
        program = Path(command)
        if not program.is_absolute():
            program = root / program
        return [str(program)]
    if not command:
        raise GroupsCommandError("groupsCommand is an empty list")
    argv = list(command)
    program = Path(argv[0])
    if not program.is_absolute() and "/" in argv[0]:
        argv[0] = str(root / program)
    return argv


def read_groups(root: Path, command: str | list[str]) -> list[dict[str, Any]]:
    argv = argv_for(root, command)
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise GroupsCommandError(
            f"groupsCommand timed out after {_TIMEOUT_SECONDS}s: {argv[0]}"
        ) from error
    except OSError as error:
        raise GroupsCommandError(f"groupsCommand could not start: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise GroupsCommandError(
            f"groupsCommand exited {completed.returncode}: {detail or argv[0]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GroupsCommandError(f"groupsCommand stdout is not JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise GroupsCommandError("groupsCommand JSON must be an object with a groups array")
    groups = payload["groups"]
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise GroupsCommandError(f"groupsCommand groups[{index}] is not an object")
    return groups
