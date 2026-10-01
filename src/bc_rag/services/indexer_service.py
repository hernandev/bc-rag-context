"""The indexer service: indexes each project whose interval is due, one at a time.

Every few seconds it re-reads the catalog and each project's interval, the config
key `every` (without running groupsCommand), and indexes the project that has waited longest
past its due time. A project with no interval is never indexed here.

SIGTERM, or a first Ctrl+C, sets `stop`: the current pass finishes the files in
flight, saves its manifest, and the service exits.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.console import Console

from bc_rag.catalog import ProjectEntry, living_projects
from bc_rag.schedule import (
    ScheduleState,
    effective_every,
    load_state,
    mark_finished,
    mark_started,
    save_state,
)

POLL_SECONDS = 5.0
QDRANT_RETRY = 30.0


@dataclass
class DueProject:
    entry: ProjectEntry
    every: str | None
    seconds: float
    source: str
    state: ScheduleState


def due_projects(now: datetime, say: Callable[[str], None]) -> list[DueProject]:
    """Projects whose interval has passed, the longest-waiting first."""
    due: list[tuple[float, DueProject]] = []
    for entry in living_projects():
        try:
            every, seconds, source = effective_every(entry)
        except ValueError as error:
            say(f"{entry.name}: schedule skipped: {error}")
            continue
        if seconds is None:
            continue
        state = load_state(entry.name)
        if not state.is_due(seconds, now):
            continue
        at = state.due_at(seconds)
        due.append(
            (at.timestamp() if at else 0.0, DueProject(entry, every, seconds, source, state))
        )
    due.sort(key=lambda row: row[0])
    return [row for _at, row in due]


def run_indexer_service(
    stop: threading.Event,
    console: Console,
    *,
    say: Callable[[str], None] = print,
    poll: float = POLL_SECONDS,
    once: bool = False,
) -> None:
    """The service loop. `once` runs a single pass (tests)."""
    from bc_rag.indexer import IndexBusyError
    from bc_rag.runtime import index_project
    from bc_rag.services import qdrant

    qdrant_down_since: float | None = None
    while not stop.is_set():
        if not qdrant.reachable():
            if qdrant_down_since is None:
                say(f"qdrant unreachable at {qdrant.url()}; waiting")
                qdrant_down_since = time.monotonic()
            if once:
                return
            stop.wait(QDRANT_RETRY)
            continue
        qdrant_down_since = None
        for item in due_projects(datetime.now(UTC), say):
            if stop.is_set():
                break
            state = item.state
            mark_started(state, every=item.every, source=item.source, pid=os.getpid())
            save_state(state)
            say(f"{item.entry.name}: indexing (every {item.every}, {item.source})")
            try:
                stats = index_project(item.entry, console=console, stop=stop)
            except IndexBusyError as error:
                mark_finished(state, result="busy", every_seconds=item.seconds, error=str(error))
                say(f"{item.entry.name}: skipped, {error}")
            except Exception as error:
                # one broken project must not stop the others.
                mark_finished(
                    state, result="error", every_seconds=item.seconds, error=str(error)
                )
                say(f"{item.entry.name}: failed: {error}")
            else:
                result = "stopped" if stats.stopped else ("error" if stats.errors else "ok")
                mark_finished(
                    state,
                    result=result,
                    every_seconds=item.seconds,
                    error="; ".join(stats.errors[:3]) or None,
                    stats={
                        "scanned": stats.scanned,
                        "indexed": stats.indexed_files,
                        "unchanged": stats.skipped_unchanged,
                        "deleted": stats.deleted_files,
                        "chunks": stats.chunks,
                        "errors": len(stats.errors),
                    },
                )
                say(
                    f"{item.entry.name}: {result}  indexed={stats.indexed_files} "
                    f"unchanged={stats.skipped_unchanged} deleted={stats.deleted_files} "
                    f"chunks={stats.chunks} errors={len(stats.errors)}"
                )
            save_state(state)
        if once:
            return
        stop.wait(poll)
