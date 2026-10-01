"""The index log: one JSONL file per project, and a live panel on a terminal.

`~/.bc-rag/{project-name}/index.jsonl` gets one line per event. Counters move only
when work completes:

- chunk stage: `chunked`, `skip_hash`, `skip_large`, `empty`, `error`
- post stage: `embedded` (a request answered)
- upsert stage: `upserted` (points written), then `file_complete` for each file whose
  last chunk just landed. Only `file_complete` counts a file as indexed.

The panel is three boxes side by side, one per stage, in pipeline order. Every box
has the same rows, so they read across, then one line per item its workers hold.
The panel draws itself four times a second; events only change numbers. Problems
(errors, skipped files) print as a plain list once the panel stops.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bc_rag.defaults import INDEX_LOG_FILENAME

ERROR_EVENTS = frozenset({"error"})
WARN_EVENTS = frozenset({"skip_large", "empty", "delete_missing"})
# what prints after the panel, or as it happens when there is no panel.
PROBLEM_EVENTS = frozenset({"error", "skip_large", "empty"})
# printed as they happen when there is no panel (the services indexer log).
PLAIN_EVENTS = frozenset({"file_complete", "deleted", "space", "done", "stopped"}) | PROBLEM_EVENTS
# a chunk-stage outcome: the file left the chunk stage.
CHUNK_OUTCOMES = frozenset({"chunked", "skip_hash", "skip_large", "empty", "error"})
VERB_WIDTH = 8
STAGES = ("chunk", "post", "upsert")
# rows every box has, in this order, one fact each:
#   working, queued, done, chunks, then FILE_ROWS file counts, then last, eta, errors,
#   then RATE_ROWS rate limits. A box without a fact for a slot leaves it blank.
FILE_ROWS = 2
RATE_ROWS = 4
FLUSH_SECONDS = 1.0


@dataclass
class _Held:
    space: str
    paths: list[str]
    chunks: int | None
    started: float


@dataclass
class _StageState:
    workers: int = 0
    queued: int = 0
    # items finished by this stage's workers, and when the first one started.
    finished: int = 0
    first_started: float = 0.0
    last_seconds: float = 0.0
    errors: int = 0
    held: dict[int, _Held] = field(default_factory=dict)


class IndexLog:
    def __init__(self, store_dir: Path, console: Console, *, catalog_name: str, root: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self.path = store_dir / INDEX_LOG_FILENAME
        self.console = console
        self.catalog_name = catalog_name
        self.root = str(root)
        self._handle = self.path.open("w", encoding="utf-8")
        self._last_flush = time.perf_counter()
        self._lock = threading.Lock()
        self._tokens = itertools.count(1)
        self.embed_label = ""
        self.files_total = 0
        self.bytes_total = 0
        self.stages = {stage: _StageState() for stage in STAGES}
        # chunk stage
        self.files_chunk_done = 0
        self.files_chunked = 0
        self.files_skipped = 0
        self.chunks_made = 0
        # post stage
        self.batches_total = 0
        self.batches_embedded = 0
        self.chunks_embedded = 0
        # upsert stage
        self.batches_upserted = 0
        self.chunks_stored = 0
        self.files_complete = 0
        self.problems: list[str] = []
        self.stopped = False
        self._live: Live | None = None

    # -- setup -------------------------------------------------------------------

    def start(self) -> None:
        if not self.console.is_terminal:
            return
        # the panel pulls its picture on a timer; events never redraw it.
        self._live = Live(
            console=self.console, refresh_per_second=4, get_renderable=self._render
        )
        self._live.start()

    def close(self) -> None:
        # stopping draws one last frame, which takes the lock, so it runs outside it.
        live, self._live = self._live, None
        if live is not None:
            live.stop()
            for line in self.problems:
                self.console.print(line, markup=False, highlight=False)
            if self.stopped:
                self.console.print("stopped  resume with bc-rag index")
        with self._lock:
            self._handle.close()

    def set_totals(self, *, files: int, nbytes: int) -> None:
        with self._lock:
            self.files_total = files
            self.bytes_total = nbytes

    def set_embed_label(self, name: str) -> None:
        """Which API's rate limits the embed box shows: voyage or jina."""
        with self._lock:
            self.embed_label = name

    def set_workers(self, *, chunk: int, post: int, upsert: int) -> None:
        with self._lock:
            for stage, count in zip(STAGES, (chunk, post, upsert), strict=True):
                self.stages[stage].workers = count

    # -- live state --------------------------------------------------------------

    def set_queue_depth(self, name: str, size: int, maxsize: int) -> None:
        del maxsize
        with self._lock:
            self.stages[name].queued = size

    def add_work(self, *, batches: int, chunks: int) -> None:
        """Batches handed to the post stage."""
        del chunks
        with self._lock:
            self.batches_total += batches

    def work_started(
        self, stage: str, *, space: str, paths: list[str], chunks: int | None = None
    ) -> int:
        """One item taken by a worker. Returns the token `work_finished` needs."""
        now = time.perf_counter()
        token = next(self._tokens)
        with self._lock:
            state = self.stages[stage]
            if not state.first_started:
                state.first_started = now
            state.held[token] = _Held(space=space, paths=paths, chunks=chunks, started=now)
        return token

    def work_finished(self, stage: str, token: int) -> None:
        now = time.perf_counter()
        with self._lock:
            state = self.stages[stage]
            held = state.held.pop(token, None)
            state.finished += 1
            if held is not None:
                state.last_seconds = now - held.started

    def event(self, event: str, **fields: Any) -> None:
        meta: dict[str, Any] = {"event": event, "catalog": self.catalog_name, "root": self.root}
        meta.update({key: value for key, value in fields.items() if value is not None})
        line = _scroll_line(event, meta)
        record = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": _level(event),
            "message": line or _plain(event, meta),
            "meta": meta,
        }
        text = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._handle.write(text)
            now = time.perf_counter()
            if now - self._last_flush >= FLUSH_SECONDS:
                # a flush per line cost more than the file check it records.
                self._handle.flush()
                self._last_flush = now
            self._count(event, meta)
            if event in PROBLEM_EVENTS and line:
                self.problems.append(line)
            if self._live is None and event in PLAIN_EVENTS and line:
                self.console.print(line, markup=False, highlight=False)

    def _count(self, event: str, meta: dict[str, Any]) -> None:
        if event in CHUNK_OUTCOMES and meta.get("path"):
            self.files_chunk_done += 1
        if event == "chunked":
            self.files_chunked += 1
            self.chunks_made += int(meta.get("chunks") or 0)
        elif event == "skip_hash":
            self.files_skipped += 1
        elif event in ("error", "skip_large", "empty"):
            stage = str(meta.get("stage") or "chunk")
            if stage in self.stages:
                self.stages[stage].errors += 1
        elif event == "embedded":
            self.batches_embedded += 1
            self.chunks_embedded += int(meta.get("texts") or 0)
        elif event == "upserted":
            self.batches_upserted += 1
            self.chunks_stored += int(meta.get("points") or 0)
        elif event == "file_complete":
            self.files_complete += 1
        elif event == "stopped":
            self.stopped = True

    # -- panel -------------------------------------------------------------------

    def files_finished(self) -> int:
        """Files that need nothing more: complete, unchanged, too large, empty, or failed."""
        left_chunk = self.files_chunk_done - self.files_chunked
        return self.files_complete + left_chunk

    def _render(self) -> Table:
        # the rate snapshot takes the API module's own lock, so it is read before ours.
        rate = _embed_rate(self.embed_label)
        with self._lock:
            return self._render_locked(time.perf_counter(), rate)

    def _render_locked(self, now: float, rate: dict[str, Any]) -> Table:
        chunk, post, upsert = (self.stages[stage] for stage in STAGES)
        # one fact per row. A box without a fact for a row leaves it blank, so the
        # same row sits on the same line in all three boxes.
        values = {
            "chunk": {
                "done": _slash(self.files_chunk_done, self.files_total),
                "chunks": str(self.chunks_made),
                "files": [
                    ("files changed", str(self.files_chunked)),
                    ("files unchanged", str(self.files_skipped)),
                ],
                "rate": [],
            },
            "post": {
                "done": _slash(self.batches_embedded, self.batches_total),
                "chunks": _slash(self.chunks_embedded, self.chunks_made),
                "files": [],
                "rate": [
                    ("requests/min", str(int(rate.get("rpm") or 0))),
                    ("tokens/min", _tok(rate.get("tpm"))),
                    ("429 retries", str(int(rate.get("retries") or 0))),
                    ("cooldown", f"{int(rate.get('cool_s') or 0)}s"),
                ],
            },
            "upsert": {
                "done": _slash(self.batches_upserted, self.batches_embedded),
                "chunks": _slash(self.chunks_stored, self.chunks_made),
                "files": [("files complete", _slash(self.files_complete, self.files_chunked))],
                "rate": [],
            },
        }
        titles = {
            "chunk": "read and split files",
            "post": f"embed with {self.embed_label}" if self.embed_label else "embed",
            "upsert": "write to Qdrant",
        }
        colors = {"chunk": "blue", "post": "magenta", "upsert": "green"}
        lines = max(state.workers for state in (chunk, post, upsert)) or 1
        boxes = Table.grid(expand=True, padding=(0, 1))
        for _ in STAGES:
            boxes.add_column(ratio=1)
        boxes.add_row(
            *(
                self._stage_box(
                    stage, titles[stage], colors[stage], values[stage], lines, now
                )
                for stage in STAGES
            )
        )
        return boxes

    def _stage_box(
        self,
        stage: str,
        title: str,
        color: str,
        values: dict[str, Any],
        lines: int,
        now: float,
    ) -> Panel:
        state = self.stages[stage]
        body = Table.grid(expand=True)
        body.add_column(style="dim", no_wrap=True)
        body.add_column(justify="right", no_wrap=True, overflow="ellipsis")
        rows = [
            ("working", f"{len(state.held)}/{state.workers}"),
            ("queued", str(state.queued)),
            ("done", values["done"]),
            ("chunks", values["chunks"]),
            *_padded(values["files"], FILE_ROWS),
            ("last", _fmt_seconds_short(state.last_seconds)),
            ("eta", _eta(state, now)),
            ("errors", str(state.errors)),
            *_padded(values["rate"], RATE_ROWS),
        ]
        for label, value in rows:
            body.add_row(label, value)
        held = Table.grid(expand=True, padding=(0, 1))
        held.add_column(no_wrap=True, style="dim")
        held.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        held.add_column(justify="right", no_wrap=True, style="dim")
        held.add_column(justify="right", no_wrap=True)
        items = sorted(state.held.values(), key=lambda item: item.started)[:lines]
        for item in items:
            more = f" +{len(item.paths) - 1}" if len(item.paths) > 1 else ""
            first = item.paths[0] if item.paths else ""
            held.add_row(
                item.space,
                f"{first}{more}",
                f"{item.chunks}ch" if item.chunks is not None else "",
                _fmt_seconds_short(now - item.started),
            )
        for _ in range(lines - len(items)):
            held.add_row("", "", "", "")
        return Panel(Group(body, Text(""), held), title=title, border_style=color)


def _padded(rows: list[tuple[str, str]], count: int) -> list[tuple[str, str]]:
    return [*rows, *[("", "")] * (count - len(rows))]


def _eta(state: _StageState, now: float) -> str:
    """Time to empty this stage's queue and free its workers, at its rate so far."""
    left = state.queued + len(state.held)
    if not left:
        return "-"
    if not state.finished or not state.first_started:
        return "--"
    per_second = state.finished / max(now - state.first_started, 1e-6)
    return _fmt_seconds(left / per_second)


def _level(event: str) -> str:
    if event in ERROR_EVENTS:
        return "error"
    if event in WARN_EVENTS:
        return "warn"
    return "info"


def _verb(word: str) -> str:
    return f"{word:<{VERB_WIDTH}}"


def _scroll_line(event: str, fields: dict[str, Any]) -> str:
    """One readable line: a problem after the panel, or any line without a panel."""
    path = str(fields.get("path") or "")
    space = str(fields.get("space") or "")
    where = f"{space:<6} {path}" if space else path
    if event == "file_complete":
        return f"{_verb('indexed')}{where}  {fields.get('chunks')} chunks written to Qdrant"
    if event == "skip_large":
        return f"{_verb('skipped')}{where}  {_fmt_bytes(fields.get('bytes'))} is over 8MB"
    if event == "empty":
        return f"{_verb('empty')}{where}  no chunks"
    if event == "error":
        stage = fields.get("stage")
        label = f"{stage} " if stage and stage != "chunk" else ""
        return f"{_verb('error')}{where}  {label}{fields.get('message')}".rstrip()
    if event == "deleted":
        return f"{_verb('deleted')}{where}"
    if event == "space":
        if fields.get("phase") == "end":
            return (
                f"{_verb('space')}{space} done  indexed {fields.get('indexed')} files  "
                f"unchanged {fields.get('unchanged')}  deleted {fields.get('deleted')}  "
                f"errors {fields.get('errors')}"
            )
        return f"{_verb('space')}{space} {fields.get('files')} files"
    if event in {"done", "stopped"}:
        tail = "  resume with bc-rag index" if event == "stopped" else ""
        return (
            f"{_verb(event)}scanned {fields.get('scanned')}  indexed {fields.get('indexed')}  "
            f"unchanged {fields.get('unchanged')}  deleted {fields.get('deleted')}  "
            f"chunks {fields.get('chunks')}  errors {fields.get('errors')}{tail}"
        )
    return ""


def _plain(event: str, fields: dict[str, Any]) -> str:
    """The JSONL message for events without a readable line."""
    extras = "  ".join(
        f"{key}={value}" for key, value in fields.items() if key not in {"event", "catalog", "root"}
    )
    return f"{event}  {extras}".strip()


def _slash(done: object, total: object) -> str:
    try:
        return f"{int(done)}/{int(total)}"
    except (TypeError, ValueError):
        return "-/-"


def _fmt_bytes(value: object) -> str:
    if value is None:
        return "?"
    size = float(value)
    if size < 1024:
        return f"{int(size)}B"
    if size < 1024**2:
        return f"{size / 1024:.1f}KB"
    if size < 1024**3:
        return f"{size / 1024**2:.1f}MB"
    return f"{size / 1024**3:.2f}GB"


def _fmt_seconds_short(seconds: float) -> str:
    """A duration that can be under a second: <1ms, 40ms, 1.2s, 3m05s."""
    if seconds <= 0:
        return "-"
    if seconds < 0.001:
        return "<1ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return _fmt_seconds(seconds)


def _fmt_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _tok(value: object) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "-"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def _embed_rate(provider: str = "") -> dict[str, Any]:
    try:
        if provider == "voyage":
            from bc_rag.voyage_api import rate_snapshot

            return rate_snapshot()
        if provider == "jina":
            from bc_rag.jina_api import embed_rate_snapshot

            return embed_rate_snapshot()
    except Exception:
        pass
    return {}
