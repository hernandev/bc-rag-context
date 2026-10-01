"""Per-project structured index log.

JSONL under `~/.bc-rag/{project-name}/index.jsonl`. Never a shared file across projects.
On a TTY, stderr is a split panel: a static status header plus a scrolling log.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bc_rag.defaults import INDEX_LOG_FILENAME

ERROR_EVENTS = frozenset({"error"})
WARN_EVENTS = frozenset({"skip_large", "empty", "delete_missing"})
FILE_EVENTS = frozenset({"skip_hash", "skip_large", "index", "chunk", "empty", "error", "deleted"})
SCROLL_EVENTS = frozenset(
    {
        "chunk",
        "index",
        "error",
        "empty",
        "skip_large",
        "deleted",
        "pipeline",
        "done",
        "stopped",
        "space",
    }
)
LOG_TAIL = 16
VERB_WIDTH = 10


class IndexLog:
    def __init__(self, store_dir: Path, console: Console, *, catalog_name: str, root: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self.path = store_dir / INDEX_LOG_FILENAME
        self.console = console
        self.catalog_name = catalog_name
        self.root = str(root)
        self._handle = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self.files_total = 0
        self.bytes_total = 0
        self.files_done = 0
        self.bytes_done = 0
        self.skipped = 0
        self.indexed = 0
        self.errors = 0
        self.chunk_workers = 0
        self.post_workers = 0
        self.upsert_workers = 0
        self.chunk_inflight = 0
        self.post_inflight = 0
        self.upsert_inflight = 0
        self.last_post_ms = 0.0
        self.last_upsert_ms = 0.0
        self.last_post_texts = 0
        self.last_post_chars = 0
        self.last_path = ""
        self.batches_total = 0
        self.batches_posted = 0
        self.batches_upserted = 0
        self.points_total = 0
        self.points_posted = 0
        self.points_upserted = 0
        self.queue_chunk = (0, 0)
        self.queue_post = (0, 0)
        self.queue_upsert = (0, 0)
        self._lines: deque[str] = deque(maxlen=LOG_TAIL)
        self._live: Live | None = None
        self.space_name = ""
        self.embed_label = ""

    def start(self) -> None:
        if not self.console.is_terminal:
            return
        self.console.clear()
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def set_totals(self, *, files: int, nbytes: int) -> None:
        with self._lock:
            self.files_total = files
            self.bytes_total = nbytes
            self._refresh()

    def set_space(self, name: str) -> None:
        """Name the space currently being written. Counters stay put."""
        with self._lock:
            self.space_name = name
            self._refresh()

    def set_embed_label(self, name: str) -> None:
        """Provider name for the post panel. Voyage must not be labeled jina."""
        with self._lock:
            self.embed_label = name
            self._refresh()

    def set_queue_depth(self, name: str, size: int, maxsize: int) -> None:
        with self._lock:
            pair = (size, maxsize)
            if name == "chunk":
                self.queue_chunk = pair
            elif name == "post":
                self.queue_post = pair
            elif name == "upsert":
                self.queue_upsert = pair
            self._refresh()

    def add_work(self, *, batches: int, points: int) -> None:
        with self._lock:
            self.batches_total += batches
            self.points_total += points
            self._refresh()

    def set_workers(self, *, chunk: int, post: int, upsert: int) -> None:
        with self._lock:
            self.chunk_workers = chunk
            self.post_workers = post
            self.upsert_workers = upsert
            self._refresh()

    def stage_enter(self, stage: str) -> None:
        with self._lock:
            if stage == "chunk":
                self.chunk_inflight += 1
            elif stage == "post":
                self.post_inflight += 1
            elif stage == "upsert":
                self.upsert_inflight += 1
            self._refresh()

    def stage_leave(self, stage: str, *, ms: float | None = None) -> None:
        with self._lock:
            if stage == "chunk":
                self.chunk_inflight = max(0, self.chunk_inflight - 1)
            elif stage == "post":
                self.post_inflight = max(0, self.post_inflight - 1)
                if ms is not None:
                    self.last_post_ms = ms
            elif stage == "upsert":
                self.upsert_inflight = max(0, self.upsert_inflight - 1)
                if ms is not None:
                    self.last_upsert_ms = ms
            self._refresh()

    def event(self, event: str, **fields: Any) -> None:
        meta: dict[str, Any] = {
            "event": event,
            "catalog": self.catalog_name,
            "root": self.root,
        }
        for key, value in fields.items():
            if value is not None:
                meta[key] = value
        message = _human(event, meta)
        line = _scroll_line(event, meta)
        level = _level(event)
        record = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": level,
            "message": message,
            "meta": meta,
        }
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._handle.flush()
            self._apply_status(event, meta)
            if self._live is not None:
                if event in SCROLL_EVENTS and line:
                    self._lines.append(line)
                self._refresh()
                return
            if message:
                if event in FILE_EVENTS:
                    self.console.print()
                self.console.print(message)

    def close(self) -> None:
        with self._lock:
            if self._live is not None:
                self._live.update(self._render())
                self._live.stop()
                self._live = None
            self._handle.close()

    def _apply_status(self, event: str, meta: dict[str, Any]) -> None:
        path = str(meta.get("path") or "")
        if path:
            self.last_path = path
        if event in {"skip_hash", "skip_large", "index", "empty", "error"}:
            self.files_done += 1
            self.bytes_done += int(meta.get("size") or meta.get("bytes") or 0)
        if event == "skip_hash":
            self.skipped += 1
        elif event == "index":
            self.indexed += 1
        elif event == "error":
            self.errors += 1
        if event == "post":
            self.last_post_texts = int(meta.get("texts") or 0)
            self.last_post_chars = int(meta.get("chars") or 0)
            self.batches_posted += 1
            self.points_posted += int(meta.get("texts") or 0)
        if event == "upsert":
            self.batches_upserted += 1
            self.points_upserted += int(meta.get("points") or 0)
        if event == "pipeline":
            self.chunk_workers = int(meta.get("chunk_workers") or self.chunk_workers)
            self.post_workers = int(meta.get("post_workers") or self.post_workers)
            self.upsert_workers = int(meta.get("upsert_workers") or self.upsert_workers)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Group:
        elapsed = time.perf_counter() - self._started
        left = max(0, self.batches_total - self.batches_upserted)
        rate = _embed_rate(self.embed_label)
        eta = "--"
        ok_ms = float(rate.get("last_ok_ms") or self.last_post_ms or 0)
        if ok_ms and self.post_workers and left:
            eta_s = (left * ok_ms / 1000.0) / max(1, self.post_workers)
            eta = _fmt_eta(eta_s)

        overview = Table.grid(expand=True, padding=(0, 4), pad_edge=True)
        overview.add_column(style="dim", justify="right", width=5, no_wrap=True)
        overview.add_column(justify="right", width=8, no_wrap=True)
        overview.add_column(style="dim", justify="right", width=4, no_wrap=True)
        overview.add_column(justify="right", width=5, no_wrap=True)
        overview.add_column(justify="left", ratio=1, no_wrap=True, overflow="ellipsis")
        overview.add_row(
            "time",
            f"{_n(int(elapsed), 5)}s",
            "err",
            _n(self.errors, 4),
            self.last_path or "—",
        )

        chunk = _panel_text(
            [
                ("files", _slash(self.files_done, self.files_total), "percent", _pct_plain(self.files_done, self.files_total)),
                ("bytes", _fmt_bytes(self.bytes_done), "of", _fmt_bytes(self.bytes_total)),
                ("in", _slash(self.chunk_inflight, self.chunk_workers), "q", str(int(self.queue_chunk[0]))),
                ("index", str(int(self.indexed)), "skip", str(int(self.skipped))),
            ]
        )

        post_rows: list[tuple[str, ...]] = [
            ("done", _slash(self.batches_posted, self.batches_total), "percent", _pct_plain(self.batches_posted, self.batches_total)),
            ("in", _slash(self.post_inflight, self.post_workers), "q", str(int(self.queue_post[0]))),
            ("last", _fmt_ms(rate.get("last_ok_ms") or self.last_post_ms), "chars", str(int(self.last_post_chars))),
            ("rpm", str(int(rate.get("rpm") or 0)), "tpm", _tok(rate.get("tpm"))),
        ]
        if rate.get("remaining_requests") is not None or rate.get("remaining_tokens") is not None:
            post_rows.append((self.embed_label or "api", _fmt_jina_remaining(rate)))
        post_rows.extend(
            [
                ("429", str(int(rate.get("retries") or 0)), "last min", str(int(rate.get("retries_min") or 0))),
                ("cool", f"{int(rate.get('cool_s') or 0)}s", "eta", eta),
            ]
        )
        post = _panel_text(post_rows)

        upsert = _panel_text(
            [
                ("done", _slash(self.batches_upserted, self.batches_total), "percent", _pct_plain(self.batches_upserted, self.batches_total)),
                ("in", _slash(self.upsert_inflight, self.upsert_workers), "q", str(int(self.queue_upsert[0]))),
                ("last", _fmt_ms(self.last_upsert_ms), "pts", str(int(self.points_upserted))),
            ]
        )

        log = Text("\n".join(self._lines) if self._lines else "…", overflow="fold")
        title = "bc-rag index" if not self.space_name else f"bc-rag index  {self.space_name}"
        return Group(
            Panel(overview, title=title, border_style="cyan"),
            Columns(
                [
                    Panel(chunk, title="chunk", border_style="blue"),
                    Panel(post, title="post", border_style="magenta"),
                    Panel(upsert, title="upsert", border_style="green"),
                ],
                equal=True,
                expand=True,
            ),
            Panel(log, title="log", border_style="dim"),
        )


def _level(event: str) -> str:
    if event in ERROR_EVENTS:
        return "error"
    if event in WARN_EVENTS:
        return "warn"
    return "info"


def _scroll_line(event: str, fields: dict[str, Any]) -> str:
    path = str(fields.get("path") or "")
    if event == "chunk":
        return f"chunk   {path}"
    if event == "index":
        return (
            f"index   {path}  {fields.get('chunks')} chunks  "
            f"{_fmt_ms(fields.get('chunk_ms'))}"
        )
    if event == "post":
        return (
            f"post    texts {_n(fields.get('texts'), 2)}          "
            f"{_n(fields.get('chars'), 6)} chars"
        )
    if event == "upsert":
        return (
            f"upsert  {_n(fields.get('points'), 2)} points  "
            f"{_w(_fmt_ms(fields.get('elapsed_ms')), 6)}"
        )
    if event == "skip_large":
        return f"skip    {path}  {_fmt_bytes(fields.get('bytes'))}"
    if event == "empty":
        return f"empty   {path}"
    if event == "error":
        return f"error   {path}  {fields.get('message')}"
    if event == "deleted":
        return f"delete  {path}"
    if event == "space":
        name = str(fields.get("space") or "")
        if fields.get("phase") == "end":
            return (
                f"space   {name}  indexed={fields.get('indexed')}  "
                f"unchanged={fields.get('unchanged')}  errors={fields.get('errors')}"
            )
        return f"space   {name}  {fields.get('files')} files"
    if event == "pipeline":
        return (
            f"pipeline chunk x{fields.get('chunk_workers')}  "
            f"post x{fields.get('post_workers')}  "
            f"upsert x{fields.get('upsert_workers')}"
        )
    if event in {"done", "stopped"}:
        return (
            f"{event:<7} scanned={fields.get('scanned')}  indexed={fields.get('indexed')}  "
            f"unchanged={fields.get('unchanged')}  errors={fields.get('errors')}"
        )
    return ""


def _human(event: str, fields: dict[str, Any]) -> str:
    path = fields.get("path")
    if event == "begin":
        return _two_line(
            "begin",
            f"catalog={fields.get('catalog')}  threads={fields.get('threads')}",
            (
                f"root={fields.get('root')}  store={fields.get('store')}  "
                f"{fields.get('files_total')} files  {_fmt_bytes(fields.get('bytes_total'))}"
            ),
        )
    if event == "pipeline":
        return _two_line(
            "pipeline",
            (
                f"chunk x{fields.get('chunk_workers')}  "
                f"post x{fields.get('post_workers')}  "
                f"upsert x{fields.get('upsert_workers')}"
            ),
            f"{fields.get('jobs')} files  queue_max={fields.get('queue_max')}",
        )
    if event == "group":
        run_total = ""
        if fields.get("run_files_total") is not None:
            run_total = (
                f"    run {fields.get('run_files_total')} files  "
                f"{_fmt_bytes(fields.get('run_bytes_total'))}"
            )
        return _two_line(
            "group",
            f"{fields.get('group')}  priority={fields.get('priority')}",
            (
                f"{fields.get('files_total')} files  "
                f"{_fmt_bytes(fields.get('bytes_total'))}{run_total}"
            ),
        )
    if event == "skip_hash":
        return _two_line("skip-hash", str(path), _progress(fields))
    if event == "skip_large":
        return _two_line(
            "skip-large",
            str(path),
            f"{_fmt_bytes(fields.get('bytes'))}    {_progress(fields)}",
        )
    if event == "chunk":
        return _two_line("chunk", str(path), str(fields.get("language") or ""))
    if event == "index":
        return _two_line(
            "index",
            str(path),
            f"{fields.get('chunks')} chunks  {_fmt_ms(fields.get('chunk_ms'))}",
        )
    if event == "post":
        return f"post      texts={fields.get('texts')}  {fields.get('chars')} chars"
    if event == "upsert":
        return f"upsert    {fields.get('points')} points  {_fmt_ms(fields.get('elapsed_ms'))}"
    if event == "empty":
        return _two_line("empty", str(path), _progress(fields) or "no chunks")
    if event == "flush":
        return ""
    if event == "delete_missing":
        return f"{'delete':<{VERB_WIDTH}} {fields.get('count')} missing path(s)"
    if event == "deleted":
        return _two_line("deleted", str(path), "removed from store")
    if event == "error":
        return _two_line("error", str(path or ""), str(fields.get("message") or ""))
    if event == "stopped":
        return _two_line(
            "stopped",
            (
                f"scanned={fields.get('scanned')}  indexed={fields.get('indexed')}  "
                f"unchanged={fields.get('unchanged')}"
            ),
            (
                f"deleted={fields.get('deleted')}  chunks={fields.get('chunks')}  "
                f"errors={fields.get('errors')}  resume with bc-rag index"
            ),
        )
    if event == "done":
        return _two_line(
            "done",
            (
                f"scanned={fields.get('scanned')}  indexed={fields.get('indexed')}  "
                f"unchanged={fields.get('unchanged')}"
            ),
            (
                f"deleted={fields.get('deleted')}  chunks={fields.get('chunks')}  "
                f"errors={fields.get('errors')}"
            ),
        )
    extras = " ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if key not in {"event", "catalog", "root"}
    )
    return f"{event} {extras}".strip()


def _two_line(verb: str, first: str, second: str) -> str:
    head = f"{verb:<{VERB_WIDTH}} {first}".rstrip()
    if not second:
        return head
    return f"{head}\n{_indent()}{second}"


def _indent() -> str:
    return " " * (VERB_WIDTH + 1)


def _progress(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    run_done = fields.get("run_files_done")
    run_total = fields.get("run_files_total")
    if run_done is not None and run_total is not None:
        parts.append(
            f"run {run_done}/{run_total}  "
            f"{_bytes_progress(fields.get('run_bytes_done'), fields.get('run_bytes_total'))}"
        )
    elif run_done is not None:
        parts.append(f"run {run_done} files  {_fmt_bytes(fields.get('run_bytes_done'))}")
    files_done = fields.get("files_done")
    files_total = fields.get("files_total")
    if files_done is not None and files_total is not None:
        parts.append(
            f"group {files_done}/{files_total}  "
            f"{_bytes_progress(fields.get('bytes_done'), fields.get('bytes_total'))}"
        )
    return "    ".join(parts)


def _bytes_progress(done: object, total: object) -> str:
    return f"({_fmt_bytes(done)}/{_fmt_bytes(total)}{_pct(done, total)})"


def _pct(done: object, total: object) -> str:
    padded = _pct_pad(done, total)
    return f" {padded}" if padded else ""


def _pct_pad(done: object, total: object) -> str:
    try:
        numerator = float(done)
        denominator = float(total)
    except (TypeError, ValueError):
        return "      "
    if denominator <= 0:
        return "      "
    return f"{100.0 * numerator / denominator:5.1f}%"


def _digits(value: object) -> int:
    try:
        return max(1, len(str(int(value))))
    except (TypeError, ValueError):
        return 1


def _n(value: object, width: int) -> str:
    try:
        return f"{int(value):>{width}d}"
    except (TypeError, ValueError):
        return f"{'?':>{width}}"


def _pair(done: object, total: object, width: int) -> str:
    return f"{_n(done, width)} / {_n(total, width)}"


def _w(text: str, width: int) -> str:
    return f"{text:>{width}}"


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


def _embed_rate(provider: str = "") -> dict[str, Any]:
    try:
        if provider == "voyage":
            from bc_rag.voyage_api import rate_snapshot

            return rate_snapshot()
        from bc_rag.jina_api import embed_rate_snapshot

        return embed_rate_snapshot()
    except Exception:
        return {
            "rpm": 0,
            "rpm_cap": 0,
            "tpm": 0,
            "tpm_cap": 0,
            "remaining_requests": None,
            "remaining_tokens": None,
            "limit_requests": None,
            "limit_tokens": None,
            "retries": 0,
            "retries_min": 0,
            "cool_s": 0,
            "last_ok_ms": 0,
        }


def _fmt_jina_remaining(rate: dict[str, Any]) -> str:
    req = rate.get("remaining_requests")
    tok = rate.get("remaining_tokens")
    if req is None and tok is None:
        return "-"
    req_part = "-" if req is None else str(int(req))
    tok_part = "-" if tok is None else _tok(tok)
    return f"{req_part}/{tok_part}"


LABEL_W = 7
VALUE_W = 13
PAIR_GAP = 8


def _panel_text(rows: list[tuple[str, ...]]) -> Text:
    lines: list[str] = []
    for row in rows:
        if len(row) == 2:
            lines.append(_stat_row(row[0], row[1]))
        else:
            lines.append(_stat_row(row[0], row[1], row[2], row[3]))
    return Text("\n".join(lines), no_wrap=True)


def _stat_row(label: str, value: str, label2: str = "", value2: str = "") -> str:
    left = f"{label:<{LABEL_W}} {value:>{VALUE_W}}"
    if not label2:
        return left
    return f"{left}{' ' * PAIR_GAP}{label2:<{LABEL_W}} {value2:>{VALUE_W}}"


def _slash(done: object, total: object) -> str:
    try:
        return f"{int(done)}/{int(total)}"
    except (TypeError, ValueError):
        return "-/-"


def _pct_plain(done: object, total: object) -> str:
    try:
        numerator = float(done)
        denominator = float(total)
    except (TypeError, ValueError):
        return "-"
    if denominator <= 0:
        return "-"
    return f"{100.0 * numerator / denominator:.1f}%"


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


def _fmt_tokens(value: object) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "      ?"
    if count >= 1_000_000:
        return f"{count / 1_000_000:5.2f}M"
    if count >= 1000:
        return f"{count / 1000:5.1f}k"
    return f"{count:6d}"


def _fmt_q(pair: tuple[int, int]) -> str:
    size, maxsize = pair
    if maxsize <= 0:
        return str(size)
    return f"{size}/{maxsize}"


def _fmt_eta(seconds: float) -> str:
    if seconds > 90:
        minutes = max(1, int(round(seconds / 60.0)))
        return f"{minutes}m"
    return f"{int(seconds)}s"


def _fmt_ms(value: object) -> str:
    if value is None:
        return "?"
    milliseconds = float(value)
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    return f"{milliseconds / 1000:.1f}s"
