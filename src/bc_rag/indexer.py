"""Index a project: every space through one pipeline of three stages.

    chunk  ->  post (embed over HTTP or locally)  ->  upsert (Qdrant)

Each stage has its own worker threads and an unbounded queue, so chunking never
waits for the network: it runs through every file of every space while posts and
upserts drain behind it. A batch carries its space, so it is embedded with that
space's model and stored in that space's collection.

A file counts as stored only when its last chunk is upserted; that is when its
row enters the space's manifest. Skip-hash is the prepared corpus pair (the
document and its Bedrock sidecar), so a facet change re-embeds the file. A file
whose size and modification time match its row is skipped without being read. A path in
the manifest that is gone from disk is deleted from Qdrant at the end of a full pass.
The manifest is saved every few seconds, so a killed run resumes from what it stored.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from rich.console import Console

from bc_rag.cache import cache_dir, move_legacy_cache
from bc_rag.catalog import ProjectEntry
from bc_rag.chunking import Chunk, chunk_file
from bc_rag.config import RagConfig
from bc_rag.corpus import (
    corpus_dir,
    corpus_key,
    delete_corpus_key,
    prepare_document,
    prune_corpus,
    sidecar_path_for,
    sidecar_sha256,
)
from bc_rag.defaults import (
    DEFAULT_MAX_FILE_BYTES,
    EMBED_HTTP_MAX_CHARS,
    EMBED_HTTP_MAX_TEXTS,
    JINA_FILE_WORKERS,
    JINA_HTTP_WORKERS,
    JINA_UPSERT_WORKERS,
    MANIFEST_SAVE_SECONDS,
    OPENAPI_LANGUAGE,
    OPENAPI_MD_DIRNAME,
    STAGE_QUEUE_MAX,
    VOYAGE_FLAT_MAX_CHARS,
    VOYAGE_FLAT_MAX_TEXTS,
)
from bc_rag.discover import SourceFile, iter_source_groups, source_for_path
from bc_rag.embeddings import Embedder
from bc_rag.facets import enrich_openapi_chunks
from bc_rag.index_log import IndexLog
from bc_rag.manifest import FileRecord, Manifest, load_manifest, save_manifest
from bc_rag.split_md import materialize_openapi_sources
from bc_rag.store import HybridStore

LONG_CHUNK_CHARS = 6000
MEDIUM_CHUNK_CHARS = 2500


class _QueueStage:
    """Named queue plus N worker threads. put() never runs the work inline."""

    def __init__(
        self,
        name: str,
        worker,
        *,
        workers: int,
        maxsize: int,
        on_depth=None,
    ) -> None:
        self.name = name
        self.maxsize = maxsize
        self._worker = worker
        self._on_depth = on_depth
        self._queue: Queue = Queue(maxsize=maxsize)
        self._sentinel = object()
        self._errors: list[BaseException] = []
        self._closed = False
        self._threads = [
            threading.Thread(target=self._loop, name=f"bc-rag-{name}-{index}", daemon=True)
            for index in range(max(1, workers))
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def qsize(self) -> int:
        return self._queue.qsize()

    def put(self, item: object) -> None:
        self._queue.put(item)
        self._emit_depth()

    def close(self) -> None:
        """Wait for every queued item, then stop the workers. Raises the first error."""
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(self._sentinel)
        for thread in self._threads:
            thread.join()
        if self._errors:
            raise self._errors[0]

    def _emit_depth(self) -> None:
        if self._on_depth is not None:
            self._on_depth(self.name, self._queue.qsize(), self.maxsize)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            self._emit_depth()
            try:
                if item is self._sentinel:
                    return
                self._worker(item)
            except Exception as error:
                self._errors.append(error)
            finally:
                self._queue.task_done()
                self._emit_depth()


def flat_http_limits(embedder: object) -> tuple[int, int]:
    """Code and other flat models pack many short files into one POST.

    Contextual models stay one document per inner list, packed separately.
    """
    if getattr(embedder, "voyage_api", False) and not getattr(embedder, "contextual", False):
        return VOYAGE_FLAT_MAX_CHARS, VOYAGE_FLAT_MAX_TEXTS
    return EMBED_HTTP_MAX_CHARS, EMBED_HTTP_MAX_TEXTS


def take_http_batch(
    chunks: list[Chunk],
    *,
    max_chars: int = EMBED_HTTP_MAX_CHARS,
    max_texts: int = EMBED_HTTP_MAX_TEXTS,
) -> list[Chunk]:
    """One request: small texts packed, a long text always alone."""
    if not chunks:
        return []
    first = chunks[0]
    total = len(first.embed_text())
    taken = [first]
    if total >= max_chars:
        return taken
    for chunk in chunks[1:]:
        if len(taken) >= max_texts:
            break
        size = len(chunk.embed_text())
        if total + size > max_chars:
            break
        taken.append(chunk)
        total += size
    return taken


@dataclass(frozen=True, slots=True)
class FileJob:
    run: _SpaceRun
    source: SourceFile
    group_name: str


@dataclass(frozen=True, slots=True)
class EmbeddedBatch:
    chunks: list[Chunk]
    dense: object
    sparse: object
    dense_texts: list[str]
    embed_ms: float


@dataclass
class IndexStats:
    scanned: int = 0
    skipped_unchanged: int = 0
    skipped_large: int = 0
    # files whose every chunk is stored in Qdrant.
    indexed_files: int = 0
    deleted_files: int = 0
    # chunks stored in Qdrant.
    chunks: int = 0
    jina_embed_tokens: int = 0
    jina_embed_calls: int = 0
    stopped: bool = False
    errors: list[str] = field(default_factory=list)

    def add(self, other: IndexStats) -> None:
        self.scanned += other.scanned
        self.skipped_unchanged += other.skipped_unchanged
        self.skipped_large += other.skipped_large
        self.indexed_files += other.indexed_files
        self.deleted_files += other.deleted_files
        self.chunks += other.chunks
        self.jina_embed_tokens += other.jina_embed_tokens
        self.jina_embed_calls += other.jina_embed_calls
        self.stopped = self.stopped or other.stopped
        self.errors.extend(other.errors)


class Indexer:
    """One space of one project: its config, embedder, and store."""

    def __init__(
        self,
        project: ProjectEntry,
        configuration: RagConfig,
        embedder: Embedder,
        store: HybridStore,
        console: Console | None = None,
        space: str | None = None,
    ) -> None:
        # the registered project: its name labels the log, its root anchors relative paths.
        self.entry = project
        self.root = project.root_path()
        self.configuration = configuration
        # the dense plus sparse embedder for this space's model.
        self.embedder = embedder
        # the Qdrant collection of this space.
        self.store = store
        self.space = space or configuration.default_space
        # progress goes to stderr, so stdout stays free for command output.
        self.console = console or Console(stderr=True)
        self._render_errors: list[tuple[str, str]] = []
        self._stop: threading.Event | None = None
        self.keep_keys: set[str] = set()

    def collect_groups(
        self, only_paths: list[str] | None, *, all_spaces: bool = False
    ) -> list[tuple[str, list[SourceFile]]]:
        """This space's files, by group, with OpenAPI specs already rendered to spec.md.

        `all_spaces` keeps every space's files, for one walk shared by a whole pass.
        """
        if only_paths is not None:
            grouped: list[tuple[str, list[SourceFile]]] = [
                (
                    "watch",
                    sources_for_paths(
                        self.root, self.configuration, set(only_paths), space=self.space
                    ),
                )
            ]
        else:
            grouped = [
                (group.name, files)
                for group, files in iter_source_groups(self.root, self.configuration)
            ]
        # a spec that cannot be rendered is reported as an error of this pass.
        self._render_errors = []

        def on_error(rel_path: str, error: Exception) -> None:
            self._render_errors.append((rel_path, str(error)))

        grouped = [
            (
                name,
                materialize_openapi_sources(
                    self.root, self.configuration, files, on_error=on_error
                ),
            )
            for name, files in grouped
        ]
        if not all_spaces:
            grouped = _only_space(grouped, self.space)
        return [(name, files) for name, files in grouped if files]

    def run(
        self,
        *,
        force: bool = False,
        only_paths: list[str] | None = None,
        log: IndexLog | None = None,
        grouped: list[tuple[str, list[SourceFile]]] | None = None,
        stop: threading.Event | None = None,
        prune: bool = True,
    ) -> IndexStats:
        """One pass over this space alone. `index_spaces` runs several through one pipeline."""
        return _run_spaces(
            [self],
            force=force,
            only_paths=only_paths,
            log=log,
            grouped={id(self): grouped} if grouped is not None else None,
            stop=stop,
            prune=prune,
        )

    def request_stop(self) -> None:
        """Finish the work in flight, then end the run. Safe from any thread."""
        if self._stop is not None:
            self._stop.set()


def _only_space(
    grouped: list[tuple[str, list[SourceFile]]], space: str
) -> list[tuple[str, list[SourceFile]]]:
    kept = [(name, [item for item in files if item.space == space]) for name, files in grouped]
    return [(name, files) for name, files in kept if files]


class _Discovery:
    """One walk of the project, shared by every space of a pass.

    Each space used to walk the whole project and keep only its own files, so two
    spaces walked it twice. Watched paths are few, so each space still resolves those.
    """

    def __init__(self) -> None:
        self._full: list[tuple[str, list[SourceFile]]] | None = None

    def groups(
        self, indexer: Indexer, only_paths: list[str] | None
    ) -> tuple[list[tuple[str, list[SourceFile]]], list[tuple[str, str]]]:
        """This space's files by group, and the render errors it reports."""
        errors: list[tuple[str, str]] = []
        if only_paths is not None:
            grouped = indexer.collect_groups(only_paths)
            errors, indexer._render_errors = indexer._render_errors, []
            return grouped, errors
        if self._full is None:
            self._full = indexer.collect_groups(None, all_spaces=True)
            # a spec that failed to render is reported once, by the first space.
            errors, indexer._render_errors = indexer._render_errors, []
        return _only_space(self._full, indexer.space), errors


def _embed_label(embedder: object) -> str:
    if getattr(embedder, "voyage_api", False):
        return "voyage"
    if getattr(embedder, "jina_api", False):
        return "jina"
    return ""


class _SpaceRun:
    """One space's share of a pass: its ledger, its store writes, and its stats."""

    def __init__(
        self,
        indexer: Indexer,
        *,
        force: bool,
        only_paths: list[str] | None,
        grouped: list[tuple[str, list[SourceFile]]] | None,
        log: IndexLog,
        discovery: _Discovery | None = None,
    ) -> None:
        self.indexer = indexer
        self.space = indexer.space
        self.log = log
        self.embedder = indexer.embedder
        self.store = indexer.store
        config = indexer.configuration
        self.config = config
        self.root = indexer.root
        self.project_dir = config.project_dir(indexer.root)
        self.manifest_path = config.manifest_path(indexer.root, self.space)
        previous = load_manifest(self.manifest_path)
        # a model swap invalidates every stored vector, so the whole collection is rebuilt.
        models_changed = (
            previous is None
            or previous.dense_model != self.embedder.dense_model
            or previous.sparse_model != self.embedder.sparse_model
        )
        self.rebuild = force or models_changed
        if self.rebuild:
            self.store.recreate_collection(self.embedder.dense_dim)
            kept: dict[str, FileRecord] = {}
            if only_paths is not None:
                # a rebuilt collection needs every file, not only the changed ones.
                only_paths = None
                grouped = None
        else:
            self.store.ensure_collection(self.embedder.dense_dim)
            kept = dict(previous.files) if previous is not None else {}
        self.wanted = set(only_paths) if only_paths is not None else None
        self.render_errors: list[tuple[str, str]] = []
        if grouped is None:
            grouped, self.render_errors = (discovery or _Discovery()).groups(indexer, only_paths)
        self.grouped = grouped
        # the next manifest starts from kept hashes, then replaces rows as files are stored.
        self.current = Manifest(
            dense_model=self.embedder.dense_model,
            sparse_model=self.embedder.sparse_model,
            files=kept,
        )
        self.jobs = [
            FileJob(run=self, source=source, group_name=name)
            for name, files in grouped
            for source in files
        ]
        self.files_total = len(self.jobs)
        self.bytes_total = sum(job.source.size for job in self.jobs)
        self.stats = IndexStats()
        self.seen: set[str] = set()
        self.keep_keys: set[str] = set()
        self.store_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.pack_lock = threading.Lock()
        self.pack_buffer: list[Chunk] = []
        self.max_chars, self.max_texts = flat_http_limits(self.embedder)
        # chunks of a file still on their way to Qdrant, and the row it gets when done.
        self.remaining_chunks: dict[str, int] = {}
        self.queued_records: dict[str, FileRecord] = {}
        # files whose old points were already removed in this pass.
        self.deleted_paths: set[str] = set()
        self.last_manifest_save = 0.0

    # -- lifecycle ---------------------------------------------------------------

    def begin(self) -> None:
        self.log.event(
            "space",
            space=self.space,
            phase="start",
            files=self.files_total,
            bytes=self.bytes_total,
            store=str(self.config.store_dir(self.root, self.space)),
        )
        for rel_path, message in self.render_errors:
            self.stats.errors.append(f"{rel_path}: {message}")
            self.log.event(
                "error", path=rel_path, space=self.space, message=f"openapi render: {message}"
            )

    def persist_manifest(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_manifest_save < MANIFEST_SAVE_SECONDS:
            return
        save_manifest(self.manifest_path, self.current)
        self.last_manifest_save = now

    def finish(self, stop: threading.Event) -> None:
        """Deletions and the final manifest, after every batch of every space is stored."""
        removed = [path for path in list(self.current.files) if path not in self.seen]
        if self.wanted is not None:
            # a watch event must not treat untouched files as deleted.
            removed = [path for path in removed if path in self.wanted]
        if stop.is_set():
            # a stopped run never treats unvisited files as deleted.
            self.stats.stopped = True
            removed = []
        if removed:
            self.log.event("delete_missing", space=self.space, count=len(removed))
            self.store.delete_paths(removed)
            for path in removed:
                record = self.current.files.pop(path, None)
                if record is not None and record.corpus_key:
                    delete_corpus_key(self.project_dir, record.corpus_key)
                self.log.event("deleted", path=path, space=self.space)
            self.stats.deleted_files = len(removed)
        self.indexer.keep_keys = self.keep_keys
        self.persist_manifest(force=True)
        self.stats.jina_embed_tokens = int(getattr(self.embedder, "jina_embed_tokens", 0) or 0)
        self.stats.jina_embed_calls = int(getattr(self.embedder, "jina_embed_calls", 0) or 0)
        self.log.event(
            "space",
            space=self.space,
            phase="end",
            scanned=self.stats.scanned,
            indexed=self.stats.indexed_files,
            unchanged=self.stats.skipped_unchanged,
            deleted=self.stats.deleted_files,
            chunks=self.stats.chunks,
            errors=len(self.stats.errors),
        )

    def abandon(self) -> None:
        """After a failure: files not fully stored lose their row, then the ledger is saved."""
        with self.store_lock:
            for path in list(self.remaining_chunks):
                self.current.files.pop(path, None)
            for path in list(self.queued_records):
                self.current.files.pop(path, None)
        try:
            self.persist_manifest(force=True)
        except Exception:
            pass

    # -- chunk stage -------------------------------------------------------------

    def _error(self, path: str, message: str) -> None:
        with self.state_lock:
            self.stats.errors.append(f"{path}: {message}")
        self.log.event("error", path=path, space=self.space, message=message)

    def chunk(self, job: FileJob) -> list[list[Chunk]]:
        """Chunk one file. Returns the batches ready to post (none when it is skipped)."""
        source = job.source
        path = source.rel_path
        with self.state_lock:
            self.stats.scanned += 1
            self.seen.add(path)
        if source.size > DEFAULT_MAX_FILE_BYTES and source.language != OPENAPI_LANGUAGE:
            # generated dumps and vendor blobs do not belong in the retrieval set.
            with self.state_lock:
                self.stats.skipped_large += 1
            self.log.event("skip_large", path=path, space=self.space, bytes=source.size)
            return []
        with self.state_lock:
            existing = None if self.rebuild else self.current.files.get(path)
        if existing is not None and self._unchanged_on_disk(source, existing):
            # same size, same modification time, same facets: not even read.
            with self.state_lock:
                self.keep_keys.add(existing.corpus_key)
                self.stats.skipped_unchanged += 1
            self.log.event("skip_hash", path=path, space=self.space, size=source.size, by="stat")
            return []
        try:
            prepared = prepare_document(self.project_dir, source)
        except (OSError, UnicodeDecodeError) as error:
            self._error(path, str(error))
            return []
        with self.state_lock:
            self.keep_keys.add(prepared.key)
        if (
            existing is not None
            and existing.sha256 == prepared.content_sha256
            and existing.sidecar_sha256 == prepared.sidecar_sha256
        ):
            with self.store_lock:
                # touched but not changed: remember the new time, so the next run
                # skips this file without reading it.
                existing.mtime_ns = source.mtime_ns
                existing.size = source.size
                existing.corpus_key = prepared.key
            with self.state_lock:
                self.stats.skipped_unchanged += 1
            self.log.event("skip_hash", path=path, space=self.space, size=source.size, by="hash")
            return []
        try:
            text = source.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            self._error(path, "not utf-8")
            return []
        started = time.perf_counter()
        chunks = chunks_for(source, text, self.config)
        for item in chunks:
            # the group's facets, the chunker's own (chat turns), then `group`.
            item.facets = {**source.facets, **item.facets}
            if source.group:
                item.facets["group"] = [source.group]
            item.group = source.group
            item.priority = source.priority
        if path.startswith(f"{OPENAPI_MD_DIRNAME}/"):
            enrich_openapi_chunks(path, chunks)
        chunk_ms = (time.perf_counter() - started) * 1000
        if not chunks:
            with self.store_lock:
                had_points = self.current.files.pop(path, None) is not None
                if had_points:
                    # a file that now yields nothing must not keep its old points.
                    self.store.delete_paths([path])
            self.log.event("empty", path=path, space=self.space, size=source.size)
            return []
        record = FileRecord(
            sha256=prepared.content_sha256,
            sidecar_sha256=prepared.sidecar_sha256,
            corpus_key=prepared.key,
            size=source.size,
            chunks=len(chunks),
            mtime_ns=source.mtime_ns,
        )
        with self.store_lock:
            # the old row goes now: if the run stops before upsert, the next run must
            # not skip this file.
            self.current.files.pop(path, None)
            self.queued_records[path] = record
            self.remaining_chunks[path] = self.remaining_chunks.get(path, 0) + len(chunks)
        self.log.event(
            "chunked",
            path=path,
            space=self.space,
            group=source.group or job.group_name,
            chunks=len(chunks),
            chunk_ms=round(chunk_ms, 1),
            size=source.size,
        )
        return self._batches(chunks)

    def _unchanged_on_disk(self, source: SourceFile, existing: FileRecord) -> bool:
        """True when the ledger row still describes this file, judged without reading it.

        Size and modification time stand in for the content hash. The sidecar hash is
        computed in memory, so a facet change still re-embeds. Any doubt means False,
        and the caller hashes the file as before.
        """
        if not source.mtime_ns or existing.mtime_ns != source.mtime_ns:
            return False
        if existing.size != source.size:
            return False
        key = corpus_key(source)
        if key != existing.corpus_key or existing.sidecar_sha256 != sidecar_sha256(source):
            return False
        document = corpus_dir(self.project_dir) / key
        return document.is_file() and sidecar_path_for(document).is_file()

    def _batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if getattr(self.embedder, "contextual", False):
            # context models never share a request across files.
            if getattr(self.embedder, "contextual_input", "chunks") == "auto":
                return [chunks]
            return _pack_one_document(chunks)
        with self.pack_lock:
            self.pack_buffer.extend(chunks)
            return self._take_packed(force=False)

    def _take_packed(self, *, force: bool) -> list[list[Chunk]]:
        batches: list[list[Chunk]] = []
        while self.pack_buffer and (force or self._pack_ready()):
            batch = take_http_batch(
                self.pack_buffer, max_chars=self.max_chars, max_texts=self.max_texts
            )
            if not batch:
                break
            del self.pack_buffer[: len(batch)]
            batches.append(batch)
        return batches

    def _pack_ready(self) -> bool:
        if len(self.pack_buffer[0].embed_text()) >= self.max_chars:
            return True
        if len(self.pack_buffer) >= self.max_texts:
            return True
        total = 0
        for item in self.pack_buffer[: self.max_texts]:
            total += len(item.embed_text())
            if total >= self.max_chars:
                return True
        return False

    def flush(self) -> list[list[Chunk]]:
        """The packed chunks still waiting for a full request, after the last file."""
        with self.pack_lock:
            return self._take_packed(force=True)

    # -- post stage --------------------------------------------------------------

    def embed(self, batch: list[Chunk]) -> EmbeddedBatch:
        started = time.perf_counter()
        embedded = self._embed(batch)
        ms = (time.perf_counter() - started) * 1000
        self.log.event(
            "embedded",
            space=self.space,
            texts=len(embedded.chunks),
            chars=sum(len(text) for text in embedded.dense_texts),
            ms=round(ms, 1),
        )
        return embedded

    def _embed(self, batch: list[Chunk]) -> EmbeddedBatch:
        dense_texts = [item.embed_text() for item in batch]
        sparse_texts = [item.sparse_text() for item in batch]
        started = time.perf_counter()
        try:
            if (
                getattr(self.embedder, "contextual", False)
                and getattr(self.embedder, "contextual_input", "chunks") == "auto"
            ):
                return self._embed_auto_document(batch)
            dense, sparse = self.embedder.embed_docs(dense_texts, sparse_texts)
        except Exception as error:
            if "too many tokens" not in str(error).lower():
                raise
            # Voyage counted more tokens than chars/2. Cut the batch and retry.
            if len(batch) > 1:
                mid = max(1, len(batch) // 2)
                left = self._embed(batch[:mid])
                right = self._embed(batch[mid:])
                return EmbeddedBatch(
                    chunks=left.chunks + right.chunks,
                    dense=left.dense + right.dense,
                    sparse=left.sparse + right.sparse,
                    dense_texts=left.dense_texts + right.dense_texts,
                    embed_ms=left.embed_ms + right.embed_ms,
                )
            smaller = _shrink_context_batch(batch)
            if len(smaller) < 2:
                raise
            extra = Counter(item.path for item in smaller)
            base = Counter(item.path for item in batch)
            with self.store_lock:
                for path, count in extra.items():
                    delta = count - base.get(path, 0)
                    if delta <= 0:
                        continue
                    self.remaining_chunks[path] = self.remaining_chunks.get(path, 0) + delta
                    record = self.queued_records.get(path)
                    if record is not None:
                        record.chunks += delta
            return self._embed(smaller)
        return EmbeddedBatch(
            chunks=batch,
            dense=dense,
            sparse=sparse,
            dense_texts=dense_texts,
            embed_ms=(time.perf_counter() - started) * 1000,
        )

    def _embed_auto_document(self, batch: list[Chunk]) -> EmbeddedBatch:
        from bc_rag.voyage_api import embed_contextual_auto

        path = batch[0].path
        text = (self.root / path).read_text(encoding="utf-8")
        pieces, _tokens = embed_contextual_auto(
            model=self.embedder.dense_model,
            document=text,
            dimensions=getattr(self.embedder, "dimensions", None),
        )
        built: list[Chunk] = []
        cursor = 0
        for piece, _vector in pieces:
            found = text.find(piece, cursor) if piece else -1
            if found < 0 and piece:
                found = text.find(piece)
            start_byte = found if found >= 0 else 0
            end_byte = start_byte + len(piece) if found >= 0 else 0
            start_line = text.count("\n", 0, start_byte) + 1 if found >= 0 else 0
            end_line = text.count("\n", 0, end_byte) + 1 if found >= 0 else 0
            built.append(
                Chunk(
                    path=path,
                    language=batch[0].language,
                    kind="auto",
                    symbol=None,
                    heading_path=None,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    text=piece,
                    facets=dict(batch[0].facets),
                    group=batch[0].group,
                    priority=batch[0].priority,
                )
            )
            if found >= 0:
                cursor = end_byte
        with self.store_lock:
            self.remaining_chunks[path] = (
                self.remaining_chunks.get(path, 0) - len(batch) + len(built)
            )
            record = self.queued_records.get(path)
            if record is not None:
                record.chunks = len(built)
        dense = [vector for _piece, vector in pieces]
        sparse = self.embedder.embed_sparse_texts([item.sparse_text() for item in built])
        return EmbeddedBatch(
            chunks=built,
            dense=dense,
            sparse=sparse,
            dense_texts=[item.embed_text() for item in built],
            embed_ms=0.0,
        )

    # -- upsert stage ------------------------------------------------------------

    def upsert(self, embedded: EmbeddedBatch) -> None:
        batch = embedded.chunks
        paths = list(dict.fromkeys(item.path for item in batch))
        started = time.perf_counter()
        stored: list[tuple[str, FileRecord]] = []
        with self.store_lock:
            fresh = [path for path in paths if path not in self.deleted_paths]
            if fresh:
                # a file's old points go before its first new batch lands.
                self.store.delete_paths(fresh)
                self.deleted_paths.update(fresh)
            self.store.upsert_chunks(batch, embedded.dense, embedded.sparse)
            for path, count in Counter(item.path for item in batch).items():
                self.remaining_chunks[path] = self.remaining_chunks.get(path, 0) - count
                if self.remaining_chunks[path] <= 0:
                    self.remaining_chunks.pop(path, None)
                    record = self.queued_records.pop(path, None)
                    if record is not None:
                        self.current.files[path] = record
                        stored.append((path, record))
            self.persist_manifest()
        with self.state_lock:
            self.stats.chunks += len(batch)
            self.stats.indexed_files += len(stored)
        self.log.event(
            "upserted",
            space=self.space,
            points=len(batch),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        for path, record in stored:
            # every chunk of this file is now in Qdrant.
            self.log.event(
                "file_complete",
                path=path,
                space=self.space,
                chunks=record.chunks,
                size=record.size,
            )


def _batch_paths(batch: list[Chunk]) -> list[str]:
    return list(dict.fromkeys(item.path for item in batch))


def _run_pipeline(runs: list[_SpaceRun], log: IndexLog, stop: threading.Event) -> None:
    """Chunk every file of every space; posts and upserts drain behind, in parallel."""
    files = sum(run.files_total for run in runs)
    api = any(_embed_label(run.embedder) for run in runs)
    if all(run.wanted is not None for run in runs) or files <= 8:
        chunk_workers = min(2, max(1, files))
        post_workers = min(2, max(1, files))
        upsert_workers = 1
    elif api:
        chunk_workers = JINA_FILE_WORKERS
        post_workers = JINA_HTTP_WORKERS
        upsert_workers = JINA_UPSERT_WORKERS
    else:
        chunk_workers = post_workers = upsert_workers = 2
    log.set_workers(chunk=chunk_workers, post=post_workers, upsert=upsert_workers)
    log.event(
        "pipeline",
        chunk_workers=chunk_workers,
        post_workers=post_workers,
        upsert_workers=upsert_workers,
        files=files,
    )

    def on_upsert(item: tuple[_SpaceRun, EmbeddedBatch]) -> None:
        run, embedded = item
        token = log.work_started(
            "upsert",
            space=run.space,
            paths=_batch_paths(embedded.chunks),
            chunks=len(embedded.chunks),
        )
        try:
            run.upsert(embedded)
        except Exception as error:
            log.event("error", stage="upsert", space=run.space, message=str(error))
            raise
        finally:
            log.work_finished("upsert", token)

    upsert_stage = _QueueStage(
        "upsert", on_upsert, workers=upsert_workers, maxsize=STAGE_QUEUE_MAX,
        on_depth=log.set_queue_depth,
    )

    def on_post(item: tuple[_SpaceRun, list[Chunk]]) -> None:
        run, batch = item
        token = log.work_started(
            "post", space=run.space, paths=_batch_paths(batch), chunks=len(batch)
        )
        try:
            embedded = run.embed(batch)
        except Exception as error:
            log.event("error", stage="post", space=run.space, message=str(error))
            raise
        finally:
            log.work_finished("post", token)
        upsert_stage.put((run, embedded))

    post_stage = _QueueStage(
        "post", on_post, workers=post_workers, maxsize=STAGE_QUEUE_MAX,
        on_depth=log.set_queue_depth,
    )

    def queue_batches(run: _SpaceRun, batches: list[list[Chunk]]) -> None:
        for batch in batches:
            log.add_work(batches=1, chunks=len(batch))
            post_stage.put((run, batch))

    def on_file(job: FileJob) -> None:
        if stop.is_set():
            job.run.stats.stopped = True
            return
        token = log.work_started("chunk", space=job.run.space, paths=[job.source.rel_path])
        try:
            batches = job.run.chunk(job)
        finally:
            log.work_finished("chunk", token)
        queue_batches(job.run, batches)

    chunk_stage = _QueueStage(
        "chunk", on_file, workers=chunk_workers, maxsize=STAGE_QUEUE_MAX,
        on_depth=log.set_queue_depth,
    )

    upsert_stage.start()
    post_stage.start()
    chunk_stage.start()
    pending: BaseException | None = None
    try:
        for run in runs:
            for job in run.jobs:
                if stop.is_set():
                    break
                chunk_stage.put(job)
        chunk_stage.close()
        for run in runs:
            queue_batches(run, run.flush())
    except KeyboardInterrupt:
        # a second Ctrl+C: abort without waiting for the queues.
        raise
    except BaseException as error:
        pending = error
    # every batch already chunked is still embedded and stored, even after an error.
    for stage in (post_stage, upsert_stage):
        try:
            stage.close()
        except BaseException as error:
            pending = pending or error
    if pending is not None:
        raise pending


def _run_spaces(
    indexers: list[Indexer],
    *,
    force: bool,
    only_paths: list[str] | None,
    log: IndexLog | None,
    grouped: dict[int, list[tuple[str, list[SourceFile]]] | None] | None,
    stop: threading.Event | None,
    prune: bool,
) -> IndexStats:
    first = indexers[0]
    own_stop = stop is None
    stop = stop if stop is not None else threading.Event()
    for indexer in indexers:
        indexer._stop = stop
    hits = 0

    def on_sigint(_signum, _frame) -> None:
        nonlocal hits
        hits += 1
        stop.set()
        if hits == 1:
            first.console.print(
                "\nstopping  finishing the work in flight, then exit  (Ctrl+C again to abort)"
            )
        else:
            raise KeyboardInterrupt

    previous_sigint = None
    if own_stop and threading.current_thread() is threading.main_thread():
        previous_sigint = signal.signal(signal.SIGINT, on_sigint)
    own_log = log is None
    if log is None:
        log = IndexLog(
            first.configuration.project_dir(first.root),
            first.console,
            catalog_name=first.entry.name,
            root=first.root,
        )
    runs: list[_SpaceRun] = []
    stats = IndexStats()
    discovery = _Discovery()
    try:
        for indexer in indexers:
            runs.append(
                _SpaceRun(
                    indexer,
                    force=force,
                    only_paths=only_paths,
                    grouped=(grouped or {}).get(id(indexer)),
                    log=log,
                    discovery=discovery,
                )
            )
        labels = [_embed_label(run.embedder) for run in runs]
        log.set_embed_label(next((label for label in labels if label), ""))
        log.set_totals(
            files=sum(run.files_total for run in runs),
            nbytes=sum(run.bytes_total for run in runs),
        )
        if own_log:
            log.start()
            log.event(
                "begin",
                cache=str(cache_dir(first.configuration.project_dir(first.root))),
                threads=os.cpu_count() or 4,
                spaces=[run.space for run in runs],
                files_total=sum(run.files_total for run in runs),
            )
        for run in runs:
            run.begin()
        try:
            _run_pipeline(runs, log, stop)
        except BaseException:
            for run in runs:
                run.abandon()
            raise
        for run in runs:
            run.finish(stop)
            stats.add(run.stats)
        full = all(run.wanted is None for run in runs)
        if prune and full and not stats.stopped:
            keep: set[str] = set()
            for run in runs:
                keep |= run.keep_keys
            pruned = prune_corpus(first.configuration.project_dir(first.root), keep)
            if pruned:
                log.event("prune_corpus", count=len(pruned))
        if own_log:
            log.event(
                "stopped" if stats.stopped else "done",
                scanned=stats.scanned,
                indexed=stats.indexed_files,
                unchanged=stats.skipped_unchanged,
                deleted=stats.deleted_files,
                chunks=stats.chunks,
                errors=len(stats.errors),
                log=str(log.path),
            )
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if own_log:
            log.close()
    return stats


class IndexBusyError(RuntimeError):
    """Another process is indexing this project right now."""


INDEX_LOCK_FILENAME = "index.lock"


def acquire_index_lock(entry: ProjectEntry):
    """One writer per project. Raises IndexBusyError naming the holder."""
    import datetime as _dt

    from bc_rag import locks
    from bc_rag.catalog import project_store_dir

    path = project_store_dir(entry.name) / INDEX_LOCK_FILENAME
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    held = locks.acquire(path, note=f"pid {os.getpid()}, since {stamp}")
    if held is None:
        holder = locks.read_note(path) or "another process"
        raise IndexBusyError(f"{entry.name} is already being indexed ({holder})")
    return held


def index_spaces(
    indexers: list[Indexer],
    *,
    force: bool = False,
    only_paths: list[str] | None = None,
    stop: threading.Event | None = None,
) -> IndexStats:
    """Index every space of a project through one pipeline, under one panel.

    Holds the project's index lock for the whole pass, and prunes the shared corpus
    folder once, with the documents every space kept.
    """
    if not indexers:
        return IndexStats()
    held = acquire_index_lock(indexers[0].entry)
    try:
        first = indexers[0]
        move_legacy_cache(first.configuration.project_dir(first.root))
        return _run_spaces(
            indexers,
            force=force,
            only_paths=only_paths,
            log=None,
            grouped=None,
            stop=stop,
            prune=True,
        )
    finally:
        held.release()


def _context_char_budget() -> int:
    from bc_rag.defaults import VOYAGE_CONTEXT_CHARS_PER_TOKEN, VOYAGE_PRECHUNK_TOKENS

    return VOYAGE_PRECHUNK_TOKENS * VOYAGE_CONTEXT_CHARS_PER_TOKEN


def _shrink_context_batch(chunks: list[Chunk]) -> list[Chunk]:
    """Split one over-budget document. A batch of several chunks is halved by the caller."""
    fitted = _split_chunks_for_context(chunks)
    if len(fitted) >= 2 or len(chunks) != 1:
        return fitted
    return _halve_chunk(chunks[0])


def _halve_chunk(chunk: Chunk) -> list[Chunk]:
    """Cut one chunk's text in half when the char budget still overshoots the tokenizer."""
    from dataclasses import replace

    text = chunk.text
    if len(text) < 2:
        return [chunk]
    mid = len(text) // 2
    left = text[:mid]
    right = text[mid:]
    left_lines = left.count("\n")
    right_start = chunk.start_line + left_lines
    return [
        replace(chunk, text=left, end_line=max(chunk.start_line, right_start)),
        replace(
            chunk,
            text=right,
            start_line=right_start,
            end_line=max(right_start, right_start + right.count("\n")),
        ),
    ]


def _split_chunks_for_context(chunks: list[Chunk]) -> list[Chunk]:
    """Cut a chunk whose embed text cannot fit in the voyage-context window."""
    from dataclasses import replace

    budget = _context_char_budget()
    fitted: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.embed_text()) <= budget:
            fitted.append(chunk)
            continue
        overhead = len(chunk.embed_text()) - len(chunk.text)
        body_budget = max(1_000, budget - max(0, overhead))
        start = 0
        text = chunk.text
        while start < len(text):
            piece = text[start : start + body_budget]
            end = start + len(piece)
            start_line = chunk.start_line + text[:start].count("\n")
            end_line = chunk.start_line + text[:end].count("\n")
            fitted.append(
                replace(
                    chunk,
                    text=piece,
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                )
            )
            start = end
    return fitted


def _pack_one_document(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Keep one file's chunks together. Split only when the request ceiling says so."""
    from bc_rag.defaults import (
        VOYAGE_CONTEXT_CHARS_PER_TOKEN,
        VOYAGE_MAX_CHUNKS,
        VOYAGE_PRECHUNK_TOKENS,
    )
    from bc_rag.voyage_api import split_token_spans

    fitted = _split_chunks_for_context(chunks)
    sizes = [
        max(1, len(chunk.embed_text()) // VOYAGE_CONTEXT_CHARS_PER_TOKEN) for chunk in fitted
    ]
    spans = split_token_spans(
        sizes, max_tokens=VOYAGE_PRECHUNK_TOKENS, max_items=VOYAGE_MAX_CHUNKS
    )
    return [fitted[start:end] for start, end in spans]


def chunks_for(source_file: SourceFile, text: str, configuration: RagConfig) -> list[Chunk]:
    """The chunks of one file. An OpenAPI spec arrives here already rendered to spec.md."""
    if source_file.language == OPENAPI_LANGUAGE:
        raise ValueError(f"{source_file.rel_path}: render OpenAPI specs to markdown first")
    space = configuration.space_named(source_file.space or configuration.default_space)
    chunk = source_file.chunk or space.chunk
    return chunk_file(
        path=source_file.path,
        rel_path=source_file.rel_path,
        language=source_file.language,
        text=text,
        max_chars=chunk.max_chars,
        min_chars=chunk.min_chars,
    )


def changed_paths(root: Path, configuration: RagConfig, rels: list[str]) -> list[str]:
    """Paths whose content or sidecar differs from the ledger of any space.

    Same rule as the index skip-hash. A path that vanished, or that no group claims
    any more, counts as changed when some space still has a record for it.
    """
    from bc_rag.corpus import prepare_document

    project_dir = configuration.project_dir(root)
    ledgers: dict[str, dict] = {}
    for space in configuration.spaces:
        previous = load_manifest(configuration.manifest_path(root, space))
        ledgers[space] = previous.files if previous is not None else {}
    out: list[str] = []
    for rel in rels:
        path = root / rel
        for space, records in ledgers.items():
            record = records.get(rel)
            source = source_for_path(rel, path, configuration, space=space)
            if source is None or not path.is_file():
                changed = record is not None
            else:
                try:
                    prepared = prepare_document(project_dir, source)
                except (OSError, UnicodeDecodeError):
                    changed = True
                else:
                    changed = (
                        record is None
                        or record.sha256 != prepared.content_sha256
                        or record.sidecar_sha256 != prepared.sidecar_sha256
                    )
            if changed:
                out.append(rel)
                break
    return out


def sources_for_paths(
    root: Path, configuration: RagConfig, wanted: set[str], *, space: str | None = None
) -> list[SourceFile]:
    """The files a full pass would see at these paths, in one space when given."""
    sources: list[SourceFile] = []
    for relative_path in sorted(wanted):
        path = root / relative_path
        source = source_for_path(relative_path, path, configuration, space=space)
        if source is None or not path.is_file():
            # missing files fall through to the deletion pass instead of being opened here.
            continue
        sources.append(source)
    sources.sort(key=lambda item: (-item.priority, item.rel_path))
    return sources
