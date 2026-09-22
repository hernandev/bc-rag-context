"""Index a project one file at a time.

Skip-hash is the prepared corpus pair: the document and its Bedrock sidecar.
A tag change rewrites the sidecar, so the pair hash misses and that file is
rewritten. A path present in the manifest and missing on disk is deleted from Qdrant.
A path present in the manifest and missing on disk is deleted from Qdrant.
A watch event never embeds the rest of the repository.

On a 20k-file Nx repo the first pass is hours of embedding, not a tree walk.
The manifest is written after every flush so a crash resumes from hashed files.
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

from bc_rag.catalog import register_project
from bc_rag.chunking import Chunk, chunk_file
from bc_rag.config import RagConfig
from bc_rag.corpus import corpus_dir, delete_corpus_key, prepare_document, prune_corpus
from bc_rag.defaults import (
    DEFAULT_MAX_FILE_BYTES,
    EMBED_HTTP_MAX_CHARS,
    EMBED_HTTP_MAX_TEXTS,
    JINA_FILE_WORKERS,
    JINA_HTTP_WORKERS,
    JINA_UPSERT_WORKERS,
    MANIFEST_SAVE_SECONDS,
    OPENAPI_LANGUAGE,
    STAGE_QUEUE_MAX,
)
from bc_rag.discover import SourceFile, iter_source_groups, source_for_path
from bc_rag.embeddings import Embedder
from bc_rag.index_log import IndexLog
from bc_rag.manifest import FileRecord, Manifest, load_manifest, save_manifest
from bc_rag.nx_tags import NxProjectIndex
from bc_rag.openapi import OpenApiError, expand_openapi_file
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


def take_http_batch(
    chunks: list[Chunk],
    *,
    max_chars: int = EMBED_HTTP_MAX_CHARS,
    max_texts: int = EMBED_HTTP_MAX_TEXTS,
) -> list[Chunk]:
    """One Jina request: small texts packed, a long text always alone."""
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
    source: SourceFile
    group_name: str
    group_files_total: int
    group_bytes_total: int


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
    indexed_files: int = 0
    deleted_files: int = 0
    chunks: int = 0
    jina_embed_tokens: int = 0
    jina_embed_calls: int = 0
    stopped: bool = False
    errors: list[str] = field(default_factory=list)


class Indexer:
    def __init__(
        self,
        root: Path,
        configuration: RagConfig,
        embedder: Embedder,
        store: HybridStore,
        console: Console | None = None,
    ) -> None:
        # keep the project root so relative paths stay stable across runs.
        self.root = root
        # keep the loaded configuration, including include, exclude, and embed models.
        self.configuration = configuration
        # keep the dense plus sparse embedder used for every rewritten file.
        self.embedder = embedder
        # keep the Qdrant store that holds one point per chunk.
        self.store = store
        # write progress on stderr so MCP stdio on stdout stays clean.
        self.console = console or Console(stderr=True)

    def run(self, *, force: bool = False, only_paths: list[str] | None = None) -> IndexStats:
        # load the previous file hashes so unchanged files can be skipped.
        manifest_path = self.configuration.manifest_path(self.root)
        previous = load_manifest(manifest_path)
        # a model swap invalidates every stored vector, so the whole collection is rebuilt.
        models_changed = (
            previous is None
            or previous.dense_model != self.embedder.dense_model
            or previous.sparse_model != self.embedder.sparse_model
        )
        rebuild = force or models_changed
        if rebuild:
            # --force and model changes drop the per-file shortcut and rewrite everything.
            only_paths = None
            self.store.recreate_collection(self.embedder.dense_dim)
            kept_files: dict[str, FileRecord] = {}
        else:
            # reuse the existing collection and the hashes of files that still match.
            self.store.ensure_collection(self.embedder.dense_dim)
            kept_files = dict(previous.files) if previous is not None else {}

        # the next manifest starts from kept hashes, then replaces rows that change.
        current = Manifest(
            dense_model=self.embedder.dense_model,
            sparse_model=self.embedder.sparse_model,
            files=kept_files,
        )

        stats = IndexStats()
        seen: set[str] = set()
        keep_keys: set[str] = set()
        stop = threading.Event()
        sigint_hits = 0

        def on_sigint(_signum, _frame) -> None:
            nonlocal sigint_hits
            sigint_hits += 1
            stop.set()
            stats.stopped = True
            if sigint_hits == 1:
                self.console.print(
                    "\nstopping  finishing the current file, then exit  (Ctrl+C again to abort)"
                )
            else:
                raise KeyboardInterrupt

        previous_sigint = signal.signal(signal.SIGINT, on_sigint)

        if only_paths is not None:
            wanted = set(only_paths)
            grouped: list[tuple[str, list[SourceFile]]] = [
                ("watch", sources_for_paths(self.root, self.configuration, wanted))
            ]
        else:
            wanted = None
            grouped = [
                (group.name, files)
                for group, files in iter_source_groups(self.root, self.configuration)
            ]
        grouped = [
            (name, materialize_openapi_sources(self.root, self.configuration, files))
            for name, files in grouped
        ]

        catalog_entry = register_project(self.root)
        catalog_name = getattr(catalog_entry, "name", None) or self.root.name
        store_dir = self.configuration.store_dir(self.root)
        log = IndexLog(
            store_dir,
            self.console,
            catalog_name=catalog_name,
            root=self.root,
        )
        run_files_total = sum(len(files) for _, files in grouped)
        run_bytes_total = sum(item.size for _, files in grouped for item in files)
        log.set_totals(files=run_files_total, nbytes=run_bytes_total)
        log.start()
        log.event(
            "begin",
            store=str(store_dir),
            corpus=str(corpus_dir(self.configuration.project_dir(self.root))),
            threads=os.cpu_count() or 4,
            files_total=run_files_total,
            bytes_total=run_bytes_total,
        )
        run_files_done = 0
        run_bytes_done = 0
        store_lock = threading.Lock()
        state_lock = threading.Lock()
        last_manifest_save = 0.0

        def persist_manifest(*, force: bool = False) -> None:
            nonlocal last_manifest_save
            now = time.perf_counter()
            if not force and now - last_manifest_save < MANIFEST_SAVE_SECONDS:
                return
            save_manifest(manifest_path, current)
            last_manifest_save = now

        jina = getattr(self.embedder, "jina_api", False)
        file_count = sum(len(files) for _name, files in grouped)
        small_job = only_paths is not None or file_count <= 8
        if small_job:
            chunk_workers = min(2, max(1, file_count))
            post_workers = min(2, max(1, file_count))
            upsert_workers = 1
        elif jina:
            chunk_workers = JINA_FILE_WORKERS
            post_workers = JINA_HTTP_WORKERS
            upsert_workers = JINA_UPSERT_WORKERS
        else:
            chunk_workers = 2
            post_workers = 2
            upsert_workers = 2
        deleted_paths: set[str] = set()
        remaining_chunks: dict[str, int] = {}
        queued_records: dict[str, FileRecord] = {}

        def embed_batch(batch: list[Chunk]) -> EmbeddedBatch:
            dense_texts = [chunk.embed_text() for chunk in batch]
            sparse_texts = [chunk.sparse_text() for chunk in batch]
            chars = sum(len(text) for text in dense_texts)
            log.stage_enter("post")
            log.event("post", texts=len(dense_texts), chars=chars)
            started = time.perf_counter()
            try:
                dense, sparse = self.embedder.embed_docs(dense_texts, sparse_texts)
            finally:
                embed_ms = (time.perf_counter() - started) * 1000
                log.stage_leave("post", ms=embed_ms)
            return EmbeddedBatch(
                chunks=batch,
                dense=dense,
                sparse=sparse,
                dense_texts=dense_texts,
                embed_ms=embed_ms,
            )

        def upsert_batch(embedded: EmbeddedBatch) -> None:
            batch = embedded.chunks
            paths = list(dict.fromkeys(chunk.path for chunk in batch))
            log.stage_enter("upsert")
            upsert_started = time.perf_counter()
            try:
                with store_lock:
                    fresh = [path for path in paths if path not in deleted_paths]
                    if fresh:
                        self.store.delete_paths(fresh)
                        deleted_paths.update(fresh)
                    self.store.upsert_chunks(batch, embedded.dense, embedded.sparse)
                    counts = Counter(chunk.path for chunk in batch)
                    for path, count in counts.items():
                        remaining_chunks[path] = remaining_chunks.get(path, 0) - count
                        if remaining_chunks[path] <= 0:
                            remaining_chunks.pop(path, None)
                            record = queued_records.pop(path, None)
                            if record is not None:
                                current.files[path] = record
                    persist_manifest()
            finally:
                upsert_ms = (time.perf_counter() - upsert_started) * 1000
                log.stage_leave("upsert", ms=upsert_ms)
            log.event("upsert", points=len(batch), elapsed_ms=round(upsert_ms, 1))

        def process_source_file(
            source_file: SourceFile,
            *,
            group_name: str,
            group_files_total: int,
            group_bytes_total: int,
            group_files_done: int,
            group_bytes_done: int,
        ) -> list[Chunk]:
            bucket = source_file.config_group or source_file.group or group_name
            with state_lock:
                stats.scanned += 1
                seen.add(source_file.rel_path)
                progress = {
                    "files_done": group_files_done,
                    "files_total": group_files_total,
                    "bytes_done": group_bytes_done,
                    "bytes_total": group_bytes_total,
                    "run_files_done": run_files_done,
                    "run_files_total": run_files_total,
                    "run_bytes_done": run_bytes_done,
                    "run_bytes_total": run_bytes_total,
                }
            if (
                source_file.size > DEFAULT_MAX_FILE_BYTES
                and source_file.language != OPENAPI_LANGUAGE
            ):
                # generated dumps and vendor blobs do not belong in the retrieval set.
                with state_lock:
                    stats.skipped_large += 1
                log.event(
                    "skip_large",
                    path=source_file.rel_path,
                    group=bucket,
                    bytes=source_file.size,
                    **progress,
                )
                return []
            file_started = time.perf_counter()
            try:
                prepared = prepare_document(self.configuration.project_dir(self.root), source_file)
                with state_lock:
                    keep_keys.add(prepared.key)
            except (OSError, UnicodeDecodeError) as error:
                with state_lock:
                    stats.errors.append(f"{source_file.rel_path}: {error}")
                log.event("error", path=source_file.rel_path, group=bucket, message=str(error))
                return []
            with state_lock:
                existing = None if rebuild else current.files.get(source_file.rel_path)
            if (
                existing is not None
                and existing.sha256 == prepared.content_sha256
                and existing.sidecar_sha256 == prepared.sidecar_sha256
            ):
                with state_lock:
                    stats.skipped_unchanged += 1
                log.event(
                    "skip_hash",
                    path=source_file.rel_path,
                    group=bucket,
                    size=source_file.size,
                    **progress,
                )
                return []
            try:
                text = source_file.path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                with state_lock:
                    stats.errors.append(f"{source_file.rel_path}: not utf-8")
                log.event("error", path=source_file.rel_path, group=bucket, message="not utf-8")
                return []
            if (
                source_file.embed is not None
                and source_file.embed.dense != self.embedder.dense_model
            ):
                message = (
                    f"{source_file.rel_path}: group embed.dense "
                    f"{source_file.embed.dense!r} does not match collection model "
                    f"{self.embedder.dense_model!r}. One Qdrant collection is one dense space."
                )
                with state_lock:
                    stats.errors.append(message)
                log.event("error", path=source_file.rel_path, group=bucket, message=message)
                return []
            try:
                log.event(
                    "chunk",
                    path=source_file.rel_path,
                    group=bucket,
                    language=source_file.language,
                    **progress,
                )
                chunk_started = time.perf_counter()
                chunks = chunks_for(self.root, source_file, text, self.configuration)
                chunk_ms = (time.perf_counter() - chunk_started) * 1000
            except OpenApiError as error:
                with state_lock:
                    stats.errors.append(f"{source_file.rel_path}: {error}")
                log.event("error", path=source_file.rel_path, group=bucket, message=str(error))
                return []
            for chunk in chunks:
                chunk.tags = list(source_file.tags)
                chunk.metadata = dict(source_file.metadata)
                chunk.group = source_file.group
                chunk.priority = source_file.priority
                from bc_rag.facets import parse_tag_clause

                for item in chunk.tags:
                    clause = parse_tag_clause(item)
                    if clause is not None:
                        chunk.metadata[clause[0]] = clause[1]
            if source_file.language == "markdown" and (
                source_file.rel_path.startswith("openapi-md/")
                or source_file.config_group
                and "openapi" in source_file.config_group
            ):
                from bc_rag.facets import enrich_openapi_chunks

                enrich_openapi_chunks(
                    source_file.rel_path,
                    chunks,
                    metadata=source_file.metadata,
                )
            if not chunks:
                with state_lock:
                    current.files.pop(source_file.rel_path, None)
                log.event("empty", path=source_file.rel_path, group=bucket)
                return []
            record = FileRecord(
                sha256=prepared.content_sha256,
                sidecar_sha256=prepared.sidecar_sha256,
                corpus_key=prepared.key,
                size=source_file.size,
                chunks=len(chunks),
            )
            with store_lock:
                # Drop the old hash now. If we stop before upsert, the next
                # index must not skip_hash this path.
                current.files.pop(source_file.rel_path, None)
                queued_records[source_file.rel_path] = record
                remaining_chunks[source_file.rel_path] = remaining_chunks.get(
                    source_file.rel_path, 0
                ) + len(chunks)
            with state_lock:
                stats.indexed_files += 1
                stats.chunks += len(chunks)
            log.event(
                "index",
                path=source_file.rel_path,
                group=bucket,
                chunks=len(chunks),
                language=source_file.language,
                chunk_ms=round(chunk_ms, 1),
                elapsed_ms=round((time.perf_counter() - file_started) * 1000, 1),
                size=source_file.size,
                **progress,
            )
            return chunks

        def chunk_job(job: FileJob) -> list[Chunk]:
            nonlocal run_files_done, run_bytes_done
            if stop.is_set():
                stats.stopped = True
                return []
            log.stage_enter("chunk")
            try:
                with state_lock:
                    run_files_done += 1
                    run_bytes_done += job.source.size
                    files_done = run_files_done
                    bytes_done = run_bytes_done
                return process_source_file(
                    job.source,
                    group_name=job.group_name,
                    group_files_total=job.group_files_total,
                    group_bytes_total=job.group_bytes_total,
                    group_files_done=files_done,
                    group_bytes_done=bytes_done,
                )
            finally:
                log.stage_leave("chunk")

        pack_lock = threading.Lock()
        pack_buffer: list[Chunk] = []

        def pack_ready(buffer: list[Chunk]) -> bool:
            if not buffer:
                return False
            first = len(buffer[0].embed_text())
            if first >= EMBED_HTTP_MAX_CHARS:
                return True
            if len(buffer) >= EMBED_HTTP_MAX_TEXTS:
                return True
            total = 0
            for chunk in buffer[:EMBED_HTTP_MAX_TEXTS]:
                total += len(chunk.embed_text())
                if total >= EMBED_HTTP_MAX_CHARS:
                    return True
            return False

        def emit_packed(*, force: bool = False) -> None:
            while pack_buffer and (force or pack_ready(pack_buffer)):
                batch = take_http_batch(pack_buffer)
                if not batch:
                    return
                del pack_buffer[: len(batch)]
                log.add_work(batches=1, points=len(batch))
                post_stage.put(batch)

        def on_file(job: FileJob) -> None:
            chunks = chunk_job(job)
            if not chunks:
                return
            with pack_lock:
                pack_buffer.extend(chunks)
                emit_packed()

        def on_post(batch: list[Chunk]) -> None:
            upsert_stage.put(embed_batch(batch))

        post_stage = _QueueStage(
            "post",
            on_post,
            workers=post_workers,
            maxsize=STAGE_QUEUE_MAX,
            on_depth=log.set_queue_depth,
        )
        upsert_stage = _QueueStage(
            "upsert",
            upsert_batch,
            workers=upsert_workers,
            maxsize=STAGE_QUEUE_MAX,
            on_depth=log.set_queue_depth,
        )
        chunk_stage = _QueueStage(
            "chunk",
            on_file,
            workers=chunk_workers,
            maxsize=STAGE_QUEUE_MAX,
            on_depth=log.set_queue_depth,
        )

        try:
            jobs: list[FileJob] = []
            for group_name, files in grouped:
                group_bytes_total = sum(item.size for item in files)
                log.event(
                    "group",
                    group=group_name,
                    priority=files[0].priority if files else 0,
                    files_total=len(files),
                    bytes_total=group_bytes_total,
                    run_files_total=run_files_total,
                    run_bytes_total=run_bytes_total,
                    chunk_workers=chunk_workers,
                    post_workers=post_workers,
                    upsert_workers=upsert_workers,
                )
                for source in files:
                    jobs.append(
                        FileJob(
                            source=source,
                            group_name=group_name,
                            group_files_total=len(files),
                            group_bytes_total=group_bytes_total,
                        )
                    )
            log.set_workers(chunk=chunk_workers, post=post_workers, upsert=upsert_workers)
            log.event(
                "pipeline",
                chunk_workers=chunk_workers,
                post_workers=post_workers,
                upsert_workers=upsert_workers,
                queue_max=STAGE_QUEUE_MAX,
                jobs=len(jobs),
            )
            upsert_stage.start()
            post_stage.start()
            chunk_stage.start()
            for job in jobs:
                if stop.is_set():
                    stats.stopped = True
                    break
                chunk_stage.put(job)
            chunk_stage.close()
            with pack_lock:
                emit_packed(force=True)
            post_stage.close()
            upsert_stage.close()

            # a path in the manifest that is no longer on disk is a deletion, not a skip.
            # Ctrl+C must not treat unvisited files as deleted.
            removed = [
                relative_path for relative_path in list(current.files) if relative_path not in seen
            ]
            if wanted is not None:
                # a watch event must not treat untouched files as deleted.
                removed = [relative_path for relative_path in removed if relative_path in wanted]
            if stop.is_set():
                removed = []
            if removed:
                log.event("delete_missing", count=len(removed))
                self.store.delete_paths(removed)
                project_dir = self.configuration.project_dir(self.root)
                for relative_path in removed:
                    record = current.files.pop(relative_path, None)
                    if record is not None and record.corpus_key:
                        delete_corpus_key(project_dir, record.corpus_key)
                    log.event("deleted", path=relative_path)
                stats.deleted_files = len(removed)
            if wanted is None and not stop.is_set():
                pruned = prune_corpus(self.configuration.project_dir(self.root), keep_keys)
                if pruned:
                    log.event("prune_corpus", count=len(pruned))

            persist_manifest(force=True)
            stats.jina_embed_tokens = int(getattr(self.embedder, "jina_embed_tokens", 0) or 0)
            stats.jina_embed_calls = int(getattr(self.embedder, "jina_embed_calls", 0) or 0)
            log.event(
                "stopped" if stats.stopped else "done",
                scanned=stats.scanned,
                indexed=stats.indexed_files,
                unchanged=stats.skipped_unchanged,
                deleted=stats.deleted_files,
                chunks=stats.chunks,
                errors=len(stats.errors),
                jina_embed_tokens=stats.jina_embed_tokens or None,
                jina_embed_calls=stats.jina_embed_calls or None,
                log=str(log.path),
            )
        finally:
            with store_lock:
                for path in list(remaining_chunks):
                    current.files.pop(path, None)
                for path in list(queued_records):
                    current.files.pop(path, None)
            try:
                persist_manifest(force=True)
            except Exception:
                pass
            signal.signal(signal.SIGINT, previous_sigint)
            log.close()

        return stats


def chunks_for(
    root: Path, source_file: SourceFile, text: str, configuration: RagConfig
) -> list[Chunk]:
    chunk = source_file.chunk or configuration.chunk
    if source_file.language == OPENAPI_LANGUAGE:
        return expand_openapi_file(
            root=root,
            source=source_file,
            text=text,
            max_chars=chunk.openapi_max_chars,
            min_chars=chunk.min_chars,
        )

    return chunk_file(
        path=source_file.path,
        rel_path=source_file.rel_path,
        language=source_file.language,
        text=text,
        max_chars=chunk.max_chars,
        min_chars=chunk.min_chars,
    )


def changed_paths(root: Path, configuration: RagConfig, rels: list[str]) -> list[str]:
    """Same rule as index skip-hash: content + sidecar vs the ledger."""
    from bc_rag.corpus import prepare_document
    from bc_rag.discover import NxProjectIndex, source_for_path

    previous = load_manifest(configuration.manifest_path(root))
    records = previous.files if previous is not None else {}
    nx_index = NxProjectIndex(root)
    project_dir = configuration.project_dir(root)
    out: list[str] = []
    for rel in rels:
        path = root / rel
        record = records.get(rel)
        if not path.is_file():
            if record is not None:
                out.append(rel)
            continue
        source = source_for_path(rel, path, configuration, nx_index=nx_index)
        if source is None:
            if record is not None:
                out.append(rel)
            continue
        try:
            prepared = prepare_document(project_dir, source)
        except (OSError, UnicodeDecodeError):
            out.append(rel)
            continue
        if (
            record is None
            or record.sha256 != prepared.content_sha256
            or record.sidecar_sha256 != prepared.sidecar_sha256
        ):
            out.append(rel)
    return out


def sources_for_paths(
    root: Path, configuration: RagConfig, wanted: set[str]
) -> list[SourceFile]:
    sources: list[SourceFile] = []
    nx_index = NxProjectIndex(root)
    for relative_path in sorted(wanted):
        path = root / relative_path
        source = source_for_path(relative_path, path, configuration, nx_index=nx_index)
        if source is None or not path.is_file():
            # missing files fall through to the deletion pass instead of being opened here.
            continue
        sources.append(source)
    sources.sort(key=lambda item: (-item.priority, item.rel_path))

    return sources
