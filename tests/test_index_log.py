import io
import json
from pathlib import Path

from rich.console import Console

from bc_rag.index_log import IndexLog, _scroll_line


def _log(tmp_path: Path, *, terminal: bool = False) -> tuple[IndexLog, io.StringIO]:
    out = io.StringIO()
    console = Console(file=out, force_terminal=terminal, width=200)
    return IndexLog(tmp_path / "store", console, catalog_name="demo", root=tmp_path), out


def _screen(log: IndexLog) -> str:
    console = Console(file=io.StringIO(), width=240, record=True)
    console.print(log._render())
    return console.export_text()


def test_the_panel_is_three_boxes_in_pipeline_order(tmp_path: Path) -> None:
    log, _ = _log(tmp_path)
    log.set_embed_label("voyage")
    log.set_workers(chunk=8, post=8, upsert=4)
    screen = _screen(log)
    first = screen.splitlines()[0]

    assert first.index("read and split files") < first.index("embed with voyage")
    assert first.index("embed with voyage") < first.index("write to Qdrant")
    body = screen.splitlines()[1:]
    for row in ("working", "queued", "done", "chunks", "last", "eta", "errors"):
        # one label per box, on the same line, so the three read across.
        line = next(line for line in body if f" {row} " in line)
        assert line.count(f" {row} ") == 3
    assert "stored" not in screen
    log.close()


def test_every_row_holds_one_value(tmp_path: Path) -> None:
    log, _ = _log(tmp_path)
    log.set_embed_label("voyage")
    log.event("chunked", path="src/a.ts", space="code", chunks=3)
    log.event("skip_hash", path="src/b.ts", space="code", size=1)
    body = _screen(log).splitlines()[1:]
    changed = next(line for line in body if "files changed" in line)
    unchanged = next(line for line in body if "files unchanged" in line)

    assert changed != unchanged
    assert "files unchanged" not in changed
    for label in ("requests/min", "tokens/min", "429 retries", "cooldown", "files complete"):
        line = next(line for line in body if label in line)
        # the box cell for this label: label, spaces, one value.
        cell = line.split(label, 1)[1].split("│", 1)[0].split()
        assert len(cell) == 1
    log.close()


def test_counters_move_only_when_work_completes(tmp_path: Path) -> None:
    log, _ = _log(tmp_path)
    log.set_totals(files=2, nbytes=10)
    log.set_workers(chunk=2, post=1, upsert=1)
    token = log.work_started("chunk", space="code", paths=["src/a.ts"])

    assert log.stages["chunk"].finished == 0
    assert log.files_chunk_done == 0
    assert "src/a.ts" in _screen(log)

    log.event("chunked", path="src/a.ts", space="code", chunks=3)
    log.work_finished("chunk", token)

    assert log.stages["chunk"].finished == 1
    assert log.stages["chunk"].held == {}
    assert log.files_chunk_done == 1
    assert log.chunks_made == 3
    assert "src/a.ts" not in _screen(log)
    log.close()


def test_a_file_is_complete_only_on_file_complete(tmp_path: Path) -> None:
    log, _ = _log(tmp_path)
    log.event("chunked", path="src/a.ts", space="code", chunks=2)
    log.event("embedded", space="code", texts=2, ms=5)
    log.event("upserted", space="code", points=2, ms=5)
    assert log.files_complete == 0

    log.event("file_complete", path="src/a.ts", space="code", chunks=2, size=10)
    assert log.files_complete == 1
    assert log.files_finished() == 1
    log.close()


def test_errors_count_in_the_box_of_their_stage(tmp_path: Path) -> None:
    log, _ = _log(tmp_path)
    log.event("error", path="src/a.ts", space="code", message="not utf-8")
    log.event("skip_large", path="big.json", space="code", bytes=9_000_000)
    log.event("error", stage="post", space="code", message="HTTP 500")

    assert log.stages["chunk"].errors == 2
    assert log.stages["post"].errors == 1
    assert log.stages["upsert"].errors == 0
    log.close()


def test_problems_print_after_the_panel_stops(tmp_path: Path) -> None:
    log, out = _log(tmp_path, terminal=True)
    log.start()
    log.event("chunked", path="src/ok.ts", space="code", chunks=1)
    log.event("error", path="src/bad.ts", space="code", message="not utf-8")
    log.close()
    last = out.getvalue().rstrip().splitlines()[-1]

    # after the escape code that shows the cursor again.
    assert last.endswith("error   code   src/bad.ts  not utf-8")


def test_every_event_reaches_the_file_by_close(tmp_path: Path) -> None:
    log, _ = _log(tmp_path)
    for index in range(50):
        log.event("skip_hash", path=f"f{index}.ts", space="code", size=1)
    log.close()
    lines = log.path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 50
    assert json.loads(lines[-1])["meta"]["path"] == "f49.ts"


def test_without_a_terminal_lines_print_as_they_happen(tmp_path: Path) -> None:
    log, out = _log(tmp_path)
    log.start()
    log.event("file_complete", path="src/a.ts", space="code", chunks=4, size=10)
    log.event("skip_hash", path="src/b.ts", space="code", size=10)

    assert "src/a.ts  4 chunks written to Qdrant" in out.getvalue()
    assert "src/b.ts" not in out.getvalue()
    log.close()


def test_file_complete_line_says_where_the_chunks_went() -> None:
    line = _scroll_line("file_complete", {"path": "docs/a.md", "space": "prose", "chunks": 3})
    assert line == "indexed prose  docs/a.md  3 chunks written to Qdrant"
