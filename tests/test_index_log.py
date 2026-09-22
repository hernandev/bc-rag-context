from bc_rag.index_log import _human, _progress


def test_progress_puts_run_totals_before_group() -> None:
    text = _progress(
        {
            "files_done": 62,
            "files_total": 155,
            "bytes_done": 2_400_000,
            "bytes_total": 6_200_000,
            "run_files_done": 62,
            "run_files_total": 12_000,
            "run_bytes_done": 2_400_000,
            "run_bytes_total": 80_000_000,
        }
    )

    assert text.startswith("run 62/12000")
    assert "(2.3MB/76.3MB 3.0%)" in text
    assert "group 62/155" in text
    assert "(2.3MB/5.9MB 38.7%)" in text
    assert "this group" not in text


def test_index_event_is_two_lines() -> None:
    message = _human(
        "index",
        {
            "path": "docs/example.md",
            "chunks": 389,
            "language": "markdown",
            "chunk_ms": 4,
            "elapsed_ms": 12,
            "files_done": 62,
            "files_total": 155,
            "bytes_done": 100,
            "bytes_total": 200,
            "run_files_done": 62,
            "run_files_total": 900,
            "run_bytes_done": 100,
            "run_bytes_total": 400,
        },
    )

    lines = message.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("index")
    assert "docs/example.md" in lines[0]
    assert "389 chunks via markdown" in lines[1]
    assert "run 62/900" in lines[1]
    assert "25.0%" in lines[1]
    assert "50.0%" in lines[1]


def test_flush_embed_is_indented_continuation() -> None:
    message = _human(
        "flush",
        {
            "step": "embed+upsert",
            "done": 8,
            "total": 389,
            "batch": 8,
            "longest_chars": 2179,
            "embed_ms": 2000,
            "upsert_ms": 6,
        },
    )

    assert message.startswith(" ")
    assert "embed 8/389" in message
    assert not message.startswith("flush")
