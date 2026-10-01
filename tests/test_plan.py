import json
from pathlib import Path

from bc_rag.config import load_config
from bc_rag.plan import chunks_for_user_path, file_strategy, iter_index_files


def test_files_plan_names_chunker_and_embed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.ts").write_text(
        "export function one() { return 1 }\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("# Note\n\nhello\n", encoding="utf-8")
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "prose",
                "spaces": {
                    "prose": {
                        "dense": {"provider": "voyage", "model": "voyage-context-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "dimensions": 1024,
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    },
                    "code": {
                        "dense": {"provider": "voyage", "model": "voyage-code-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "dimensions": 1024,
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    },
                },
                "groups": [
                    {"name": "docs", "space": "prose", "include": ["docs/**/*.md"]},
                    {"name": "src", "space": "code", "include": ["src/**/*.ts"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    config, _path = load_config(tmp_path)
    rows = [file_strategy(config, source) for source in iter_index_files(tmp_path, config)]
    by_path = {row["path"]: row for row in rows}
    assert by_path["docs/note.md"]["chunker"] == "markdown"
    assert by_path["docs/note.md"]["embed"] == "contextual"
    assert by_path["docs/note.md"]["model"] == "voyage-context-4"
    assert by_path["src/one.ts"]["chunker"] == "code"
    assert by_path["src/one.ts"]["embed"] == "voyage"
    assert by_path["src/one.ts"]["space"] == "code"


def test_chunks_command_prints_real_code_chunks(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.ts").write_text(
        "export function one() { return 1 }\nexport function two() { return 2 }\n",
        encoding="utf-8",
    )
    (tmp_path / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "code",
                "spaces": {
                    "code": {
                        "dense": {"provider": "voyage", "model": "voyage-code-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    }
                },
                "groups": [{"name": "src", "space": "code", "include": ["src/**/*.ts"]}],
            }
        ),
        encoding="utf-8",
    )
    config, _path = load_config(tmp_path)
    source, chunks = chunks_for_user_path(tmp_path, config, "src/one.ts")
    assert source.rel_path == "src/one.ts"
    assert any("function one" in chunk.text for chunk in chunks)
    assert any("function two" in chunk.text for chunk in chunks)
