from pathlib import Path

from bc_rag.config import ChunkConfig, ModelSpec, RagConfig, SpaceConfig
from bc_rag.discover import SourceFile
from bc_rag.plan import file_selected, unknown_file_filters


def _source(group: str, space: str) -> SourceFile:
    return SourceFile(
        path=Path("docs/a.md"),
        rel_path="docs/a.md",
        language="markdown",
        size=1,
        group=group,
        space=space,
    )


def _config() -> RagConfig:
    return RagConfig(
        default_space="prose",
        groups=[],
        spaces={
            "prose": SpaceConfig(
                dense=ModelSpec(provider="voyage", model="voyage-context-4"),
                sparse=ModelSpec(provider="local", model="Qdrant/bm25"),
                rerank=ModelSpec(provider="jina", model="jina-reranker-v3.5"),
                chunk=ChunkConfig(),
            ),
            "code": SpaceConfig(
                dense=ModelSpec(provider="voyage", model="voyage-code-4"),
                sparse=ModelSpec(provider="local", model="Qdrant/bm25"),
                rerank=ModelSpec(provider="jina", model="jina-reranker-v3.5"),
                chunk=ChunkConfig(),
            ),
        },
    )


def test_group_or_space_filters_combine() -> None:
    config = _config()
    prose = _source("docs-reference-02-engine", "prose")
    code = _source("libs-engine-core-engine-core-action", "code")
    assert file_selected(config, prose, {"docs-reference-02-engine"}, None)
    assert not file_selected(config, code, {"docs-reference-02-engine"}, None)
    assert file_selected(config, prose, {"docs-reference-02-engine", "docs-loose"}, {"prose"})
    assert not file_selected(config, prose, {"docs-reference-02-engine"}, {"code"})
    assert file_selected(config, code, None, {"code", "prose"})


def test_unknown_group_and_space_are_named() -> None:
    config = _config()
    config.groups = []
    assert unknown_file_filters(config, ["missing"], None) == "unknown group: missing"
    assert unknown_file_filters(config, None, ["missing"]) == "unknown space: missing"
    assert unknown_file_filters(config, None, ["prose"]) is None
