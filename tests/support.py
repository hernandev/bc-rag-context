"""Small configs that satisfy the required spaces schema."""

from bc_rag.config import ChunkConfig, ModelSpec, RagConfig, SourceGroup, SpaceConfig
from bc_rag.defaults import DEFAULT_INCLUDE


def space_config(groups: list[SourceGroup] | None = None, **kwargs) -> RagConfig:
    spaces = kwargs.pop(
        "spaces",
        {
            "prose": SpaceConfig(
                dense=ModelSpec(provider="local", model="jinaai/jina-embeddings-v2-base-en"),
                sparse=ModelSpec(provider="local", model="Qdrant/bm25"),
                rerank=ModelSpec(provider="jina", model="jina-reranker-v3.5"),
                chunk=ChunkConfig(),
            ),
            "code": SpaceConfig(
                dense=ModelSpec(provider="voyage", model="voyage-code-4"),
                sparse=ModelSpec(provider="local", model="Qdrant/bm25"),
                rerank=ModelSpec(provider="jina", model="jina-reranker-v3.5"),
                dimensions=1024,
                chunk=ChunkConfig(),
            ),
        },
    )
    return RagConfig(
        default_space=kwargs.pop("default_space", "prose"),
        spaces=spaces,
        groups=groups or [],
        **kwargs,
    )


def default_files_config(**kwargs) -> RagConfig:
    """One prose space and one group that includes the old default file types."""
    groups = kwargs.pop(
        "groups",
        [SourceGroup(name="default", space="prose", include=list(DEFAULT_INCLUDE))],
    )
    return space_config(groups=groups, **kwargs)
