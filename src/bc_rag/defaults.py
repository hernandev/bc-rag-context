"""Built-in defaults. A project can run with no `.bc-rag.json` at all."""

from __future__ import annotations

CONFIG_FILENAME = ".bc-rag.json"
# Legacy on-disk folder inside a project. Live stores are ~/.bc-rag/{name}/.
STORE_DIRNAME = ".bc-rag"
COLLECTION_NAME = "chunks"
QDRANT_HTTP_URL = "http://127.0.0.1:32321"
QDRANT_HTTP_PORT = 32321
QDRANT_GRPC_PORT = 32322
# One shared MCP process. Chats connect here instead of spawning stdio.
MCP_HTTP_HOST = "127.0.0.1"
MCP_HTTP_PORT = 32323
MCP_HTTP_PATH = "/mcp"
QDRANT_DOCKER_CONTEXT = "orbstack"
QDRANT_COMPOSE_PROJECT = "bc-rag"
QDRANT_IMAGE = "qdrant/qdrant:v1.19.1"
MANIFEST_FILENAME = "manifest.json"
INDEX_LOG_FILENAME = "index.jsonl"
OPENAPI_MD_DIRNAME = "openapi-md"
CORPUS_DIRNAME = "corpus"
JINA_STORE_DIRNAME = "jina"

DEFAULT_INCLUDE: list[str] = [
    "**/*.ts",
    "**/*.tsx",
    "**/*.mts",
    "**/*.cts",
    "**/*.vue",
    "**/*.md",
    "**/*.mdx",
]

# Entrypoints only. Split path/schema fragments are pulled in by Redocly bundle.
DEFAULT_OPENAPI_INCLUDE: list[str] = [
    "**/openapi.yaml",
    "**/openapi.yml",
    "**/openapi.json",
    "**/swagger.yaml",
    "**/swagger.yml",
    "**/swagger.json",
]

DEFAULT_EXCLUDE: list[str] = [
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/coverage/**",
    "**/out/**",
    "**/.git/**",
    "**/.bc-rag/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.next/**",
    "**/.turbo/**",
    "**/.nx/**",
    "**/.angular/**",
    "**/.output/**",
    "**/.vite/**",
    "**/.cache/**",
    "**/tmp/**",
    "**/storybook-static/**",
    "**/vendor/**",
]

# Always pruned, even if a user include glob would otherwise match.
HARD_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".bc-rag",
        ".venv",
        "node_modules",
        ".next",
        ".turbo",
        ".nx",
        ".angular",
        "dist",
        "coverage",
        "build",
        "out",
    }
)

# English retrieval model with 8192-token context. Vendor OpenAPI descriptions
# and internal markdown are the hard queries: same idea, different words.
# BM25 stays in the collection as a backstop when a name is already known.
DEFAULT_DENSE_MODEL = "jinaai/jina-embeddings-v2-base-en"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
# 8k-context reranker, so a whole OpenAPI operation can be scored, not truncated.
DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v1-turbo-en"
JINA_API_URL = "https://api.jina.ai"
JINA_API_KEY_ENV = "JINA_API_KEY"
JINA_RPM = 500
JINA_TPM = 2_000_000
JINA_HTTP_WORKERS = 8
JINA_GROUP_WORKERS = 8
JINA_FILE_WORKERS = 8
JINA_UPSERT_WORKERS = 4
# 0 = unbounded. Chunking must never block on HTTP or Qdrant.
STAGE_QUEUE_MAX = 0
MANIFEST_SAVE_SECONDS = 2.0
# Tiny files are one chunk each. Pack several into one HTTP body, but never
# wait on that HTTP before chunking the next file.
FLUSH_CHUNK_TARGET = 32
# One POST packs many short chunks. Each string still has to fit the model
# window (v5-text-small: 32k tokens). A long chunk always goes alone.
# ~10k tokens. 96k chars × 8 workers blew the 2M TPM cap and 429-stormed.
EMBED_HTTP_MAX_CHARS = 40_000
EMBED_HTTP_MAX_TEXTS = 64

DEFAULT_CHUNK_MAX_CHARS = 2400
DEFAULT_OPENAPI_MAX_CHARS = 16000
DEFAULT_CHUNK_MIN_CHARS = 40
DEFAULT_PREFETCH = 80
DEFAULT_LIMIT = 10
DEFAULT_MAX_FILE_BYTES = 8_000_000

DENSE_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "jinaai/jina-embeddings-v2-small-en": 512,
    "jinaai/jina-embeddings-v2-base-en": 768,
    "jina-embeddings-v2-base-en": 768,
    "jinaai/jina-embeddings-v2-base-code": 768,
    "jina-embeddings-v2-base-code": 768,
    "jina-embeddings-v5-text-small": 1024,
    "jina-embeddings-v5-text-nano": 768,
    "jina-embeddings-v5-omni-small": 1024,
    "jina-embeddings-v4": 2048,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5-Q": 768,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
}

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".md": "markdown",
    ".mdx": "markdown",
    ".vue": "vue",
    ".py": "python",
    ".json": "json",
}

CODE_LANGUAGES: frozenset[str] = frozenset(
    {"typescript", "tsx", "javascript", "python", "vue"}
)
MARKDOWN_LANGUAGES: frozenset[str] = frozenset({"markdown"})
JSON_LANGUAGE = "json"
OPENAPI_LANGUAGE = "openapi"

CATALOG_DIRNAME = ".bc-rag"
CATALOG_FILENAME = "catalog.json"
