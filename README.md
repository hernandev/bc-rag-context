# bc-rag-context

This project is **bc-rag-context**.

The binary is **bc-rag**.

**bc-rag** is also the nickname for the tool.

Local hybrid retrieval over TypeScript, Markdown, and OpenAPI.

Clone this repo, put `bin/` on your `PATH`, then run `bc-rag` against a project folder.

## What it is for

You point it at a repository.

It indexes:

- TypeScript and TSX, split on functions and classes
- Markdown, split on headings
- OpenAPI entrypoints, after Redocly fully dereferences split files, **one retrieval document per operation**

Then you search by meaning.

Vendor OpenAPI files that name the same feature differently are the main case.

Sparse BM25 is still stored. It is the backstop for when you already know an identifier.

## No Ollama

Embeddings run in-process through [FastEmbed](https://github.com/qdrant/fastembed) (ONNX Runtime).

The first index downloads the models into the FastEmbed cache (usually `~/.cache/fastembed`).

Nothing else has to be running. No Ollama. No Qdrant Docker. Qdrant runs embedded on disk under `~/.bc-rag/{project-name}/qdrant`. `.bc-rag.json` stays in the project folder.

## Install

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd /path/to/bc-rag-context
uv sync
chmod +x bin/bc-rag
```

Put the `bin` directory on your `PATH`:

```bash
export PATH="/path/to/bc-rag-context/bin:$PATH"
```

OpenAPI split files need the [Redocly CLI](https://redocly.com/docs/cli) on `PATH`, or `npx` so `bc-rag` can run `@redocly/cli`.

## Commands

```bash
bc-rag init [folder]          # write .bc-rag.json (optional, defaults work)
bc-rag index [folder]         # index. Unchanged files are skipped by sha256
bc-rag query "how do I write overlay keys"
bc-rag watch [folder]         # index, then rewrite only files that change or disappear
bc-rag status [folder]
bc-rag projects list          # catalog used by MCP
bc-rag projects forget NAME
bc-rag mcp                    # one stdio MCP server for every cataloged project
```

`bc-rag index` hashes each file. A later run embeds only files whose bytes changed, and deletes paths that vanished.

`bc-rag watch` does the same for a single file event. It does not re-embed the rest of the repository.

## One MCP server, many repos

Index each repository once:

```bash
bc-rag index /path/to/engine
bc-rag index /path/to/portal
```

Each successful index registers the root in `~/.bc-rag/catalog.json`.

Register **one** MCP server in Cursor, Claude, or any local agent:

```json
{
  "mcpServers": {
    "bc-rag": {
      "command": "bc-rag",
      "args": ["mcp"]
    }
  }
}
```

Tools:

- `list_projects`
- `search(query, project?, limit?)` — omit `project` to search every living catalog entry

You do not register one MCP server per repository.

## Config

`.bc-rag.json` is optional. Missing file means the defaults.

```json
{
  "groups": [
    {
      "name": "source",
      "priority": 50,
      "include": ["libs/**/src/**/*.{ts,tsx,mts}"],
      "exclude": ["**/*.spec.ts"]
    },
    {
      "name": "vendor-openapi",
      "kind": "openapi",
      "priority": 100,
      "include": ["docs/providers/_openapi/*.json"]
    }
  ]
}
```

Include and exclude are expanded on disk. Git is not the file list. `node_modules` and `dist` are still pruned by directory name.

The group `name` is the tag on every chunk. Ingest order is `priority` (high first), not JSON list order.

## Models

Default dense model: `jinaai/jina-embeddings-v2-base-en` (768 dimensions, 8192-token context).

Default reranker: `jinaai/jina-reranker-v1-turbo-en` (8k context, so a whole operation can be scored).

A query pulls 80 dense + 80 BM25 candidates, fuses them with Reciprocal Rank Fusion, then the reranker reorders.

That is the path for "I do not know what they called it."

## Scale this is built for

One Nx workspace, as a starting target:

| Corpus | Size |
| --- | --- |
| Vendor documentation | 3,538,000 tokens |
| Own system documentation | 1,637,000 tokens |
| TypeScript | 9,512,000 tokens in 20,749 files |
| Other (json / vue / mixed) | 981,000 tokens |
| Packages | 180 |

That is about **15.7 million tokens**.

Average TypeScript file is about **460 tokens**, so most source files become **one chunk**.

Expect **tens of thousands of Qdrant points**, not millions.

Qdrant-on-disk is fine at that size.

The long pole is the **first embed**, on CPU, through FastEmbed.

A killed `bc-rag index` **resumes**: each flush writes `~/.bc-rag/{project-name}/manifest.json` with sha256 per file, and the next run skips those files.

`bc-rag index` uses every CPU (`os.cpu_count()`) for FastEmbed ONNX. There is no cap. Dense and sparse embed one after the other. Embed batches are 1, 2, or 8 texts depending on chunk length.

Do **not** add `**/*.json` to `include` on an Nx repo. That pulls `project.json`, `tsconfig`, and lockfile noise. OpenAPI JSON goes through `openapi.include` (entrypoints only).

`.nx/`, `dist/`, `tmp/`, `.angular/` are excluded by default.
