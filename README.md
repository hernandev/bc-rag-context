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
bc-rag init --root [folder]          # write .bc-rag.json (optional, defaults work)
bc-rag index --root [folder]         # index. Unchanged files are skipped by sha256
bc-rag index --all --every 5m         # every cataloged project, again every 5 minutes
bc-rag query --query "how do I write overlay keys"
bc-rag watch --root [folder]         # index, then rewrite only files that change or disappear
bc-rag status --root [folder]
bc-rag projects list                  # catalog used by MCP
bc-rag projects forget --name NAME
bc-rag files list --group NAME --space prose
bc-rag mcp                            # stdio, one chat
bc-rag mcp --http                     # one shared listener on 127.0.0.1:32323/mcp
bc-rag config set voyage-api-key KEY  # stored in ~/.bc-rag/config
bc-rag daemon status                  # the background indexer and MCP listener
```

`bc-rag index` hashes each file. A later run embeds only files whose bytes changed, and deletes paths that vanished.

`bc-rag watch` does the same for a single file event. It does not re-embed the rest of the repository.

## One MCP server, many repos

Index each repository once:

```bash
bc-rag index --root /path/to/engine
bc-rag index --root /path/to/portal
```

Each successful index registers the root in `~/.bc-rag/catalog.json`.

Start one shared listener, then point every client at that address. Each chat then connects over HTTP instead of starting its own process.

```bash
bc-rag mcp --http
```

That listens on `http://127.0.0.1:32323/mcp`. You rarely start it by hand: [the daemon](#the-daemon) runs it.

Claude's remote MCP form asks for `https`. Issue a certificate with mkcert, then the same process also listens on `https://bc-rag.localhost:32324/mcp`:

```bash
bin/bc-rag-mcp-cert
```

That writes `~/.bc-rag/mcp.pem` and `~/.bc-rag/mcp.key` for the name `bc-rag.localhost`. `mkcert -install` has to have been run once already, so the certificate is trusted. Restart the daemon after the files appear (`bc-rag daemon restart`) so the running listener picks them up.

```json
{
  "mcpServers": {
    "bc-rag": {
      "type": "http",
      "url": "http://127.0.0.1:32323/mcp"
    }
  }
}
```

Tools:

- `list_projects`
- `search` — dense vectors, then rerank. `project` and `space` are required.
- `search_sparse` — BM25 only. No dense vector and no rerank. `project` and `space` are required.
- `list_tags`

You do not register one MCP server per repository.

## API keys

Keys live in `~/.bc-rag/config`, a JSON file only you can read:

```bash
bc-rag config set voyage-api-key pa-...
bc-rag config list                     # every setting, where it comes from, secrets masked
bc-rag config get voyage-api-key --reveal
bc-rag config unset voyage-api-key
```

A setting missing from that file falls back to its environment variable: `VOYAGE_AI_API_KEY` (or `VOYAGE_API_KEY`), `JINA_API_KEY`, `BC_RAG_DAEMON_INDEX_EVERY`.

## The daemon

Like the Nx daemon, nothing is installed. The first `bc-rag` command you run starts one background process per user, and it keeps running after the terminal closes. It runs two children and restarts either one when it exits:

| Child | Command |
|---|---|
| index | `bc-rag index --all --every 5m` — every project in `~/.bc-rag/catalog.json` |
| mcp | `bc-rag mcp --http` — `127.0.0.1:32323/mcp`, and `https://bc-rag.localhost:32324/mcp` when `~/.bc-rag/mcp.pem` and `mcp.key` exist |

A repository joins the index the first time you run `bc-rag index --root` there. Set the interval with `bc-rag config set daemon-index-every 10m`.

The daemon inherits the environment of the command that started it, so `node`, `pnpm` and `docker` come from your own PATH. When bc-rag's own source changes, the next command replaces the running daemon, so it never serves old code.

```bash
bc-rag daemon status
bc-rag daemon restart                 # pick up a new PATH or setting
bc-rag daemon stop
BC_RAG_DAEMON=false bc-rag query ...  # run one command without starting it
```

The log is `~/.bc-rag/logs/daemon.log`. After a reboot the daemon is down until the next `bc-rag` command, exactly like Nx.

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

`search` pulls dense candidates, then the reranker reorders them.

`search_sparse` is BM25 only. No dense vector and no rerank. That is the path for an identifier or an exact error string.

## Spaces

A `.bc-rag.json` can name more than one vector space. Each group points at one space. Each space names its own dense model.

A model whose name starts with `voyage-context` uses the Voyage contextual HTTP endpoint. `input` `chunks` sends this repo's chunks as one batch per file. `input` `auto` lets Voyage split the file. Every other model keeps today's embed path.

`search` and `search_sparse` require `project` and `space`. A file with no `embed.spaces` has one space named `default`, and that collection name stays the one already in use.

```json
"embed": {
  "spaces": {
    "prose": { "dense": "voyage-context-4", "dimensions": 1024, "input": "chunks" },
    "code": { "dense": "voyage-code-4", "dimensions": 1024 }
  },
  "defaultSpace": "prose"
}
```

```json
{ "name": "docs", "space": "prose", "include": ["docs/**/*.md"] }
```

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
