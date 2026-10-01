# bc-rag-context

Local hybrid search over TypeScript, Markdown, and OpenAPI, for one or more repositories, served to Claude over MCP.

The project is **bc-rag-context**. The command is **bc-rag**.

- You register a repository once (`bc-rag project add`).
- bc-rag splits its files into chunks, embeds each chunk, and stores the vectors in Qdrant.
- Claude Code, Claude Desktop, and other MCP clients search those chunks through one shared MCP listener on `http://127.0.0.1:32323/mcp`.
- Qdrant runs in Docker, started and kept up by bc-rag. You can point bc-rag at your own Qdrant instead.
- Dense embeddings come from the Voyage API, the Jina API, or local FastEmbed models, chosen per space. BM25 (keyword) vectors are always computed locally.

---

## Contents

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Quick start](#3-quick-start)
4. [Command reference](#4-command-reference)
5. [Projects and the registry](#5-projects-and-the-registry)
6. [The project file `.bc-rag.json`](#6-the-project-file-bc-ragjson)
7. [Spaces and models](#7-spaces-and-models)
8. [Groups and facets](#8-groups-and-facets)
9. [Indexing](#9-indexing)
10. [OpenAPI specs](#10-openapi-specs)
11. [Search](#11-search)
12. [Services](#12-services)
13. [Background indexing](#13-background-indexing)
14. [MCP for Claude](#14-mcp-for-claude)
15. [Settings](#15-settings)
16. [Environment variables](#16-environment-variables)
17. [Troubleshooting](#17-troubleshooting)
18. [Scale and limits](#18-scale-and-limits)
19. [Development](#19-development)

---

## 1. Requirements

- Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/).
- Docker, when bc-rag runs Qdrant for you (the default, setting `qdrant=docker`).
- An API key for each provider your spaces use: Voyage (`voyage-*` and `rerank-*` models) or Jina (`jina-*` models). Local FastEmbed models need no key.
- [mkcert](https://github.com/FiloSottile/mkcert), only for the optional HTTPS listener.

---

## 2. Install

```bash
git clone <this repository> bc-rag-context
cd bc-rag-context
uv sync
```

Put `bin/` on your `PATH`. `bin/bc-rag` runs the command from this checkout and keeps your current folder, so it works from inside any repository:

```bash
export PATH="/path/to/bc-rag-context/bin:$PATH"
```

The first `bc-rag` command writes `~/.bc-rag/config` with the default settings (see [Settings](#15-settings)). It prints one dim line when it does.

---

## 3. Quick start

```bash
cd ~/code/my-repo
bc-rag project add                                    # registers the folder, writes a starter .bc-rag.json
$EDITOR .bc-rag.json                                  # describe your spaces and groups (section 6)
bc-rag config set voyage-api-key pa-... --global      # or jina-api-key, for the providers you use
bc-rag index                                          # starts Qdrant if needed, then indexes
bc-rag search "how are overlay keys written"
```

Then connect Claude (section 14).

---

## 4. Command reference

The verbs at the top work on one project's index. The nouns group the rest, like `git remote` and `git config`.

Every command that works on one project takes these two flags. Without them, the project that holds the current folder is used; outside any project, the command exits with code 2 and lists the registered project names. Passing both is an error (exit 2).

| Option | Meaning |
|---|---|
| `--project NAME`, `-p` | A registered project, by name. |
| `--root PATH`, `-r` | A folder inside a registered project. Defaults to the current folder. |

```text
Index
   bc-rag index
      --project, -p NAME
      --root, -r PATH
      --force                Rebuild every space, ignoring file hashes.
      --path PATH            Only these project-relative paths. Repeatable.
      --watch                Keep running and reindex files as they change.
      --debounce SECONDS     With --watch: seconds of quiet before hashing. Default 0.
      --skip-initial         With --watch: skip the full pass before watching.

   bc-rag search QUERY
      --project, -p NAME
      --root, -r PATH
      --space, -s SPACE      Space to search. Default: defaultSpace.
      --limit, -n COUNT      Hits to return. Default: the space's.
      --sparse               BM25 only. No dense vector and no rerank.
      --no-rerank            Skip the rerank pass.
      --facet, -f KEY=VALUE  Keep matches. Repeat a key for any-of; different keys all match.
      --exclude, -x KEY=VALUE  Drop matches. Repeatable.
      --json                 Print JSON.

   bc-rag status             Each space with its points and files, and the last and next background run.
      --project, -p NAME
      --root, -r PATH
      --json                 Print JSON.

   bc-rag facets             Facets declared in .bc-rag.json and stored in each space.
      --project, -p NAME
      --root, -r PATH
      --key, -k KEY          One key, with every one of its values.
      --json                 Print JSON.

Sources
   bc-rag sources files      Repo files the groups select, with space, group, chunker, and model.
      --project, -p NAME
      --root, -r PATH
      --group, -g GROUP      Any of these groups. Repeatable.
      --space, -s SPACE      Any of these spaces. Repeatable.
      --raw                  One path per line, nothing else.
      --jsonl                One JSON object per file.

   bc-rag sources groups     Groups in .bc-rag.json, highest priority first.
      --project, -p NAME
      --root, -r PATH
      --facet, -f KEY=VALUE  Keep groups with this facet. Repeatable.
      --exclude, -x KEY=VALUE  Drop groups with this facet. Repeatable.
      --timing               Time each group's file search instead, slowest first.
      --json                 Print JSON.

   bc-rag sources chunks PATH   The chunks one repo file becomes.
      --project, -p NAME
      --root, -r PATH

Projects
   bc-rag project list       Every registered project, its interval, and its last index.

   bc-rag project add        Register a folder. Writes .bc-rag.json if missing.
      --root, -r PATH        The project folder. Default: the current folder.
      --project, -p NAME     The name to register. Default: the folder name.

   bc-rag project clear      Delete the stored index: collections and cache. Stays registered.
      --project, -p NAME
      --root, -r PATH
      --yes, -y              Do not ask first.

   bc-rag project remove     Unregister the project. Its stored index stays on disk and in Qdrant.
      --project, -p NAME
      --root, -r PATH

Config (like git config and npm config: this project by default, --global for all)
   bc-rag config list        Every key, its value, and where it comes from.
      --project, -p NAME
      --root, -r PATH
      --global               Only the global file, ~/.bc-rag/config.

   bc-rag config get KEY     The value in effect, and where it comes from.
      --project, -p NAME
      --root, -r PATH
      --global               Only the global file.
      --reveal               Print a secret in full.

   bc-rag config set KEY VALUE   Your setting for this project, like every 10m.
      --project, -p NAME
      --root, -r PATH
      --global               Write ~/.bc-rag/config instead, for every project.

   bc-rag config unset KEY   Drop your setting for this project. The next source applies again.
      --project, -p NAME
      --root, -r PATH
      --global               Drop it from ~/.bc-rag/config instead.

Models
   bc-rag models list        Cached models, and which ones this project uses.
      --project, -p NAME
      --root, -r PATH

   bc-rag models add         Download local models into the cache.
      --name MODEL           Model id. Repeatable. Default: this project's local models.
      --project, -p NAME
      --root, -r PATH

   bc-rag models remove      Delete models from the cache.
      --name MODEL           Model id. Repeatable. Required.

Services
   bc-rag services list      Each service, its state, pid, and log.
      --json                 Print JSON.

   bc-rag services start [NAME]     qdrant, indexer, or mcp. Omit for all three.
      --recreate             Recreate the Qdrant container from the pinned image.

   bc-rag services stop [NAME]      qdrant, indexer, or mcp. Omit for all three.

   bc-rag services restart [NAME]   qdrant, indexer, or mcp. Omit for all three.

   bc-rag services run NAME         Run one service in the foreground: qdrant, indexer, or mcp.
      --log-file PATH        Write output to this rotating log instead of the screen.
```

`bc-rag --help` prints every command, under these headings. A group run without a subcommand (for example `bc-rag sources`) prints its help and exits with code 2.

---

## 5. Projects and the registry

A folder is a bc-rag project only after `bc-rag project add` registers it in `~/.bc-rag/catalog.json`. No other command adds a project.

`project add`:

1. writes a starter `.bc-rag.json` when the folder has none (it never overwrites one),
2. loads the file, so a broken file fails here instead of during background indexing,
3. registers the folder. A folder that is already registered, or a `--project NAME` another folder uses, is refused (exit 1). The default name is the folder name, or `parent-folder` when another project already has that name.

Every other command that works on one project finds it this way:

1. `--project NAME`: the registered project with that name (or that absolute root path).
2. Otherwise, starting at `--root` or the current folder, bc-rag walks up through the parent folders to the nearest registered root. Any subfolder of a project works.
3. Nothing found: exit 2 with

```text
not a bc-rag project: /path/you/were/in
pass --project NAME. Registered: my-repo, other-repo
or run `bc-rag project add` here (it writes a starter .bc-rag.json when the folder has none)
```

`project clear` deletes what is stored for the project (its Qdrant collections and `~/.bc-rag/{name}/`'s space folders and cache) and keeps it registered. `project remove` only unregisters: the vectors stay in Qdrant and the files stay under `~/.bc-rag/{name}/`; run `project clear` first to drop them.

Each project keeps its generated data under `~/.bc-rag/{name}/`:

| Path | What |
|---|---|
| `{space folder}/manifest.json` | The hash ledger of one space: one row per file, so unchanged files are skipped. The folder name is the space's backend id (section 7). |
| `cache/corpus/` | Each indexed document and its sidecar (section 9). |
| `cache/openapi-md/` | The rendered `spec.md` of each OpenAPI spec (section 10). |
| `index.jsonl` | The event log of index runs. |
| `schedule.json` | The last and next background run (section 13). |
| `index.lock` | Held while a process indexes the project, so two indexers never write at once. |

`cache/` is internal: indexing writes it, and rebuilds anything missing from the repo without an embedding API call. Before it existed, `corpus/` and `openapi-md/` sat directly in `~/.bc-rag/{name}/`; the next index run renames them into `cache/`.

---

## 6. The project file `.bc-rag.json`

The file is required. It sits at the project root and is shared with your team. Comments (`//` and `/* */`) are allowed; trailing commas are not.

Top-level keys:

| Key | Required | Meaning |
|---|---|---|
| `spaces` | yes | Vector spaces by name (section 7). |
| `defaultSpace` | yes | The space a command uses when none is named. Must be a key of `spaces`. |
| `groups` | no | Named sets of files (section 8). |
| `groupsCommand` | no | A program that prints more groups (below). |
| `exclude` | no | Globs no group may include. |
| `schedule` | no | `{ "every": "15m" }`: the team's background indexing interval (section 13). |
| `facetKeys` | no | One line per facet key you use, telling search clients what the key means. |
| `searchHints` | no | Filter examples for this project, shown to Claude by `list_projects`. |

Keys are snake_case or camelCase: `groupsCommand`/`groups_command`, `defaultSpace`/`default_space`, `facetKeys`/`facet_keys`, `searchHints`/`search_hints`, and inside `chunk`, `maxChars`/`max_chars` and `minChars`/`min_chars`.

Any other key is an error. These old keys fail with a message that says where the setting lives now: `embed`, `chunk`, `openapi`, `include`, `tags`, `facets`, `dense`, `sparse`, `rerank`, `retrieve` (at the top level), `useJinaApi`, `jinaApi`, `jinaDense`, `jinaRerank`, and `qdrant_url`/`qdrantUrl`:

```text
qdrant_url is not a project key. Run `bc-rag config set qdrant-url URL --global`.
```

### 6.1 `groupsCommand`

A program whose standard output is `{"groups": [...]}`, in the same shape as `groups`. Its groups are added after the static ones.

- A string is a path to the program, relative to the project root.
- A list is the argv, with no shell: `["node", "tools/scripts/bc-rag-groups.mjs"]`. A first item that contains `/` is resolved against the project root.
- It runs with the project root as its working folder and has 180 seconds.
- It runs for every command that lists files or groups: `index`, `sources ...`, and each background run. `search`, `status`, `project clear`, and the MCP tools never run it.
- A group name already used by a static group, a non-zero exit, output that is not JSON, or output without a `groups` array fails the command with a message naming the problem.

### 6.2 A trimmed real example

```jsonc
{
  "exclude": ["**/*.spec.ts", "**/*.test.ts", "**/__tests__/**"],
  "defaultSpace": "prose",
  "schedule": { "every": "15m" },
  "facetKeys": {
    "area": "product area: engine, admin, portal, ...",
    "provider": "integration slug; on the vendor spec and on the code that integrates it",
    "vendor": "integration slug; on the vendor spec only"
  },
  "searchHints": [
    "Olo's own API spec: {\"vendor\": \"olo\"}",
    "Code that calls Olo: {\"provider\": \"olo\", \"content_type\": \"source-code\"}"
  ],
  "spaces": {
    "prose": {
      "dimensions": 1024,
      "chunk": { "min_chars": 40, "max_chars": 80000 },
      "retrieve": { "prefetch": 80, "limit": 10, "rerank": true },
      "dense": { "provider": "voyage", "model": "voyage-context-4" },
      "sparse": { "provider": "local", "model": "Qdrant/bm25" },
      "rerank": { "provider": "voyage", "model": "rerank-2.5" }
    },
    "code": {
      "dimensions": 1024,
      "chunk": { "min_chars": 40, "max_chars": 80000 },
      "dense": { "provider": "voyage", "model": "voyage-code-4" },
      "sparse": { "provider": "local", "model": "Qdrant/bm25" },
      "rerank": { "provider": "voyage", "model": "rerank-2.5" }
    }
  },
  "groupsCommand": ["node", "tools/scripts/bc-rag-groups.mjs"],
  "groups": [
    {
      "name": "docs-reference-02-engine",
      "space": "prose",
      "priority": 80,
      "facets": { "scope": "internal", "area": "engine", "content_type": "documentation" },
      "include": ["docs/bigcolony/reference/02-engine/**/*.md"]
    },
    {
      "name": "apps-bigcolony-admin-web",
      "space": "code",
      "priority": 57,
      "facets": { "scope": "internal", "area": "admin", "content_type": "source-code" },
      "include": ["apps/bigcolony-admin-web/src/**/*.{ts,tsx,mts,vue}"]
    }
  ]
}
```

`bc-rag sources groups` lists the groups after `groupsCommand` has been merged; `--json` prints them in full.

---

## 7. Spaces and models

A space is one Qdrant collection with one dense model. Every group sends its files to one space. Two spaces let prose and code use different models, for example `voyage-context-4` for documents and `voyage-code-4` for source.

| Key | Meaning |
|---|---|
| `dense` | `{ "provider": "voyage" \| "jina" \| "local", "model": "..." }`. Required. |
| `sparse` | The BM25 model. Use `{ "provider": "local", "model": "Qdrant/bm25" }`. Required. |
| `rerank` | Optional reranker, same shape as `dense`. |
| `dimensions` | Output size, sent to the Voyage API. When unset, the model's own default is used. |
| `chunk` | Required. `max_chars` (200 to 80000, default 2400) and `min_chars` (0 to 2000, default 40). |
| `retrieve` | `prefetch` (1 to 400, default 80), `limit` (1 to 50, default 10), `rerank` (default `true`). |

Where a model runs:

| Model id | Runs through |
|---|---|
| `voyage-*` (dense), `rerank-*` (rerank) | the Voyage API. `voyage-context-*` models embed a whole document's chunks together. |
| `jina-*`, provider `jina` | the Jina API. |
| anything else, provider `local` | FastEmbed on this machine, for example `jinaai/jina-embeddings-v2-base-en`. |

A provider and id that do not belong together are rejected when the file loads, for example a `voyage` provider with a non-Voyage id, or provider `local` with an API id.

**Collections.** Each space's folder under `~/.bc-rag/{name}/` is named by its backend id, `{space}--{mode}--{dense model}--d{dimensions}--{sparse model}--c{max_chars}`, and its Qdrant collection is `bcrag_{project}_{space}_{mode}_{hash of the backend id}`. Changing any of those parts (space name, provider, dense model, dimensions, sparse model, `chunk.max_chars`) starts a new, empty collection; the next `bc-rag index` fills it. Changing `min_chars` does not; run `bc-rag index --force` to re-chunk.

---

## 8. Groups and facets

### 8.1 Groups

| Field | Meaning |
|---|---|
| `name` | Unique within the project. Stored on every chunk as the facet `group`. |
| `space` | Which space the files go to. |
| `include` | Globs, relative to the project root. `**`, `{a,b}`, and dotfiles are matched. |
| `exclude` | Globs removed from this group. |
| `priority` | Higher claims first. Default 0. |
| `kind` | `auto` (default: by file extension) or `openapi` (every match is an OpenAPI spec). |
| `facets` | Labels for every file the group claims (below). |
| `enabled` | `false` keeps the group in the file but out of the index. |

Globs are expanded on the filesystem; Git is not consulted. These folders are always pruned, whatever the globs say: `.angular`, `.bc-rag`, `.git`, `.next`, `.nx`, `.turbo`, `.venv`, `build`, `coverage`, `dist`, `node_modules`, `out`.

Files are read with these languages: TypeScript (`.ts`, `.mts`, `.cts`, `.tsx`), JavaScript (`.js`, `.mjs`, `.cjs`, `.jsx`), Vue, Python, Markdown (`.md`, `.mdx`), and JSON. Other extensions are skipped unless the group's `kind` is `openapi`.

Groups are processed highest `priority` first, then by name. Within one space, the first group that matches a file claims it. A file claimed in two spaces is indexed in both.

### 8.2 Facets

A facet is a key with one value or a list of values. It is the only label format, from the config file to search results:

```json
{ "area": "engine", "scope": "internal", "provider": ["olo", "toast"] }
```

- Keys start with a letter and use letters, digits, `_` and `-`.
- Values are non-empty strings. bc-rag stores and returns every value as a list.
- A file carries only the facets of the group that claimed it.

bc-rag sets these keys itself. A group may not use them:

| Key | Set on |
|---|---|
| `group` | every chunk: the name of the group that claimed the file |
| `specSlug` | OpenAPI chunks: the spec file |
| `method`, `apiPath`, `operationId` | OpenAPI chunks: the operation |
| `apiTag` | OpenAPI chunks: every OpenAPI tag of the operation |
| `at`, `day`, `role` | chat transcript turns in Markdown |

Searching by facets: values under one key are any-of, and different keys must all match. `exclude` has the same shape and removes matches.

```bash
bc-rag search "create a basket" --facet vendor=olo
bc-rag search "location sync" -f area=engine -f area=admin -x scope=external
```

An OR across two different keys ("vendor olo, or area engine") cannot be written as one filter; run two searches.

`bc-rag facets` lists every facet key and value, the ones declared by enabled groups next to the ones stored per space, with point counts. A key with more than 50 values collapses to one row; `--key NAME` lists all of them.

---

## 9. Indexing

`bc-rag index` runs every space through one pipeline:

1. **Discover.** Expand each enabled group's globs once, for every space, as described in section 8.
2. **OpenAPI.** Render each spec of an `openapi` group to one `spec.md` (section 10).
3. **Corpus.** Write each file as a document plus a sidecar `{document}.metadata.json` holding `{"metadataAttributes": <the file's facets>}` under `~/.bc-rag/{name}/cache/corpus/`. This is the layout an Amazon Bedrock knowledge base reads.
4. **Skip unchanged.** A file whose size and modification time match its row in the space's `manifest.json`, and whose facets are unchanged, is skipped without being read. That is the rule `git status` uses. Otherwise the document and sidecar hashes are compared; a file whose content and facets are unchanged is still skipped, and its new modification time is recorded. An edit that keeps both the size and the modification time is missed; `--force` re-reads everything.
5. **Chunk.** TypeScript, JavaScript, Vue, and Python are split on functions and classes with tree-sitter. Markdown is split on headings and keeps the heading path; chat transcripts (`<details><summary>{time} - user|agent: ...</summary>`) are split per turn, and an agent turn carries the user question it answers. `package.json` and `project.json` are rendered as short Markdown. Text longer than `max_chars` is split with a 200-character overlap.
6. **Embed and store.** Each chunk's dense input is a short header (file, symbol, kind, section, and a `Facets:` line) plus the text. The BM25 input is the text alone. Both vectors and the chunk's facets go to Qdrant.

Other rules:

- **Deletions.** A file in the manifest that is gone from disk, or that no group claims any more, has its points deleted. A file that now yields no chunks has its points deleted too.
- **Size cap.** A file over 8 MB is skipped, except OpenAPI specs.
- **Resume.** The manifest is saved every 2 seconds. A killed run resumes where it stopped: files already recorded are skipped.
- **Ctrl+C.** The first Ctrl+C finishes the files in flight, saves the manifest, and stops without deleting anything. A second one aborts at once (exit 130).
- **One writer.** A second index run of the same project fails with `... is already being indexed (pid N, since ...)` while the first one holds `index.lock`.
- **Panel.** On a terminal, three boxes side by side, one per stage: `read and split files`, `embed with <provider>`, `write to Qdrant`. Each box shows the same rows (`working`, `queued`, `done`, `chunks`, `files`, `last`, `eta`, `errors`), then one line per item its workers hold. Problems print as a plain list when the run ends.
- **Log.** Every event is a line in `~/.bc-rag/{name}/index.jsonl`. `file_complete` marks a file whose every chunk is written to Qdrant.
- **Rebuild.** `--force` recreates every space's collection. A different dense or sparse model than the manifest records does the same.
- **Some paths.** `--path FILE` (repeatable) indexes only those files.

Preview without writing vectors: `bc-rag sources files` shows what would be indexed, with its space, group, chunker, and model, and `bc-rag sources chunks PATH` prints the chunks of one file.

`bc-rag index --watch` indexes once, then watches the project and reindexes only the files whose content or facets change.

---

## 10. OpenAPI specs

A group with `"kind": "openapi"` treats every matched file as an OpenAPI spec (JSON or YAML):

1. `$ref` pointers are inlined with prance; reference cycles are cut instead of failing.
2. The spec is rendered to one `spec.md` under `~/.bc-rag/{name}/cache/openapi-md/`, with one `## METHOD /path` section per operation, listing its operation id, tags, parameters, request body, and responses.
3. A `source.sha256` file next to `spec.md` records the spec it came from. An edited spec is rendered again on the next run.
4. The Markdown chunker splits `spec.md` on those headings. Every chunk of an operation, including its sub-sections, carries `specSlug`, `method`, `apiPath`, `operationId`, and `apiTag`.

---

## 11. Search

`bc-rag search "query"`:

- **Dense** (default): embeds the query with the space's dense model and retrieves by meaning. When the space reranks (`retrieve.rerank` is true and a `rerank` model is set), Qdrant returns `max(prefetch, limit x 4)` candidates and the reranker keeps the best `limit`.
- **`--sparse`**: BM25 only, for identifiers, routes, paths, and exact error text. No dense vector and no rerank.
- `--space NAME` (default `defaultSpace`), `--limit N` (default the space's `retrieve.limit`), `--no-rerank`, `--facet key=value`, `--exclude key=value`, `--json`.

The MCP `search` tool follows the same space rules.

---

## 12. Services

bc-rag runs three background services, all kept up by one supervisor process per user:

| Service | What |
|---|---|
| `qdrant` | The Qdrant container (`qdrant=docker`), or a check that your own Qdrant answers (`qdrant=external`). |
| `indexer` | Indexes each project whose interval is due (section 13). |
| `mcp` | The MCP listener: `http://127.0.0.1:32323/mcp`, plus `https://bc-rag.localhost:32324/mcp` when the certificate files exist (section 14). |

Commands:

- `bc-rag services list` shows each service's state, pid, detail, and log.
- `bc-rag services start [NAME]`, `stop [NAME]`, `restart [NAME]` act on one service, or on all three without a name.
- A stopped service **stays stopped**, across commands and restarts, until you start it. `services stop indexer` pauses background indexing while MCP keeps serving.
- `bc-rag services run NAME` runs one service in the foreground with its log on screen, for debugging. It refuses while the supervisor already runs that service.

**Autostart.** The commands that need the index (`project index build`, `watch`, `search`, `show`, `facets`, `delete`) start the supervisor when it is not running, wait up to 30 seconds for Qdrant, and replace a supervisor that runs older bc-rag code. They never start a service you stopped. Turn this off with `bc-rag settings set autostart false`, or for one command with `BC_RAG_AUTOSTART=false`.

**Environment.** The services inherit the environment of the shell that started the supervisor (`PATH` for `node` and `docker`, and API keys you keep in environment variables). `bc-rag services restart` without a name starts a new supervisor from your current shell. `services restart NAME` restarts one service under the supervisor's existing environment.

**Stopping.** The indexer gets 120 seconds to finish its current file and save its manifest; mcp gets 10 seconds. After that the supervisor kills them. A service that exits on its own is restarted after 2, 4, 8, ... up to 60 seconds. When the mcp port is taken, the mcp service exits with code 3 and the supervisor retries every 60 seconds:

```text
mcp service already runs on 127.0.0.1:32323; run `bc-rag services stop mcp` first
```

**Docker Qdrant.** With `qdrant=docker`, bc-rag creates the container `bc-rag-qdrant` from `qdrant/qdrant:v1.19.1`, with its data on the Docker volume `bc-rag-qdrant-data` and its ports on loopback only:

```bash
docker run -d --name bc-rag-qdrant --label bc-rag=qdrant --restart unless-stopped \
  -p 127.0.0.1:32321:6333 -p 127.0.0.1:32322:6334 \
  -v bc-rag-qdrant-data:/qdrant/storage qdrant/qdrant:v1.19.1
```

It uses your current Docker context unless the setting `docker-context` names one. A container named `bc-rag-qdrant` that bc-rag did not create (no `bc-rag=qdrant` label) is reported and never touched. `services start qdrant --recreate` replaces the container with the pinned image; the volume, and so the data, is kept.

**Your own Qdrant.** `bc-rag settings set qdrant external`, `settings set qdrant-url https://...`, and, when it needs one, `settings set qdrant-api-key ...`. bc-rag then never calls Docker.

**Logs.** `~/.bc-rag/logs/supervisor.log`, `indexer.log`, and `mcp.log`. Each rotates at 10 MB and keeps two older copies. The container's log is `docker logs bc-rag-qdrant`.

**Files.** `~/.bc-rag/services.json` holds what you asked for (written only by `services start/stop/restart`). `~/.bc-rag/supervisor.json` holds what is running (written only by the supervisor).

---

## 13. Background indexing

Background indexing is off for a project until it has an interval. There is no built-in interval.

1. **Team default:** `"schedule": { "every": "15m" }` in `.bc-rag.json`.
2. **Your override:** `bc-rag project set every 5m` stores your interval in the registry, or `bc-rag project set every off` to turn it off for you. `bc-rag project unset every` follows the team default again.

```text
every  5m   (your override; project default 15m)
every  15m  (project default)
every  none (no schedule; set schedule.every in .bc-rag.json or run bc-rag project set every 10m)
```

The indexer service indexes one project at a time. A project's next run is due one interval after its last run finished, so runs never overlap and restarting the services never triggers an extra run. `project show` prints the last and next run, and `services list` shows what the indexer is doing.

A project that is being indexed by another command (for example a manual `project index build`) is skipped until its next due time.

---

## 14. MCP for Claude

The mcp service listens on `http://127.0.0.1:32323/mcp`. Every chat shares it; you register one MCP server for all your projects.

**Claude Code** (`.mcp.json` or your user config):

```json
{
  "mcpServers": {
    "bc-rag": { "type": "http", "url": "http://127.0.0.1:32323/mcp" }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`), through `mcp-remote`:

```json
{
  "mcpServers": {
    "bc-rag": {
      "command": "npx",
      "args": ["mcp-remote", "http://127.0.0.1:32323/mcp", "--transport", "http-first"]
    }
  }
}
```

**HTTPS.** Forms that only accept `https` URLs can use `https://bc-rag.localhost:32324/mcp`. Run `mkcert -install` once, then:

```bash
bin/bc-rag-mcp-cert          # writes ~/.bc-rag/mcp.pem and ~/.bc-rag/mcp.key
bc-rag services restart mcp  # the listener picks the files up
```

**Tools.** All four are read-only.

| Tool | What |
|---|---|
| `list_projects` | Every registered project, its spaces with their dense and rerank models, its `facetKeys`, and its `searchHints`. A project whose config fails to load is listed with an `error` instead of hiding the others. |
| `search` | Dense search, then rerank when the space reranks. Takes `query`, `project`, `space`, and optionally `limit`, `facets`, and `exclude`. Each hit carries its `facets`; the result says whether it was `reranked`. |
| `search_sparse` | BM25 only, same parameters. |
| `list_facets` | The facet keys stored in a space with their meaning, point count, and values (first 50 per key). With `key`, every value of that key (up to 1000, or `limit`). |

A failing tool returns its error text to Claude, for example:

```text
Error executing tool search: ValueError: unknown project: nope. Registered projects: bigcolony-workspaces
```

---

## 15. Settings

`~/.bc-rag/config` is a JSON file readable only by you (mode 0600). Every command fills in the missing non-secret settings with their defaults.

| Key | Default | Secret | Environment variable | Meaning |
|---|---|---|---|---|
| `voyage-api-key` | | yes | `VOYAGE_AI_API_KEY`, `VOYAGE_API_KEY` | Voyage key for dense embeddings and rerank. |
| `jina-api-key` | | yes | `JINA_API_KEY` | Jina key, for spaces that use the Jina API. |
| `autostart` | `true` | no | `BC_RAG_AUTOSTART`, `BC_RAG_DAEMON` | Start the services when an index or search command needs them. |
| `qdrant` | `docker` | no | `BC_RAG_QDRANT` | `docker`: bc-rag runs Qdrant. `external`: you run it. |
| `qdrant-url` | `http://127.0.0.1:32321` | no | `BC_RAG_QDRANT_URL` | Qdrant's HTTP URL. |
| `qdrant-api-key` | | yes | `QDRANT_API_KEY` | Sent as the `api-key` header, for a Qdrant that asks for one. |
| `docker-context` | empty | no | `BC_RAG_DOCKER_CONTEXT` | Docker context for the container. Empty means the current one. |

Where a value comes from:

- A secret: the file first, then its environment variable.
- Any other setting: its environment variable first, then the file, then the default. The file always holds these keys, so the environment variable is a one-shot override.

`settings list` shows every value with its source, secrets masked. `settings get KEY --reveal` prints a secret in full. `settings set` validates the value: `autostart` takes true/false/on/off/yes/no/1/0, `qdrant` takes `docker` or `external`, `qdrant-url` must start with `http://` or `https://`.

---

## 16. Environment variables

| Variable | Effect |
|---|---|
| `BC_RAG_HOME` | Use this folder instead of `~/.bc-rag`. |
| `BC_RAG_AUTOSTART`, `BC_RAG_DAEMON` | `false` turns autostart off for one command. |
| `BC_RAG_QDRANT`, `BC_RAG_QDRANT_URL`, `BC_RAG_DOCKER_CONTEXT`, `QDRANT_API_KEY` | Override the matching setting. |
| `VOYAGE_AI_API_KEY` (or `VOYAGE_API_KEY`), `JINA_API_KEY` | API keys, when the settings file has none. |
| `FASTEMBED_CACHE_PATH` (or `BC_RAG_MODELS`) | Use this folder instead of `~/.cache/bc-rag/fastembed` for local models. |

---

## 17. Troubleshooting

- **Start with `bc-rag services list`.** It shows whether the supervisor runs, whether it runs current code, each service's state, and the path of each log.
- **`not a bc-rag project: ...`** The folder is not inside a registered project. Run `bc-rag project add` at the project root, or pass `--project NAME`.
- **`groupsCommand exited 1: ...`** The message carries the program's own error output. For example, an Nx-based script can fail when a long-running Nx daemon's socket path is too long; stop the Nx daemon (`nx daemon --stop`) or make the script read the workspace files directly.
- **`mcp service already runs on 127.0.0.1:32323`** Something already listens on the port: usually the supervised mcp service. `bc-rag services stop mcp` before `services run mcp`, or find the owner with `lsof -nP -iTCP:32323 -sTCP:LISTEN`.
- **`qdrant unreachable at ...`** `bc-rag services start qdrant`. With `qdrant=external`, check `qdrant-url` and `qdrant-api-key`.
- **`empty index. Run bc-rag project index build first.`** Nothing is stored in that space yet, or its collection changed (section 7).
- **An index run that did less than expected.** Read `~/.bc-rag/{name}/index.jsonl`: one JSON line per event, including `error`, `skip_large`, `empty`, `skip_hash` (its `by` says `stat` or `hash`), `file_complete`, `deleted`, and the final `done` or `stopped` with totals.
- **`... is not a valid manifest`** The ledger file is damaged. Move it aside to re-hash every file, or run `project index build --force`. bc-rag never treats a damaged ledger as a first run, because that would drop the collection.

---

## 18. Scale and limits

One Nx workspace, as a starting target:

| Corpus | Size |
| --- | --- |
| Vendor documentation | 3,538,000 tokens |
| Own system documentation | 1,637,000 tokens |
| TypeScript | 9,512,000 tokens in 20,749 files |
| Other (json / vue / mixed) | 981,000 tokens |
| Packages | 180 |

That is about 15.7 million tokens. The average TypeScript file is about 460 tokens, so most source files become one chunk, and the index holds tens of thousands of Qdrant points, not millions. The long step is the first embedding pass; later runs only embed changed files.

Limits worth knowing:

- **Jina API** truncates an input that is longer than the model's window instead of failing.
- **Voyage contextual models** (`voyage-context-4`) refuse a document over 32,000 tokens. bc-rag packs each file's chunks into requests of at most 20,000 estimated tokens and splits a chunk that alone is too long.
- **Embedding batches.** Voyage flat models (for example `voyage-code-4`) take up to 256 texts and 480,000 characters per request; every other model takes up to 64 texts and 40,000 characters.
- **Facet listing.** A facet key returns at most 10,000 distinct values.

Do not add `**/*.json` to a group's `include` on an Nx repository: it pulls in `project.json`, `tsconfig`, and lockfiles. List OpenAPI specs in a `kind: openapi` group instead.

---

## 19. Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
```

`uv sync` without `--extra dev` removes pytest and ruff from the environment.

Module map of `src/bc_rag/`:

| Module | Holds |
|---|---|
| `cli/` | The command tree: `_app.py` (root, help, project lookup), one module per group. |
| `catalog.py` | The registry, project lookup, and per-project intervals. |
| `config.py` | `.bc-rag.json` models, loading, and validation. |
| `facets.py` | The facet format, validation, and the OpenAPI facets. |
| `discover.py` | Group globs and which group claims a file. |
| `chunking.py` | The chunkers. |
| `corpus.py` | Corpus documents and sidecars. |
| `indexer.py` | The index pipeline, the manifest checks, and the project lock. |
| `embeddings.py`, `voyage_api.py`, `jina_api.py`, `models.py` | Embedding and rerank clients, and the FastEmbed cache. |
| `store.py` | Qdrant collections, payloads, facet indexes and filters. |
| `query.py` | Search and rerank. |
| `mcp_server.py` | The MCP tools and listeners. |
| `services/` | The supervisor, Docker Qdrant, the indexer and mcp services, and service logs. |
| `schedule.py` | Intervals and the per-project run state. |
| `openapi.py`, `split_md.py` | OpenAPI rendering. |
| `overview.py`, `reset.py` | `project index show`, `facets`, and `delete`. |
| `usersettings.py`, `fileio.py`, `locks.py`, `duration.py` | Settings, atomic writes, file locks, durations. |
