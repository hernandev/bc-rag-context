"""One MCP server for every registered project, shared by every chat over HTTP.

The mcp service (`bc-rag services ...`) listens on `http://127.0.0.1:32323/mcp`.
When `~/.bc-rag/mcp.pem` and `~/.bc-rag/mcp.key` both exist, the same process also
listens on `https://bc-rag.localhost:32324/mcp`.

Tools: `list_projects`, `search`, `search_sparse`, `list_facets`. A failing tool
returns its error text to the client; nothing is reduced to "Error executing tool".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from bc_rag.catalog import find_project, living_projects
from bc_rag.defaults import (
    MCP_HTTP_HOST,
    MCP_HTTP_PATH,
    MCP_HTTP_PORT,
    MCP_HTTPS_PORT,
    MCP_TLS_CERT_FILENAME,
    MCP_TLS_HOST,
    MCP_TLS_KEY_FILENAME,
)
from bc_rag.facets import RESERVED_KEYS, normalize_facets
from bc_rag.store import Hit

log = logging.getLogger("bc_rag.mcp")

LIST_FACETS_OVERVIEW_LIMIT = 50
LIST_FACETS_KEY_LIMIT = 1000


def run_mcp(*, host: str = MCP_HTTP_HOST, port: int = MCP_HTTP_PORT) -> None:
    """Serve the tools over HTTP until SIGTERM or Ctrl+C. Also HTTPS when cert files exist."""
    if not host or host.strip() != host or any(ch.isspace() for ch in host):
        raise ValueError(f"invalid mcp host: {host!r}")
    if port < 1 or port > 65535:
        raise ValueError(f"invalid mcp port: {port}")
    mcp = build_server()
    tls = mcp_tls_files()
    if tls is None:
        url = f"http://{host}:{port}{MCP_HTTP_PATH}"
        print(f"bc-rag mcp {url}", file=sys.stderr, flush=True)
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            streamable_http_path=MCP_HTTP_PATH,
            # Each chat is its own request. No session table to leak when a chat closes.
            stateless_http=True,
            json_response=True,
        )
        return
    cert, key = tls
    _serve_http_and_https(mcp, host=host, port=port, cert=cert, key=key)


def server_instructions() -> str:
    reserved = "\n".join(f"- {key}: {meaning}" for key, meaning in RESERVED_KEYS.items())
    return (
        "Search local source and docs indexed by bc-rag.\n"
        "Call list_projects first: it names each project's spaces, and its facetKeys and "
        "hints explain the project's filters. A space is one collection; search one at "
        "a time.\n"
        "search embeds the query, retrieves by meaning, and reranks when the space has a "
        "reranker. search_sparse is BM25: use it for identifiers, routes, paths, and exact "
        "error text.\n"
        'Filter with facets, an object of key -> value or list of values, such as '
        '{"area": "engine", "provider": ["olo", "toast"]}. Values under one key are '
        "any-of; different keys must all match. exclude has the same shape and removes "
        "matches. To match one key OR another key, run two searches.\n"
        "Values are exact strings. Call list_facets for the space first: it lists each "
        "key with its meaning and values.\n"
        f"Keys bc-rag sets itself:\n{reserved}\n"
        "Project keys are described by the project's facetKeys."
    )


FacetsArg = Annotated[
    dict[str, str | list[str]] | None,
    Field(
        description=(
            "Only hits whose facets match. key -> value or list of values. Values under "
            "one key are any-of; different keys must all match. Use values list_facets "
            'returned. Examples: {"vendor": "olo"}; {"area": "engine", "apiTag": '
            '["Baskets", "Orders"]}.'
        )
    ),
]
ExcludeArg = Annotated[
    dict[str, str | list[str]] | None,
    Field(
        description=(
            'Remove hits whose facets match. Same shape as facets: {"scope": "external"}.'
        )
    ),
]
ProjectArg = Annotated[str, Field(description="Project name from list_projects.")]
SpaceArg = Annotated[str, Field(description="Space name from list_projects.")]
LimitArg = Annotated[
    int | None,
    Field(ge=1, le=50, description="Hits to return. Omit for the space's default."),
]


@contextlib.contextmanager
def _surfaced(tool: str) -> Iterator[None]:
    """Turn any failure into a ToolError whose text reaches the client."""
    from mcp.server.mcpserver.exceptions import ToolError

    try:
        yield
    except ToolError:
        raise
    except Exception as error:
        log.exception("tool %s failed", tool)
        raise ToolError(f"{type(error).__name__}: {error}") from error


def _entry(project: str):
    entries = living_projects()
    found = find_project(project, entries)
    if found is None:
        known = ", ".join(entry.name for entry in entries) or "none"
        raise ValueError(f"unknown project: {project}. Registered projects: {known}")
    return found


def _search_result(result, project: str, space: str, facets, exclude) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": result.query,
        "project": project,
        "space": space,
        "reranked": result.reranked,
        "hits": [_hit_dict(hit) for hit in result.hits],
    }
    if (facets or exclude) and not result.hits:
        payload["hint"] = (
            "no hit matched these facets. Check the values with list_facets, or drop a key."
        )
    return payload


def build_server() -> Any:
    """The MCP server with every tool registered. No listener yet."""
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    mcp = MCPServer("bc-rag", instructions=server_instructions())
    read_only = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

    @mcp.tool(annotations=read_only)
    def list_projects() -> list[dict[str, Any]]:
        """List registered projects, their spaces and models, facet key meanings, and hints."""
        from bc_rag.config import load_config

        with _surfaced("list_projects"):
            rows: list[dict[str, Any]] = []
            for entry in living_projects():
                try:
                    configuration, _path = load_config(entry.root_path(), groups=False)
                except Exception as error:
                    # one broken project must not hide the others.
                    rows.append(
                        {"name": entry.name, "root": entry.root, "spaces": [], "error": str(error)}
                    )
                    continue
                rows.append(
                    {
                        "name": entry.name,
                        "root": entry.root,
                        "spaces": [
                            {
                                "name": name,
                                "dense": f"{spec.dense.provider} {spec.dense.model}",
                                "rerank": (
                                    f"{spec.rerank.provider} {spec.rerank.model}"
                                    if spec.rerank is not None and spec.retrieve.rerank
                                    else None
                                ),
                            }
                            for name, spec in configuration.spaces.items()
                        ],
                        "facetKeys": configuration.facet_keys,
                        "hints": configuration.search_hints,
                    }
                )
            return rows

    @mcp.tool(annotations=read_only)
    def search(
        query: Annotated[str, Field(description="A question, or words that describe the code.")],
        project: ProjectArg,
        space: SpaceArg,
        limit: LimitArg = None,
        facets: FacetsArg = None,
        exclude: ExcludeArg = None,
    ) -> dict[str, Any]:
        """Search by meaning: embed the query, retrieve, then rerank when the space has a
        reranker."""
        from bc_rag.query import search_projects

        with _surfaced("search"):
            wanted = normalize_facets(facets, where="facets")
            unwanted = normalize_facets(exclude, where="exclude")
            result = search_projects(
                query,
                project=project,
                space=space,
                limit=limit,
                facets=wanted,
                exclude=unwanted,
            )
            return _search_result(result, project, space, wanted, unwanted)

    @mcp.tool(annotations=read_only)
    def search_sparse(
        query: Annotated[
            str,
            Field(description="An identifier, route, path fragment, or exact error text."),
        ],
        project: ProjectArg,
        space: SpaceArg,
        limit: LimitArg = None,
        facets: FacetsArg = None,
        exclude: ExcludeArg = None,
    ) -> dict[str, Any]:
        """BM25 keyword search. No dense vector and no rerank. Not for questions about
        meaning."""
        from bc_rag.query import search_projects

        with _surfaced("search_sparse"):
            wanted = normalize_facets(facets, where="facets")
            unwanted = normalize_facets(exclude, where="exclude")
            result = search_projects(
                query,
                project=project,
                space=space,
                limit=limit,
                facets=wanted,
                exclude=unwanted,
                sparse=True,
            )
            return _search_result(result, project, space, wanted, unwanted)

    @mcp.tool(annotations=read_only)
    def list_facets(
        project: ProjectArg,
        space: SpaceArg,
        key: Annotated[
            str | None,
            Field(description="One facet key. Omit for every key and a sample."),
        ] = None,
        limit: Annotated[
            int | None,
            Field(ge=1, le=10_000, description="Values to return per key."),
        ] = None,
    ) -> dict[str, Any]:
        """List the facets stored in one space: each key with its meaning, how many
        points carry it, and its values. Pass key to list one key's values."""
        with _surfaced("list_facets"):
            return facets_listing(_entry(project), space, key=key, limit=limit)

    return mcp


def facets_listing(entry, space: str, *, key: str | None, limit: int | None) -> dict[str, Any]:
    """The list_facets answer. Reads Qdrant only; no embedder, no groupsCommand."""
    from bc_rag.config import load_config
    from bc_rag.store import HybridStore

    root = entry.root_path()
    config, _path = load_config(root, groups=False)
    config.space_named(space)
    meanings = {**config.facet_keys, **RESERVED_KEYS}
    store = HybridStore(
        url=config.qdrant_http_url(),
        collection=config.qdrant_collection(root, space),
        read_only=True,
    )
    try:
        stored = store.facet_keys()
        if key is not None:
            if key not in stored:
                known = ", ".join(stored) or "none"
                raise ValueError(
                    f"facet key {key!r} is not stored in space {space!r}. Keys here: {known}"
                )
            cap = limit or LIST_FACETS_KEY_LIMIT
            values = store.facet_values(key, limit=cap + 1)
            return {
                "project": entry.name,
                "space": space,
                "key": key,
                "meaning": meanings.get(key),
                "points": stored[key],
                "values": [{"value": value, "points": count} for value, count in values[:cap]],
                "truncated": len(values) > cap,
            }
        cap = limit or LIST_FACETS_OVERVIEW_LIMIT
        keys = []
        for name, points in stored.items():
            values = store.facet_values(name)
            keys.append(
                {
                    "key": name,
                    "meaning": meanings.get(name),
                    "points": points,
                    "distinct": len(values),
                    "values": [value for value, _count in values[:cap]],
                    "truncated": len(values) > cap,
                }
            )
        return {"project": entry.name, "space": space, "keys": keys}
    finally:
        store.close()


def mcp_tls_files() -> tuple[Path, Path] | None:
    """Certificate and key for the HTTPS listener, or None when either file is missing."""
    from bc_rag.catalog import user_dir

    directory = user_dir()
    cert = directory / MCP_TLS_CERT_FILENAME
    key = directory / MCP_TLS_KEY_FILENAME
    if cert.is_file() and key.is_file():
        return cert, key
    return None


def mcp_transport_security() -> Any:
    """Hosts and origins the shared listener accepts.

    The plain HTTP listener stays on 127.0.0.1. Claude connects to
    https://bc-rag.localhost, so that name has to be allowed too.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    names = ("127.0.0.1", "localhost", "[::1]", MCP_TLS_HOST)
    hosts = [*names, *[f"{name}:*" for name in names]]
    origins = [
        f"{scheme}://{name}{suffix}"
        for scheme in ("http", "https")
        for name in names
        for suffix in ("", ":*")
    ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def _serve_http_and_https(mcp: Any, *, host: str, port: int, cert: Path, key: Path) -> None:
    """One process, two listeners, one session manager.

    uvicorn's MCP helper starts a single socket and cannot take a certificate.
    Two servers on one Starlette app would start that session manager twice,
    and the second start raises. Lifespan runs once around both servers.
    """
    import uvicorn

    http_url = f"http://{host}:{port}{MCP_HTTP_PATH}"
    https_url = f"https://{MCP_TLS_HOST}:{MCP_HTTPS_PORT}{MCP_HTTP_PATH}"
    print(f"bc-rag mcp {http_url}", file=sys.stderr, flush=True)
    print(f"bc-rag mcp {https_url}", file=sys.stderr, flush=True)
    app = mcp.streamable_http_app(
        streamable_http_path=MCP_HTTP_PATH,
        json_response=True,
        stateless_http=True,
        transport_security=mcp_transport_security(),
        host=host,
    )
    log_level = str(mcp.settings.log_level).lower()
    http_server = _SignalQuietServer(
        uvicorn.Config(app, host=host, port=port, log_level=log_level, lifespan="off")
    )
    https_server = _SignalQuietServer(
        uvicorn.Config(
            app,
            host=host,
            port=MCP_HTTPS_PORT,
            log_level=log_level,
            lifespan="off",
            ssl_certfile=str(cert),
            ssl_keyfile=str(key),
        )
    )
    sockets = _bound_loopback_sockets(MCP_HTTPS_PORT)
    try:
        asyncio.run(_serve_both(app, http_server, https_server, sockets))
    finally:
        for sock in sockets:
            sock.close()


class _SignalQuietServer:
    """A uvicorn server that does not install its own signal handlers."""

    def __init__(self, config: Any) -> None:
        import uvicorn

        self._server = uvicorn.Server(config)
        self.should_exit = False

    @property
    def should_exit(self) -> bool:
        return bool(self._server.should_exit)

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._server.should_exit = value

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        with _no_signal_capture(self._server):
            await self._server.serve(sockets=sockets)


@contextlib.contextmanager
def _no_signal_capture(server: Any):
    """Stop one server from replacing the process signal handlers."""

    @contextlib.contextmanager
    def capture_signals():
        yield

    previous = server.capture_signals
    server.capture_signals = capture_signals
    try:
        yield
    finally:
        server.capture_signals = previous


async def _serve_both(
    app: Any,
    http_server: _SignalQuietServer,
    https_server: _SignalQuietServer,
    sockets: list[socket.socket],
) -> None:
    """Run both listeners until SIGINT or SIGTERM."""

    def stop(_signum: int, _frame: Any) -> None:
        http_server.should_exit = True
        https_server.should_exit = True

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    async with app.router.lifespan_context(app):
        async with asyncio.TaskGroup() as group:
            group.create_task(http_server.serve())
            group.create_task(https_server.serve(sockets=sockets))


def _bound_loopback_sockets(port: int) -> list[socket.socket]:
    """Sockets for 127.0.0.1 and ::1.

    `bc-rag.localhost` resolves to both. A client that tries IPv6 first
    never reaches a listener bound only to 127.0.0.1.
    """
    bound: list[socket.socket] = []
    v4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        v4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        v4.bind(("127.0.0.1", port))
    except OSError:
        v4.close()
        raise
    bound.append(v4)
    try:
        v6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return bound
    try:
        v6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        v6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        v6.bind(("::1", port))
    except OSError:
        v6.close()
        return bound
    bound.append(v6)
    return bound


def _hit_dict(hit: Hit) -> dict[str, Any]:
    return {
        "project": hit.project,
        "path": hit.path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "language": hit.language,
        "kind": hit.kind,
        "symbol": hit.symbol,
        "heading_path": hit.heading_path,
        "facets": hit.facets,
        "score": hit.score,
        "text": hit.text,
    }
