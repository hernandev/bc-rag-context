"""One MCP server for every cataloged project.

`bc-rag mcp` speaks over stdin for a single chat.
`bc-rag mcp --http` listens on 127.0.0.1 so every chat shares that process.
When `~/.bc-rag/mcp.pem` and `~/.bc-rag/mcp.key` both exist, the same process
also listens on `https://bc-rag.localhost:32324/mcp`.
`search` and `search_sparse` require `project` and `space`.
`space` is required. The names are the keys of `spaces`.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Any

from bc_rag.catalog import living_projects
from bc_rag.defaults import (
    MCP_HTTP_HOST,
    MCP_HTTP_PATH,
    MCP_HTTP_PORT,
    MCP_HTTPS_PORT,
    MCP_TLS_CERT_FILENAME,
    MCP_TLS_HOST,
    MCP_TLS_KEY_FILENAME,
)
from bc_rag.query import search_projects
from bc_rag.store import Hit


def run_mcp(
    *,
    http: bool = False,
    host: str = MCP_HTTP_HOST,
    port: int = MCP_HTTP_PORT,
) -> None:
    from mcp.server import MCPServer

    mcp = MCPServer(
        "bc-rag",
        instructions=(
            "Search local source and docs indexed by bc-rag-context. "
            "Call list_projects first, then list_tags for the space you will search. "
            "search and search_sparse require project and space. "
            "search is dense then rerank. search_sparse is BM25 only. "
            "Pass tags only with values list_tags returned. "
            "The tags list is or of alternatives. A nested list is and. "
            "Prefix a tag with - to exclude that value."
        ),
    )

    @mcp.tool()
    def list_projects() -> list[dict[str, Any]]:
        """List indexed projects and the space names search will accept."""
        from pathlib import Path

        from bc_rag.config import load_config

        rows: list[dict[str, Any]] = []
        for entry in living_projects():
            configuration, _path = load_config(Path(entry.root))
            spaces = list(configuration.spaces)
            rows.append({"name": entry.name, "root": entry.root, "spaces": spaces})
        return rows

    @mcp.tool()
    def search(
        query: str,
        project: str,
        space: str,
        limit: int = 8,
        groups: list[str] | None = None,
        tags: list | None = None,
    ) -> dict[str, Any]:
        """Dense search, then rerank. Does not use BM25.

        Call list_tags for this project and space before you invent a tag value.

        tags is one list of alternatives. The hit may match any alternative.
        A string is one alternative. A nested list is one alternative whose tags must all match.
        A tag prefixed with - excludes that value. Put the minus on that tag, inside the alternative it belongs to.

        One tag. Only vendor Olo:
        ["vendor:olo"]

        Or. Vendor Olo or vendor Yext:
        ["vendor:olo", "vendor:yext"]

        And. Engine area and the reference system, both required:
        [["area:engine", "system:bigcolony-reference"]]

        Or of an and. Vendor Olo, or engine and the reference system:
        ["vendor:olo", ["area:engine", "system:bigcolony-reference"]]

        Exclude. Internal, and not vendor Olo:
        [["scope:internal", "-vendor:olo"]]

        groups is a separate list. A hit may be in any of those group names.
        ["docs-providers-_openapi-olo", "libs-engine-providers-engine-providers-olo"]

        Args:
            query: Natural language question or an identifier.
            project: Catalog name or absolute root. From list_projects.
            space: Vector space. Required. From list_projects.
            limit: Maximum hits to return.
            groups: Group names. Any of them may match.
            tags: Alternatives as a list of strings or nested lists. See the examples above.
        """
        result = search_projects(
            query,
            project=project,
            space=space,
            limit=limit,
            use_rerank=True,
            groups=groups,
            tags=tags,
        )
        return {
            "query": result.query,
            "project": project,
            "space": space,
            "groups": groups,
            "tags": tags,
            "hits": [_hit_dict(hit) for hit in result.hits],
        }

    @mcp.tool()
    def search_sparse(
        query: str,
        project: str,
        space: str,
        limit: int = 8,
        groups: list[str] | None = None,
        tags: list | None = None,
    ) -> dict[str, Any]:
        """BM25 keyword search only. No dense vector and no rerank.

        Use for an identifier, a path fragment, or an exact error string.
        Do not use for a question about what the code means.

        tags works the same way as on search.
        The list is or of alternatives. A nested list is and. A leading - excludes that value.

        One tag:
        ["vendor:olo"]

        Or:
        ["vendor:olo", "vendor:yext"]

        And:
        [["area:engine", "system:bigcolony-reference"]]

        Or of an and:
        ["vendor:olo", ["area:engine", "system:bigcolony-reference"]]

        Exclude:
        [["scope:internal", "-vendor:olo"]]

        Args:
            query: Tokens to match.
            project: Catalog name or absolute root. From list_projects.
            space: Vector space. Required. From list_projects.
            limit: Maximum hits to return.
            groups: Group names. Any of them may match.
            tags: Alternatives as a list of strings or nested lists. See the examples above.
        """
        result = search_projects(
            query,
            project=project,
            space=space,
            limit=limit,
            use_rerank=False,
            groups=groups,
            tags=tags,
            sparse=True,
        )
        return {
            "query": result.query,
            "project": project,
            "space": space,
            "groups": groups,
            "tags": tags,
            "hits": [_hit_dict(hit) for hit in result.hits],
        }

    @mcp.tool()
    def list_tags(project: str | None = None, space: str | None = None) -> dict[str, Any]:
        """List tags stored on indexed points in one space, with point counts.

        Call this before search when you need a tag. Copy a tag string into search unchanged.

        Each tag is already key:value, for example area:engine or vendor:olo.
        Do not split it. Pass that same string in the search tags list.

        Args:
            project: Catalog name or absolute root. Omit to use the first catalog entry.
            space: Vector space. Omit to use defaultSpace. prose and code are different collections.
        """
        from pathlib import Path

        from bc_rag.catalog import find_project
        from bc_rag.runtime import open_session

        entries = living_projects()
        if project:
            found = find_project(project, entries)
            if found is None:
                raise ValueError(f"unknown project: {project}")
            entries = [found]
        if not entries:
            return {"tags": []}
        entry = entries[0]
        session = open_session(Path(entry.root), need_reranker=False, write=False, space=space)
        try:
            tags = session.store.facet_values("tags")
        finally:
            session.close()
        return {
            "project": entry.name,
            "space": session.space,
            "tags": [{"tag": name, "points": count} for name, count in tags],
        }

    if http:
        if not host or host.strip() != host or any(ch.isspace() for ch in host):
            raise ValueError(f"invalid mcp host: {host!r}")
        if port < 1 or port > 65535:
            raise ValueError(f"invalid mcp port: {port}")
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
        return
    mcp.run(transport="stdio")


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
        "group": (hit.payload or {}).get("group"),
        "tags": (hit.payload or {}).get("tags") or [],
        "score": hit.score,
        "text": hit.text,
    }
