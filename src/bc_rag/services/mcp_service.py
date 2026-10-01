"""The mcp service: the one MCP listener every chat shares.

`http://127.0.0.1:32323/mcp`, plus `https://bc-rag.localhost:32324/mcp` when
`~/.bc-rag/mcp.pem` and `~/.bc-rag/mcp.key` exist. A taken port exits 3 with a
message, so the supervisor waits instead of restarting in a tight loop.
"""

from __future__ import annotations

import socket

from bc_rag.defaults import MCP_HTTP_HOST, MCP_HTTP_PORT, MCP_HTTPS_PORT

EXIT_PORT_IN_USE = 3


def port_in_use(port: int, hosts: tuple[str, ...] = ("127.0.0.1", "::1")) -> str | None:
    """`host:port` of the first loopback address that already accepts connections."""
    for host in hosts:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                if probe.connect_ex((host, port)) == 0:
                    return f"{host}:{port}" if family == socket.AF_INET else f"[{host}]:{port}"
        except OSError:
            continue
    return None


def busy_port(port: int = MCP_HTTP_PORT) -> str | None:
    """The first of the service's ports that something already listens on."""
    from bc_rag.mcp_server import mcp_tls_files

    ports = [port]
    if mcp_tls_files() is not None:
        ports.append(MCP_HTTPS_PORT)
    for candidate in ports:
        found = port_in_use(candidate)
        if found is not None:
            return found
    return None


def run_mcp_service(host: str = MCP_HTTP_HOST, port: int = MCP_HTTP_PORT) -> int:
    """Serve until SIGTERM or Ctrl+C. Returns 3 when a port is taken."""
    from bc_rag.mcp_server import run_mcp

    taken = busy_port(port)
    if taken is not None:
        print(
            f"mcp service already runs on {taken}; run `bc-rag services stop mcp` first",
            flush=True,
        )
        return EXIT_PORT_IN_USE
    run_mcp(host=host, port=port)
    return 0
