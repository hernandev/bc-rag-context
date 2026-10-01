from pathlib import Path

from bc_rag.defaults import MCP_HTTP_HOST, MCP_HTTP_PATH, MCP_HTTP_PORT, MCP_TLS_HOST
from bc_rag.mcp_server import mcp_tls_files, mcp_transport_security, run_mcp


class _FakeServer:
    def __init__(self, name: str, instructions: str) -> None:
        self.name = name
        self.instructions = instructions

    def tool(self):
        def decorate(fn):
            return fn

        return decorate

    def run(self, transport: str = "stdio", **kwargs) -> None:
        self.transport = transport
        self.kwargs = kwargs


def test_run_mcp_stdio_is_the_default(monkeypatch) -> None:
    created: list[_FakeServer] = []

    def factory(*args, **kwargs):
        server = _FakeServer(*args, **kwargs)
        created.append(server)
        return server

    monkeypatch.setattr("mcp.server.MCPServer", factory)
    run_mcp()
    assert created[0].transport == "stdio"
    assert created[0].kwargs == {}


def test_run_mcp_http_listens_on_the_shared_port(monkeypatch) -> None:
    created: list[_FakeServer] = []

    def factory(*args, **kwargs):
        server = _FakeServer(*args, **kwargs)
        created.append(server)
        return server

    monkeypatch.setattr("mcp.server.MCPServer", factory)
    run_mcp(http=True)
    server = created[0]
    assert server.transport == "streamable-http"
    assert server.kwargs["host"] == MCP_HTTP_HOST
    assert server.kwargs["port"] == MCP_HTTP_PORT
    assert server.kwargs["streamable_http_path"] == MCP_HTTP_PATH
    assert server.kwargs["stateless_http"] is True
    assert server.kwargs["json_response"] is True


def test_https_starts_only_when_both_cert_files_exist(
    monkeypatch, isolate_bc_rag_home: Path
) -> None:
    started: list[tuple[Path, Path]] = []

    def fake_serve(_mcp, *, host: str, port: int, cert: Path, key: Path) -> None:
        assert host == MCP_HTTP_HOST
        assert port == MCP_HTTP_PORT
        started.append((cert, key))

    monkeypatch.setattr("bc_rag.mcp_server._serve_http_and_https", fake_serve)
    monkeypatch.setattr("mcp.server.MCPServer", _FakeServer)

    run_mcp(http=True)
    assert started == []
    assert mcp_tls_files() is None

    cert = isolate_bc_rag_home / "mcp.pem"
    cert.write_text("cert")
    run_mcp(http=True)
    assert started == []

    key = isolate_bc_rag_home / "mcp.key"
    key.write_text("key")
    run_mcp(http=True)
    assert started == [(cert, key)]


def test_tls_name_passes_host_and_origin_checks() -> None:
    from mcp.server.transport_security import TransportSecurityMiddleware

    middleware = TransportSecurityMiddleware(mcp_transport_security())
    assert middleware._validate_host(f"{MCP_TLS_HOST}:32324") is True
    assert middleware._validate_host("127.0.0.1:32323") is True
    assert middleware._validate_origin(f"https://{MCP_TLS_HOST}:32324") is True
    assert middleware._validate_origin(None) is True


def test_run_mcp_http_rejects_a_bad_port(monkeypatch) -> None:
    monkeypatch.setattr("mcp.server.MCPServer", _FakeServer)
    try:
        run_mcp(http=True, port=0)
    except ValueError as exc:
        assert "port" in str(exc)
    else:
        raise AssertionError("expected ValueError")
