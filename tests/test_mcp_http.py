from bc_rag.defaults import MCP_HTTP_HOST, MCP_HTTP_PATH, MCP_HTTP_PORT
from bc_rag.mcp_server import run_mcp


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


def test_run_mcp_http_rejects_a_bad_port(monkeypatch) -> None:
    monkeypatch.setattr("mcp.server.MCPServer", _FakeServer)
    try:
        run_mcp(http=True, port=0)
    except ValueError as exc:
        assert "port" in str(exc)
    else:
        raise AssertionError("expected ValueError")
