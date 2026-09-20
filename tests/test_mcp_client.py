"""Tests for the plain-HTTPS MCP client. A fake MCP server via httpx.MockTransport."""

import json

import httpx
import pytest

from services.orchestrator import azure_http, mcp_client

SECRET_URL = "https://bookingcar-1.zohomcp.eu/mcp/s3cr3tkey0123456789/message"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(azure_http, "_pause", lambda s: None)


class FakeServer:
    """Speaks just enough MCP: initialize, tools/list (two pages), tools/call."""

    def __init__(self, sse=False, session="sess-1"):
        self.sse = sse
        self.session = session
        self.seen = []

    def reply(self, msg, request):
        asked = json.loads(request.content).get("method")
        headers = {"mcp-session-id": self.session} if asked == "initialize" and self.session else {}
        if self.sse:
            body = f"event: message\ndata: {json.dumps(msg)}\n\n"
            return httpx.Response(200, headers={**headers, "content-type": "text/event-stream"}, text=body)
        return httpx.Response(200, headers=headers, json=msg)

    def __call__(self, request):
        body = json.loads(request.content)
        self.seen.append((body, request.headers))  # httpx.Headers: case-insensitive, like HTTP
        method, rid = body.get("method"), body.get("id")
        if rid is None:  # a notification
            return httpx.Response(202)
        if method == "initialize":
            return self.reply({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2025-03-26", "serverInfo": {"name": "zoho-bookings", "version": "1"},
                "capabilities": {"tools": {}}}}, request)
        if method == "tools/list":
            if not body["params"].get("cursor"):
                return self.reply({"jsonrpc": "2.0", "id": rid, "result": {
                    "tools": [{"name": "getAvailability", "inputSchema": {"type": "object"}}], "nextCursor": "p2"}}, request)
            return self.reply({"jsonrpc": "2.0", "id": rid, "result": {
                "tools": [{"name": "bookAppointment", "inputSchema": {"type": "object"}}]}}, request)
        if method == "tools/call":
            name = body["params"]["name"]
            if name == "fails":
                return self.reply({"jsonrpc": "2.0", "id": rid, "result": {
                    "isError": True, "content": [{"type": "text", "text": "slot not available"}]}}, request)
            if name == "structured":
                return self.reply({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": "{}"}], "structuredContent": {"booking_id": "Z-1"}}}, request)
            return self.reply({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": json.dumps({"echo": body["params"]["arguments"]})}]}}, request)
        return self.reply({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no such method"}}, request)


def client(server):
    return mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(server)))


@pytest.mark.parametrize("sse", [False, True], ids=["json", "event-stream"])
def test_handshake_then_calls_carry_the_session(sse):
    server = FakeServer(sse=sse)
    c = client(server)
    assert c.call_tool("getAvailability", {"service_id": "1"}) == {"echo": {"service_id": "1"}}
    methods = [b.get("method") for b, _ in server.seen]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]
    assert server.seen[0][0]["params"]["clientInfo"]["name"] == "autoassist"
    assert server.seen[2][1]["mcp-session-id"] == "sess-1"
    assert c.server_info == {"protocol": "2025-03-26", "name": "zoho-bookings", "version": "1"}


def test_the_handshake_happens_once():
    server = FakeServer()
    c = client(server)
    c.call_tool("a", {})
    c.call_tool("b", {})
    assert [b.get("method") for b, _ in server.seen].count("initialize") == 1


def test_a_server_without_sessions_works():
    server = FakeServer(session=None)
    c = client(server)
    c.call_tool("a", {})
    assert "mcp-session-id" not in server.seen[-1][1]


def test_tools_are_listed_across_pages():
    assert [t["name"] for t in client(FakeServer()).list_tools()] == ["getAvailability", "bookAppointment"]


def test_a_tool_error_raises_with_the_servers_reason():
    with pytest.raises(mcp_client.McpError, match="slot not available"):
        client(FakeServer()).call_tool("fails", {})


def test_structured_content_is_preferred():
    assert client(FakeServer()).call_tool("structured", {}) == {"booking_id": "Z-1"}


def test_a_json_rpc_error_raises():
    c = client(FakeServer())
    c._ensure_initialized()
    with pytest.raises(mcp_client.McpError, match="no such method"):
        c._rpc("resources/list")


def test_the_key_in_the_url_never_appears_in_an_error():
    """Zoho puts the access key in the URL path. Errors show the host only."""
    def refuse(request):
        return httpx.Response(403, text="forbidden")

    c = mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(mcp_client.McpError) as e:
        c.list_tools()
    assert "s3cr3tkey" not in str(e.value)
    assert "bookingcar-1.zohomcp.eu" in str(e.value)


def test_an_unreachable_server_error_hides_the_key_too():
    def down(request):
        raise httpx.ConnectError(f"cannot connect to {request.url}")

    c = mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(down)))
    with pytest.raises(mcp_client.McpError) as e:
        c.list_tools()
    assert "s3cr3tkey" not in str(e.value)
    assert e.value.__cause__ is None, "the original error, which names the URL, is not chained"


def test_throttling_is_retried():
    server = FakeServer()
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] == 1 else server(request)

    c = mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(flaky)))
    assert c.call_tool("a", {"x": 1}) == {"echo": {"x": 1}}


class FakeAuth:
    def __init__(self):
        self.calls = []

    def token(self, force_refresh=False):
        self.calls.append(force_refresh)
        return f"tok-{len(self.calls)}"


def test_the_sign_in_token_is_sent():
    server = FakeServer()
    auth = FakeAuth()
    c = mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(server)), auth=auth)
    c.call_tool("a", {})
    assert all(h["authorization"].startswith("Bearer tok-") for _, h in server.seen)


def test_a_rejected_token_is_refreshed_once():
    server = FakeServer()
    auth = FakeAuth()
    state = {"first": True}

    def expired_once(request):
        if state["first"]:
            state["first"] = False
            return httpx.Response(401, text="Authentication required")
        return server(request)

    c = mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(expired_once)), auth=auth)
    assert c.call_tool("a", {"x": 1}) == {"echo": {"x": 1}}
    assert True in auth.calls, "a forced refresh happened"


def test_a_token_that_keeps_failing_gives_up():
    auth = FakeAuth()

    def always_401(request):
        return httpx.Response(401, text="Authentication required")

    c = mcp_client.McpClient(SECRET_URL, http_client=httpx.Client(transport=httpx.MockTransport(always_401)), auth=auth)
    with pytest.raises(mcp_client.McpError, match="401"):
        c.list_tools()
    assert auth.calls.count(True) == 1, "refreshes once, not forever"
