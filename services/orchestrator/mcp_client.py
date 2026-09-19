"""A minimal MCP client over plain HTTPS, for vendor MCP servers such as Zoho Bookings.

MCP (Model Context Protocol) is JSON-RPC 2.0 over HTTP. A client needs three
calls, and this is all of them:

    initialize        handshake; the server may hand back an Mcp-Session-Id
                      header, which every later request must carry
    tools/list        what the server offers, with a JSON Schema per tool
    tools/call        run one tool: {"name": ..., "arguments": {...}}

The server may answer a POST with plain JSON or with a short Server-Sent
Events stream ("data: {...}" lines); both are handled.

No MCP SDK, for the reasons in docs/decisions/007: it is a handful of JSON
requests, and the official Python SDK needs Python 3.10+.

THE URL IS A SECRET. Vendor MCP servers such as Zoho's put the access key in
the URL path. Anyone holding it can book and cancel appointments. It comes from
the environment, and it never appears in a log line or an exception here -
only the host does.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading

import httpx

from . import azure_http

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"  # the newest we speak; the server answers with its own
CLIENT_INFO = {"name": "autoassist", "version": "1.0"}


class McpError(Exception):
    """The server refused a request, or a tool reported an error."""


class McpClient:
    """One connection to one MCP server. Safe to share between threads."""

    def __init__(self, url: str, http_client: httpx.Client | None = None, auth=None):
        """auth: optional object with token(force_refresh=False) -> str, for
        servers that require OAuth on top of the URL (Zoho does)."""
        self._url = url
        self._auth = auth
        self.host = httpx.URL(url).host  # the only part of the URL that is safe to show
        self._http = http_client or azure_http.new_client()
        self._ids = itertools.count(1)
        self._session: str | None = None
        self._ready = False
        self._lock = threading.Lock()
        self.server_info: dict = {}

    # ------------------------------------------------------------ protocol

    def _post(self, payload: dict) -> httpx.Response:
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        if self._auth:
            headers["Authorization"] = f"Bearer {self._auth.token()}"
        refreshed = False
        for attempt in range(azure_http.MAX_RETRIES + 1):
            try:
                r = self._http.post(self._url, headers=headers, content=json.dumps(payload))
            except httpx.TransportError as e:
                if attempt == azure_http.MAX_RETRIES:
                    raise McpError(f"could not reach the MCP server at {self.host}: {type(e).__name__}") from None
                azure_http._pause(azure_http._backoff(attempt, None))
                continue
            if r.status_code == 401 and self._auth and not refreshed:
                # The token was revoked or expired early: get a fresh one, once.
                headers["Authorization"] = f"Bearer {self._auth.token(force_refresh=True)}"
                refreshed = True
                continue
            if r.status_code in azure_http.RETRY_STATUSES and attempt < azure_http.MAX_RETRIES:
                azure_http._pause(azure_http._backoff(attempt, r.headers.get("retry-after")))
                continue
            if r.status_code >= 400:
                # Deliberately not the URL: it holds the key.
                raise McpError(f"MCP server at {self.host} answered HTTP {r.status_code}: {r.text[:200]}")
            return r
        raise AssertionError("unreachable")

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        request_id = next(self._ids)
        r = self._post({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        message = _response_for(r, request_id)
        if "error" in message:
            err = message["error"]
            raise McpError(f"{method} failed: {err.get('code')} {err.get('message')}")
        return message.get("result") or {}

    def _notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method})

    def _ensure_initialized(self) -> None:
        with self._lock:
            if self._ready:
                return
            r = self._post({
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "initialize",
                "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO},
            })
            self._session = r.headers.get("mcp-session-id") or self._session
            result = _response_for(r, None).get("result") or {}
            self.server_info = {
                "protocol": result.get("protocolVersion"),
                **(result.get("serverInfo") or {}),
            }
            self._notify("notifications/initialized")
            self._ready = True
            log.info("MCP connected to %s: %s", self.host, self.server_info)

    # ------------------------------------------------------------ what callers use

    def list_tools(self) -> list[dict]:
        """Every tool, following pagination cursors."""
        self._ensure_initialized()
        tools: list[dict] = []
        cursor = None
        while True:
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> dict | list | str:
        """Run a tool. Returns its structured result if it gave one, else its
        text (parsed as JSON when it is JSON). Raises McpError if the tool
        reported an error."""
        self._ensure_initialized()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text").strip()
        if result.get("isError"):
            raise McpError(f"{name}: {text[:500] or 'the tool reported an error'}")
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        try:
            return json.loads(text)
        except ValueError:
            return text

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> McpClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _response_for(r: httpx.Response, request_id) -> dict:
    """The JSON-RPC response in an HTTP reply, whether sent as JSON or as SSE."""
    if "text/event-stream" in r.headers.get("content-type", ""):
        found = None
        for line in r.text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                msg = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                if request_id is None or msg.get("id") == request_id:
                    found = msg
        if found is None:
            raise McpError("the MCP server's event stream held no response")
        return found
    if not r.content:
        return {}
    return r.json()
