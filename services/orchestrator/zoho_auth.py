"""Sign-in for the Zoho Bookings MCP server (OAuth 2.1, as the MCP spec requires).

The server URL identifies the server; every request must also carry a Bearer
token. Zoho advertises the flow at the standard discovery addresses:

    /.well-known/oauth-protected-resource     which authorization server, which scopes
    /.well-known/oauth-authorization-server   endpoints; grants: authorization_code
                                              + refresh_token; PKCE S256; public
                                              clients; dynamic client registration

One-time setup (scripts/zoho_login.py): AutoAssist registers itself as a client,
the workshop owner approves it in their browser, and the refresh token is saved
to .env. From then on ZohoAuth trades the refresh token for short-lived access
tokens as needed. Nobody's password passes through the app.

Tokens are secrets. Nothing here prints or logs one.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import pathlib
import secrets
import threading
import time
import urllib.parse

import httpx

from . import azure_http

log = logging.getLogger(__name__)

SCOPES = "ZohoBookings.data.CREATE ZohoMCP.tool.execute"
REFRESH_MARGIN_SECONDS = 120


class ZohoAuthError(Exception):
    pass


def discover(mcp_url: str, http: httpx.Client | None = None) -> dict:
    """The authorization server's metadata, found the way the MCP spec says."""
    http = http or azure_http.new_client()
    host = httpx.URL(mcp_url).host
    resource = http.get(f"https://{host}/.well-known/oauth-protected-resource").json()
    issuer = (resource.get("authorization_servers") or [f"https://{host}"])[0].rstrip("/")
    meta = http.get(f"{issuer}/.well-known/oauth-authorization-server").json()
    for needed in ("authorization_endpoint", "token_endpoint"):
        if needed not in meta:
            raise ZohoAuthError(f"authorization server metadata has no {needed}")
    return meta


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def authorize_url(meta: dict, client_id: str, redirect_uri: str, challenge: str, state: str, resource: str) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": resource,  # RFC 8707: the token is for this MCP server only
    }
    return f"{meta['authorization_endpoint']}?{urllib.parse.urlencode(params)}"


def register_client(meta: dict, redirect_uri: str, http: httpx.Client) -> dict:
    """Dynamic client registration (RFC 7591): AutoAssist introduces itself."""
    if not meta.get("registration_endpoint"):
        raise ZohoAuthError("this server does not offer dynamic client registration")
    r = http.post(meta["registration_endpoint"], json={
        "client_name": "AutoAssist",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": SCOPES,
    })
    if r.status_code >= 400:
        raise ZohoAuthError(f"client registration failed: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def exchange(meta: dict, form: dict, http: httpx.Client) -> dict:
    """POST to the token endpoint. Errors never include the form: it holds secrets."""
    r = http.post(meta["token_endpoint"], data=form, headers={"Accept": "application/json"})
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code >= 400 or "access_token" not in body:
        reason = body.get("error_description") or body.get("error") or f"HTTP {r.status_code}"
        raise ZohoAuthError(f"token request ({form.get('grant_type')}) failed: {reason}")
    return body


class ZohoAuth:
    """Hands out a valid access token, refreshing it from the refresh token.

    If Zoho rotates the refresh token, the new one is kept in memory and saved
    back to the .env it came from, when there is one (on a laptop). A deployed
    container keeps it for its lifetime; see docs/decisions/008.
    """

    def __init__(self, mcp_url: str, client_id: str, refresh_token: str, client_secret: str = "",
                 env_file: pathlib.Path | None = None, http: httpx.Client | None = None):
        self._mcp_url = mcp_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._env_file = env_file
        self._http = http or azure_http.new_client()
        self._meta: dict | None = None
        self._access: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env_file: pathlib.Path | None = None) -> ZohoAuth | None:
        url = os.getenv("ZOHO_MCP_URL", "").strip()
        client_id = os.getenv("ZOHO_MCP_CLIENT_ID", "").strip()
        refresh = os.getenv("ZOHO_MCP_REFRESH_TOKEN", "").strip()
        if not (url and client_id and refresh):
            return None
        return cls(url, client_id, refresh, os.getenv("ZOHO_MCP_CLIENT_SECRET", "").strip(), env_file)

    def token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if force_refresh or not self._access or time.time() > self._expires_at - REFRESH_MARGIN_SECONDS:
                self._refresh()
            return self._access  # type: ignore[return-value]

    def _refresh(self) -> None:
        if self._meta is None:
            self._meta = discover(self._mcp_url, self._http)
        form = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self._client_id,
            "resource": self._mcp_url,
        }
        if self._client_secret:
            form["client_secret"] = self._client_secret
        body = exchange(self._meta, form, self._http)
        self._access = body["access_token"]
        self._expires_at = time.time() + float(body.get("expires_in") or 3600)
        rotated = body.get("refresh_token")
        if rotated and rotated != self._refresh_token:
            self._refresh_token = rotated
            log.warning("Zoho rotated the refresh token")
            if self._env_file:
                set_env_var(self._env_file, "ZOHO_MCP_REFRESH_TOKEN", rotated)


def set_env_var(path: pathlib.Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file, replacing an existing line or appending one."""
    lines = path.read_text().splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n")
