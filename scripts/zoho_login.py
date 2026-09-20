"""Connect AutoAssist to Zoho Bookings. Run once; the workshop owner approves in a browser.

Usage:
    python3 scripts/zoho_login.py

What it does:
  1. Finds Zoho's sign-in endpoints from the MCP server URL in .env
  2. Registers AutoAssist as an OAuth client, the first time
  3. Opens Zoho's approval page in your browser - you sign in and approve there
  4. Catches the answer on http://127.0.0.1:8765/callback
  5. Saves ZOHO_MCP_CLIENT_ID and ZOHO_MCP_REFRESH_TOKEN into .env

Tokens go straight into .env and are never printed. Nothing here sees your
Zoho password: you type it into Zoho's own page.
"""

from __future__ import annotations

import http.server
import os
import pathlib
import secrets
import sys
import threading
import urllib.parse
import webbrowser

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
load_dotenv(ENV)
sys.path.insert(0, str(ROOT))

from services.orchestrator import azure_http  # noqa: E402
from services.orchestrator.zoho_auth import (  # noqa: E402
    ZohoAuthError,
    authorize_url,
    discover,
    exchange,
    pkce_pair,
    register_client,
    set_env_var,
)

PORT = 8765
REDIRECT = f"http://127.0.0.1:{PORT}/callback"
WAIT_SECONDS = 600


def wait_for_callback(expected_state: str) -> str:
    """Serve one request on the redirect address and return the code in it."""
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the name http.server calls
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if urllib.parse.urlparse(self.path).path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got and got.get("state") == expected_state
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = ("AutoAssist is connected to Zoho Bookings. You can close this tab."
                   if ok else "Zoho did not approve the connection. You can close this tab and try again.")
            self.wfile.write(f"<html><body style='font-family:sans-serif;padding:40px'><h2>{msg}</h2></body></html>".encode())
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):  # the request line carries the code: keep it off the console
            pass

    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    timer = threading.Timer(WAIT_SECONDS, server.shutdown)
    timer.start()
    server.serve_forever()
    timer.cancel()

    if got.get("error"):
        raise ZohoAuthError(f"Zoho said: {got.get('error_description') or got['error']}")
    if got.get("state") != expected_state:
        raise ZohoAuthError("no approval arrived (timed out, or the state did not match)")
    if "code" not in got:
        raise ZohoAuthError("the reply had no authorization code")
    return got["code"]


def main() -> int:
    url = os.getenv("ZOHO_MCP_URL", "").strip()
    if not url:
        print("ZOHO_MCP_URL is not set in .env")
        return 1

    http = azure_http.new_client()
    try:
        meta = discover(url, http)
        print("found Zoho's sign-in endpoints")

        client_id = os.getenv("ZOHO_MCP_CLIENT_ID", "").strip()
        if not client_id:
            reg = register_client(meta, REDIRECT, http)
            client_id = reg["client_id"]
            set_env_var(ENV, "ZOHO_MCP_CLIENT_ID", client_id)
            if reg.get("client_secret"):
                set_env_var(ENV, "ZOHO_MCP_CLIENT_SECRET", reg["client_secret"])
            print("registered AutoAssist as a Zoho OAuth client (saved to .env)")
        else:
            print("using the client already registered in .env")

        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(24)
        link = authorize_url(meta, client_id, REDIRECT, challenge, state, url)

        # The link is never printed: its `resource` parameter is the MCP server
        # URL, and that URL holds the access key. (It was printed once, on
        # 20 Sep, and the key had to be regenerated.)
        print("\nOpening Zoho in your browser. Sign in there and approve AutoAssist.", flush=True)
        if not webbrowser.open(link):
            link_file = ENV.parent / ".zoho-login-link"
            link_file.write_text(link + "\n")
            link_file.chmod(0o600)
            print(f"No browser opened. The link is in {link_file.name} (readable only by you; delete it after).")
        print(f"waiting up to {WAIT_SECONDS // 60} minutes for your approval...", flush=True)

        code = wait_for_callback(state)
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": url,
        }
        secret = os.getenv("ZOHO_MCP_CLIENT_SECRET", "").strip()
        if secret:
            form["client_secret"] = secret
        tokens = exchange(meta, form, http)
    except ZohoAuthError as e:
        print(f"\nnot connected: {e}")
        return 1

    if not tokens.get("refresh_token"):
        print("\nZoho gave an access token but no refresh token; the app could not stay connected.")
        return 1
    set_env_var(ENV, "ZOHO_MCP_REFRESH_TOKEN", tokens["refresh_token"])
    print(f"\nconnected. Refresh token saved to .env (not shown). "
          f"Access tokens last {int(tokens.get('expires_in') or 0) // 60} minutes; scopes: {tokens.get('scope', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
