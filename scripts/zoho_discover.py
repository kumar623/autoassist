"""Show what the Zoho Bookings MCP server offers. Read-only: it books nothing.

Usage:
    python3 scripts/zoho_discover.py            # every tool and its inputs
    python3 scripts/zoho_discover.py --schemas  # plus each tool's full JSON Schema

Needs ZOHO_MCP_URL in .env. The URL contains the access key - it is never
printed; only its host is.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from services.orchestrator.mcp_client import McpClient, McpError  # noqa: E402
from services.orchestrator.zoho_auth import ZohoAuth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemas", action="store_true", help="print each tool's full input schema")
    args = parser.parse_args()

    url = os.getenv("ZOHO_MCP_URL", "").strip()
    if not url:
        print("ZOHO_MCP_URL is not set. Add it to .env (it is gitignored) - never to the repo.")
        return 1

    try:
        auth = ZohoAuth.from_env(env_file=ROOT / ".env")
        if auth is None:
            print("Not signed in to Zoho yet. Run: python3 scripts/zoho_login.py")
            return 1
        with McpClient(url, auth=auth) as mcp:
            tools = mcp.list_tools()
            print(f"server : {mcp.host}")
            print(f"info   : {mcp.server_info}")
            print(f"tools  : {len(tools)}\n")
            for t in tools:
                schema = t.get("inputSchema") or {}
                props = schema.get("properties") or {}
                required = set(schema.get("required") or [])
                print(f"- {t['name']}")
                if t.get("description"):
                    print(f"    {' '.join(t['description'].split())[:220]}")
                for name, spec in props.items():
                    kind = spec.get("type", "?")
                    mark = "*" if name in required else " "
                    desc = " ".join((spec.get("description") or "").split())[:110]
                    print(f"    {mark} {name:<22} {kind:<8} {desc}")
                if args.schemas:
                    print("    " + json.dumps(schema, indent=2).replace("\n", "\n    "))
                print()
    except McpError as e:
        print(f"MCP error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
