"""Who the agents are and what each one is allowed to touch.

The page shows this beside the chat, so that what the system is doing is
visible while it does it rather than only afterwards in the trace. It is read
from the same files `agents/deploy_agents.py` deploys from, and from the same
tool schemas the agents are actually given - not written out again here, because
a hand-copied panel is a panel that quietly goes out of date and then tells the
interviewer something that is not true.

`raise_ticket` appearing under diagnostics is not a mistake: it can hand over
when the library does not cover something important.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

from . import tools
from .cache import TimedCache

log = logging.getLogger(__name__)

DEFINITIONS = pathlib.Path(__file__).resolve().parents[2] / "agents" / "definitions"

# The order they run in, which is also the order they are shown in.
ORDER = ["triage", "diagnostics", "booking", "escalation"]

# What each one is for, in a sentence a customer could read. The definitions'
# own `description` is written for the Foundry console and says roughly this;
# these are shorter, because the panel is narrow.
PURPOSE = {
    "triage": "Reads the message and decides who should answer it.",
    "diagnostics": "Explains fault codes and warning lights, from the service documents.",
    "booking": "Finds free slots and books, moves or cancels an appointment.",
    "escalation": "Hands over to a human service advisor, with a written summary.",
}

# Why it is on the route, in the orchestrator's terms rather than the agent's.
ROLE = {
    "triage": "Runs first, on its own. Returns JSON, never prose.",
    "diagnostics": "Runs in parallel with booking when both are needed.",
    "booking": "Runs in parallel with diagnostics when both are needed.",
    "escalation": "Runs last, after the others, so it can summarise what was said.",
}

# Where a tool call actually goes. Worth showing beside the tool name: "booking
# calls get_available_slots" says nothing about the fact that the call leaves
# this process, speaks MCP to Zoho's server, and lands in the workshop's real
# calendar. The booking row is read at call time, not hard-coded, because
# BOOKING_BACKEND switches it between Zoho and a local file.
SEARCH_BACKEND = "Azure AI Search"
LOCAL_BACKEND = "local store"


def _backend_of(tool: str) -> str:
    if tool == "search_service_docs":
        return SEARCH_BACKEND
    if tool == "raise_ticket":
        return LOCAL_BACKEND
    return "Zoho Bookings · MCP" if os.getenv("BOOKING_BACKEND", "file") == "zoho" else LOCAL_BACKEND


_ROSTER = TimedCache(600, name="roster")


def _first_sentence(text: str) -> str:
    head = text.split(". ")[0].strip().rstrip(".")
    return head + "." if head else ""


def _read(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        log.warning("could not read agent definition %s: %s", path.name, e)
        return None


def triage_backend() -> str:
    """Which classifier is deciding the route, for the panel to say out loud."""
    return "jev" if os.getenv("TRIAGE_BACKEND", "agent").strip().lower() == "jev" else "agent"


def load(deployed: dict | None = None) -> dict:
    """The four agents, their tools, and whether each is actually deployed.

    `deployed` is app.STATE["agent_ids"]. An agent in the repo but not in
    Foundry is shown as not deployed rather than hidden: that difference is
    worth seeing, and it is exactly what /ready complains about.
    """
    agents = _ROSTER.get_or_call("agents", _load_definitions)
    live = deployed or {}
    return {
        "agents": [
            {**a, "deployed": a["name"] in live,
             "tools": [{**t, "backend": _backend_of(t["name"])} for t in a["tools"]]}
            for a in agents
        ],
        "count": len(agents),
        "triage_backend": triage_backend(),
    }


def _load_definitions() -> list[dict]:
    found = []
    for name in ORDER:
        d = _read(DEFINITIONS / f"{name}.json")
        if d is None:
            continue
        found.append({
            "name": name,
            "model": d.get("model", ""),
            "temperature": d.get("temperature"),
            "purpose": PURPOSE.get(name) or _first_sentence(d.get("description", "")),
            "role": ROLE.get(name, ""),
            "returns": "JSON" if d.get("response_format") == "json_object" else "an answer",
            "tools": [
                {"name": t, "does": _first_sentence(_tool_description(t)), "backend": _backend_of(t)}
                for t in d.get("tools") or []
            ],
        })
    if not found:
        log.warning("no agent definitions found under %s", DEFINITIONS)
    return found


def _tool_description(name: str) -> str:
    schema = tools.SCHEMAS.get(name)
    if not schema:
        return ""
    return (schema.get("function") or {}).get("description", "")
