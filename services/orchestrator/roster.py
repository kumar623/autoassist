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
import pathlib

from . import jev_triage, tools, zoho_bookings
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

# With TRIAGE_BACKEND=jev the triage agent is the fallback, not the first step,
# and the panel on the live page should not say otherwise.
TRIAGE_ROLE_WITH_JEV = ("Jev decides first, with four probabilities; this agent answers only "
                        "when Jev cannot. Returns JSON, never prose.")

# Where a tool call actually goes. Worth showing beside the tool name: "booking
# calls get_available_slots" says nothing about the fact that the call leaves
# this process, speaks MCP to Zoho's server, and lands in the workshop's real
# calendar. The booking row asks tools.BACKEND - where bookings actually go -
# rather than reading BOOKING_BACKEND a second time: two readings of one setting
# can disagree ("Zoho" was lower-cased by one and not the other), and then the
# panel says local while bookings land in Zoho.
SEARCH_BACKEND = "Azure AI Search"
LOCAL_BACKEND = "local store"
ZOHO_BACKEND = "Zoho Bookings · MCP"


def _backend_of(tool: str) -> str:
    if tool == "search_service_docs":
        return SEARCH_BACKEND
    if tool == "raise_ticket":
        return LOCAL_BACKEND
    return ZOHO_BACKEND if tools.BACKEND is zoho_bookings else LOCAL_BACKEND


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
    """Which classifier is deciding the route, for the panel to say out loud.

    Asks jev_triage rather than reading TRIAGE_BACKEND again: switched on with
    no key, Jev is never called, the agent answers every message, and a
    /metrics that still said "jev" would be the silent fallback STATS exists
    to expose.
    """
    return "jev" if jev_triage.configured() else "agent"


def load(deployed: dict | None = None) -> dict:
    """The four agents, their tools, and whether each is actually deployed.

    `deployed` is app.STATE["agent_ids"]. An agent in the repo but not in
    Foundry is shown as not deployed rather than hidden: that difference is
    worth seeing, and it is exactly what /ready complains about.
    """
    agents = _ROSTER.get_or_call("agents", _load_definitions)
    live = deployed or {}
    backend = triage_backend()
    return {
        "agents": [
            {**a, "deployed": a["name"] in live,
             **({"role": TRIAGE_ROLE_WITH_JEV} if a["name"] == "triage" and backend == "jev" else {}),
             "tools": [{**t, "backend": _backend_of(t["name"])} for t in a["tools"]]}
            for a in agents
        ],
        "triage_backend": backend,
        # Whether the page's Jev toggle can do anything on this server. Picking
        # it without a key still works - the agent answers and says why - but
        # the page can say so before anyone clicks.
        "jev_available": jev_triage.available(),
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
                # No backend here: it is added per request by load(), because
                # these definitions are cached and the backend is not.
                {"name": t, "does": _first_sentence(_tool_description(t))}
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
