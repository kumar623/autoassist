"""The functions the agents can call, and their schemas.

Each entry has two halves:
  SCHEMAS  - what the model sees: name, description, parameters. This is the
             only thing that tells the model when to call it, so the wording
             matters as much as any prompt.
  HANDLERS - what actually runs, here in our process.

The agent never touches Azure AI Search directly. It asks for a search; we run
it with our own retrieval code (relevance floor, source cap) and hand back
text. That is what lets us guarantee what reaches the model.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from . import booking, retrieval

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- handlers


def search_service_docs(query: str, doc_type: str | None = None) -> str:
    result = retrieval.search(query, doc_type=doc_type)
    return retrieval.format_for_agent(result)


def get_available_slots(date: str | None = None) -> str:
    return json.dumps(booking.get_slots(on_date=date), indent=2)


def book_service_slot(slot_id: str, registration: str, issue: str = "") -> str:
    return json.dumps(booking.book_slot(slot_id, registration, issue), indent=2)


def move_service_booking(reference: str, new_slot_id: str) -> str:
    return json.dumps(booking.move_booking(reference, new_slot_id), indent=2)


def cancel_service_booking(reference: str) -> str:
    return json.dumps(booking.cancel_booking(reference), indent=2)


def look_up_booking(reference: str) -> str:
    return json.dumps(booking.get_booking(reference), indent=2)


def raise_ticket(summary: str, urgency: str = "normal", registration: str = "") -> str:
    return json.dumps(booking.create_ticket(summary, urgency, registration), indent=2)


HANDLERS: dict[str, Callable[..., str]] = {
    "search_service_docs": search_service_docs,
    "get_available_slots": get_available_slots,
    "book_service_slot": book_service_slot,
    "move_service_booking": move_service_booking,
    "cancel_service_booking": cancel_service_booking,
    "look_up_booking": look_up_booking,
    "raise_ticket": raise_ticket,
}


# ---------------------------------------------------------------- schemas


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


SCHEMAS: dict[str, dict] = {
    "search_service_docs": _tool(
        "search_service_docs",
        "Search the workshop's service library: fault code definitions, maintenance "
        "schedules and service bulletins. Call this before answering ANY question "
        "about a vehicle, including safety questions and questions you think you "
        "already know the answer to. Returns the relevant documents, or a clear "
        "statement that the library does not cover the topic.",
        {
            "query": {
                "type": "string",
                "description": "What to look for, in the customer's own words plus any "
                "fault code they gave. E.g. 'P0420 catalytic converter' or "
                "'spongy brake pedal'.",
            },
            "doc_type": {
                "type": "string",
                "enum": ["dtc", "maintenance", "bulletin"],
                "description": "Optional filter. 'dtc' for fault codes, 'maintenance' for "
                "service intervals, 'bulletin' for known issues. Leave it out to "
                "search everything, which is usually right.",
            },
        },
        ["query"],
    ),
    "get_available_slots": _tool(
        "get_available_slots",
        "List free service slots. Call this before offering the customer any time. "
        "Never state availability you have not looked up.",
        {
            "date": {
                "type": "string",
                "description": "Optional. 'YYYY-MM-DD', a weekday name like 'saturday', "
                "or 'tomorrow'. Leave it out for the next few open days.",
            }
        },
        [],
    ),
    "book_service_slot": _tool(
        "book_service_slot",
        "Reserve a slot. This is the ONLY way a booking is made. Never tell a "
        "customer they are booked unless this returned ok=true, and always give "
        "them the exact reference it returned - never invent one.",
        {
            "slot_id": {
                "type": "string",
                "description": "The slot_id from get_available_slots, exactly as given.",
            },
            "registration": {
                "type": "string",
                "description": "The vehicle registration number. Ask for it if you do not have it.",
            },
            "issue": {
                "type": "string",
                "description": "Short description of what needs looking at.",
            },
        },
        ["slot_id", "registration"],
    ),
    "move_service_booking": _tool(
        "move_service_booking",
        "Move an existing booking to a different free slot, in one step. This is the "
        "ONLY way to change a booking's time - never cancel and rebook. The reference "
        "stays the same. If the new slot is not free, nothing changes and the booking "
        "stays where it was.",
        {
            "reference": {"type": "string", "description": "The existing booking reference, e.g. AA-4K2P9X."},
            "new_slot_id": {
                "type": "string",
                "description": "The slot_id from get_available_slots, exactly as given.",
            },
        },
        ["reference", "new_slot_id"],
    ),
    "cancel_service_booking": _tool(
        "cancel_service_booking",
        "Cancel an existing booking using its reference. Only when the customer asks "
        "to cancel. To change the time, use move_service_booking instead.",
        {"reference": {"type": "string", "description": "Booking reference, e.g. AA-4K2P9X."}},
        ["reference"],
    ),
    "look_up_booking": _tool(
        "look_up_booking",
        "Look up a booking by its reference to confirm the details.",
        {"reference": {"type": "string", "description": "Booking reference, e.g. AA-4K2P9X."}},
        ["reference"],
    ),
    "raise_ticket": _tool(
        "raise_ticket",
        "Raise a ticket for a human service advisor. Use this when the question "
        "involves a safety system, when the library does not cover something "
        "important, or when the customer is unhappy.",
        {
            "summary": {
                "type": "string",
                "description": "What the customer needs, in one or two plain sentences. "
                "Include the symptom and anything they told you about the vehicle.",
            },
            "urgency": {
                "type": "string",
                "enum": ["normal", "urgent", "safety"],
                "description": "'safety' for brakes, steering, airbags, fuel leaks, smoke "
                "or fire - these get a callback within the hour.",
            },
            "registration": {"type": "string", "description": "Vehicle registration if known."},
        },
        ["summary"],
    ),
}


def schemas_for(names: list[str]) -> list[dict]:
    missing = [n for n in names if n not in SCHEMAS]
    if missing:
        raise KeyError(f"Unknown tool(s): {', '.join(missing)}. Known: {', '.join(SCHEMAS)}")
    return [SCHEMAS[n] for n in names]


def execute(name: str, arguments: str | dict) -> str:
    """Run a tool call and always return a string.

    Never raises. A tool that blows up must hand the model a readable error so
    it can tell the customer something honest, rather than killing the run.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        return f"ERROR: no such tool '{name}'."

    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError as e:
        return f"ERROR: could not read the arguments for {name}: {e}"

    if not isinstance(args, dict):
        return f"ERROR: arguments for {name} must be an object, got {type(args).__name__}."

    try:
        return handler(**args)
    except TypeError as e:
        return f"ERROR: wrong arguments for {name}: {e}"
    except Exception as e:  # noqa: BLE001 - deliberate: never kill a run
        log.exception("tool %s failed", name)
        return f"ERROR: {name} failed: {type(e).__name__}: {e}"
