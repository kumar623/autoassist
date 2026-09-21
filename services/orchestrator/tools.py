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
import os
from typing import Callable

from . import booking, retrieval, zoho_bookings

# Which calendar bookings live in. "zoho" puts them in the workshop's real Zoho
# Bookings calendar, which also emails the customer; "file" is the JSON file in
# the container, which is lost on every deploy. Both offer the same functions,
# so every rule above them - the registration check, one booking per vehicle per
# day, safe moves - works either way.
BACKEND = zoho_bookings if os.getenv("BOOKING_BACKEND", "file").lower() == "zoho" else booking

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- handlers


def search_service_docs(query: str, doc_type: str | None = None) -> str:
    result = retrieval.search(query, doc_type=doc_type)
    return retrieval.format_for_agent(result)


# How many times to put in front of a customer at once. The service is an hour
# long on a fifteen-minute grid, so a free day is thirty-odd slots, and the
# agent decides which of them to read out. On the live app on 20 Sep it read out
# the first four: "9:00, 9:15, 9:30 and 9:45" - four times inside one hour, for
# a job that takes an hour. An earlier run of the same question offered 09:00,
# 10:30, 12:00, 14:00, 15:30 and 17:00, which is a useful answer. Which one the
# customer got was luck.
#
# So the spread is chosen here, in code, and handed over as `offer_these`. The
# full list stays, because the customer may name any time in it and booking
# needs that slot's id.
OFFER_COUNT = 6


def spread(slots: list, count: int = OFFER_COUNT) -> list:
    """`count` slots spaced evenly across the list, keeping the first and last."""
    if len(slots) <= count:
        return list(slots)
    step = (len(slots) - 1) / (count - 1)
    return [slots[round(i * step)] for i in range(count)]


def get_available_slots(date: str | None = None) -> str:
    result = BACKEND.get_slots(on_date=date)
    slots = result.get("slots") or []
    if len(slots) > OFFER_COUNT:
        result["offer_these"] = spread(slots)
        # Said in the tool's own output rather than in the agent's prompt. A
        # tool's output is a second prompt and it wins - that is finding 7, and
        # here it is being used deliberately rather than by accident.
        result["how_to_offer"] = (
            f"Read out the {OFFER_COUNT} times in offer_these, not the first few in slots: they are "
            "spread across the day, and a run of quarter-hour starts is no use for a job that takes "
            "an hour. If the customer names a different time, look it up in slots and book that."
        )
    return json.dumps(result, indent=2)


def book_service_slot(slot_id: str, registration: str, issue: str = "", customer_name: str = "",
                      customer_email: str = "", customer_phone: str = "") -> str:
    customer = {"name": customer_name.strip(), "email": customer_email.strip(), "phone": customer_phone.strip()}
    # Refused here, in code, for either backend. Told only in the prompt, the
    # agent booked without a phone number anyway (live app, 20 Sep) - and the
    # workshop's calendar rejects those, so the customer would have been told
    # they had an appointment they did not have.
    missing = [what for what, value in
               (("name", customer["name"]), ("email address", customer["email"]), ("phone number", customer["phone"]))
               if not value]
    if missing:
        return json.dumps({"ok": False, "error": f"Not booked. The workshop needs the customer's "
                                                 f"{', '.join(missing)}. Ask for it, then book."}, indent=2)
    return json.dumps(BACKEND.book_slot(slot_id, registration, issue, customer), indent=2)


def move_service_booking(reference: str, new_slot_id: str, registration: str) -> str:
    return json.dumps(BACKEND.move_booking(reference, new_slot_id, registration), indent=2)


def cancel_service_booking(reference: str, registration: str) -> str:
    return json.dumps(BACKEND.cancel_booking(reference, registration), indent=2)


def look_up_booking(reference: str, registration: str) -> str:
    return json.dumps(BACKEND.get_booking(reference, registration), indent=2)


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


# The same two parameters on every tool that reads or changes an existing
# booking. The registration is all that stands between a guessed reference and
# someone else's booking, so the model is told where it must come from - once,
# rather than three copies that could drift apart.
BOOKING_REFERENCE = {"type": "string", "description": "Booking reference, e.g. AA-4K2P9X."}
MATCHING_REGISTRATION = {
    "type": "string",
    "description": "The registration the booking was made with. Ask the customer for it - "
    "never copy it from a booking you looked up. A booking is only shown or changed "
    "when the reference and this registration match.",
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
            "customer_name": {
                "type": "string",
                "description": "The customer's name, for the appointment. Ask for it.",
            },
            "customer_email": {
                "type": "string",
                "description": "The customer's email address. The workshop's calendar emails the "
                "confirmation there, so the booking cannot be made without it.",
            },
            "customer_phone": {
                "type": "string",
                "description": "The customer's phone number, so the workshop can call about the "
                "appointment. The calendar refuses a booking without one.",
            },
        },
        ["slot_id", "registration", "customer_name", "customer_email", "customer_phone"],
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
            "registration": MATCHING_REGISTRATION,
        },
        ["reference", "new_slot_id", "registration"],
    ),
    "cancel_service_booking": _tool(
        "cancel_service_booking",
        "Cancel an existing booking using its reference. Only when the customer asks "
        "to cancel. To change the time, use move_service_booking instead.",
        {
            "reference": BOOKING_REFERENCE,
            "registration": MATCHING_REGISTRATION,
        },
        ["reference", "registration"],
    ),
    "look_up_booking": _tool(
        "look_up_booking",
        "Look up a booking by its reference to confirm the details.",
        {
            "reference": BOOKING_REFERENCE,
            "registration": MATCHING_REGISTRATION,
        },
        ["reference", "registration"],
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
