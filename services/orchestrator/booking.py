"""Service slot booking.

Storage is a JSON file for now. Week 3 swaps it for Azure Table Storage - the
interface here (get_slots / book_slot / cancel_booking) stays the same, so that
is a change to this file only.

The rule that matters: book_slot is the only thing that creates a booking, and
it returns the reference. The agent must never invent a confirmation. A made-up
booking reference is the booking equivalent of a fabricated citation.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import string
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

STORE = pathlib.Path(os.getenv("BOOKING_STORE", "data/bookings.json"))
_LOCK = threading.Lock()

OPENING_HOURS = ["09:00", "10:30", "12:00", "14:00", "15:30", "17:00"]
DAYS_AHEAD = 21


@dataclass
class Booking:
    reference: str
    slot_id: str
    date: str
    time: str
    registration: str
    issue: str
    created_at: str
    status: str = "confirmed"


def _load() -> dict:
    if not STORE.exists():
        return {"bookings": {}}
    try:
        return json.loads(STORE.read_text())
    except json.JSONDecodeError:
        # A corrupt store should not take the service down, but it must be loud.
        return {"bookings": {}, "_warning": "store was unreadable and has been reset"}


def _save(data: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, indent=2))


def _slot_id(d: str, t: str) -> str:
    return f"{d}-{t.replace(':', '')}"


def _reference() -> str:
    return "AA-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _workshop_open(d: date) -> bool:
    return d.weekday() != 6  # closed Sundays


def get_slots(on_date: str | None = None, days: int = 3) -> dict:
    """Free slots, either for one date or for the next few open days.

    on_date: 'YYYY-MM-DD', or a weekday name like 'saturday', or None.
    """
    data = _load()
    taken = {
        b["slot_id"]
        for b in data["bookings"].values()
        if b.get("status") == "confirmed"
    }

    today = date.today()
    wanted: list[date] = []

    if on_date:
        parsed = _parse_date(on_date, today)
        if parsed is None:
            return {
                "ok": False,
                "error": f"Could not understand the date '{on_date}'. "
                "Use YYYY-MM-DD or a weekday name.",
            }
        if parsed < today:
            return {"ok": False, "error": f"{parsed.isoformat()} is in the past."}
        if not _workshop_open(parsed):
            return {
                "ok": True,
                "slots": [],
                "note": f"The workshop is closed on {parsed.strftime('%A %d %B')} (Sundays).",
            }
        wanted = [parsed]
    else:
        d = today
        while len(wanted) < days and (d - today).days <= DAYS_AHEAD:
            d += timedelta(days=1)
            if _workshop_open(d):
                wanted.append(d)

    slots = []
    for d in wanted:
        for t in OPENING_HOURS:
            sid = _slot_id(d.isoformat(), t)
            if sid not in taken:
                slots.append(
                    {
                        "slot_id": sid,
                        "date": d.isoformat(),
                        "day": d.strftime("%A"),
                        "time": t,
                    }
                )

    return {"ok": True, "slots": slots, "count": len(slots)}


def _parse_date(text: str, today: date) -> date | None:
    text = text.strip().lower()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        pass

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if text.startswith("next "):
        text = text[5:]
    if text in weekdays:
        target = weekdays.index(text)
        ahead = (target - today.weekday()) % 7
        return today + timedelta(days=ahead or 7)
    if text == "tomorrow":
        return today + timedelta(days=1)
    return None


def book_slot(slot_id: str, registration: str, issue: str) -> dict:
    """Reserve a slot. The ONLY way a booking comes into existence."""
    if not slot_id or not registration:
        return {"ok": False, "error": "slot_id and registration are both required."}

    with _LOCK:
        data = _load()

        for ref, b in data["bookings"].items():
            if b["slot_id"] == slot_id and b.get("status") == "confirmed":
                return {
                    "ok": False,
                    "error": f"Slot {slot_id} was taken already. Offer the customer a different time.",
                }

        try:
            d, t = slot_id.rsplit("-", 1)
            datetime.strptime(d, "%Y-%m-%d")
            time_str = f"{t[:2]}:{t[2:]}"
        except (ValueError, IndexError):
            return {"ok": False, "error": f"'{slot_id}' is not a valid slot id."}

        booking = Booking(
            reference=_reference(),
            slot_id=slot_id,
            date=d,
            time=time_str,
            registration=registration.upper().replace(" ", ""),
            issue=issue or "not stated",
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        data["bookings"][booking.reference] = asdict(booking)
        _save(data)

    return {
        "ok": True,
        "reference": booking.reference,
        "date": booking.date,
        "day": datetime.strptime(booking.date, "%Y-%m-%d").strftime("%A"),
        "time": booking.time,
        "registration": booking.registration,
        "message": f"Booked. Reference {booking.reference}.",
    }


def cancel_booking(reference: str) -> dict:
    with _LOCK:
        data = _load()
        b = data["bookings"].get(reference.upper().strip())
        if b is None:
            return {"ok": False, "error": f"No booking found with reference {reference}."}
        if b["status"] == "cancelled":
            return {"ok": False, "error": f"Booking {reference} was already cancelled."}
        b["status"] = "cancelled"
        _save(data)
    return {"ok": True, "reference": reference, "message": f"Booking {reference} cancelled."}


def get_booking(reference: str) -> dict:
    b = _load()["bookings"].get(reference.upper().strip())
    if b is None:
        return {"ok": False, "error": f"No booking found with reference {reference}."}
    return {"ok": True, **b}


# ---------------------------------------------------------------- tickets


TICKETS = pathlib.Path(os.getenv("TICKET_STORE", "data/tickets.json"))


def create_ticket(summary: str, urgency: str = "normal", registration: str = "") -> dict:
    """Raise a ticket for a human. Used when the agent must not answer itself."""
    if urgency not in ("normal", "urgent", "safety"):
        urgency = "normal"

    with _LOCK:
        data = json.loads(TICKETS.read_text()) if TICKETS.exists() else {"tickets": {}}
        ref = "TK-" + "".join(random.choices(string.digits, k=6))
        data["tickets"][ref] = {
            "reference": ref,
            "summary": summary,
            "urgency": urgency,
            "registration": registration.upper().replace(" ", ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "open",
        }
        TICKETS.parent.mkdir(parents=True, exist_ok=True)
        TICKETS.write_text(json.dumps(data, indent=2))

    callback = "within 1 hour" if urgency == "safety" else "within one working day"
    return {
        "ok": True,
        "reference": ref,
        "urgency": urgency,
        "message": f"Ticket {ref} raised. A service advisor will call {callback}.",
    }
