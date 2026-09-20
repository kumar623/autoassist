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


def book_slot(slot_id: str, registration: str, issue: str, customer: dict | None = None) -> dict:
    """Reserve a slot. The ONLY way a booking comes into existence.

    `customer` (name, email, phone) is ignored here and used by the Zoho
    backend, so tools.py can call either one the same way.
    """
    if not slot_id or not registration:
        return {"ok": False, "error": "slot_id and registration are both required."}

    reg = registration.upper().replace(" ", "")

    with _LOCK:
        data = _load()

        for b in data["bookings"].values():
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

        # One booking per vehicle per day, enforced here rather than asked for in
        # a prompt. Asked to "move it to 3:30", the booking agent booked 3:30 and
        # left the original in place - twice out of twice - despite being told to
        # cancel the old one.
        #
        # The refusal does NOT name the existing booking's reference or time. A
        # registration is printed on the car; anyone can read it. Handing out the
        # reference here would let a stranger look up, move or cancel a booking
        # knowing nothing but a number plate (red team, 20 Sep).
        for b in data["bookings"].values():
            if b.get("status") == "confirmed" and b.get("registration") == reg and b.get("date") == d:
                return {
                    "ok": False,
                    "error": (
                        f"{reg} already has a booking on {d}. Only one booking per vehicle per "
                        f"day. If the customer wants a different time, ask for their booking "
                        f"reference and use move_service_booking. Otherwise tell them they are "
                        f"already booked that day."
                    ),
                }

        booking = Booking(
            reference=_reference(),
            slot_id=slot_id,
            date=d,
            time=time_str,
            registration=reg,
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


def move_booking(reference: str, new_slot_id: str, registration: str) -> dict:
    """Move a booking to another slot in one step. Changes nothing unless it works.

    Moving used to be cancel_booking then book_slot, run by the agent. Twice in
    local replays it stopped in between: once to ask "shall I book 3:30?", once
    because the new time was taken - after cancelling. Either way the customer
    was left with no booking. Here the new slot is checked before anything
    changes, under the same lock, and the reference stays the same.
    """
    ref = reference.upper().strip()

    with _LOCK:
        data = _load()
        b = _owned(data, ref, registration)
        if b is None or b.get("status") != "confirmed":
            return _not_found(reference, registration)

        if new_slot_id == b["slot_id"]:
            return {"ok": False, "error": f"Booking {ref} is already at that time. Nothing changed."}

        for other in data["bookings"].values():
            if other["slot_id"] == new_slot_id and other.get("status") == "confirmed":
                return {
                    "ok": False,
                    "error": f"Slot {new_slot_id} is taken. Booking {ref} is unchanged and still "
                    f"at {b['date']} {b['time']}. Offer the customer a different time.",
                }

        try:
            d, t = new_slot_id.rsplit("-", 1)
            datetime.strptime(d, "%Y-%m-%d")
            time_str = f"{t[:2]}:{t[2:]}"
        except (ValueError, IndexError):
            return {"ok": False, "error": f"'{new_slot_id}' is not a valid slot id. Booking {ref} is unchanged."}

        was = f"{b['date']} {b['time']}"
        b.update(slot_id=new_slot_id, date=d, time=time_str)
        _save(data)

    return {
        "ok": True,
        "reference": ref,
        "date": d,
        "day": datetime.strptime(d, "%Y-%m-%d").strftime("%A"),
        "time": time_str,
        "message": f"Moved. Reference {ref} is now {d} at {time_str} (was {was}).",
    }


def cancel_booking(reference: str, registration: str) -> dict:
    with _LOCK:
        data = _load()
        b = _owned(data, reference.upper().strip(), registration)
        if b is None:
            return _not_found(reference, registration)
        if b["status"] == "cancelled":
            return {"ok": False, "error": f"Booking {reference} was already cancelled."}
        b["status"] = "cancelled"
        _save(data)
    return {"ok": True, "reference": reference, "message": f"Booking {reference} cancelled."}


def get_booking(reference: str, registration: str) -> dict:
    b = _owned(_load(), reference.upper().strip(), registration)
    if b is None:
        return _not_found(reference, registration)
    return {"ok": True, **b}


# ---------------------------------------------------------------- who owns a booking
#
# Looking up, moving and cancelling all need the reference AND the registration
# the booking was made with - like an airline's booking code plus surname.
# Before this, the reference alone was enough: in the red team on 20 Sep, a
# stranger holding a reference was told another customer's registration and
# fault, and could cancel their appointment. Enforced here, in code, because
# "only show a customer their own booking" as a prompt rule is a request, not
# a guarantee.


def _norm(registration: str | None) -> str:
    return (registration or "").upper().replace(" ", "")


def _owned(data: dict, reference: str, registration: str) -> dict | None:
    """The booking, only if this registration made it. Otherwise None."""
    b = data["bookings"].get(reference)
    if b is None or not _norm(registration) or b.get("registration") != _norm(registration):
        return None
    return b


def _not_found(reference: str, registration: str) -> dict:
    """The same answer whether the reference does not exist or belongs to someone
    else, so guessing references does not reveal which ones are real."""
    return {
        "ok": False,
        "error": f"No booking found with reference {reference} for registration {_norm(registration) or '(none given)'}. "
        "Check both with the customer.",
    }


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
