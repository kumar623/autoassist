"""Bookings in Zoho Bookings, through its MCP server.

A drop-in replacement for booking.py's slot and booking functions, so the tool
layer, the agent prompts and every rule built on top keep working:

    get_slots(on_date, days)                    -> {"ok", "slots": [...]}
    book_slot(slot_id, registration, issue, customer) -> {"ok", "reference", ...}
    get_booking(reference, registration)
    move_booking(reference, new_slot_id, registration)
    cancel_booking(reference, registration)

Why Zoho rather than the JSON file: appointments live in the workshop's real
calendar, they survive a deploy, they are shared by every replica, and Zoho
emails the customer on booking, reschedule and cancellation. That is the
confirmation email we would otherwise have had to build.

WHAT ZOHO DOES NOT KEEP FOR US
Zoho holds a customer name, email and phone. It has no field for a vehicle
registration, so the registration goes in the appointment's notes and is read
back from there.

    KNOWN LIMITATION, measured on 20 Sep 2026: rescheduleAppointment WIPES the
    notes. After a move, the registration is gone and even the owner can no
    longer look the booking up - the ownership check refuses everyone. The
    reschedule tool takes no notes argument, so it cannot be written back.
    The fix is a custom booking field ("Vehicle Registration", single line) on
    the service in Zoho Bookings; it is sent as additional_fields and read from
    customer_more_info, which is customer data rather than notes. Until that
    field exists Zoho ignores additional_fields (verified: it came back empty),
    so moving a booking is blocked here rather than quietly breaking ownership.

Every rule that depends on the registration still runs here:

  - a booking is only shown, moved or cancelled when the reference AND the
    registration match (red team, finding 12)
  - one booking per vehicle per day

QUIRKS, ALL FOUND BY TESTING AGAINST THE LIVE ACCOUNT (20 Sep 2026)
  - bookAppointment fails with "invalid phone_number" unless a phone number is
    given, though the booking form marks it optional. So the agent asks for one.
  - getAvailability nests its slots under data.response.returnvalue.data, beside
    a key misspelled "reponse".
  - fetchAppointment needs a from_time/to_time range; "upcoming" and
    customer_email filters return "No Match Found".
  - Times are "dd-MMM-yyyy HH:mm:ss", dates "dd-MMM-yyyy", slots "10:30 AM".
  - Cancelling is updateAppointmentStatus action="cancel"; the status reads
    "cancel" afterwards.
  - FETCH_AVAILABILITY takes at most SEVEN dates. An eighth is refused with
    {"status": "failure", "message": "Maximum Date Size Exceeded"} - in the
    body, not as an error, so it has to be read for rather than waited for.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from functools import lru_cache

from .booking import _parse_date  # one date parser for both backends
from .cache import TimedCache
from .mcp_client import McpClient, McpError
from .zoho_auth import ZohoAuth

log = logging.getLogger(__name__)

SERVICE_ID = os.getenv("ZOHO_SERVICE_ID", "")
STAFF_ID = os.getenv("ZOHO_STAFF_ID", "")
TIMEZONE = os.getenv("ZOHO_TIMEZONE", "Asia/Kolkata")
SEARCH_DAYS = 21  # how far ahead to look for open days, matching booking.py

# A custom booking field on the service survives a reschedule; the notes do not.
# Empty until the field exists in Zoho Bookings - see the note above.
REGISTRATION_FIELD = os.getenv("ZOHO_REGISTRATION_FIELD", "")
REGISTRATION_NOTE = "Vehicle registration: {registration}"
_REGISTRATION_IN_NOTES = re.compile(r"vehicle registration:\s*([A-Z0-9]+)", re.IGNORECASE)

_DAY = "%d-%b-%Y"
_MOMENT = "%d-%b-%Y %H:%M:%S"


# The booking agent asked Zoho for the same day's availability twice inside one
# reply (1.2s each, measured 20 Sep). Cleared the moment anything is booked,
# moved or cancelled, so a slot that has just gone never looks free.
_AVAILABILITY = TimedCache(float(os.getenv("ZOHO_CACHE_SECONDS", "60")), "zoho-availability")


class ZohoUnavailable(Exception):
    """Zoho could not be reached. The caller turns this into a readable error."""


# ---------------------------------------------------------------- plumbing


@lru_cache(maxsize=1)
def _mcp() -> McpClient:
    url = os.environ["ZOHO_MCP_URL"]
    auth = ZohoAuth.from_env()
    if auth is None:
        raise ZohoUnavailable("not signed in to Zoho: run scripts/zoho_login.py")
    return McpClient(url, auth=auth)


def _call(tool: str, arguments: dict) -> dict:
    """One Zoho tool call, unwrapped to the part that carries the answer."""
    try:
        out = _mcp().call_tool(f"ZohoBookings_{tool}", arguments)
    except McpError as e:
        raise ZohoUnavailable(str(e)) from None
    if not isinstance(out, dict):
        raise ZohoUnavailable(f"{tool} returned something unexpected")
    return ((out.get("data") or {}).get("response") or {}).get("returnvalue", out)


def _failed(result: dict) -> str | None:
    """Zoho's own message when it refused, else None."""
    if isinstance(result, dict) and result.get("status") == "failure":
        return str(result.get("message") or "Zoho refused the request")
    return None


def _norm(registration: str | None) -> str:
    return (registration or "").upper().replace(" ", "")


def _slot_id(when: datetime) -> str:
    return f"{when.date().isoformat()}-{when.strftime('%H%M')}"


def _slot_moment(slot_id: str) -> datetime | None:
    try:
        day, hhmm = slot_id.rsplit("-", 1)
        return datetime.strptime(f"{day} {hhmm[:2]}:{hhmm[2:]}:00", "%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        return None


def _appointment(record: dict) -> dict:
    """One Zoho appointment in the shape booking.py returns."""
    start = record.get("start_time") or ""
    try:
        when = datetime.strptime(start, _MOMENT)
    except ValueError:
        when = None
    notes = record.get("notes") or ""
    # The custom field first: it survives a reschedule, the notes do not.
    more = record.get("customer_more_info") or {}
    from_field = more.get(REGISTRATION_FIELD) if REGISTRATION_FIELD and isinstance(more, dict) else None
    found = _REGISTRATION_IN_NOTES.search(notes)
    return {
        "reference": record.get("booking_id", ""),
        "slot_id": _slot_id(when) if when else "",
        "date": when.date().isoformat() if when else "",
        "day": when.strftime("%A") if when else "",
        "time": when.strftime("%H:%M") if when else "",
        "registration": _norm(from_field) if from_field else (_norm(found.group(1)) if found else ""),
        "issue": notes.split("|", 1)[-1].strip() if "|" in notes else notes.strip(),
        "status": "confirmed" if record.get("status") == "upcoming" else str(record.get("status") or ""),
    }


def _not_found(reference: str, registration: str) -> dict:
    """Same answer whether it does not exist or belongs to someone else."""
    return {
        "ok": False,
        "error": f"No booking found with reference {reference} for registration "
        f"{_norm(registration) or '(none given)'}. Check both with the customer.",
    }


# ---------------------------------------------------------------- slots


def _slots_on(day: date) -> list[dict]:
    return _AVAILABILITY.get_or_call(("slots", day), lambda: _slots_on_now(day))


def _slots_on_now(day: date) -> list[dict]:
    result = _call("getAvailability", {"query_params": {
        "service_id": SERVICE_ID, "selected_date": day.strftime(_DAY),
        **({"staff_id": STAFF_ID} if STAFF_ID else {}),
    }})
    out = []
    for text in result.get("data") or []:
        try:
            when = datetime.combine(day, datetime.strptime(text.strip(), "%I:%M %p").time())
        except ValueError:
            continue
        out.append({"slot_id": _slot_id(when), "date": day.isoformat(), "day": when.strftime("%A"),
                    "time": when.strftime("%H:%M")})
    return out


# Zoho refuses a FETCH_AVAILABILITY carrying more than seven dates. SEARCH_DAYS
# is 21, so the question is asked a week at a time.
_MAX_DATE_LIST = 7


def _opens_on(days: list[date]) -> dict[date, bool]:
    """Which of these days the workshop opens at all, a week per call.

    The refusal is checked for, not merely survived. Zoho reports "Maximum Date
    Size Exceeded" inside a normal-looking body, so a refusal read as data says
    "no day is open" - which is indistinguishable from a full calendar. That is
    how asking for the next free slot came to answer "nothing for three weeks"
    on the live app rather than failing out loud (20 Sep).
    """
    out: dict[date, bool] = {}
    for start in range(0, len(days), _MAX_DATE_LIST):
        week = days[start:start + _MAX_DATE_LIST]
        result = _AVAILABILITY.get_or_call(("open-days", week[0], week[-1]), lambda w=week: _call(
            "fetchAvailability_or_getRecentAppointment", {"body": {"data": json.dumps({
                "action": "FETCH_AVAILABILITY", "service_id": SERVICE_ID,
                "date_list": [d.strftime(_DAY) for d in w],
            })}}))
        refused = _failed(result)
        if refused:
            raise ZohoUnavailable(refused)
        for d in week:
            out[d] = result.get(d.strftime(_DAY)) is True
    return out


def get_slots(on_date: str | None = None, days: int = 3) -> dict:
    """Free slots from Zoho, for one date or the next few days that have any."""
    today = date.today()
    try:
        if on_date:
            wanted = _parse_date(on_date, today)
            if wanted is None:
                return {"ok": False, "error": f"Could not understand the date '{on_date}'. "
                                              "Use YYYY-MM-DD or a weekday name."}
            if wanted < today:
                return {"ok": False, "error": f"{wanted.isoformat()} is in the past."}
            slots = _slots_on(wanted)
            if slots:
                return {"ok": True, "slots": slots, "count": len(slots)}

            # An empty list means either "shut that day" or "open and full", and
            # they are not the same answer. Told "no free slots on Saturday" the
            # customer asks about the Saturday after; told the workshop does not
            # open on Saturdays, they pick another day. Asked only when the day
            # came back empty, so an ordinary answer still costs one call.
            if not _opens_on([wanted])[wanted]:
                return {"ok": True, "slots": [], "count": 0, "closed": True,
                        "note": f"The workshop is closed on {wanted.strftime('%A %d %B')} - it does "
                                f"not open on {wanted.strftime('%A')}s at all. Say so and offer "
                                "another day; do not describe this as being fully booked."}
            return {"ok": True, "slots": [], "count": 0,
                    "note": f"The workshop is open on {wanted.strftime('%A %d %B')} but has nothing "
                            "free. Offer another day."}

        # No date given: the next few days that have anything free. The opening
        # days are asked for a week at a time and only while they are still
        # needed, so a customer who takes the first slot costs one extra call.
        ahead = [today + timedelta(days=n) for n in range(1, SEARCH_DAYS + 1)]
        slots: list[dict] = []
        found_days = 0
        for start in range(0, len(ahead), _MAX_DATE_LIST):
            if found_days >= days:
                break
            week = ahead[start:start + _MAX_DATE_LIST]
            opens = _opens_on(week)
            for d in week:
                if found_days >= days:
                    break
                if not opens.get(d):
                    continue
                day_slots = _slots_on(d)
                if day_slots:
                    slots.extend(day_slots)
                    found_days += 1
        return {"ok": True, "slots": slots, "count": len(slots)}
    except ZohoUnavailable as e:
        log.warning("Zoho unavailable while listing slots: %s", e)
        return {"ok": False, "error": f"The booking calendar could not be reached: {e}"}


# ---------------------------------------------------------------- bookings


def _bookings_on(day: date) -> list[dict]:
    """Every appointment that day, as booking.py-shaped records."""
    result = _call("fetchAppointment", {"body": {"data": json.dumps({
        "from_time": f"{day.strftime(_DAY)} 00:00:00",
        "to_time": f"{day.strftime(_DAY)} 23:59:59",
    })}})
    records = result.get("response")
    if not isinstance(records, list):  # "No Match Found" when the day is empty
        return []
    return [_appointment(r) for r in records if isinstance(r, dict)]


def book_slot(slot_id: str, registration: str, issue: str, customer: dict | None = None) -> dict:
    """Book in Zoho. Zoho emails the customer; it needs their name, email and phone."""
    customer = customer or {}
    reg = _norm(registration)
    name = (customer.get("name") or "").strip()
    email = (customer.get("email") or "").strip()
    phone = (customer.get("phone") or "").strip()

    if not slot_id or not reg:
        return {"ok": False, "error": "slot_id and registration are both required."}
    missing = [what for what, value in (("name", name), ("email", email), ("phone number", phone)) if not value]
    if missing:
        return {"ok": False, "error": f"The booking calendar needs the customer's {', '.join(missing)}. Ask for it."}

    when = _slot_moment(slot_id)
    if when is None:
        return {"ok": False, "error": f"'{slot_id}' is not a valid slot id."}

    try:
        # One booking per vehicle per day, the same rule as the file backend.
        for existing in _bookings_on(when.date()):
            if existing["registration"] == reg and existing["status"] == "confirmed":
                return {"ok": False, "error": (
                    f"{reg} already has a booking on {when.date().isoformat()}. Only one booking per "
                    f"vehicle per day. If the customer wants a different time, ask for their booking "
                    f"reference and use move_service_booking. Otherwise tell them they are already booked."
                )}

        result = _call("bookAppointment", {"body": {
            "service_id": SERVICE_ID,
            **({"staff_id": STAFF_ID} if STAFF_ID else {}),
            "from_time": when.strftime(_MOMENT),
            "customer_details": json.dumps({"name": name, "email": email, "phone_number": phone}),
            "notes": f"{REGISTRATION_NOTE.format(registration=reg)} | {issue or 'not stated'}",
            **({"additional_fields": json.dumps({REGISTRATION_FIELD: reg})} if REGISTRATION_FIELD else {}),
            "timezone": TIMEZONE,
        }})
    except ZohoUnavailable as e:
        return {"ok": False, "error": f"The booking calendar could not be reached: {e}"}

    refused = _failed(result)
    if refused:
        return {"ok": False, "error": f"Zoho did not accept the booking: {refused}"}

    _AVAILABILITY.clear()  # that slot is gone now
    booked = _appointment(result)
    return {"ok": True, "reference": booked["reference"], "date": booked["date"], "day": booked["day"],
            "time": booked["time"], "registration": reg,
            "message": f"Booked. Reference {booked['reference']}. Zoho has emailed {email}."}


def _owned(reference: str, registration: str) -> dict | None:
    """The appointment, only if this registration is the one in its notes."""
    result = _call("getAppointment", {"query_params": {"booking_id": reference.strip()}})
    if _failed(result) or not isinstance(result, dict):
        return None
    record = result.get("response") if isinstance(result.get("response"), dict) else result
    booked = _appointment(record or {})
    if not booked["reference"] or not _norm(registration) or booked["registration"] != _norm(registration):
        return None
    return booked


def get_booking(reference: str, registration: str) -> dict:
    try:
        booked = _owned(reference, registration)
    except ZohoUnavailable as e:
        return {"ok": False, "error": f"The booking calendar could not be reached: {e}"}
    if booked is None:
        return _not_found(reference, registration)
    return {"ok": True, **booked}


def move_booking(reference: str, new_slot_id: str, registration: str) -> dict:
    when = _slot_moment(new_slot_id)
    try:
        booked = _owned(reference, registration)
        if booked is None:
            return _not_found(reference, registration)
        if booked["status"] != "confirmed":
            return {"ok": False, "error": f"Booking {reference} is {booked['status']}, so it cannot be moved."}
        if when is None:
            return {"ok": False, "error": f"'{new_slot_id}' is not a valid slot id. Booking {reference} is unchanged."}
        if new_slot_id == booked["slot_id"]:
            return {"ok": False, "error": f"Booking {reference} is already at that time. Nothing changed."}

        if not REGISTRATION_FIELD:
            # Moving would wipe the notes holding the registration, leaving a
            # booking nobody can look up or cancel. Refuse instead.
            return {"ok": False, "error": (
                "This calendar cannot move a booking yet: the workshop's booking form needs a "
                "'Vehicle Registration' field. Cancel this booking and make a new one at the "
                f"time they want. Booking {reference} is unchanged."
            )}

        result = _call("rescheduleAppointment", {"body": {
            "booking_id": booked["reference"], "start_time": when.strftime(_MOMENT),
            **({"staff_id": STAFF_ID} if STAFF_ID else {}),
        }})
    except ZohoUnavailable as e:
        return {"ok": False, "error": f"The booking calendar could not be reached: {e}"}

    refused = _failed(result)
    if refused:
        return {"ok": False, "error": f"That time could not be taken: {refused}. Booking {reference} is unchanged "
                                      f"and still at {booked['date']} {booked['time']}."}
    _AVAILABILITY.clear()  # one slot freed, another taken
    moved = _appointment(result.get("response") if isinstance(result.get("response"), dict) else result)
    was = f"{booked['date']} {booked['time']}"
    return {"ok": True, "reference": booked["reference"], "date": moved["date"] or booked["date"],
            "day": moved["day"] or booked["day"], "time": moved["time"],
            "message": f"Moved. Reference {booked['reference']} is now {moved['date']} at {moved['time']} (was {was})."}


def cancel_booking(reference: str, registration: str) -> dict:
    try:
        booked = _owned(reference, registration)
        if booked is None:
            return _not_found(reference, registration)
        if booked["status"] != "confirmed":
            return {"ok": False, "error": f"Booking {reference} was already {booked['status']}."}
        result = _call("updateAppointmentStatus", {"body": {"booking_id": booked["reference"], "action": "cancel"}})
    except ZohoUnavailable as e:
        return {"ok": False, "error": f"The booking calendar could not be reached: {e}"}

    refused = _failed(result)
    if refused:
        return {"ok": False, "error": f"Zoho did not cancel it: {refused}"}
    _AVAILABILITY.clear()  # that slot is free again
    return {"ok": True, "reference": booked["reference"],
            "message": f"Booking {booked['reference']} cancelled. Zoho has emailed the customer."}
