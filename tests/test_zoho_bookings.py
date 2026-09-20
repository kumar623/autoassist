"""Tests for the Zoho Bookings backend.

Zoho is replaced by a fake that answers in the shapes the live account returned
on 20 Sep 2026, including its quirks: slots as "10:30 AM" strings, appointments
only inside a date range, failures as {"status": "failure", "message": ...}.

The rules that matter are the ones carried over from the file backend: the
registration must match before anything is shown or changed, and one vehicle
cannot hold two bookings on one day.
"""

import json
from datetime import date, timedelta

import pytest

from services.orchestrator import zoho_bookings as z

TOMORROW = date.today() + timedelta(days=1)
DAY_ZOHO = TOMORROW.strftime("%d-%b-%Y")
SLOT = f"{TOMORROW.isoformat()}-1030"

APPOINTMENT = {
    "booking_id": "#TE-00002",
    "service_id": "273103000000034065",
    "staff_id": "273103000000034009",
    "start_time": f"{DAY_ZOHO} 10:30:00",
    "end_time": f"{DAY_ZOHO} 11:30:00",
    "status": "upcoming",
    "notes": "Vehicle registration: AP31BP2133 | clutch slipping",
}


@pytest.fixture
def zoho(monkeypatch):
    """Records calls; `answers` maps a tool name to a reply or a function."""
    state = {"calls": [], "answers": {}}

    def fake_call(tool, arguments):
        state["calls"].append((tool, arguments))
        answer = state["answers"].get(tool, {})
        return answer(arguments) if callable(answer) else answer

    monkeypatch.setattr(z, "_call", fake_call)
    monkeypatch.setattr(z, "SERVICE_ID", "273103000000034065")
    monkeypatch.setattr(z, "STAFF_ID", "273103000000034009")
    monkeypatch.setattr(z, "REGISTRATION_FIELD", "Vehicle Registration")
    z._AVAILABILITY.clear()
    return state


def sent(state, tool):
    return [args for name, args in state["calls"] if name == tool]


# ---------------------------------------------------------------- slots


def test_slots_come_back_in_our_own_shape(zoho):
    zoho["answers"]["getAvailability"] = {"time_zone": "Asia/Kolkata", "data": ["10:30 AM", "02:15 PM"]}
    r = z.get_slots(on_date=TOMORROW.isoformat())
    assert r["ok"]
    assert [s["slot_id"] for s in r["slots"]] == [f"{TOMORROW.isoformat()}-1030", f"{TOMORROW.isoformat()}-1415"]
    assert [s["time"] for s in r["slots"]] == ["10:30", "14:15"]
    assert sent(zoho, "getAvailability")[0]["query_params"]["selected_date"] == DAY_ZOHO


def test_a_day_the_workshop_opens_but_is_full_says_nothing_free(zoho):
    zoho["answers"]["getAvailability"] = {"data": []}
    zoho["answers"]["fetchAvailability_or_getRecentAppointment"] = {DAY_ZOHO: True}
    r = z.get_slots(on_date=TOMORROW.isoformat())
    assert r["ok"] and r["slots"] == [] and "nothing free" in r["note"]
    assert not r.get("closed") and "closed" not in r["note"].lower()


def test_a_closed_day_says_closed_not_fully_booked(zoho):
    """Zoho answers an empty slot list whether the workshop is shut that day or
    merely full, and those are not the same answer to a customer. Asked "what
    slots are free on Saturday?" the live app said "there are no free slots on
    Saturday 26 September" (20 Sep) - which reads as "fully booked", so the
    customer asks again about the Saturday after. The workshop does not open on
    Saturdays at all, and saying so is what stops the second question.
    """
    zoho["answers"]["getAvailability"] = {"data": []}
    zoho["answers"]["fetchAvailability_or_getRecentAppointment"] = {DAY_ZOHO: False}
    r = z.get_slots(on_date=TOMORROW.isoformat())
    assert r["ok"] and r["slots"] == [] and r["closed"] is True
    assert "closed" in r["note"].lower()
    assert "nothing free" not in r["note"].lower()


def test_the_opening_days_are_asked_in_chunks_zoho_accepts(zoho):
    """Zoho refuses a date_list longer than seven with "Maximum Date Size
    Exceeded" (live account, 20 Sep). SEARCH_DAYS is 21, so the question has to
    be split - unsplit, the refusal came back as an ordinary dict, every day
    read as closed, and the customer was told the workshop had nothing free for
    three weeks.
    """
    sizes = []

    def answer(arguments):
        dates = json.loads(arguments["body"]["data"])["date_list"]
        sizes.append(len(dates))
        if len(dates) > 7:
            return {"status": "failure", "message": "Maximum Date Size Exceeded"}
        return dict.fromkeys(dates, True)

    zoho["answers"]["fetchAvailability_or_getRecentAppointment"] = answer
    zoho["answers"]["getAvailability"] = {"data": ["09:00 AM"]}
    r = z.get_slots(days=3)
    assert sizes, "never asked which days the workshop opens"
    assert max(sizes) <= 7, f"sent {max(sizes)} dates; Zoho refuses more than seven"
    assert r["ok"] and r["count"] > 0


def test_a_refused_availability_question_is_not_an_empty_calendar(zoho):
    """A refusal arrives in the body, not as an exception. Counted as data it
    means "no day is open", which is indistinguishable from a full calendar and
    is how the customer got told there was nothing free for three weeks."""
    zoho["answers"]["fetchAvailability_or_getRecentAppointment"] = {
        "status": "failure", "message": "Maximum Date Size Exceeded"}
    r = z.get_slots(days=3)
    assert not r["ok"], "a refusal was reported as an empty calendar"
    assert "Maximum Date Size Exceeded" in r["error"]


def test_only_open_days_are_looked_up(zoho):
    """One call asks which days are open, so closed days cost nothing."""
    day2 = (TOMORROW + timedelta(days=1)).strftime("%d-%b-%Y")
    zoho["answers"]["fetchAvailability_or_getRecentAppointment"] = {DAY_ZOHO: False, day2: True}
    zoho["answers"]["getAvailability"] = {"data": ["09:00 AM"]}
    r = z.get_slots(days=1)
    assert r["ok"] and len(r["slots"]) == 1
    assert [a["query_params"]["selected_date"] for a in sent(zoho, "getAvailability")] == [day2]


def test_a_date_we_cannot_read_is_refused(zoho):
    assert not z.get_slots(on_date="whenever")["ok"]


def test_a_past_date_is_refused(zoho):
    assert not z.get_slots(on_date="2020-01-01")["ok"]


def test_zoho_being_down_is_a_readable_error_not_a_crash(zoho):
    def boom(arguments):
        raise z.ZohoUnavailable("MCP server at zohomcp.eu answered HTTP 503")

    zoho["answers"]["getAvailability"] = boom
    r = z.get_slots(on_date=TOMORROW.isoformat())
    assert not r["ok"] and "could not be reached" in r["error"]


# ---------------------------------------------------------------- booking


CUSTOMER = {"name": "Krishna", "email": "k@example.com", "phone": "9876543210"}


def test_a_booking_sends_zoho_what_it_needs(zoho):
    zoho["answers"]["fetchAppointment"] = {"response": "No Match Found"}
    zoho["answers"]["bookAppointment"] = APPOINTMENT
    r = z.book_slot(SLOT, "ap 31 bp 2133", "clutch slipping", CUSTOMER)
    assert r["ok"] and r["reference"] == "#TE-00002" and r["time"] == "10:30"
    body = sent(zoho, "bookAppointment")[0]["body"]
    assert body["from_time"] == f"{DAY_ZOHO} 10:30:00"
    assert '"phone_number": "9876543210"' in body["customer_details"]
    assert "Vehicle registration: AP31BP2133" in body["notes"], "the registration must survive in Zoho"
    assert "emailed k@example.com" in r["message"]


@pytest.mark.parametrize("missing", ["name", "email", "phone"])
def test_zoho_needs_the_customers_details(zoho, missing):
    """Zoho refuses without a phone number, whatever its booking form says."""
    customer = {**CUSTOMER, missing: ""}
    r = z.book_slot(SLOT, "AP31BP2133", "service", customer)
    assert not r["ok"] and missing.replace("phone", "phone number") in r["error"]
    assert sent(zoho, "bookAppointment") == []


def test_one_booking_per_vehicle_per_day(zoho):
    zoho["answers"]["fetchAppointment"] = {"response": [APPOINTMENT]}
    r = z.book_slot(f"{TOMORROW.isoformat()}-1415", "AP31BP2133", "service", CUSTOMER)
    assert not r["ok"] and "already has a booking" in r["error"]
    assert sent(zoho, "bookAppointment") == []
    assert "#TE-00002" not in r["error"], "a number plate must not reveal the reference"


def test_another_vehicle_may_book_the_same_day(zoho):
    zoho["answers"]["fetchAppointment"] = {"response": [APPOINTMENT]}
    zoho["answers"]["bookAppointment"] = {**APPOINTMENT, "booking_id": "#TE-00003"}
    assert z.book_slot(f"{TOMORROW.isoformat()}-1415", "AP31XX0000", "service", CUSTOMER)["ok"]


def test_zohos_refusal_is_passed_on(zoho):
    zoho["answers"]["fetchAppointment"] = {"response": "No Match Found"}
    zoho["answers"]["bookAppointment"] = {"status": "failure", "message": "invalid phone_number"}
    r = z.book_slot(SLOT, "AP31BP2133", "service", CUSTOMER)
    assert not r["ok"] and "invalid phone_number" in r["error"]


def test_an_invented_slot_id_is_refused(zoho):
    assert not z.book_slot("slot_10:30_2020-09-21", "AP31BP2133", "service", CUSTOMER)["ok"]


# ---------------------------------------------------------------- who owns it


def test_the_owner_sees_their_booking(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    r = z.get_booking("#TE-00002", "ap31bp2133")
    assert r["ok"] and r["time"] == "10:30" and r["issue"] == "clutch slipping"


def test_a_stranger_is_told_nothing(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    r = z.get_booking("#TE-00002", "AP31XX0000")
    assert not r["ok"]
    assert "AP31BP2133" not in r["error"] and "clutch" not in r["error"]


def test_a_wrong_registration_looks_like_a_missing_booking(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    wrong = z.get_booking("#TE-00002", "AP31XX0000")["error"].replace("#TE-00002", "REF")
    zoho["answers"]["getAppointment"] = {"status": "failure", "message": "not found"}
    missing = z.get_booking("#TE-99999", "AP31XX0000")["error"].replace("#TE-99999", "REF")
    assert wrong == missing


def test_a_stranger_cannot_cancel_or_move(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    assert not z.cancel_booking("#TE-00002", "AP31XX0000")["ok"]
    assert not z.move_booking("#TE-00002", f"{TOMORROW.isoformat()}-1415", "AP31XX0000")["ok"]
    assert sent(zoho, "updateAppointmentStatus") == [] and sent(zoho, "rescheduleAppointment") == []


# ---------------------------------------------------------------- moving and cancelling


def test_moving_asks_zoho_to_reschedule(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    zoho["answers"]["rescheduleAppointment"] = {**APPOINTMENT, "start_time": f"{DAY_ZOHO} 14:15:00"}
    r = z.move_booking("#TE-00002", f"{TOMORROW.isoformat()}-1415", "AP31BP2133")
    assert r["ok"] and r["time"] == "14:15" and r["reference"] == "#TE-00002"
    assert sent(zoho, "rescheduleAppointment")[0]["body"]["start_time"] == f"{DAY_ZOHO} 14:15:00"


def test_a_refused_move_leaves_the_booking_alone(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    zoho["answers"]["rescheduleAppointment"] = {"status": "failure", "message": "slot not available"}
    r = z.move_booking("#TE-00002", f"{TOMORROW.isoformat()}-1415", "AP31BP2133")
    assert not r["ok"]
    assert "unchanged" in r["error"] and "10:30" in r["error"]


def test_moving_to_the_same_time_changes_nothing(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    assert not z.move_booking("#TE-00002", SLOT, "AP31BP2133")["ok"]
    assert sent(zoho, "rescheduleAppointment") == []


def test_cancelling_uses_the_status_action(zoho):
    zoho["answers"]["getAppointment"] = APPOINTMENT
    zoho["answers"]["updateAppointmentStatus"] = {**APPOINTMENT, "status": "cancel"}
    r = z.cancel_booking("#TE-00002", "AP31BP2133")
    assert r["ok"] and "emailed" in r["message"]
    assert sent(zoho, "updateAppointmentStatus")[0]["body"] == {"booking_id": "#TE-00002", "action": "cancel"}


def test_an_already_cancelled_booking_is_not_cancelled_twice(zoho):
    zoho["answers"]["getAppointment"] = {**APPOINTMENT, "status": "cancel"}
    r = z.cancel_booking("#TE-00002", "AP31BP2133")
    assert not r["ok"] and "already" in r["error"]
    assert sent(zoho, "updateAppointmentStatus") == []


def test_moving_is_refused_until_zoho_has_a_registration_field(zoho, monkeypatch):
    """Measured on 20 Sep: rescheduleAppointment wipes the notes, and the
    registration lives there. A moved booking would belong to nobody - not even
    its owner could cancel it. So refuse the move rather than break ownership."""
    monkeypatch.setattr(z, "REGISTRATION_FIELD", "")
    zoho["answers"]["getAppointment"] = APPOINTMENT
    r = z.move_booking("#TE-00002", f"{TOMORROW.isoformat()}-1415", "AP31BP2133")
    assert not r["ok"]
    assert "Vehicle Registration" in r["error"] and "unchanged" in r["error"]
    assert sent(zoho, "rescheduleAppointment") == []


def test_moving_works_once_the_field_is_configured(zoho, monkeypatch):
    monkeypatch.setattr(z, "REGISTRATION_FIELD", "Vehicle Registration")
    zoho["answers"]["getAppointment"] = APPOINTMENT
    zoho["answers"]["rescheduleAppointment"] = {**APPOINTMENT, "start_time": f"{DAY_ZOHO} 14:15:00"}
    assert z.move_booking("#TE-00002", f"{TOMORROW.isoformat()}-1415", "AP31BP2133")["ok"]


def test_the_registration_goes_into_the_custom_field_when_there_is_one(zoho):
    zoho["answers"]["fetchAppointment"] = {"response": "No Match Found"}
    zoho["answers"]["bookAppointment"] = APPOINTMENT
    z.book_slot(SLOT, "AP31BP2133", "service", CUSTOMER)
    body = sent(zoho, "bookAppointment")[0]["body"]
    assert '"Vehicle Registration": "AP31BP2133"' in body["additional_fields"]


def test_the_custom_field_is_read_even_when_a_move_wiped_the_notes(zoho):
    """The exact failure on 20 Sep: after a reschedule the notes were empty."""
    zoho["answers"]["getAppointment"] = {**APPOINTMENT, "notes": "",
                                         "customer_more_info": {"Vehicle Registration": "AP31BP2133"}}
    assert z.get_booking("#TE-00002", "AP31BP2133")["ok"]


def test_the_zoho_backend_says_who_was_emailed(zoho):
    """Zoho does send one, so the agent may say so - and only then."""
    zoho["answers"]["fetchAppointment"] = {"response": "No Match Found"}
    zoho["answers"]["bookAppointment"] = APPOINTMENT
    r = z.book_slot(SLOT, "AP31BP2133", "service", CUSTOMER)
    assert "emailed k@example.com" in r["message"]


def test_availability_for_a_day_is_asked_once(zoho):
    """The booking agent asked Zoho twice in one reply, at 1.2s each."""
    zoho["answers"]["getAvailability"] = {"data": ["10:30 AM"]}
    z.get_slots(on_date=TOMORROW.isoformat())
    z.get_slots(on_date=TOMORROW.isoformat())
    assert len(sent(zoho, "getAvailability")) == 1


@pytest.mark.parametrize("change", ["book", "move", "cancel"])
def test_any_change_to_the_calendar_forgets_what_was_cached(zoho, change):
    """A slot that has just gone must never still look free."""
    zoho["answers"]["getAvailability"] = {"data": ["10:30 AM", "02:15 PM"]}
    zoho["answers"]["fetchAppointment"] = {"response": "No Match Found"}
    zoho["answers"]["getAppointment"] = APPOINTMENT
    zoho["answers"]["bookAppointment"] = APPOINTMENT
    zoho["answers"]["rescheduleAppointment"] = {**APPOINTMENT, "start_time": f"{DAY_ZOHO} 14:15:00"}
    zoho["answers"]["updateAppointmentStatus"] = {**APPOINTMENT, "status": "cancel"}
    z.get_slots(on_date=TOMORROW.isoformat())

    if change == "book":
        z.book_slot(SLOT, "AP31XX0000", "service", CUSTOMER)
    elif change == "move":
        z.move_booking("#TE-00002", f"{TOMORROW.isoformat()}-1415", "AP31BP2133")
    else:
        z.cancel_booking("#TE-00002", "AP31BP2133")

    z.get_slots(on_date=TOMORROW.isoformat())
    assert len(sent(zoho, "getAvailability")) == 2, "the calendar was read again after it changed"
