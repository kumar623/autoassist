"""Tests for the booking store.

No Azure needed - these run offline and in CI on every pull request.

The important ones are the negative cases. A booking system that happily
double-books, or accepts a slot id it invented, is worse than no booking system,
because the customer turns up and there is no appointment.
"""

import os
import pathlib
import tempfile

import pytest

os.environ.setdefault("BOOKING_STORE", str(pathlib.Path(tempfile.gettempdir()) / "aa_test_bookings.json"))
os.environ.setdefault("TICKET_STORE", str(pathlib.Path(tempfile.gettempdir()) / "aa_test_tickets.json"))

from services.orchestrator import booking  # noqa: E402


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    monkeypatch.setattr(booking, "STORE", tmp_path / "bookings.json")
    monkeypatch.setattr(booking, "TICKETS", tmp_path / "tickets.json")
    yield


def first_slot():
    return booking.get_slots(days=2)["slots"][0]


def test_slots_are_returned_for_open_days():
    r = booking.get_slots(days=3)
    assert r["ok"]
    assert r["count"] > 0
    assert all(s["day"] != "Sunday" for s in r["slots"]), "workshop is closed on Sundays"


def test_booking_returns_a_reference():
    s = first_slot()
    r = booking.book_slot(s["slot_id"], "AP31AB1234", "check engine light")
    assert r["ok"]
    assert r["reference"].startswith("AA-")
    assert r["time"] == s["time"]


def test_a_booked_slot_disappears_from_availability():
    before = booking.get_slots(days=2)["count"]
    s = first_slot()
    booking.book_slot(s["slot_id"], "AP31AB1234", "service")
    after = booking.get_slots(days=2)["count"]
    assert after == before - 1


def test_double_booking_is_refused():
    s = first_slot()
    assert booking.book_slot(s["slot_id"], "AP31AB1234", "first")["ok"]
    second = booking.book_slot(s["slot_id"], "AP31CD5678", "second")
    assert not second["ok"]
    assert "taken" in second["error"].lower()


def test_invented_slot_id_is_refused():
    r = booking.book_slot("not-a-slot", "AP31AB1234", "service")
    assert not r["ok"]


def test_registration_is_required():
    s = first_slot()
    r = booking.book_slot(s["slot_id"], "", "service")
    assert not r["ok"]


def test_registration_is_normalised():
    s = first_slot()
    r = booking.book_slot(s["slot_id"], "ap 31 ab 1234", "service")
    assert r["registration"] == "AP31AB1234"


def test_cancelling_frees_the_slot():
    s = first_slot()
    ref = booking.book_slot(s["slot_id"], "AP31AB1234", "service")["reference"]
    before = booking.get_slots(days=2)["count"]
    assert booking.cancel_booking(ref)["ok"]
    assert booking.get_slots(days=2)["count"] == before + 1


def test_cancelling_twice_is_refused():
    s = first_slot()
    ref = booking.book_slot(s["slot_id"], "AP31AB1234", "service")["reference"]
    booking.cancel_booking(ref)
    assert not booking.cancel_booking(ref)["ok"]


def test_unknown_reference_is_refused():
    assert not booking.cancel_booking("AA-NOPE00")["ok"]
    assert not booking.get_booking("AA-NOPE00")["ok"]


def test_past_dates_are_refused():
    assert not booking.get_slots(on_date="2020-01-01")["ok"]


def test_unparseable_date_is_refused():
    r = booking.get_slots(on_date="whenever")
    assert not r["ok"]
    assert "could not understand" in r["error"].lower()


def test_sunday_returns_no_slots_with_a_reason():
    r = booking.get_slots(on_date="sunday")
    assert r["ok"]
    assert r["slots"] == []
    assert "closed" in r["note"].lower()


def test_safety_ticket_promises_an_hour():
    r = booking.create_ticket("spongy brakes", "safety", "AP31AB1234")
    assert r["ok"]
    assert r["reference"].startswith("TK-")
    assert "1 hour" in r["message"]


def test_unknown_urgency_falls_back_to_normal():
    r = booking.create_ticket("something", "extremely-urgent")
    assert r["urgency"] == "normal"


def _two_slots_same_day():
    slots = booking.get_slots(days=1)["slots"]
    return slots[0], slots[1]


def test_a_vehicle_cannot_be_booked_twice_on_one_day():
    """Local replay, 19 Sep: asked to 'move it to 3:30', the agent booked 3:30 and
    left 2:00 in place. Two out of two runs, despite being told to cancel."""
    a, b = _two_slots_same_day()
    first = booking.book_slot(a["slot_id"], "AP31BP2133", "wiper blades")
    second = booking.book_slot(b["slot_id"], "ap 31 bp 2133", "wiper blades")
    assert not second["ok"]
    assert first["reference"] in second["error"], "the agent needs the reference to cancel it"
    assert "move_service_booking" in second["error"]


def test_moving_a_booking_is_cancel_then_book():
    a, b = _two_slots_same_day()
    ref = booking.book_slot(a["slot_id"], "AP31BP2133", "wiper blades")["reference"]
    assert booking.cancel_booking(ref)["ok"]
    moved = booking.book_slot(b["slot_id"], "AP31BP2133", "wiper blades")
    assert moved["ok"]


def test_the_same_vehicle_can_book_another_day():
    slots = booking.get_slots(days=2)["slots"]
    day1 = slots[0]
    day2 = next(s for s in slots if s["date"] != day1["date"])
    assert booking.book_slot(day1["slot_id"], "AP31BP2133", "service")["ok"]
    assert booking.book_slot(day2["slot_id"], "AP31BP2133", "service")["ok"]


def test_different_vehicles_can_book_the_same_day():
    a, b = _two_slots_same_day()
    assert booking.book_slot(a["slot_id"], "AP31BP2133", "service")["ok"]
    assert booking.book_slot(b["slot_id"], "AP31CD5678", "service")["ok"]


# ---------------------------------------------------------------- moving
#
# Moving used to be cancel then book, done by the agent. In local replays it
# stopped between the two - once to ask, once because the new time was taken -
# and the customer was left with nothing. move_booking checks first, then changes.


def test_moving_keeps_the_reference_and_changes_the_time():
    a, b = _two_slots_same_day()
    ref = booking.book_slot(a["slot_id"], "AP31BP2133", "wipers")["reference"]
    moved = booking.move_booking(ref, b["slot_id"])
    assert moved["ok"]
    assert moved["reference"] == ref
    assert booking.get_booking(ref)["time"] == b["time"]


def test_moving_frees_the_old_slot():
    a, b = _two_slots_same_day()
    ref = booking.book_slot(a["slot_id"], "AP31BP2133", "wipers")["reference"]
    booking.move_booking(ref, b["slot_id"])
    free = {s["slot_id"] for s in booking.get_slots(days=1)["slots"]}
    assert a["slot_id"] in free
    assert b["slot_id"] not in free


def test_moving_to_a_taken_slot_changes_nothing():
    """The exact failure: new time taken, and the original must survive."""
    a, b = _two_slots_same_day()
    ref = booking.book_slot(a["slot_id"], "AP31BP2133", "wipers")["reference"]
    booking.book_slot(b["slot_id"], "AP31ZZ0001", "someone else")
    moved = booking.move_booking(ref, b["slot_id"])
    assert not moved["ok"]
    assert "unchanged" in moved["error"]
    kept = booking.get_booking(ref)
    assert kept["status"] == "confirmed" and kept["slot_id"] == a["slot_id"]


def test_moving_to_an_invented_slot_changes_nothing():
    a, _ = _two_slots_same_day()
    ref = booking.book_slot(a["slot_id"], "AP31BP2133", "wipers")["reference"]
    assert not booking.move_booking(ref, "slot_10:30_2020-09-21")["ok"]
    assert booking.get_booking(ref)["slot_id"] == a["slot_id"]


def test_a_cancelled_or_unknown_booking_cannot_be_moved():
    a, b = _two_slots_same_day()
    ref = booking.book_slot(a["slot_id"], "AP31BP2133", "wipers")["reference"]
    booking.cancel_booking(ref)
    assert not booking.move_booking(ref, b["slot_id"])["ok"]
    assert not booking.move_booking("AA-NOPE00", b["slot_id"])["ok"]
