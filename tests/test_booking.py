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
