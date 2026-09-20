"""Tests for the tool layer.

The contract that matters: execute() must NEVER raise. A tool that throws kills
the whole run; a tool that returns an error string lets the model tell the
customer something honest.
"""

import os
import pathlib
import tempfile

import pytest

os.environ.setdefault("BOOKING_STORE", str(pathlib.Path(tempfile.gettempdir()) / "aa_tools_bookings.json"))
os.environ.setdefault("TICKET_STORE", str(pathlib.Path(tempfile.gettempdir()) / "aa_tools_tickets.json"))

from services.orchestrator import booking, tools  # noqa: E402


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    monkeypatch.setattr(booking, "STORE", tmp_path / "bookings.json")
    monkeypatch.setattr(booking, "TICKETS", tmp_path / "tickets.json")
    yield


def test_every_schema_has_a_handler():
    assert set(tools.SCHEMAS) == set(tools.HANDLERS)


def test_schemas_are_well_formed():
    for name, s in tools.SCHEMAS.items():
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] == name
        assert fn["description"], f"{name} has no description - the model uses this to decide when to call it"
        params = fn["parameters"]
        assert params["type"] == "object"
        for req in params["required"]:
            assert req in params["properties"], f"{name} requires '{req}' but does not define it"


def test_schemas_for_rejects_unknown_names():
    with pytest.raises(KeyError):
        tools.schemas_for(["search_service_docs", "make_tea"])


def test_unknown_tool_returns_an_error_not_an_exception():
    out = tools.execute("make_tea", "{}")
    assert out.startswith("ERROR:")


def test_broken_json_arguments_return_an_error():
    out = tools.execute("get_available_slots", "{not json")
    assert out.startswith("ERROR:")


def test_wrong_arguments_return_an_error():
    out = tools.execute("book_service_slot", '{"wrong": "shape"}')
    assert out.startswith("ERROR:")


def test_non_object_arguments_return_an_error():
    out = tools.execute("get_available_slots", '["a", "list"]')
    assert out.startswith("ERROR:")


def test_get_slots_returns_json():
    out = tools.execute("get_available_slots", "{}")
    assert '"ok": true' in out.lower()


def test_booking_through_the_tool_layer():
    import json

    slots = json.loads(tools.execute("get_available_slots", "{}"))
    slot_id = slots["slots"][0]["slot_id"]
    out = json.loads(
        tools.execute(
            "book_service_slot",
            json.dumps({"slot_id": slot_id, "registration": "AP31AB1234", "issue": "noise",
                        "customer_name": "Krishna", "customer_email": "k@example.com",
                        "customer_phone": "9876543210"}),
        )
    )
    assert out["ok"]
    assert out["reference"].startswith("AA-")


def test_the_booking_backend_is_chosen_by_configuration(monkeypatch):
    """BOOKING_BACKEND=zoho puts bookings in the workshop's real calendar."""
    import importlib

    monkeypatch.setenv("BOOKING_BACKEND", "zoho")
    reloaded = importlib.reload(tools)
    try:
        assert reloaded.BACKEND.__name__.endswith("zoho_bookings")
        monkeypatch.setenv("BOOKING_BACKEND", "file")
        assert importlib.reload(tools).BACKEND.__name__.endswith("booking")
    finally:
        monkeypatch.delenv("BOOKING_BACKEND", raising=False)
        importlib.reload(tools)


def test_both_backends_offer_the_same_booking_functions():
    from services.orchestrator import booking as file_backend
    from services.orchestrator import zoho_bookings

    for name in ("get_slots", "book_slot", "get_booking", "move_booking", "cancel_booking"):
        assert callable(getattr(file_backend, name)) and callable(getattr(zoho_bookings, name)), name


@pytest.mark.parametrize("missing,words", [
    ("customer_name", "name"), ("customer_email", "email address"), ("customer_phone", "phone number"),
])
def test_a_booking_without_the_customers_details_is_refused(missing, words):
    """Live app, 20 Sep: told only in the prompt, the agent booked without a
    phone number. The workshop's calendar rejects that, so the customer would
    have been promised an appointment that does not exist."""
    import json

    slots = json.loads(tools.execute("get_available_slots", "{}"))
    args = {"slot_id": slots["slots"][0]["slot_id"], "registration": "AP31AB1234", "issue": "service",
            "customer_name": "Krishna", "customer_email": "k@example.com", "customer_phone": "9876543210"}
    args[missing] = "   "
    out = json.loads(tools.execute("book_service_slot", json.dumps(args)))
    assert out["ok"] is False
    assert words in out["error"]
    assert "reference" not in out


def test_a_booking_with_every_detail_goes_through():
    import json

    slots = json.loads(tools.execute("get_available_slots", "{}"))
    out = json.loads(tools.execute("book_service_slot", json.dumps({
        "slot_id": slots["slots"][0]["slot_id"], "registration": "AP31AB1234", "issue": "service",
        "customer_name": "Krishna", "customer_email": "k@example.com", "customer_phone": "9876543210"})))
    assert out["ok"] and out["reference"].startswith("AA-")
