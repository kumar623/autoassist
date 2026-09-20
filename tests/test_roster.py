"""The agents panel is read from the definitions, not written out again.

A hand-copied panel is a panel that goes quietly out of date and then tells an
interviewer something that is not true. These tests are mostly about that: what
the page shows has to be what the deploy script deploys.
"""

import pytest

from services.orchestrator import roster, tools


@pytest.fixture(autouse=True)
def fresh():
    roster._ROSTER.clear()
    yield
    roster._ROSTER.clear()


def test_all_four_agents_are_listed_in_the_order_they_run():
    assert [a["name"] for a in roster.load()["agents"]] == ["triage", "diagnostics", "booking", "escalation"]


def test_every_agent_says_what_it_is_for():
    for a in roster.load()["agents"]:
        assert a["purpose"].endswith("."), a["name"]
        assert a["role"], a["name"]
        assert a["model"], a["name"]


def test_the_tools_are_the_ones_the_agent_is_actually_given():
    """Not a list typed into the page. If deploy_agents.py gives diagnostics a
    new tool, the panel shows it without anyone remembering to."""
    import json
    import pathlib

    for a in roster.load()["agents"]:
        deployed = json.loads(pathlib.Path(f"agents/definitions/{a['name']}.json").read_text())
        assert [t["name"] for t in a["tools"]] == (deployed.get("tools") or [])


def test_triage_has_no_tools_and_returns_json():
    triage = roster.load()["agents"][0]
    assert triage["tools"] == []
    assert triage["returns"] == "JSON"


def test_diagnostics_can_hand_over_as_well_as_search():
    """raise_ticket under diagnostics is not a mistake: it hands over when the
    library does not cover something important."""
    diagnostics = roster.load()["agents"][1]
    assert {t["name"] for t in diagnostics["tools"]} == {"search_service_docs", "raise_ticket"}


def test_every_tool_explains_itself():
    for a in roster.load()["agents"]:
        for t in a["tools"]:
            assert t["does"], f"{a['name']}.{t['name']}"
            assert t["does"].endswith(".")
            assert len(t["does"]) < 160, "the panel is narrow"


def test_a_tool_description_comes_from_the_schema_the_agent_is_given():
    booking = roster.load()["agents"][2]
    slots = next(t for t in booking["tools"] if t["name"] == "get_available_slots")
    real = tools.SCHEMAS["get_available_slots"]["function"]["description"]
    assert slots["does"].rstrip(".") in real


def test_an_agent_that_is_not_deployed_is_shown_as_such():
    """/ready complains about exactly this, and hiding it on the page would make
    a half-deployed system look complete."""
    partial = roster.load({"triage": "x", "diagnostics": "y"})
    state = {a["name"]: a["deployed"] for a in partial["agents"]}
    assert state == {"triage": True, "diagnostics": True, "booking": False, "escalation": False}


def test_a_missing_definition_file_does_not_break_the_page(monkeypatch, tmp_path):
    monkeypatch.setattr(roster, "DEFINITIONS", tmp_path)
    roster._ROSTER.clear()
    assert roster.load() == {"agents": [], "count": 0}


def test_a_broken_definition_file_is_skipped(monkeypatch, tmp_path):
    (tmp_path / "triage.json").write_text("{ this is not json")
    (tmp_path / "diagnostics.json").write_text('{"name": "diagnostics", "model": "m", "tools": []}')
    monkeypatch.setattr(roster, "DEFINITIONS", tmp_path)
    roster._ROSTER.clear()

    loaded = roster.load()
    assert [a["name"] for a in loaded["agents"]] == ["diagnostics"]


# ------------------------------------------------- where a tool call really goes


def test_every_tool_says_where_it_goes():
    """"booking calls get_available_slots" says nothing about the call leaving
    this process and landing in the workshop's real calendar."""
    for a in roster.load()["agents"]:
        for t in a["tools"]:
            assert t["backend"], f"{a['name']}.{t['name']}"


def test_the_search_tool_goes_to_azure_search():
    diagnostics = roster.load()["agents"][1]
    search = next(t for t in diagnostics["tools"] if t["name"] == "search_service_docs")
    assert search["backend"] == roster.SEARCH_BACKEND


def test_booking_tools_say_mcp_when_zoho_is_the_backend(monkeypatch):
    monkeypatch.setenv("BOOKING_BACKEND", "zoho")
    booking = roster.load()["agents"][2]
    assert all("MCP" in t["backend"] for t in booking["tools"])


def test_booking_tools_say_local_when_the_file_backend_is_in_use(monkeypatch):
    """The panel has to follow BOOKING_BACKEND, not assume it."""
    monkeypatch.setenv("BOOKING_BACKEND", "file")
    booking = roster.load()["agents"][2]
    assert all(t["backend"] == roster.LOCAL_BACKEND for t in booking["tools"])


def test_the_backend_is_not_cached_from_an_earlier_call(monkeypatch):
    """The definitions are cached for ten minutes; the backend is an env var and
    can change under a running process when the revision changes."""
    monkeypatch.setenv("BOOKING_BACKEND", "zoho")
    assert "MCP" in roster.load()["agents"][2]["tools"][0]["backend"]
    monkeypatch.setenv("BOOKING_BACKEND", "file")
    assert "MCP" not in roster.load()["agents"][2]["tools"][0]["backend"]
