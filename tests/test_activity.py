"""What the panel is told while a message is being answered.

The page draws from these events, so they are the panel's contract: if an agent
starts and never finishes, a card spins for ever; if the pre-search is not
reported, the panel says diagnostics answered without looking anything up, which
is the one claim this project cares most about being able to make honestly.
"""

import json

import pytest

from services.orchestrator import azure_http
from services.orchestrator import router as _router
from services.orchestrator.runner import TurnResult, emit

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}


def collector():
    seen = []
    return seen, seen.append


def agents_answering(intents=("diagnostics",), safety=False):
    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", on_event=None, **_):
        emit(on_event, kind="agent", name=agent_name, state="working")
        t = TurnResult(agent_name=agent_name, status="completed")
        t.answer = json.dumps({"intents": list(intents), "safety": safety}) if agent_name == "triage" \
            else f"answer from {agent_name}"
        t.prompt_tokens, t.duration_ms = 500, 1200
        emit(on_event, kind="agent", name=agent_name, state="done", ms=t.duration_ms,
             tokens=t.prompt_tokens, ok=True)
        return t
    return ask


@pytest.fixture(autouse=True)
def fake_azure(monkeypatch):
    """No Azure. Tests that need a different route replace `ask` again."""
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", agents_answering())


# ----------------------------------------------------------------- emit()


def test_emit_with_nobody_listening_does_nothing():
    emit(None, kind="agent", name="triage")


def test_a_panel_that_throws_does_not_break_the_answer():
    """This is called from inside the agent threads. A request must not fail
    because the drawing went wrong."""
    def broken(event):
        raise RuntimeError("the page is gone")

    emit(broken, kind="agent", name="triage")  # must not raise


# -------------------------------------------------------------- the events


def test_the_route_is_announced_before_any_specialist_runs():
    seen, on_event = collector()
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_event=on_event)

    kinds = [e["kind"] for e in seen]
    assert kinds[0] == "route"
    route = seen[0]
    assert route["agents"] == ["diagnostics"]
    assert route["triage_skipped"] is True


def test_the_pre_search_is_reported_as_a_search():
    """It is a real search, run before the agent. If the panel did not show it,
    diagnostics would look like it answered from nothing - finding 1."""
    seen, on_event = collector()
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_event=on_event)

    searches = [e for e in seen if e["kind"] == "tool" and e["name"] == "search_service_docs"]
    assert [e["state"] for e in searches] == ["running", "done"]
    assert searches[0]["agent"] == "diagnostics"
    assert searches[1]["failed"] is False


def test_a_failed_pre_search_says_so(monkeypatch):
    def boom(**kw):
        raise RuntimeError("search is down")

    monkeypatch.setattr(_router.tools, "search_service_docs", boom)
    seen, on_event = collector()
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_event=on_event)

    done = [e for e in seen if e["kind"] == "tool" and e["state"] == "done"]
    assert done and done[0]["failed"] is True


def test_every_agent_that_starts_also_finishes(monkeypatch):
    """A card that never gets its 'done' spins for ever."""
    monkeypatch.setattr(_router, "ask", agents_answering(intents=("diagnostics", "booking")))
    seen, on_event = collector()
    _router.handle(None, "P0420 is showing, can I book in for Saturday?",
                   agent_ids=ALL_IDS, on_event=on_event)

    started = [e["name"] for e in seen if e["kind"] == "agent" and e["state"] == "working"]
    finished = [e["name"] for e in seen if e["kind"] == "agent" and e["state"] != "working"]
    assert sorted(started) == sorted(finished)
    assert set(started) == {"triage", "diagnostics", "booking"}


def test_an_agent_that_throws_still_reports_that_it_stopped(monkeypatch):
    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", on_event=None, **_):
        emit(on_event, kind="agent", name=agent_name, state="working")
        if agent_name == "diagnostics":
            emit(on_event, kind="agent", name=agent_name, state="failed")
            raise azure_http.Throttled(429, "no quota", "POST", "https://x", retry_after=20.0)
        emit(on_event, kind="agent", name=agent_name, state="done", ok=True)
        t = TurnResult(agent_name=agent_name, status="completed")
        t.answer = json.dumps({"intents": ["diagnostics"], "safety": False})
        return t

    monkeypatch.setattr(_router, "ask", ask)
    seen, on_event = collector()
    _router.handle(None, "my wipers squeak", agent_ids=ALL_IDS, on_event=on_event)

    diagnostics = [e for e in seen if e.get("name") == "diagnostics" and e["kind"] == "agent"]
    assert [e["state"] for e in diagnostics] == ["working", "failed"]


def test_a_safety_route_says_so(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents_answering(intents=("diagnostics",), safety=True))
    seen, on_event = collector()
    _router.handle(None, "my brakes have stopped working", agent_ids=ALL_IDS, on_event=on_event)

    route = next(e for e in seen if e["kind"] == "route")
    assert route["safety"] is True
    assert "escalation" in route["agents"]


def test_a_standing_ticket_is_explained_rather_than_just_missing(monkeypatch):
    """Escalation's card greys out. Without this the panel would look as though
    the safety route had quietly stopped working."""
    monkeypatch.setattr(_router, "ask", agents_answering(intents=("booking",), safety=True))
    seen, on_event = collector()
    _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, on_event=on_event,
                   history=[{"role": "customer", "text": "my brakes have failed"},
                            {"role": "assistant", "text": "Ticket reference TK-013051."}])

    route = next(e for e in seen if e["kind"] == "route")
    assert route["ticket_stands"] == "TK-013051"
    assert "escalation" not in route["agents"]


def test_small_talk_says_no_agent_ran():
    seen, on_event = collector()
    _router.handle(None, "hello", agent_ids=ALL_IDS, on_event=on_event)

    route = next(e for e in seen if e["kind"] == "route")
    assert route["agents"] == []
    assert "without agents" in route["reason"]
    assert not [e for e in seen if e["kind"] == "agent"]


def test_a_cached_answer_says_it_cost_nothing(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents_answering())
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    seen, on_event = collector()
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_event=on_event)

    route = next(e for e in seen if e["kind"] == "route")
    assert route["cached"] is True
    assert route["agents"] == []
    assert not [e for e in seen if e["kind"] == "agent"], "no agent ran, so no card should light up"


def test_events_are_plain_json(monkeypatch):
    """They go down an SSE stream as JSON. Anything that cannot be serialised
    would break the whole connection, not just one event."""
    monkeypatch.setattr(_router, "ask", agents_answering(intents=("diagnostics", "booking")))
    seen, on_event = collector()
    _router.handle(None, "P0420 is showing, can I book in for Saturday?",
                   agent_ids=ALL_IDS, on_event=on_event)

    for event in seen:
        json.dumps(event)
