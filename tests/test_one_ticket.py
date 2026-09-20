"""One conversation, one ticket.

From the live app on 20 September. "i have a problem with my gear shifting" was
flagged as a safety issue, and it stayed flagged for the rest of the
conversation - triage is told to judge the flag on the new message alone, and it
did not. Escalation therefore ran on every turn and raised a ticket on every
turn:

    "i have a problem with my gear shifting"  -> TK-013051
    "i need to book appointment"              -> TK-892738
    "tomoroow"                                -> TK-953064

Three advisors' worth of work for one car, and the customer was given three
different references for one problem.
"""

import json

import pytest

from services.orchestrator import router as _router
from services.orchestrator.runner import TurnResult

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}

RAISED = "I have raised a ticket for a service advisor. The ticket reference is TK-013051. Someone will contact you within one hour."


def agents(intents=("diagnostics", "escalation"), safety=True):
    """Fake agents that record who was asked."""
    called = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", **_):
        called.append(agent_name)
        t = TurnResult(agent_name=agent_name, status="completed")
        if agent_name == "triage":
            t.answer = json.dumps({"intents": list(intents), "safety": safety})
        elif agent_name == "escalation":
            t.answer = "I have raised a ticket. The reference is TK-999999."
        else:
            t.answer = f"answer from {agent_name}"
        return t

    ask.called = called
    return ask


def conversation(*, reply=RAISED):
    return [
        {"role": "customer", "text": "i have a problem with my gear shifting"},
        {"role": "assistant", "text": reply},
    ]


@pytest.fixture(autouse=True)
def documents(monkeypatch):
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")


# ------------------------------------------------------- finding the ticket


def test_a_reference_in_the_conversation_is_found():
    assert _router.ticket_already_raised(conversation()) == "TK-013051"


def test_a_conversation_with_no_ticket_has_none():
    assert _router.ticket_already_raised([
        {"role": "customer", "text": "what does P0420 mean"},
        {"role": "assistant", "text": "The catalytic converter is worn."},
    ]) is None


def test_no_history_has_none():
    assert _router.ticket_already_raised(None) is None
    assert _router.ticket_already_raised([]) is None


def test_a_reference_the_customer_typed_does_not_count():
    """Only what we told them. Otherwise anyone could turn escalation off by
    typing a ticket number."""
    assert _router.ticket_already_raised([
        {"role": "customer", "text": "my ticket is TK-013051, my brakes have failed"},
    ]) is None


def test_a_long_answer_does_not_lose_its_reference():
    """_recent trims turns to 600 characters. A reference past that would mean a
    second ticket for the same problem."""
    padded = "The gearbox needs a look. " * 40 + " The ticket reference is TK-013051."
    assert len(padded) > 600
    assert _router.ticket_already_raised([{"role": "assistant", "text": padded}]) == "TK-013051"


# ------------------------------------------------------- one ticket per problem


def test_escalation_does_not_run_again_once_a_ticket_stands(monkeypatch):
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, history=conversation())

    assert "escalation" not in ask.called, "a second advisor was called for the same car"
    assert "TK-013051" in r.reply, "the customer is told the reference they already have"
    assert "TK-999999" not in r.reply, "and not a new one"


def test_the_first_time_escalation_does_run(monkeypatch):
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)

    _router.handle(None, "i have a problem with my gear shifting", agent_ids=ALL_IDS)

    assert "escalation" in ask.called


def test_the_other_specialists_still_run(monkeypatch):
    """Skipping escalation must not skip the answer the customer asked for."""
    ask = agents(intents=("booking", "escalation"))
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, history=conversation())

    assert "booking" in ask.called
    assert "answer from booking" in r.reply


def test_the_standing_ticket_comes_before_the_rest(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents(intents=("booking", "escalation")))
    r = _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, history=conversation())
    assert r.reply.index("TK-013051") < r.reply.index("answer from booking")


def test_asking_for_a_person_again_is_answered_without_an_agent(monkeypatch):
    """The whole route was escalation, so there is nothing else to say - but the
    customer must still get an answer, and the safety warning with it."""
    ask = agents(intents=("escalation",))
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "can I speak to someone please", agent_ids=ALL_IDS, history=conversation())

    assert ask.called == ["triage"], "no specialist needed to run"
    assert "TK-013051" in r.reply
    assert "Do not drive" in r.reply, "it is still a safety conversation"
    assert "could not get an answer" not in r.reply


def test_a_standing_ticket_without_a_safety_flag_is_just_the_reference(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents(intents=("escalation",), safety=False))
    r = _router.handle(None, "has anyone looked at this yet", agent_ids=ALL_IDS,
                       history=conversation())
    assert r.reply == _router.TICKET_STANDS.format(reference="TK-013051")


def test_skipping_escalation_saves_the_turn(monkeypatch):
    """It cost ~3,000 tokens and ~5s of every turn of that conversation."""
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)

    first = _router.handle(None, "i have a problem with my gear shifting", agent_ids=ALL_IDS)
    later = _router.handle(None, "tomorrow please", agent_ids=ALL_IDS, history=conversation())

    assert "escalation" in first.agents_used
    assert "escalation" not in later.agents_used
