"""Triage by probabilities instead of by JSON, and the choice between them.

Jev is off unless TRIAGE_BACKEND=jev, and it falls back to the agent for
anything it cannot answer. These tests are mostly about the falling back: a
classifier that is down must cost a little latency, not a customer's message.
"""

import json

import pytest

from services.orchestrator import jev_triage, typesafe
from services.orchestrator import router as _router

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}


def answered(**probs):
    """A Jev reply with the four nouls."""
    full = {"needs_diagnostics": 0.0, "needs_booking": 0.0, "needs_escalation": 0.0, "safety": 0.0}
    full.update(probs)
    return {"answers": {k: {"type": "noul", "noul": v} for k, v in full.items()},
            "usage": {"input_tokens": 1316, "output_tokens": 20}}


@pytest.fixture
def jev_on(monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    return monkeypatch


# ------------------------------------------------------------ the switch


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TRIAGE_BACKEND", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    assert not jev_triage.configured()


def test_on_only_with_a_key(monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "jev")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert not jev_triage.configured(), "switched on but nothing to call"


def test_on_when_both_are_set(jev_on):
    assert jev_triage.configured()


# ------------------------------------------------------ what it decides


def test_probabilities_become_a_route(jev_on, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.98, needs_booking=0.97))
    c = jev_triage.classify("P0420 is showing, can I book in for Saturday?")
    assert c.intents == ["diagnostics", "booking"]
    assert not c.safety


def test_nothing_over_the_bar_is_other_not_empty(jev_on, monkeypatch):
    """TriageDecision.route() then sends it to diagnostics, as it always has."""
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered())
    assert jev_triage.classify("what time do you close").intents == ["other"]


def test_the_safety_bar_is_higher_than_the_intent_bar(jev_on, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.9, safety=0.65))
    assert not jev_triage.classify("the aircon rattles").safety, "0.65 is under the 0.7 safety cut"


def test_the_numbers_are_kept_for_the_trace(jev_on, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.98, safety=0.04))
    c = jev_triage.classify("my wipers squeak")
    assert c.probabilities["needs_diagnostics"] == 0.98
    assert "diagnostics 0.98" in c.reason()
    assert c.tokens == 1316


# ------------------------------------------------------ falling back


@pytest.mark.parametrize("boom", [
    typesafe.TypeSafeUnavailable("no key"),
    typesafe.TypeSafeUnavailable("api.typesafe.ai is rate limiting this key (HTTP 429)"),
    RuntimeError("something nobody predicted"),
])
def test_anything_going_wrong_falls_back(jev_on, monkeypatch, boom):
    def raise_it(*a, **k):
        raise boom

    monkeypatch.setattr(typesafe, "ask", raise_it)
    assert jev_triage.classify("my brakes feel spongy") is None


def test_a_missing_answer_falls_back(jev_on, monkeypatch):
    """Three of four answered is not an answer: a missing safety noul read as
    0.0 would route a brake failure as an ordinary question."""
    half = answered(needs_diagnostics=0.9)
    del half["answers"]["safety"]
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: half)
    assert jev_triage.classify("my brakes feel spongy") is None


def test_the_router_uses_the_agent_when_jev_returns_nothing(jev_on, monkeypatch):
    monkeypatch.setattr(jev_triage, "classify", lambda *a, **k: None)
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    asked = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", **_):
        asked.append(agent_name)
        from services.orchestrator.runner import TurnResult
        t = TurnResult(agent_name=agent_name, status="completed")
        t.answer = json.dumps({"intents": ["diagnostics"], "safety": False}) if agent_name == "triage" else "Answer."
        return t

    monkeypatch.setattr(_router, "ask", ask)
    r = _router.handle(None, "my wipers squeak", agent_ids=ALL_IDS)
    assert "triage" in asked, "the agent did the classifying"
    assert r.decision.backend == "agent"


# ------------------------------------------------- the backstops still run


def test_the_keyword_net_still_catches_what_jev_scores_low(jev_on, monkeypatch):
    """Jev scored a routine Hinglish complaint at 0.75 and the docs warn that
    non-English accuracy is lower. The regex does not care what language it is
    reading, so it stays."""
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.9, safety=0.02))
    c = jev_triage.classify("my brakes feel spongy")
    d = _router._decision_from_jev(c, "my brakes feel spongy")
    assert d.safety and d.safety_source == "keyword"
    assert "escalation" in d.route()


def test_both_agreeing_is_recorded_as_both(jev_on, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.9, safety=0.95))
    c = jev_triage.classify("my brakes feel spongy")
    assert _router._decision_from_jev(c, "my brakes feel spongy").safety_source == "both"


def test_jev_alone_is_recorded_as_jev(jev_on, monkeypatch):
    """A safety issue no keyword covers - the case only a model can catch."""
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.9, safety=0.93))
    c = jev_triage.classify("something feels badly wrong")
    assert _router._decision_from_jev(c, "something feels badly wrong").safety_source == "jev"


def test_the_booking_backstop_still_runs(jev_on, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.9, needs_booking=0.1))
    c = jev_triage.classify("book at 2 pm , viper blades")
    d = _router._decision_from_jev(c, "book at 2 pm , viper blades")
    assert "booking" in d.route() and d.booking_added_by_keyword


# ------------------------------------------------------ the registration


@pytest.mark.parametrize("text,expected", [
    ("ap31bd1213", "AP31BD1213"),
    ("my reg is AP 31 BD 1213", "AP31BD1213"),
    ("AP31BP2133 is the number", "AP31BP2133"),
])
def test_a_registration_is_taken_by_regex_not_by_the_model(text, expected):
    """Jev returns typed answers, so it cannot hand back a value it was never
    given options for. A regex is better at this anyway, being exact."""
    assert jev_triage.registration_in(text, None) == expected


def test_a_registration_from_earlier_in_the_conversation_is_found():
    history = [{"role": "customer", "text": "ap31bd1213"},
               {"role": "assistant", "text": "Thank you. Which day suits you?"}]
    assert jev_triage.registration_in("tomorrow please", history) == "AP31BD1213"


def test_no_registration_is_none():
    assert jev_triage.registration_in("what does P0420 mean", None) is None


# ------------------------------------------------------------ the state


def test_the_state_is_named_fields():
    state = jev_triage.state_for("1 pm", [
        {"role": "customer", "text": "book me in"},
        {"role": "assistant", "text": "Which time suits you?"},
    ])
    assert state["new_message"] == "1 pm"
    assert state["assistant_last_asked"] == "Which time suits you?"


def test_every_question_points_at_the_message_it_judges():
    for name, q in jev_triage.QUESTIONS.items():
        assert "`new_message`" in q["instructions"]["question"], name


# ----------------------------------------------- a fallback must be visible


@pytest.fixture
def fresh_stats():
    jev_triage.STATS.update(answered=0, fell_back=0)
    yield jev_triage.STATS
    jev_triage.STATS.update(answered=0, fell_back=0)


def test_an_answer_is_counted(jev_on, monkeypatch, fresh_stats):
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: answered(needs_diagnostics=0.9))
    jev_triage.classify("my wipers squeak")
    assert fresh_stats == {"answered": 1, "fell_back": 0}


def test_a_fallback_is_counted(jev_on, monkeypatch, fresh_stats):
    """Silent by design - the customer is answered either way - which is exactly
    why it needs counting. A revoked key would otherwise send every message back
    to the agent with nothing to say so."""
    def down(*a, **k):
        raise typesafe.TypeSafeUnavailable("HTTP 401: invalid api key")

    monkeypatch.setattr(typesafe, "ask", down)
    jev_triage.classify("my wipers squeak")
    assert fresh_stats == {"answered": 0, "fell_back": 1}


def test_an_unanswered_question_is_counted_as_a_fallback(jev_on, monkeypatch, fresh_stats):
    half = answered(needs_diagnostics=0.9)
    del half["answers"]["safety"]
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: half)
    jev_triage.classify("my brakes feel spongy")
    assert fresh_stats["fell_back"] == 1


def test_metrics_report_both(fresh_stats):
    from services.orchestrator import app as app_module

    fresh_stats.update(answered=40, fell_back=2)
    m = app_module.metrics()
    assert m["jev_answered"] == 40 and m["jev_fell_back"] == 2
    assert m["triage_backend"] in ("agent", "jev")
