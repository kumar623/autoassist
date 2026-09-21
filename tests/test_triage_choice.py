"""Choosing the classifier per message, and watching the other one read it too.

The page lets a visitor pick Jev or the triage agent for one message, or tick
"compare both". The choice is the demo, so it has to be real: the classifier
picked is the one that routes, and when it cannot be, the reply says so.

The comparison is the part with the sharper rule. It is display only - it must
never change the route, the reply, the tickets or the answer cache - and it
must never fail the request it is shown beside. Most of these tests are about
that line.
"""

import contextlib
import json
import pathlib
import sys
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.orchestrator import app as app_module
from services.orchestrator import azure_http, jev_triage, limits, roster, typesafe
from services.orchestrator import router as _router
from services.orchestrator.app import ChatRequest
from services.orchestrator.runner import TurnResult, emit

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}
ROOT = pathlib.Path(__file__).resolve().parents[1]

BRAKE_FLUID = "How often should brake fluid be changed?"


def jev_says(**probs):
    """A fake typesafe.ask that answers the four nouls, and remembers being asked."""
    full = {"needs_diagnostics": 0.0, "needs_booking": 0.0, "needs_escalation": 0.0, "safety": 0.0}
    full.update(probs)
    answer = {"answers": {k: {"type": "noul", "noul": v} for k, v in full.items()},
              "usage": {"input_tokens": 1316, "output_tokens": 20}}
    asked = []

    def ask(state, questions, **kwargs):
        asked.append(state["new_message"])
        return answer

    ask.asked = asked
    return ask


def agents(intents=("diagnostics",), safety=False):
    """A fake runner.ask: triage writes the given JSON, specialists answer.

    Emits the same panel events the real one does when it is given somewhere to
    send them, so a test can see which turns lit up the page.
    """
    called = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", on_event=None, **_):
        called.append(agent_name)
        emit(on_event, kind="agent", name=agent_name, state="working")
        t = TurnResult(agent_name=agent_name, status="completed")
        if agent_name == "triage":
            t.answer = json.dumps({"intents": list(intents), "safety": safety, "reason": "the model's reason"})
            t.prompt_tokens, t.completion_tokens, t.duration_ms = 880, 40, 2100
        else:
            t.answer = f"answer from {agent_name}"
            t.prompt_tokens, t.completion_tokens, t.duration_ms = 4000, 200, 5000
        emit(on_event, kind="agent", name=agent_name, state="done", ms=t.duration_ms,
             tokens=t.prompt_tokens + t.completion_tokens, ok=True)
        return t

    ask.called = called
    return ask


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No Azure and no TypeSafe. Jev is not asked unless a test says it may be."""
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", agents())
    monkeypatch.setattr(typesafe, "ask", lambda *a, **k: pytest.fail("Jev should not have been asked"))


@pytest.fixture
def jev_key(monkeypatch):
    """A key, and nothing else: the server's default is still the agent."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")


@pytest.fixture
def jev_default(monkeypatch, jev_key):
    monkeypatch.setenv("TRIAGE_BACKEND", "jev")


@pytest.fixture
def fresh_stats():
    jev_triage.STATS.update(answered=0, fell_back=0)
    yield jev_triage.STATS
    jev_triage.STATS.update(answered=0, fell_back=0)


def handle(message="my wipers squeak", **kw):
    seen = []
    out = _router.handle(None, message, agent_ids=ALL_IDS, on_event=seen.append, **kw)
    return out, seen


def the(seen, kind):
    found = [e for e in seen if e["kind"] == kind]
    assert len(found) <= 1, f"more than one {kind} event"
    return found[0] if found else None


# ------------------------------------------------------ the choice is honoured


def test_the_agent_when_asked_for_even_where_jev_is_the_default(jev_default, monkeypatch):
    out, seen = handle(triage="agent")
    assert "triage" in _router.ask.called
    assert out.decision.backend == "agent"
    assert out.triage["requested"] == "agent" and out.triage["used"] == "agent"
    assert out.triage["why"] is None


def test_jev_when_asked_for_on_a_server_whose_default_is_the_agent(jev_key, monkeypatch):
    """A key is all Jev needs; TRIAGE_BACKEND only says what "auto" means."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    out, seen = handle(triage="jev")
    assert "triage" not in _router.ask.called, "the agent did not classify"
    assert out.decision.backend == "jev"
    assert out.triage["used"] == "jev" and out.triage["why"] is None


def test_jev_asked_for_without_a_key_is_answered_by_the_agent_and_says_so():
    """Never silently: the page must not show the agent's route under Jev's name."""
    out, seen = handle(triage="jev")
    assert out.decision.backend == "agent"
    route = the(seen, "route")
    assert route["triage"]["requested"] == "jev"
    assert route["triage"]["used"] == "agent"
    assert route["triage"]["why"] == _router.NO_JEV_KEY


def test_jev_asked_for_but_unable_to_answer_is_answered_by_the_agent_and_says_so(jev_key, monkeypatch):
    def down(*a, **k):
        raise typesafe.TypeSafeUnavailable("api.typesafe.ai is rate limiting this key (HTTP 429)")

    monkeypatch.setattr(typesafe, "ask", down)
    out, _ = handle(triage="jev")
    assert "triage" in _router.ask.called, "the agent is the retry"
    assert out.triage["used"] == "agent" and out.triage["why"] == _router.JEV_DID_NOT_ANSWER


def test_an_unknown_choice_is_refused_by_the_router():
    with pytest.raises(ValueError):
        _router.handle(None, "my wipers squeak", agent_ids=ALL_IDS, triage="gpt")


# ------------------------------------------------------------- auto is today


def test_auto_asks_the_agent_on_a_server_whose_default_is_the_agent(jev_key):
    """A key alone does not switch Jev on for everyone; TRIAGE_BACKEND does."""
    out, _ = handle()
    assert "triage" in _router.ask.called
    assert (out.triage["requested"], out.triage["used"], out.triage["why"]) == ("auto", "agent", None)


def test_auto_asks_jev_where_jev_is_the_default(jev_default, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    out, _ = handle()
    assert "triage" not in _router.ask.called
    assert out.decision.backend == "jev"


def test_saying_auto_is_the_same_as_saying_nothing():
    """Every existing caller - the eval runner, the red team, curl - sends no
    choice. What they get must not depend on the choice having been invented."""
    a, seen_a = handle()
    b, seen_b = handle(triage="auto", compare=False)
    assert (a.reply, a.agents_used, a.decision.route()) == (b.reply, b.agents_used, b.decision.route())
    assert a.comparison is None and b.comparison is None
    assert [e["kind"] for e in seen_a] == [e["kind"] for e in seen_b]


def test_the_route_event_only_gained_a_field():
    """The page and anything else reading the stream keep every field they had,
    with the value it had."""
    _, seen = handle()
    route = the(seen, "route")
    assert set(route) == {"kind", "agents", "safety", "safety_source", "reason", "triage_skipped",
                          "ticket_stands", "backend", "probabilities", "triage"}
    assert route["agents"] == ["diagnostics"] and route["backend"] == "agent"
    assert route["probabilities"] is None and route["safety"] is False


def test_an_auto_answer_is_cached_under_the_plain_question():
    handle("what does P0420 mean")
    assert _router.ANSWERS.get("what does p0420 mean") is not None


# ----------------------------------------------- what the keyword net did


def test_the_brake_fluid_question_shows_jev_said_no_and_the_keyword_said_yes(jev_key, monkeypatch):
    """The live case: Jev scored "how often should brake fluid be changed" at
    0.05 on safety, and it was escalated because the keyword net saw "brake".
    The page has to be able to show exactly that, from the router's own values."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.97, safety=0.05))
    out, seen = handle(BRAKE_FLUID, triage="jev")

    reading = the(seen, "route")["triage"]["reading"]
    assert reading["probabilities"]["safety"] == 0.05
    assert reading["own_safety"] is False
    assert reading["own_route"] == ["diagnostics"]
    assert reading["keyword_match"] == "brake"
    assert reading["safety"] is True and reading["safety_source"] == "keyword"
    assert reading["route"] == ["diagnostics", "escalation"]
    assert reading["keyword_changed"] is True
    assert out.decision.route() == ["diagnostics", "escalation"], "the policy did not change"


def test_the_agents_own_json_is_shown_apart_from_the_keyword_net():
    out, seen = handle(BRAKE_FLUID)
    reading = out.triage["reading"]
    assert reading["backend"] == "agent"
    assert reading["own_intents"] == ["diagnostics"] and reading["own_safety"] is False
    assert reading["keyword_match"] == "brake" and reading["keyword_changed"] is True
    assert reading["prompt_tokens"] == 880 and reading["completion_tokens"] == 40


def test_a_booking_word_the_classifier_missed_is_reported_as_the_nets_doing(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents(intents=("diagnostics",)))
    out, _ = handle("book at 2 pm, wiper blades")
    reading = out.triage["reading"]
    assert reading["booking_match"] == "book"
    assert reading["own_route"] == ["diagnostics"]
    assert reading["route"] == ["diagnostics", "booking"]
    assert reading["keyword_changed"] is True


def test_when_the_classifier_and_the_net_agree_the_net_changed_nothing(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents(intents=("diagnostics",), safety=True))
    out, _ = handle("my brakes feel spongy")
    reading = out.triage["reading"]
    assert reading["keyword_match"] == "brakes" and reading["safety_source"] == "both"
    assert reading["keyword_changed"] is False


def test_nothing_is_matched_on_an_ordinary_message():
    out, _ = handle()
    reading = out.triage["reading"]
    assert reading["keyword_match"] is None and reading["booking_match"] is None
    assert reading["keyword_changed"] is False


def test_a_classifier_that_did_not_answer_is_not_shown_as_having_answered(monkeypatch):
    """The keyword fallback decided, and the reading says so rather than
    presenting the fallback's route as the agent's opinion."""
    monkeypatch.setattr(_router, "ask", lambda *a, agent_name="", **k: _unparseable(agent_name))
    out, _ = handle(BRAKE_FLUID)
    reading = out.triage["reading"]
    assert reading["ok"] is False and reading["error"] == "its JSON could not be read"
    assert reading["own_route"] is None
    assert reading["route"] == ["diagnostics", "escalation"], "the fallback's route is still shown: it ran"


def _unparseable(agent_name):
    t = TurnResult(agent_name=agent_name, status="completed")
    t.answer = "not json" if agent_name == "triage" else "Answer."
    return t


# --------------------------------------------------------------- comparing


def test_compare_asks_the_other_classifier_too(jev_key, monkeypatch):
    jev = jev_says(needs_diagnostics=0.98)
    monkeypatch.setattr(typesafe, "ask", jev)
    out, seen = handle(compare=True)

    assert jev.asked == ["my wipers squeak"], "Jev read the message"
    compared = the(seen, "compare")
    assert compared["chosen"]["backend"] == "agent" and compared["other"]["backend"] == "jev"
    assert compared["other"]["ok"] is True
    assert out.comparison == {k: v for k, v in compared.items() if k != "kind"}


def test_the_route_reply_and_tickets_follow_the_chosen_classifier_only(jev_key, monkeypatch):
    """Jev, compared, would have escalated - which is a ticket. The agent was
    chosen and did not, so escalation never runs and nobody is called."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.9, needs_escalation=0.95))
    plain, _ = handle()
    _router.ask.called.clear()

    out, seen = handle(compare=True)

    assert "escalation" not in _router.ask.called
    assert out.decision.route() == ["diagnostics"]
    assert out.reply == plain.reply
    assert out.agents_used == plain.agents_used
    compared = the(seen, "compare")
    assert compared["other"]["route"] == ["diagnostics", "escalation"]
    assert "route" in compared["differences"] and "own_route" in compared["differences"]


def test_a_compared_agent_turn_is_not_part_of_the_answer(jev_key, monkeypatch):
    """Its tokens are spent and reported, but not as the answer's cost, and its
    turn is not in the trace the customer's reply is judged by."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    plain, _ = handle(triage="jev")
    out, _ = handle(triage="jev", compare=True)

    assert _router.ask.called.count("triage") == 1, "the agent ran once, for the comparison"
    assert out.agents_used == plain.agents_used == ["diagnostics"]
    assert out.total_tokens == plain.total_tokens
    assert out.comparison_tokens == 920
    assert [t["agent"] for t in out.trace()] == ["diagnostics"]


def test_the_compared_agent_does_not_light_the_triage_card(jev_key, monkeypatch):
    """The card belongs to the classifier that routed."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    _, seen = handle(triage="jev", compare=True)
    triage_cards = [e for e in seen if e["kind"] == "agent" and e["name"] == "triage"]
    assert [e.get("backend") for e in triage_cards] == ["jev"]


def test_disagreement_before_the_net_is_shown_even_when_the_net_makes_them_agree(jev_key, monkeypatch):
    """The brake fluid case, compared: the agent flags it, Jev does not, and the
    keyword net escalates it either way. The outcome agrees; the classifiers did
    not, and that is the difference worth seeing."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.97, safety=0.05))
    monkeypatch.setattr(_router, "ask", agents(intents=("diagnostics",), safety=True))
    out, seen = handle(BRAKE_FLUID, triage="jev", compare=True)

    compared = the(seen, "compare")
    assert compared["chosen"]["own_safety"] is False and compared["other"]["own_safety"] is True
    assert set(compared["differences"]) == {"own_safety", "own_route"}
    assert compared["chosen"]["route"] == compared["other"]["route"] == ["diagnostics", "escalation"]


def test_what_each_costs_is_worked_out_as_the_eval_does(jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    out, _ = handle(compare=True)
    agent, jev = out.comparison["chosen"], out.comparison["other"]
    assert jev["tokens"] == 1316 and jev["cost_usd"] == pytest.approx(1316 / 1e6 * 0.042)
    assert agent["tokens"] == 920 and agent["cost_usd"] == pytest.approx(920 / 1e6 * 0.40)
    assert "floor" in agent["cost_basis"] and "free" in jev["cost_basis"]


def test_the_page_prices_the_classifiers_as_the_eval_does():
    """Two tables of the same prices; this holds them to each other."""
    sys.path.insert(0, str(ROOT / "evals"))
    import compare_triage as ct

    assert _router.PRICE_PER_MTOK == {"agent": ct.PRICE_PER_MTOK["triage"], "jev": ct.PRICE_PER_MTOK["jev"]}


def test_comparing_with_jev_without_a_key_says_so():
    out, seen = handle(compare=True)
    other = the(seen, "compare")["other"]
    assert other["backend"] == "jev" and other["ok"] is False
    assert other["error"] == _router.NO_JEV_KEY
    assert out.reply, "the answer is unaffected"


def test_a_compared_jev_is_not_counted_as_routing(jev_key, monkeypatch, fresh_stats):
    """/metrics reads jev_answered and jev_fell_back as what routing did. A
    comparison routes nothing, so it moves neither."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    handle(compare=True)
    assert fresh_stats == {"answered": 0, "fell_back": 0}


# ------------------------------------------------ a comparison cannot fail you


def test_a_comparison_that_blows_up_does_not_fail_the_request(jev_key, monkeypatch, caplog):
    """And its exception text goes nowhere: not the page, not the log."""
    def boom(*a, **k):
        raise RuntimeError("https://api.example.invalid/?key=ts-SECRET")

    monkeypatch.setattr(_router, "_classify_with_jev", boom)
    plain, _ = handle()
    out, seen = handle(compare=True)

    assert out.reply == plain.reply and out.error is None
    other = the(seen, "compare")["other"]
    assert other["ok"] is False and other["error"] == "the comparison failed"
    assert "SECRET" not in json.dumps(seen) and "SECRET" not in json.dumps(out.comparison)
    assert "SECRET" not in caplog.text


def test_a_compared_agent_that_blows_up_does_not_fail_the_request(jev_key, monkeypatch, caplog):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    real = agents()

    def ask(client, agent_id, prompt, agent_name="", **kw):
        if agent_name == "triage":
            raise RuntimeError("Client error for url 'https://x.invalid/?api-key=SECRET'")
        return real(client, agent_id, prompt, agent_name=agent_name, **kw)

    monkeypatch.setattr(_router, "ask", ask)
    out, _ = handle(triage="jev", compare=True)
    assert out.reply == "answer from diagnostics"
    assert out.comparison["other"]["error"] == "the comparison failed"
    assert "SECRET" not in json.dumps(out.comparison) and "SECRET" not in caplog.text


def test_a_throttled_comparison_says_so(jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    real = agents()

    def ask(client, agent_id, prompt, agent_name="", **kw):
        if agent_name == "triage":
            raise azure_http.Throttled(429, "no quota", "POST", "https://x", retry_after=20.0)
        return real(client, agent_id, prompt, agent_name=agent_name, **kw)

    monkeypatch.setattr(_router, "ask", ask)
    out, _ = handle(triage="jev", compare=True)
    assert out.comparison["other"]["error"] == "Azure had no quota for it"
    assert not out.throttled, "the answer was not throttled; only the comparison was"


def test_a_slow_comparison_is_left_behind_rather_than_waited_for(jev_key, monkeypatch):
    def slow(*a, **k):
        time.sleep(0.3)
        return jev_says(needs_diagnostics=0.98)(*a, **k)

    monkeypatch.setattr(typesafe, "ask", slow)
    monkeypatch.setattr(_router, "COMPARE_GRACE_SECONDS", 0.01)
    started = time.time()
    out, _ = handle(compare=True)
    assert time.time() - started < 0.25
    assert out.comparison["other"]["error"] == "still running when the answer was ready"
    assert out.reply == "answer from diagnostics"


def test_a_comparison_is_still_shown_when_the_chosen_agent_was_throttled(jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.9, safety=0.95))

    def throttled(*a, **k):
        raise azure_http.Throttled(429, "no quota", "POST", "https://x", retry_after=20.0)

    monkeypatch.setattr(_router, "ask", throttled)
    out, seen = handle(BRAKE_FLUID, compare=True)
    assert out.throttled and out.reply == _router.SAFETY_FALLBACK, "the warning still comes first"
    compared = the(seen, "compare")
    assert compared["chosen"]["ok"] is False and compared["chosen"]["error"] == "Azure had no quota for it"
    assert compared["other"]["ok"] is True


def test_the_compare_event_is_plain_json(jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.9, needs_booking=0.8))
    _, seen = handle("P0420 is showing, can I book in for Saturday?", compare=True)
    for event in seen:
        json.dumps(event)


# ------------------------------------------ when no classifier would have run


def test_small_talk_is_not_compared(jev_key):
    out, seen = handle("hello", compare=True)
    assert the(seen, "compare") is None and out.comparison is None
    assert _router.ask.called == []
    assert the(seen, "route")["triage"]["used"] == "none"


def test_a_bare_fault_code_is_not_compared_and_says_the_toggle_did_not_apply(jev_key):
    out, seen = handle("what does P0420 mean", triage="jev", compare=True)
    assert the(seen, "compare") is None and out.comparison is None
    assert "triage" not in _router.ask.called
    triage = the(seen, "route")["triage"]
    assert triage["requested"] == "jev" and triage["used"] == "none"
    assert "fault code" in triage["why"]


def test_a_cache_hit_runs_no_classifier_and_is_not_compared(jev_default, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    handle("what does P0420 mean")
    out, seen = handle("what does P0420 mean", triage="jev")
    assert out.cached
    assert the(seen, "compare") is None
    assert the(seen, "route")["triage"]["used"] == "none"


# ------------------------------------------------------------ the cache


def test_a_comparison_is_never_answered_from_the_cache():
    handle("what does P0420 mean")
    out, _ = handle("what does P0420 mean", compare=True)
    assert not out.cached


def test_a_comparison_never_fills_the_cache():
    handle("what does P0420 mean", compare=True)
    assert _router.ANSWERS.get("what does p0420 mean") is None


def test_a_jev_routed_answer_is_not_served_to_someone_who_chose_the_agent(jev_key, monkeypatch):
    """Today only fast-routed answers are kept, and no classifier reads those.
    Suppose the rule is widened one day: an answer Jev routed must still never
    reach a visitor who picked the agent, or the toggle would show them no
    difference at all."""
    monkeypatch.setattr(_router, "worth_caching", lambda *a: True)
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))

    first, _ = handle(triage="jev")
    assert first.decision.backend == "jev"
    _router.ask.called.clear()

    second, _ = handle(triage="agent")
    assert not second.cached
    assert "triage" in _router.ask.called, "the agent read it"

    third, _ = handle(triage="jev")
    assert third.cached, "the same choice still shares the answer"


def test_picking_the_servers_own_classifier_shares_its_cache():
    """It changes nothing about how the answer is made."""
    handle("what does P0420 mean")
    out, _ = handle("what does P0420 mean", triage="agent")
    assert out.cached


def test_the_other_classifiers_key_cannot_be_reached_by_typing():
    """The classifier sits on a line of its own, and whitespace in a message is
    collapsed, so no message can land on another choice's entry."""
    assert _router.cache_key("what does P0420 mean", "jev") == "jev\nwhat does p0420 mean"
    assert _router.cache_key("jev\nwhat does P0420 mean") != _router.cache_key("what does P0420 mean", "jev")


# -------------------------------------------------------------- telemetry


@pytest.fixture
def spans(monkeypatch):
    recorded = []

    @contextlib.contextmanager
    def span(name, **attributes):
        s = {"name": name, **attributes}
        recorded.append(s)
        yield s

    def set_(s, **attributes):
        if s is not None:
            s.update({k: v for k, v in attributes.items() if v is not None})

    monkeypatch.setattr(_router.telemetry, "span", span)
    monkeypatch.setattr(_router.telemetry, "set", set_)
    return recorded


def test_the_choice_and_the_nets_part_are_recorded_with_the_decision(spans):
    handle(BRAKE_FLUID, triage="jev")
    decision = next(s for s in spans if s["name"] == "routing.decision")
    assert decision["triage_choice"] == "jev"
    assert decision["triage_fallback"] == _router.NO_JEV_KEY
    assert decision["classifier_safety"] is False and decision["keyword_match"] == "brake"
    request = next(s for s in spans if s["name"] == "chat.request")
    assert request["triage_choice"] == "jev"


def test_a_comparison_has_its_own_span_so_decisions_are_still_counted_once(jev_key, monkeypatch, spans):
    """The runbook counts routing.decision spans by safety_source; a second one
    per compared message would double every number it reads."""
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.9, needs_escalation=0.95))
    handle(compare=True)
    assert [s["name"] for s in spans].count("routing.decision") == 1
    compare = next(s for s in spans if s["name"] == "routing.compare")
    assert compare["chosen"] == "agent" and compare["other"] == "jev"
    assert compare["agrees"] is False and "route" in compare["differences"]
    assert next(s for s in spans if s["name"] == "chat.request")["compared"] is True


# ------------------------------------------------------------------ the API


def test_a_plain_request_is_auto_and_does_not_compare():
    req = ChatRequest(message="my wipers squeak")
    assert req.triage == "auto" and req.compare is False


@pytest.mark.parametrize("body", [
    {"message": "hi", "triage": "gpt"},
    {"message": "hi", "triage": "LLM"},
    {"message": "hi", "triage": None},
    {"message": "hi", "compare": "maybe"},
])
def test_an_unknown_choice_is_not_a_valid_request(body):
    with pytest.raises(ValidationError):
        ChatRequest(**body)


@pytest.fixture
def api(monkeypatch):
    """The real router behind the endpoints, with Azure and TypeSafe faked."""
    monkeypatch.setitem(app_module.STATE, "client", object())
    monkeypatch.setitem(app_module.STATE, "agent_ids", ALL_IDS)
    monkeypatch.setattr(app_module, "LIMITS", limits.Limits(limits.Visitors(100, 100), limits.InFlight(4)))
    for key in app_module.METRICS:
        app_module.METRICS[key] = 0
    return TestClient(app_module.app)


def test_an_unknown_choice_is_refused_before_any_agent_runs(api, monkeypatch):
    monkeypatch.setattr(app_module.routing, "handle", lambda *a, **k: pytest.fail("no agent should run"))
    assert api.post("/chat", json={"message": "hi", "triage": "gpt"}).status_code == 422
    assert api.post("/chat/stream", json={"message": "hi", "triage": "gpt"}).status_code == 422


def test_chat_shows_the_comparison_without_streaming(api, jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.97, safety=0.05))
    body = api.post("/chat", json={"message": BRAKE_FLUID, "triage": "jev", "compare": True}).json()

    assert body["triage"]["requested"] == "jev" and body["triage"]["used"] == "jev"
    assert body["triage"]["reading"]["keyword_match"] == "brake"
    assert body["comparison"]["chosen"]["backend"] == "jev"
    assert body["comparison"]["other"]["backend"] == "agent"
    assert body["safety"] is True


def test_the_stream_carries_the_choice_and_the_comparison(api, jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))

    def ask_streaming(client, agent_id, prompt, on_delta, agent_name="", **kw):
        turn = agents()(client, agent_id, prompt, agent_name=agent_name)
        on_delta(turn.answer)
        return turn

    monkeypatch.setattr(_router, "ask_streaming", ask_streaming)
    r = api.post("/chat/stream", json={"message": "my wipers squeak", "triage": "jev", "compare": True})
    events = [json.loads(line[5:]) for line in r.text.split("\n\n") if line.startswith("data:")]

    compared = [e for e in events if e["type"] == "activity" and e["kind"] == "compare"]
    assert len(compared) == 1 and compared[0]["other"]["backend"] == "agent"
    done = events[-1]
    assert done["type"] == "done"
    assert done["triage"]["used"] == "jev" and done["comparison"]["other"]["ok"] is True
    kinds = [e.get("kind") for e in events if e["type"] == "activity"]
    assert kinds.index("compare") > kinds.index("route"), "the card follows the route"


def test_metrics_count_the_toggle_and_what_comparing_cost(api, jev_key, monkeypatch):
    monkeypatch.setattr(typesafe, "ask", jev_says(needs_diagnostics=0.98))
    api.post("/chat", json={"message": "my wipers squeak", "triage": "jev", "compare": True})
    api.post("/chat", json={"message": "my wipers squeak", "triage": "agent"})
    api.post("/chat", json={"message": "my wipers squeak"})

    m = api.get("/metrics").json()
    assert (m["triage_choice_jev"], m["triage_choice_agent"], m["triage_compare"]) == (1, 1, 1)
    assert m["compare_tokens"] == 920, "the agent's turn, spent on the comparison"


def test_the_page_can_tell_whether_jev_is_there_to_pick(jev_key):
    assert roster.load()["jev_available"] is True
