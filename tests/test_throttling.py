"""What the customer gets when Azure has no token quota left.

Throttling is not a failure: nothing is broken, there is simply no quota this
minute. The old behaviour treated it as one - three retries of up to 20s on each
of the several calls a message makes, the 90s request timeout, and then "Sorry -
I could not get an answer for that just now", which took a minute and a half to
say nothing useful.

These tests pin the new behaviour: stop quickly, say we are busy, and never let
a safety warning depend on a model being available.
"""

import pytest

from services.orchestrator import azure_http
from services.orchestrator import router as _router
from services.orchestrator.runner import TurnResult

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}


def throttle(*, agents=(), retry_after=20.0):
    """A fake `ask` that is throttled for `agents` and answers for the rest."""
    called = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", **_):
        called.append(agent_name)
        if agent_name in agents:
            raise azure_http.Throttled(429, "S0: rate limit exceeded", "POST", "https://x/runs",
                                       retry_after=retry_after)
        t = TurnResult(agent_name=agent_name, status="completed")
        t.answer = f"answer from {agent_name}"
        return t

    ask.called = called
    return ask


def triage_says(intents, safety=False):
    import json
    return json.dumps({"intents": intents, "safety": safety})


# ------------------------------------------------------------ the reply


def test_a_throttled_answer_says_we_are_busy_not_that_it_failed(monkeypatch):
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", throttle(agents=("diagnostics",)))

    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    assert r.reply == _router.BUSY_REPLY
    assert "could not get an answer" not in r.reply
    assert r.throttled
    assert r.retry_after == 20


def test_a_safety_warning_does_not_depend_on_a_model_being_available(monkeypatch):
    """"We are busy, try later" is not an answer to a brake problem."""
    ask = throttle(agents=("triage", "diagnostics", "escalation"))
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "my brakes have stopped working", agent_ids=ALL_IDS)

    assert r.reply == _router.SAFETY_FALLBACK
    assert "Do not drive" in r.reply
    assert r.decision.safety and r.decision.safety_source == "keyword"


def test_a_throttle_is_reported_even_when_triage_itself_answered(monkeypatch):
    """Triage's answer is routing JSON the customer never sees. Counting it as
    "something was answered" reported no throttling on exactly the messages that
    had been throttled - and /metrics is where the decision to ask Azure for
    more quota comes from."""
    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", **_):
        if agent_name == "triage":
            t = TurnResult(agent_name="triage", status="completed")
            t.answer = triage_says(["diagnostics"])
            return t
        raise azure_http.Throttled(429, "no quota", "POST", "https://x", retry_after=30.0)

    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "my car makes a rattling noise", agent_ids=ALL_IDS)

    assert r.reply == _router.BUSY_REPLY
    assert r.throttled, "the reply says busy, so the counters must say busy too"
    assert r.retry_after == 30


def test_azures_own_wording_does_not_reach_the_customer(monkeypatch):
    """The trace goes out in the body of a public endpoint, and Azure's message
    names the deployment and the pricing tier."""
    azure_says = "429: Requests to ChatCompletions under S0 exceeded the token rate limit"
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", throttle(agents=("diagnostics",)))

    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    written = str(r.trace())

    assert "S0" not in written and "pricing" not in written
    assert azure_says not in written
    assert "throttled" in written, "that it happened is still in the trace"


def test_triage_being_throttled_stops_there(monkeypatch):
    """All four agents share one deployment. If triage could not get a token,
    neither will the specialists - and each attempt costs the customer time."""
    ask = throttle(agents=("triage",))
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "my wipers squeak", agent_ids=ALL_IDS)

    assert ask.called == ["triage"], "no specialist was asked"
    assert r.reply == _router.BUSY_REPLY
    assert r.throttled


def test_a_throttled_triage_is_not_recorded_as_a_parse_failure(monkeypatch):
    """It parsed nothing because it never ran. Counting it would hide the real
    number, which is the one that says whether triage is reliable."""
    monkeypatch.setattr(_router, "ask", throttle(agents=("triage",)))
    r = _router.handle(None, "my wipers squeak", agent_ids=ALL_IDS)
    assert not r.decision.parse_failed


def test_one_throttled_specialist_does_not_lose_the_others_answer(monkeypatch):
    ask = throttle(agents=("booking",))
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "P0420 is showing, can I book in for Saturday?", agent_ids=ALL_IDS)

    assert "answer from diagnostics" in r.reply
    assert "Part of this reply is missing" in r.reply
    assert not r.throttled, "something was answered, so this is not a busy reply"


def test_a_reply_with_nothing_in_it_at_all_is_still_the_old_apology(monkeypatch):
    """Throttling is not the only way to get no answer, and the two read differently."""
    def broken(client, agent_id, prompt, timeout=90.0, agent_name="", **_):
        t = TurnResult(agent_name=agent_name, status="failed")
        t.error = "run failed"
        return t

    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", broken)

    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    assert "could not get an answer" in r.reply
    assert not r.throttled


def test_a_throttled_reply_costs_no_tokens_and_is_not_cached(monkeypatch):
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", throttle(agents=("diagnostics",)))

    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    assert _router.ANSWERS.get("what does p0420 mean") is None


def test_azures_own_advice_is_passed_on(monkeypatch):
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask", throttle(agents=("diagnostics",), retry_after=45.0))
    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    assert r.retry_after == 45


# --------------------------------------------------------------- streaming


def test_a_throttle_before_the_stream_opens_shows_the_busy_reply(monkeypatch):
    def ask_streaming(client, agent_id, prompt, on_delta, timeout=90.0, agent_name="", on_status=None, **_):
        raise azure_http.Throttled(429, "no quota", "POST", "https://x/runs", retry_after=20.0)

    written = []
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask_streaming", ask_streaming)

    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_delta=written.append)

    assert written == [], "nothing was shown, so nothing has to be taken back"
    assert r.reply == _router.BUSY_REPLY


def test_a_half_written_answer_is_kept_and_said_to_be_half_written(monkeypatch):
    """The customer can already see it, and it was grounded. Taking it away
    would be worse than telling them it stops early."""
    def ask_streaming(client, agent_id, prompt, on_delta, timeout=90.0, agent_name="", on_status=None, **_):
        on_delta("The catalytic converter is")
        t = TurnResult(agent_name=agent_name, status="failed")
        t.answer, t.throttled, t.retry_after = "The catalytic converter is", True, 20.0
        t.error = "throttled: no quota"
        return t

    written = []
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")
    monkeypatch.setattr(_router, "ask_streaming", ask_streaming)

    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_delta=written.append)

    assert written == ["The catalytic converter is"]
    assert r.reply.startswith("The catalytic converter is")
    assert "Part of this reply is missing" in r.reply


# ----------------------------------------------------- the turn it produces


def test_a_throttled_turn_is_marked_as_such_not_just_failed():
    e = azure_http.Throttled(429, "rate limit", "POST", "https://x", retry_after=12.0)
    from services.orchestrator.runner import throttled_turn

    t = throttled_turn("diagnostics", e)
    assert t.throttled and t.retry_after == 12.0
    assert not t.ok
    assert "throttled" in t.error
    assert t.answer == ""


@pytest.mark.parametrize("bad", [
    TurnResult(agent_name="diagnostics", status="failed"),
    TurnResult(agent_name="diagnostics", status="completed"),
])
def test_an_ordinary_turn_is_not_throttled(bad):
    assert not bad.throttled
    assert bad.retry_after == 0.0
