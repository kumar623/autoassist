"""What may be kept and handed to the next person who asks, and what may not.

The cache itself is four lines. The rules about what is allowed into it are the
part worth testing, because getting them wrong means handing one customer's
answer to another - which is the same failure the red team found in the booking
tools (docs/evaluation.md, findings 11 and 12), reached from a different
direction.
"""

import pytest

from services.orchestrator import router as _router
from services.orchestrator.runner import TurnResult

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}


@pytest.fixture(autouse=True)
def documents(monkeypatch):
    monkeypatch.setattr(_router.tools, "search_service_docs",
                        lambda query, doc_type=None: f"2 CANDIDATE document(s) for {query}")


def answering(text="The catalytic converter is worn (fault code list, P0420).", intents=("diagnostics",),
              safety=False, status="completed"):
    """A fake `ask`: triage returns the given route, specialists return `text`."""
    import json
    called = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name=""):
        called.append(agent_name)
        t = TurnResult(agent_name=agent_name, status=status)
        if agent_name == "triage":
            t.status = "completed"
            t.answer = json.dumps({"intents": list(intents), "safety": safety})
        else:
            t.answer = text
            t.prompt_tokens, t.completion_tokens = 4000, 200
        return t

    ask.called = called
    return ask


# ------------------------------------------------------------- the key


@pytest.mark.parametrize("a,b", [
    ("what does P0420 mean", "What does P0420 mean?"),
    ("what does P0420 mean", "  what  does   P0420   mean  "),
    ("what does P0420 mean?", "WHAT DOES P0420 MEAN!"),
])
def test_the_same_question_asked_differently_is_the_same_question(a, b):
    assert _router.cache_key(a) == _router.cache_key(b)


def test_a_negation_is_not_the_same_question():
    """No word is dropped when normalising, and this is why."""
    assert _router.cache_key("is it safe to drive") != _router.cache_key("is it not safe to drive")


def test_different_fault_codes_are_different_questions():
    assert _router.cache_key("what does P0420 mean") != _router.cache_key("what does P0171 mean")


@pytest.mark.parametrize("message", [
    "my reg is AP31BD1213, what does P0420 mean",
    "ap 31 bd 1213 - is this serviceable",
    "email me at krishna.kumar@example.com about P0420",
    "call me on 09876543210",
    "what happened to booking AA-M8T481",
    "my booking is #TE-00006",
])
def test_anything_personal_is_never_a_cache_key(message):
    """The answer belongs to one person, and the question should not sit in
    memory next to a reply that quotes it."""
    assert _router.cache_key(message) is None


def test_a_very_long_message_is_not_cached():
    assert _router.cache_key("the car rattles " * 40) is None


def test_the_cache_can_be_turned_off(monkeypatch):
    """The eval runner needs this: an eval answered from an earlier eval's cache
    measures nothing."""
    monkeypatch.setattr(_router.ANSWERS, "seconds", 0)
    assert _router.cache_key("what does P0420 mean") is None


# ---------------------------------------------------------- serving a hit


def test_the_second_person_to_ask_does_not_pay_for_it(monkeypatch):
    ask = answering()
    monkeypatch.setattr(_router, "ask", ask)

    first = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    asked_once = list(ask.called)
    second = _router.handle(None, "What does P0420 mean?", agent_ids=ALL_IDS)

    assert second.reply == first.reply
    assert ask.called == asked_once, "no agent ran the second time"
    assert second.cached and not first.cached
    assert second.total_tokens == 0
    assert first.total_tokens > 0


def test_a_cached_answer_is_still_grounded(monkeypatch):
    """Reporting it as unsearched would fire the runbook's ungrounded-answer
    alert on every cache hit."""
    monkeypatch.setattr(_router, "ask", answering())
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    second = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    assert second.searched
    assert second.agents_used == ["diagnostics"]


def test_the_trace_says_it_came_from_the_cache(monkeypatch):
    monkeypatch.setattr(_router, "ask", answering())
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    trace = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS).trace()

    assert trace[0]["agent"] == "cache" and trace[0]["status"] == "hit"
    assert trace[0]["tokens"] == 0
    assert any(t["agent"] == "diagnostics" for t in trace[1:]), "what produced it is still shown"
    assert any(c["name"] == "search_service_docs" for t in trace[1:] for c in t["tools"])


def test_a_cached_answer_arrives_in_one_piece(monkeypatch):
    """There is nothing to wait for, so there is nothing to stream."""
    monkeypatch.setattr(_router, "ask", answering())
    monkeypatch.setattr(_router, "ask_streaming",
                        lambda *a, **k: pytest.fail("no agent should run"))

    first = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    written = []
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS, on_delta=written.append)

    assert written == [first.reply]


def test_an_expired_answer_is_asked_again(monkeypatch):
    ask = answering()
    monkeypatch.setattr(_router, "ask", ask)
    monkeypatch.setattr(_router.ANSWERS, "seconds", 0.01)

    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    import time
    time.sleep(0.02)
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    assert ask.called.count("diagnostics") == 2


# ------------------------------------------------------- what is never kept


def test_a_conversation_is_never_answered_from_the_cache(monkeypatch):
    """With something before it, "yes" does not mean what it meant last time."""
    ask = answering()
    monkeypatch.setattr(_router, "ask", ask)

    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    before = list(ask.called)
    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS,
                       history=[{"role": "customer", "text": "my car is a 2019 Corvale"}])

    assert not r.cached
    assert ask.called != before, "the agents ran again"


def test_an_answer_given_in_a_conversation_is_not_kept_for_strangers(monkeypatch):
    monkeypatch.setattr(_router, "ask", answering())
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS,
                   history=[{"role": "customer", "text": "my car is a 2019 Corvale"}])
    assert _router.ANSWERS.get("what does p0420 mean") is None


def test_only_questions_that_route_in_code_are_kept(monkeypatch):
    """Triage is a model. On an ambiguous symptom it flags a safety issue on most
    runs and not on all of them, and caching the run where it did not would hand
    that one roll of the dice to everyone for ten minutes. fast_route decides in
    code, so there is no judgement to freeze."""
    monkeypatch.setattr(_router, "ask", answering())

    r = _router.handle(None, "the car pulls hard to the left when I slow down", agent_ids=ALL_IDS)

    assert r.reply, "it is still answered"
    assert _router.ANSWERS.get("the car pulls hard to the left when i slow down") is None


def test_a_fault_code_question_is_kept(monkeypatch):
    """The one the demo actually repeats, and the reason any of this exists."""
    monkeypatch.setattr(_router, "ask", answering())
    _router.handle(None, "What does P0420 mean and is it safe to drive?", agent_ids=ALL_IDS)
    assert _router.ANSWERS.get("what does p0420 mean and is it safe to drive") is not None


def test_a_fault_code_with_a_booking_in_it_is_not_kept(monkeypatch):
    """fast_route sends it to triage, and the answer names slots and days."""
    monkeypatch.setattr(_router, "ask", answering(intents=("diagnostics", "booking")))
    _router.handle(None, "P0420 is showing, can I book in for Saturday?", agent_ids=ALL_IDS)
    assert _router.ANSWERS.get("p0420 is showing, can i book in for saturday") is None


def test_a_safety_answer_is_never_kept(monkeypatch):
    """The flag depends on the message, the reply carries a warning, and a stale
    warning is worse than none."""
    monkeypatch.setattr(_router, "ask", answering(intents=("diagnostics",), safety=True))
    r = _router.handle(None, "there is a smell of petrol", agent_ids=ALL_IDS)
    assert r.decision.safety
    assert _router.ANSWERS.get("there is a smell of petrol") is None


def test_a_booking_answer_is_never_kept(monkeypatch):
    """It names slots, times and sometimes a reference. It belongs to one person."""
    monkeypatch.setattr(_router, "ask", answering(text="I have booked you in for 2pm, reference AA-M8T481.",
                                                  intents=("booking",)))
    _router.handle(None, "can I come in tomorrow afternoon", agent_ids=ALL_IDS)
    assert _router.ANSWERS.get("can i come in tomorrow afternoon") is None


def test_an_escalation_answer_is_never_kept(monkeypatch):
    monkeypatch.setattr(_router, "ask", answering(intents=("diagnostics", "escalation")))
    _router.handle(None, "nobody has called me back", agent_ids=ALL_IDS)
    assert _router.ANSWERS.get("nobody has called me back") is None


def test_a_withheld_answer_is_never_kept(monkeypatch):
    monkeypatch.setattr(_router, "ask",
                        answering(text="A spongy pedal is normal and it is safe to keep driving (TSB-032).",
                                  intents=("diagnostics",), safety=True))
    r = _router.handle(None, "the brake pedal feels spongy", agent_ids=ALL_IDS)
    assert r.withheld == ["diagnostics"]
    assert _router.ANSWERS.get("the brake pedal feels spongy") is None


def test_a_failed_answer_is_never_kept(monkeypatch):
    monkeypatch.setattr(_router, "ask", answering(status="failed"))
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    assert _router.ANSWERS.get("what does p0420 mean") is None


def test_an_ungrounded_answer_is_never_kept(monkeypatch):
    """Finding 1 is an answer with no search behind it. Caching one would serve
    it to everybody for ten minutes instead of once."""
    monkeypatch.setattr(_router.tools, "search_service_docs",
                        lambda query, doc_type=None: (_ for _ in ()).throw(RuntimeError("search is down")))
    monkeypatch.setattr(_router, "ask", answering())
    r = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    assert not r.searched
    assert _router.ANSWERS.get("what does p0420 mean") is None


def test_an_empty_answer_is_never_kept(monkeypatch):
    monkeypatch.setattr(_router, "ask", answering(text="   "))
    _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    assert _router.ANSWERS.get("what does p0420 mean") is None


def test_small_talk_is_not_cached_because_it_never_gets_that_far(monkeypatch):
    monkeypatch.setattr(_router, "ask", lambda *a, **k: pytest.fail("no agent should run"))
    r = _router.handle(None, "hello", agent_ids=ALL_IDS)
    assert not r.cached and r.reply == _router.GREETING_REPLY
    assert _router.ANSWERS.get("hello") is None


def test_the_stored_record_is_not_the_live_turns(monkeypatch):
    """TurnResults are mutable and shared between threads; _withhold_reassurance
    edits them in place. What is kept is a record of what a reply reported."""
    monkeypatch.setattr(_router, "ask", answering())
    first = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)

    first.turns[-1].answer = "tampered with after the fact"

    second = _router.handle(None, "what does P0420 mean", agent_ids=ALL_IDS)
    assert "tampered" not in second.reply
