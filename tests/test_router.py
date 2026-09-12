"""Tests for routing.

Routing lives in code rather than inside an agent precisely so it can be tested
like this - no Azure, no tokens, no flakiness, runs in CI on every pull request.

The safety cases are the ones that matter. Week 1 and week 2 both produced
failures where a model dropped a safety rule under pressure from other
instructions. These assertions cannot be talked out of it.
"""

import pytest

from services.orchestrator.router import TriageDecision, parse_triage


def d(text: str, message: str = "") -> TriageDecision:
    return parse_triage(text, message)


# ---------------------------------------------------------------- parsing


def test_plain_json_is_parsed():
    r = d('{"intents": ["booking"], "safety": false, "registration": "AP31AB1234", "date": "saturday", "reason": "wants a slot"}')
    assert r.intents == ["booking"]
    assert r.registration == "AP31AB1234"
    assert r.date == "saturday"
    assert not r.parse_failed


def test_markdown_fenced_json_is_parsed():
    r = d('```json\n{"intents": ["diagnostics"], "safety": false}\n```')
    assert r.intents == ["diagnostics"]
    assert not r.parse_failed


def test_unparseable_output_falls_back_safely():
    r = d("I think they want a booking!", "when can I come in")
    assert r.parse_failed
    assert r.intents == ["diagnostics"]
    assert not r.safety


def test_unparseable_output_still_catches_safety():
    r = d("no idea", "my brakes have stopped working")
    assert r.parse_failed
    assert r.safety
    assert "escalation" in r.route()


def test_empty_intents_becomes_other():
    assert d('{"intents": [], "safety": false}').intents == ["other"]


def test_unknown_intents_are_dropped():
    r = d('{"intents": ["diagnostics", "make_tea"], "safety": false}')
    assert r.intents == ["diagnostics"]


def test_empty_strings_become_none():
    r = d('{"intents": ["booking"], "safety": false, "registration": "", "date": ""}')
    assert r.registration is None
    assert r.date is None


# ---------------------------------------------------------------- safety


@pytest.mark.parametrize("message", [
    "my brakes feel spongy",
    "the steering is making a noise",
    "airbag light is on",
    "there is a smell of petrol",
    "smoke coming from the bonnet",
    "my seat belt will not retract",
    "I lost control on a wet road",
    "the tyre blew on the highway",
])
def test_keyword_check_catches_safety_even_when_triage_misses_it(message):
    r = d('{"intents": ["diagnostics"], "safety": false}', message)
    assert r.safety, f"keyword check missed: {message}"
    assert r.safety_source == "keyword"
    assert "escalation" in r.route()


def test_triage_alone_can_flag_safety():
    r = d('{"intents": ["diagnostics"], "safety": true}', "something feels badly wrong")
    assert r.safety
    assert r.safety_source == "triage"
    assert "escalation" in r.route()


def test_both_sources_agreeing_is_recorded():
    r = d('{"intents": ["diagnostics"], "safety": true}', "my brakes feel spongy")
    assert r.safety_source == "both"


def test_ordinary_questions_are_not_flagged():
    for msg in ["what does P0420 mean", "when is the timing belt due", "book me in on saturday"]:
        r = d('{"intents": ["diagnostics"], "safety": false}', msg)
        assert not r.safety, msg


def test_brake_fluid_interval_is_not_a_safety_emergency():
    """A maintenance question that mentions brakes is still flagged.

    The keyword check is deliberately blunt: it errs towards escalating. This
    test records that trade-off rather than pretending it does not exist - the
    cost is an unnecessary ticket, and that is the cheap direction to be wrong in.
    """
    r = d('{"intents": ["diagnostics"], "safety": false}', "how often should brake fluid be changed")
    assert r.safety
    assert r.safety_source == "keyword"


# ---------------------------------------------------------------- routing


def test_single_intent_routes_to_one_agent():
    assert d('{"intents": ["diagnostics"], "safety": false}').route() == ["diagnostics"]


def test_diagnostics_runs_before_booking():
    r = d('{"intents": ["booking", "diagnostics"], "safety": false}')
    assert r.route() == ["diagnostics", "booking"]


def test_escalation_runs_last():
    r = d('{"intents": ["escalation", "diagnostics"], "safety": false}')
    assert r.route() == ["diagnostics", "escalation"]


def test_safety_forces_escalation_even_when_not_listed():
    r = d('{"intents": ["diagnostics"], "safety": true}')
    assert r.route() == ["diagnostics", "escalation"]


def test_safety_does_not_duplicate_escalation():
    r = d('{"intents": ["diagnostics", "escalation"], "safety": true}')
    assert r.route().count("escalation") == 1


def test_other_still_gets_an_attempt():
    assert d('{"intents": ["other"], "safety": false}').route() == ["diagnostics"]


def test_all_three_intents_are_ordered():
    r = d('{"intents": ["escalation", "booking", "diagnostics"], "safety": false}')
    assert r.route() == ["diagnostics", "booking", "escalation"]


# ---------------------------------------------------------------- parallel planning


from services.orchestrator.router import _plan  # noqa: E402


def test_one_specialist_is_one_wave():
    assert _plan(["diagnostics"]) == [["diagnostics"]]


def test_diagnostics_and_booking_run_together():
    """The whole point of the change: these two do not depend on each other."""
    assert _plan(["diagnostics", "booking"]) == [["diagnostics", "booking"]]


def test_escalation_waits_for_diagnostics():
    assert _plan(["diagnostics", "escalation"]) == [["diagnostics"], ["escalation"]]


def test_escalation_waits_for_booking():
    assert _plan(["booking", "escalation"]) == [["booking"], ["escalation"]]


def test_escalation_waits_for_both():
    assert _plan(["diagnostics", "booking", "escalation"]) == [
        ["diagnostics", "booking"],
        ["escalation"],
    ]


def test_escalation_alone_does_not_wait_for_absent_agents():
    """It only waits for things that are actually in this route."""
    assert _plan(["escalation"]) == [["escalation"]]


def test_every_specialist_runs_exactly_once():
    for route in (
        ["diagnostics"],
        ["diagnostics", "booking"],
        ["diagnostics", "escalation"],
        ["booking", "escalation"],
        ["diagnostics", "booking", "escalation"],
        ["escalation"],
    ):
        flat = [s for wave in _plan(route) for s in wave]
        assert sorted(flat) == sorted(route), route
        assert len(flat) == len(set(flat)), f"duplicate in {route}"


def test_planning_terminates_on_an_empty_route():
    assert _plan([]) == []


def test_a_dependency_cycle_degrades_to_sequential(monkeypatch):
    """Guards a future edit. A cycle must not hang the request."""
    from services.orchestrator import router

    monkeypatch.setattr(
        router,
        "NEEDS_BEFORE_IT_CAN_START",
        {"diagnostics": {"booking"}, "booking": {"diagnostics"}},
    )
    waves = router._plan(["diagnostics", "booking"])
    flat = [s for w in waves for s in w]
    assert sorted(flat) == ["booking", "diagnostics"]


def test_safety_route_still_reaches_escalation_after_planning():
    """The safety guarantee must survive the parallel change."""
    r = d('{"intents": ["diagnostics"], "safety": true}', "my brakes feel spongy")
    flat = [s for wave in _plan(r.route()) for s in wave]
    assert "escalation" in flat
    # and it must run after diagnostics, so its summary knows what was said
    waves = _plan(r.route())
    assert waves[-1] == ["escalation"]


# ---------------------------------------------------------------- concurrent execution


import time  # noqa: E402

from services.orchestrator import router as _router  # noqa: E402
from services.orchestrator.runner import TurnResult  # noqa: E402


def _fake_ask(delay=0.3, fail=()):
    """Stand-in for a real agent turn: sleeps, then returns a result.

    Sleeping is the point - a real turn is mostly waiting on Azure, so if the
    wall clock for two agents is close to one delay rather than two, they really
    did run at the same time.
    """
    calls = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name=""):
        calls.append((agent_name, time.time()))
        time.sleep(delay)
        if agent_name in fail:
            raise RuntimeError(f"{agent_name} blew up")
        t = TurnResult(agent_name=agent_name, status="completed")
        t.answer = f"answer from {agent_name}"
        t.prompt_tokens = 100
        return t

    ask.calls = calls
    return ask


def _decision(intents, safety=False):
    import json as _json
    return parse_triage(_json.dumps({"intents": intents, "safety": safety}), "")


IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}


def test_independent_specialists_really_run_at_the_same_time(monkeypatch):
    fake = _fake_ask(delay=0.4)
    monkeypatch.setattr(_router, "ask", fake)

    start = time.time()
    turns = _router._run_specialists(
        None, IDS, ["diagnostics", "booking"], "msg", _decision(["diagnostics", "booking"]), 90.0
    )
    elapsed = time.time() - start

    assert len(turns) == 2
    # Sequential would be ~0.8s. Allow generous headroom for a slow CI runner
    # while still failing if they ran one after the other.
    assert elapsed < 0.7, f"took {elapsed:.2f}s - they ran sequentially"


def test_dependent_specialists_still_run_in_order(monkeypatch):
    fake = _fake_ask(delay=0.2)
    monkeypatch.setattr(_router, "ask", fake)

    _router._run_specialists(
        None, IDS, ["diagnostics", "escalation"], "msg", _decision(["diagnostics", "escalation"]), 90.0
    )

    names = [name for name, _ in fake.calls]
    assert names == ["diagnostics", "escalation"]
    starts = dict(fake.calls)
    assert starts["escalation"] > starts["diagnostics"] + 0.15, "escalation did not wait"


def test_results_come_back_in_route_order_not_finish_order(monkeypatch):
    """Booking finishes first; the reply must still read diagnostics first."""
    def ask(client, agent_id, prompt, timeout=90.0, agent_name=""):
        time.sleep(0.05 if agent_name == "booking" else 0.35)
        t = TurnResult(agent_name=agent_name, status="completed")
        t.answer = f"answer from {agent_name}"
        return t

    monkeypatch.setattr(_router, "ask", ask)
    turns = _router._run_specialists(
        None, IDS, ["diagnostics", "booking"], "msg", _decision(["diagnostics", "booking"]), 90.0
    )
    assert [t.agent_name for t in turns] == ["diagnostics", "booking"]


def test_one_specialist_failing_does_not_lose_the_other(monkeypatch):
    monkeypatch.setattr(_router, "ask", _fake_ask(delay=0.1, fail={"booking"}))

    turns = _router._run_specialists(
        None, IDS, ["diagnostics", "booking"], "msg", _decision(["diagnostics", "booking"]), 90.0
    )

    by_name = {t.agent_name: t for t in turns}
    assert by_name["diagnostics"].ok
    assert not by_name["booking"].ok
    assert "blew up" in by_name["booking"].error
    assert by_name["diagnostics"].answer  # the good answer survived


def test_prompts_are_built_before_any_thread_starts(monkeypatch):
    """Guards against prompts depending on thread timing.

    _context_for reads the results list. If it were called inside a thread while
    another thread appended, the same request could produce different prompts on
    different runs.
    """
    seen = []

    def spy(specialist, message, decision, so_far):
        seen.append((specialist, len(so_far)))
        return f"prompt for {specialist}"

    monkeypatch.setattr(_router, "_context_for", spy)
    monkeypatch.setattr(_router, "ask", _fake_ask(delay=0.05))

    _router._run_specialists(
        None, IDS, ["diagnostics", "booking"], "msg", _decision(["diagnostics", "booking"]), 90.0
    )

    # Both prompts built with zero prior results - i.e. before either ran.
    assert seen == [("diagnostics", 0), ("booking", 0)]


def test_escalation_sees_what_the_earlier_wave_said(monkeypatch):
    seen = {}

    def spy(specialist, message, decision, so_far):
        seen[specialist] = [t.agent_name for t in so_far]
        return f"prompt for {specialist}"

    monkeypatch.setattr(_router, "_context_for", spy)
    monkeypatch.setattr(_router, "ask", _fake_ask(delay=0.05))

    _router._run_specialists(
        None, IDS, ["diagnostics", "booking", "escalation"], "msg",
        _decision(["diagnostics", "booking", "escalation"]), 90.0,
    )

    assert seen["diagnostics"] == []
    assert seen["booking"] == []
    assert seen["escalation"] == ["diagnostics", "booking"]


# ---------------------------------------------------------------- run states


def test_every_azure_terminal_state_is_recognised():
    """Regression: 'incomplete' was missing, so a run that had already stopped
    was polled until the 90s timeout and reported as a timeout. The eval suite
    found it on its first run (injection-01).

    If Azure adds a state, this list must grow or we hang again.
    """
    from services.orchestrator.runner import TERMINAL

    for state in ("completed", "failed", "cancelled", "cancelling", "expired", "incomplete"):
        assert state in TERMINAL, f"{state} would cause a poll-until-timeout hang"


def test_in_progress_states_are_not_terminal():
    from services.orchestrator.runner import TERMINAL

    for state in ("queued", "in_progress", "requires_action"):
        assert state not in TERMINAL
