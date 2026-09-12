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
