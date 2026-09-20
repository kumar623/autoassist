"""Tests for the scoring in evals/compare_triage.py.

The comparison itself needs Azure and a TypeSafe key, so it cannot run in CI.
The arithmetic that turns two sets of answers into a recommendation can, and it
is the part that would quietly mislead if it were wrong - a threshold sweep that
counts a miss as a false alarm would recommend the wrong number, and the whole
point of the exercise is to pick that number from data.
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import compare_triage as ct  # noqa: E402

CASES = [json.loads(line) for line in (ROOT / "evals" / "routing_set.jsonl").read_text().splitlines()
         if line.strip()]


def case(cid="x", route=("diagnostics",), safety=False, jev=None, **kw):
    c = ct.Compared(id=cid, message="m", expect_route=list(route), expect_safety=safety, **kw)
    c.jev = jev or {}
    return c


# ------------------------------------------------------------- the corpus


def test_the_routing_set_is_well_formed():
    assert len(CASES) >= 30, "too small to conclude anything from"
    for c in CASES:
        assert set(c["expect_route"]) <= {"diagnostics", "booking", "escalation"}, c["id"]
        assert isinstance(c["expect_safety"], bool), c["id"]
        assert c["message"].strip() and c["note"].strip(), c["id"]


def test_every_case_has_a_distinct_id():
    ids = [c["id"] for c in CASES]
    assert len(set(ids)) == len(ids)


def test_both_answers_are_represented():
    """A corpus that is all one answer measures nothing."""
    flagged = sum(c["expect_safety"] for c in CASES)
    assert 5 <= flagged <= len(CASES) - 5


def test_the_documented_false_positive_is_labelled_as_the_truth():
    """The keyword net flags "how often should brake fluid be changed" and the
    project accepts that. The label has to say what is true, not what we do, or
    the measurement just confirms the current behaviour."""
    brake_fluid = next(c for c in CASES if c["id"] == "notsafety-brake-fluid")
    assert brake_fluid["expect_safety"] is False


def test_the_smoke_cases_exist():
    assert ct.SMOKE <= {c["id"] for c in CASES}


# ---------------------------------------------------- reading Jev's answers


@pytest.mark.parametrize("cut,expected", [
    (0.05, ["diagnostics", "booking"]),
    (0.50, ["diagnostics"]),
    (0.95, ["other"]),
])
def test_the_route_depends_on_the_threshold(cut, expected):
    c = case(jev={"needs_diagnostics": 0.9, "needs_booking": 0.2, "needs_escalation": 0.01})
    assert c.jev_intents(cut) == expected


def test_the_route_is_in_the_order_the_router_runs_them():
    c = case(jev={"needs_escalation": 0.9, "needs_booking": 0.9, "needs_diagnostics": 0.9})
    assert c.jev_intents(0.5) == ["diagnostics", "booking", "escalation"]


def test_nothing_chosen_is_other_not_empty():
    assert case(jev={}).jev_intents(0.5) == ["other"]


def test_intent_scoring_ignores_order():
    c = case(route=("booking", "diagnostics"),
             jev={"needs_diagnostics": 0.9, "needs_booking": 0.9})
    c.triage_intents = ["diagnostics", "booking"]
    assert c.triage_right_on_intents()
    assert c.jev_right_on_intents(0.5)


# --------------------------------------------------------- the safety sweep


def test_the_sweep_counts_hits_misses_and_false_alarms():
    results = [
        case("a", safety=True, jev={"safety": 0.9}),    # caught
        case("b", safety=True, jev={"safety": 0.01}),   # missed
        case("c", safety=False, jev={"safety": 0.9}),   # false alarm
        case("d", safety=False, jev={"safety": 0.01}),  # correctly quiet
    ]
    s = ct.sweep(results, 0.5)
    assert (s["caught"], s["missed"], s["false_alarms"]) == (1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_a_lower_threshold_never_catches_less():
    """Monotonicity. If this fails the sweep is not measuring what it says."""
    results = [case(str(i), safety=True, jev={"safety": i / 10}) for i in range(11)]
    caught = [ct.sweep(results, c)["caught"] for c in ct.THRESHOLDS]
    assert caught == sorted(caught, reverse=True)


def test_a_case_jev_did_not_answer_is_left_out_rather_than_counted_as_safe():
    results = [case("a", safety=True, jev={}), case("b", safety=True, jev={"safety": 0.9})]
    s = ct.sweep(results, 0.5)
    assert (s["caught"], s["missed"]) == (1, 0), "the unanswered case is not a miss or a hit"


def test_the_chosen_threshold_prefers_catching_over_staying_quiet():
    """Missing a safety issue is the expensive direction to be wrong in - the
    same reason SAFETY_WORDS is deliberately blunt."""
    results = [
        case("a", safety=True, jev={"safety": 0.07}),   # only a low cut catches this
        case("b", safety=False, jev={"safety": 0.06}),  # and a low cut also flags this
    ]
    assert ct.best_cut(results) == 0.05


def test_the_chosen_threshold_breaks_ties_on_false_alarms():
    results = [case(str(i), safety=True, jev={"safety": 0.99}) for i in range(3)]
    assert ct.best_cut(results) == 0.05, "nothing is missed at any cut, so the lowest wins"


def test_todays_numbers_are_counted_the_same_way():
    results = [
        case("a", safety=True), case("b", safety=True),
        case("c", safety=False), case("d", safety=False),
    ]
    results[0].triage_safety = True    # caught
    results[1].triage_safety = False   # missed
    results[2].triage_safety = True    # false alarm
    results[3].triage_safety = False   # correctly quiet
    now = ct.today_safety(results)
    assert (now["caught"], now["missed"], now["false_alarms"]) == (1, 1, 1)


def test_a_triage_turn_that_failed_is_not_counted_as_a_judgement():
    """triage_safety stays None when the turn errored, and None is not False."""
    broken = case("a", safety=True)
    broken.triage_error = "run failed"
    now = ct.today_safety([broken])
    assert (now["caught"], now["missed"], now["false_alarms"]) == (0, 0, 0)


# -------------------------------------------------------------- the report


def test_the_report_survives_jev_not_being_configured():
    """Without a key the triage numbers still have to come out."""
    results = [case("a", safety=True)]
    results[0].triage_safety = True
    results[0].triage_intents = ["diagnostics"]
    results[0].triage_ms, results[0].triage_tokens = 2100, 990
    results[0].jev_error = "TYPESAFE_API_KEY is not set"

    summary = ct.report(results, elapsed=1.0, asked_jev=False)
    assert summary["safety"]["today"]["caught"] == 1
    assert summary["safety"]["jev_by_threshold"] == []
    assert summary["latency_ms"]["triage_median"] == 2100


def test_the_report_is_json_serialisable():
    """It is written to evals/results/ with json.dumps."""
    r = case("a", safety=True, jev={"safety": 0.9, "needs_diagnostics": 0.8})
    r.triage_safety, r.triage_intents = True, ["diagnostics"]
    json.dumps(ct.report([r], elapsed=1.0, asked_jev=True))


def test_the_questions_ask_both_sides_the_same_thing():
    """The safety wording is lifted from the triage agent's own prompt."""
    safety = ct.QUESTIONS["safety"]
    assert safety["type"] == "noul"
    for word in ("brakes", "steering", "airbags", "seat belts", "smoke"):
        assert word in safety["criteria"]["true"].lower(), word
    # The sticking flag, said out loud, because it is the failure being measured.
    assert "EARLIER" in safety["criteria"]["false"]


def test_intents_are_three_yes_no_questions_not_one_choice():
    """Triage is multi-intent: "P0420 is showing, can I come in Saturday?" is
    both. A choice question returns exactly one option."""
    assert {"needs_diagnostics", "needs_booking", "needs_escalation"} <= set(ct.QUESTIONS)
    assert all(ct.QUESTIONS[f"needs_{n}"]["type"] == "noul" for n in ct.SPECIALISTS)
