"""Measure Jev against the triage agent we actually run.

Usage:
    python3 evals/compare_triage.py                 # the whole routing set
    python3 evals/compare_triage.py --smoke         # three cases, to check the wiring
    python3 evals/compare_triage.py --only gearshift
    python3 evals/compare_triage.py --save          # write the report to evals/results/

Triage is the one part of this system that is pure classification: it reads a
message and returns JSON saying who should handle it and whether it looks like a
safety issue. It costs ~2.1s and ~1,000 tokens of gpt-4.1-mini per message, out
of the same quota the specialists need.

Jev answers typed questions with calibrated probabilities instead of text. This
script asks both the same thing about the same input and reports where they
agree, where each one is right, and what each costs. It changes nothing: no
module in services/orchestrator imports typesafe.py, and this script never
touches the live app.

The question it is really trying to answer is not "is Jev cheaper" - it is
whether a probability makes the safety flag better. Today the flag is a boolean
from a model OR-ed with a keyword regex, and the failures recorded in
docs/evaluation.md are calibration failures: the flag sticking across turns,
warnings on medium-severity codes. A probability can be thresholded; a model's
judgement cannot. So the safety section sweeps the threshold and reports
precision and recall at each, rather than picking one and hoping.

Needs Azure (for the triage agent) and TYPESAFE_API_KEY (for Jev). Without the
key it still runs, reports the triage side, and says Jev was not asked.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()
# A cached answer measures nothing, and the router is imported below.
os.environ["ANSWER_CACHE_SECONDS"] = "0"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from services.orchestrator import router as routing  # noqa: E402
from services.orchestrator import runner, typesafe  # noqa: E402
from services.orchestrator.foundry import FoundryAgents  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROUTING_SET = ROOT / "evals" / "routing_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

SMOKE = {"safety-brakes", "both-p0420-saturday", "followup-not-safety"}

# Where a probability would be cut into a yes or a no. Swept rather than chosen:
# the whole point of the exercise is to pick this from data.
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

# $ per million input tokens. Output tokens are free on Jev and charged on
# gpt-4.1-mini, so the mini figure is a floor rather than a quote.
PRICE_PER_MTOK = {"triage": 0.40, "jev": 0.042}

SPECIALISTS = ("diagnostics", "booking", "escalation")

# Where the safety probability is cut for the escalation rule. 0.7 separated the
# labelled set perfectly on both v3 runs: 14/14 caught, no false alarms.
SAFETY_CUT = 0.7

# One threshold per intent. Three questions with different base rates and
# different costs of being wrong do not share a number - the same argument as
# separating the routing and safety cuts. Tuned on the routing set, where 94 of
# 1,331 combinations reached the best score and the good region was wide
# (diagnostics 0.05-0.3, booking 0.05-0.8, escalation 0.3-0.8), so this is a
# plateau rather than a knife edge. A held-out half still favoured tuned cuts
# over a flat 0.2, by about one case in twenty - real, but most of the gain is
# the router rules above, not these numbers.
INTENT_CUTS = {"diagnostics": 0.15, "booking": 0.10, "escalation": 0.40}


# ------------------------------------------------------------ the questions
#
# Two shapes, kept side by side so the difference is measurable rather than
# asserted. `--shape v1` reproduces the first run; `--shape v2` is the same
# judgements asked the way TypeSafe's own guidance asks for them.
#
# What changed, and why:
#
#  1. STRUCTURED STATE. v1 sent routing._triage_input() - a prose block written
#     for a chat model, with a preamble and "NEW MESSAGE:" at the bottom. That is
#     failure mode 4 (indirection) and 5 (large state full of irrelevant detail)
#     on TypeSafe's own jaggedness page for jev-1.13. The docs ask for named
#     fields: "Use an object for most requests so each part of the state has a
#     descriptive name and its relationships remain clear."
#
#  2. STRUCTURED INSTRUCTIONS that point at a field. Every question now names
#     `new_message` in backticks and says in `focus` what the rest of the state
#     is for. Failure mode 1 is literal reading: Jev answers the question you
#     wrote, so the question has to say which part of the state it is about.
#
#  3. NO THUMB ON THE SCALE. v1's safety criteria ended "If unsure, lean towards
#     yes" - copied from triage.json, where it belongs because a chat model
#     returns a boolean and cannot express doubt. Jev returns a probability, and
#     the leaning is then applied twice: once by the model and again by the
#     threshold sweep. A calibrated number with the policy in code is the whole
#     point of the exercise. The bias belongs in the threshold, not the question.
#
#  4. ANCHORED CRITERIA. Concrete boundary cases, in the manner of
#     hotchpotch/jev-reranker, which writes explicit anchors into its criteria
#     ("topic overlap alone should receive 0.1"). Ours name the real failures:
#     a maintenance question that merely mentions brakes, and a safety problem
#     mentioned in an EARLIER turn.

QUESTIONS_V1 = {
    "needs_diagnostics": typesafe.noul(
        "Does the customer want something explained about the vehicle itself?",
        true_means="A fault code, a warning light, a noise, a smell, a symptom, when a part is "
                   "due for replacement, or what a service includes.",
        false_means="They only want an appointment arranged, or only want a person, or are "
                    "asking about something other than the vehicle.",
    ),
    "needs_booking": typesafe.noul(
        "Does the customer want an appointment arranged, changed, cancelled or confirmed?",
        true_means="Booking, rescheduling, cancelling, confirming, or asking what times are free. "
                   "Also a bare date, time or registration given in answer to a question the "
                   "assistant just asked about a booking.",
        false_means="They are only describing a problem with the vehicle, with no mention of "
                    "coming in.",
    ),
    "needs_escalation": typesafe.noul(
        "Should a human service advisor take this over?",
        true_means="They asked for a person, they are clearly unhappy or complaining, there has "
                   "been an accident or a breakdown, the vehicle is undriveable, or it is an "
                   "insurance or warranty dispute.",
        false_means="An ordinary question a workshop assistant can answer, or an ordinary "
                    "booking request.",
    ),
    "safety": typesafe.noul(
        "Does the NEW MESSAGE describe something that could make the vehicle unsafe to drive?",
        true_means="Brakes, steering, airbags, seat belts, a smell of fuel, smoke, fire, a wheel "
                   "or suspension problem, losing control, a crash, a warning light the customer "
                   "describes as red, or anything else suggesting the vehicle is dangerous to "
                   "drive now. If unsure, lean towards yes.",
        false_means="A maintenance question that merely mentions a safety part, a fault code the "
                    "documents call low or medium severity, a comfort or convenience problem, or "
                    "an appointment request. A safety problem mentioned EARLIER in the "
                    "conversation does not make a new message about something else unsafe.",
    ),
}

# The same four judgements, asked the way the docs ask for them.
FOCUS = ("Judge `new_message` only. `assistant_last_asked` and `conversation_so_far` are "
         "background, there to resolve what a short reply refers to - they are not the thing "
         "being judged.")

QUESTIONS_V2 = {
    "needs_diagnostics": {
        "type": "noul",
        "instructions": {
            "question": "Does `new_message` ask for something about the vehicle to be explained?",
            "focus": FOCUS,
        },
        "criteria": {
            "true": {
                "covers": "A fault code, a warning light, a noise, a smell, a symptom, when a part "
                          "is due for replacement, or what a service includes.",
                "also_true_when": "The customer describes a fault AND asks to come in - both are "
                                  "wanted, and this question is only about the explaining half.",
            },
            "false": {
                "covers": "They only want an appointment arranged, only want a person, or are "
                          "asking about something that is not the vehicle.",
                "example": "A bare registration, date or time given in answer to a booking "
                           "question is not asking for an explanation.",
            },
        },
    },
    "needs_booking": {
        "type": "noul",
        "instructions": {
            "question": "Does `new_message` ask to arrange, change, cancel or confirm an appointment?",
            "focus": FOCUS,
        },
        "criteria": {
            "true": {
                "covers": "Booking, rescheduling, cancelling, confirming, or asking which times "
                          "are free.",
                "also_true_when": "`new_message` is a bare date, time, registration or yes/no that "
                                  "answers `assistant_last_asked`, and `assistant_last_asked` was "
                                  "about an appointment.",
            },
            "false": "They only describe a problem with the vehicle, with no mention of coming in.",
        },
    },
    "needs_escalation": {
        "type": "noul",
        "instructions": {
            "question": "Should a human service advisor take `new_message` over from the assistant?",
            "focus": FOCUS,
        },
        "criteria": {
            "true": "They asked for a person, they are clearly unhappy or complaining, there has "
                    "been an accident or a breakdown, the vehicle is undriveable, or it is an "
                    "insurance or warranty dispute.",
            "false": {
                "covers": "An ordinary question a workshop assistant can answer, or an ordinary "
                          "booking request.",
                "note": "A fault being serious is judged by the safety question, not this one.",
            },
        },
    },
    "safety": {
        "type": "noul",
        "instructions": {
            "question": "Does `new_message` describe something that could make the vehicle unsafe "
                        "to drive now?",
            "focus": FOCUS,
        },
        # No "lean towards yes": the leaning is the threshold's job, in code.
        "criteria": {
            "true": {
                "covers": "Brakes, steering, airbags, seat belts, a smell of fuel, smoke, fire, a "
                          "wheel or suspension problem, losing control, a crash, or a warning "
                          "light the customer calls red.",
                "test": "Would a technician tell them to stop driving until it is looked at?",
            },
            "false": {
                "covers": "A comfort or convenience problem, or an appointment request.",
                "maintenance": "Asking WHEN a safety part is due - brake fluid, brake pads - is a "
                               "maintenance question, not a fault. Nothing is wrong with the car.",
                "severity": "A fault code the documents call low or medium severity.",
                "earlier_turns": "A safety problem in `conversation_so_far` does NOT make "
                                 "`new_message` unsafe when `new_message` is about something else, "
                                 "such as booking an appointment.",
            },
        },
    },
}

# v3: v2's structure, with the safety question asked conservatively again.
#
# Removing "if unsure, lean towards yes" from v2 was meant to fix a double bias -
# the model leaning AND the threshold leaning. Measured, it was the wrong call:
# v1 held 14/14 at a cut of 0.5, v2 missed one there, because the leaning was
# spreading the true cases further from the false ones rather than corrupting
# them. It is not a thumb on the scale; it asks a different and legitimate
# question - "could this plausibly be unsafe?" rather than "is this unsafe?" -
# and plausibility is the right question when missing a brake failure costs more
# than an unnecessary ticket. That is the same trade SAFETY_WORDS already makes
# in router.py, written down rather than assumed.
QUESTIONS_V3 = copy.deepcopy(QUESTIONS_V2)
QUESTIONS_V3["safety"]["instructions"]["question"] = (
    "Could `new_message` plausibly describe something that makes the vehicle unsafe to drive now?"
)
QUESTIONS_V3["safety"]["criteria"]["true"]["when_unsure"] = (
    "If it is genuinely unclear whether the fault affects safety, this is true. Missing a brake "
    "failure costs more than an unnecessary check."
)

# v4: v3, with the two questions the misses actually point at.
#
# Measured on v3 (both runs), the remaining routing errors were not threshold
# errors. They were the questions meaning the wrong thing:
#
#  - needs_diagnostics asked whether the customer wants something EXPLAINED.
#    "book at 2 pm , viper blades" scored 0.03: they are not asking for an
#    explanation, they are naming the part for the job. But diagnostics is
#    exactly who should handle a wiper complaint - that message going to booking
#    alone is the 19 Sep live failure the keyword backstop exists for. The
#    question now asks whether the message NAMES a fault, symptom or part,
#    which is what the router actually needs to know.
#
#  - needs_escalation fired at 0.22-0.30 on ordinary faults (a warm air
#    conditioner, a hard-starting engine, a gearbox complaint). Its false
#    criteria now say plainly that a fault being serious is not escalation:
#    a human is wanted for a PERSON-shaped reason, and severity is the safety
#    question's job.
QUESTIONS_V4 = copy.deepcopy(QUESTIONS_V3)
QUESTIONS_V4["needs_diagnostics"]["instructions"]["question"] = (
    "Does `new_message` name a vehicle fault, symptom, warning light, fault code, or a part "
    "the workshop should say something about?"
)
QUESTIONS_V4["needs_diagnostics"]["criteria"]["true"] = {
    "covers": "A fault code, a warning light, a noise, a smell, a symptom, a part that needs "
              "attention, when a part is due for replacement, or what a service includes.",
    "naming_is_enough": "Naming the part or fault is enough. 'book at 2 pm, wiper blades' names "
                        "a part, so this is true even though the customer only asked to book.",
    "also_true_when": "The customer describes a fault AND asks to come in - both are wanted, and "
                      "this question is only about the vehicle half.",
}
QUESTIONS_V4["needs_diagnostics"]["criteria"]["false"] = {
    "covers": "Nothing about the vehicle is mentioned at all: only scheduling, only a request for "
              "a person, or a subject that is not the vehicle.",
    "example": "A bare registration, date, time or yes/no answering a booking question names no "
               "fault and no part.",
}
QUESTIONS_V4["needs_escalation"]["criteria"]["false"] = {
    "covers": "An ordinary question a workshop assistant can answer, or an ordinary booking request.",
    "severity_is_not_escalation": "A fault being serious, dangerous or expensive is NOT a reason "
                                  "for this to be true. Whether the car is unsafe is a different "
                                  "question and is asked separately.",
    "needs_a_person": "This is true only when a PERSON is wanted: they asked for one, they are "
                      "complaining, or the situation is a dispute, an accident or a breakdown.",
}

SHAPES = {"v1": QUESTIONS_V1, "v2": QUESTIONS_V2, "v3": QUESTIONS_V3, "v4": QUESTIONS_V4}


def state_for(case: dict, shape: str):
    """What Jev is given to judge.

    v1 is the chat model's own prompt, which made the comparison fair and Jev's
    job harder. v2 is an object with named fields, which is what the docs ask
    for and what the questions point at.
    """
    if shape == "v1":
        return routing._triage_input(case["message"], case.get("history"))

    history = case.get("history") or []
    said = [turn["text"] for turn in history if turn.get("role") == "assistant"]
    return {
        "new_message": case["message"],
        "assistant_last_asked": said[-1] if said else None,
        "conversation_so_far": [f"{t.get('role')}: {t.get('text')}" for t in history],
    }


@dataclass
class Compared:
    id: str
    message: str
    expect_route: list
    expect_safety: bool
    note: str = ""
    split: str = "core"

    triage_intents: list = field(default_factory=list)
    triage_safety: bool | None = None
    triage_source: str = ""
    triage_ms: int = 0
    triage_tokens: int = 0
    triage_error: str = ""

    jev: dict = field(default_factory=dict)   # question -> probability
    jev_ms: int = 0
    jev_tokens: int = 0
    jev_error: str = ""

    def jev_intents(self, cut, safety_cut: float = SAFETY_CUT) -> list:
        """Jev's route, through the same rules router.route() applies.

        `cut` is one threshold for all three intents, or a dict per intent.

        The rules are not decoration. The labels in routing_set.jsonl are the
        route that should RUN, and router.route() forces escalation onto any
        safety-flagged message and sends an otherwise empty route to diagnostics
        ("other still gets a helpful attempt"). Scoring Jev's raw nouls against
        post-rule labels marked three cases wrong that the real system would
        have got right - a comparison of the model against the model plus its
        orchestrator, which is not the question anyone asked.
        """
        cuts = cut if isinstance(cut, dict) else dict.fromkeys(SPECIALISTS, cut)
        chosen = [name for name in SPECIALISTS
                  if (self.jev.get(f"needs_{name}") or 0.0) >= cuts[name]]
        if (self.jev.get("safety") or 0.0) >= safety_cut and "escalation" not in chosen:
            chosen.append("escalation")
        return chosen or ["diagnostics"]

    def triage_right_on_intents(self) -> bool:
        return set(self.triage_intents) == set(self.expect_route)

    def jev_right_on_intents(self, cut: float) -> bool:
        return set(self.jev_intents(cut)) == set(self.expect_route)


# ------------------------------------------------------------------ the run


def ask_triage(client, agent_ids: dict, case: dict, timeout: float, out: Compared) -> None:
    """Exactly what the live app does, including the keyword backstops."""
    started = time.time()
    try:
        turn = runner.ask(
            client, agent_ids["triage"],
            routing._triage_input(case["message"], case.get("history")),
            timeout=timeout, agent_name="triage",
        )
    except Exception as e:  # noqa: BLE001 - one case failing must not stop the run
        out.triage_error = f"{type(e).__name__}: {e}"
        out.triage_ms = int((time.time() - started) * 1000)
        return

    out.triage_ms = turn.duration_ms
    out.triage_tokens = turn.prompt_tokens + turn.completion_tokens

    # A failed turn is not a missing answer: _handle_inner falls back to
    # parse_triage("", message), which still runs the keyword checks. Recording
    # the error and stopping scored triage worse than the live app behaves -
    # Azure's content filter killed the run on the injection case, and the real
    # router still flagged the brakes in it. Scoring the harness instead of the
    # system is how an eval flatters or libels the thing it measures.
    answer = turn.answer if turn.ok else ""
    if not turn.ok:
        out.triage_error = turn.error or turn.status

    # parse_triage, not json.loads: the keyword checks that add booking and
    # force safety are part of today's behaviour and belong in the comparison.
    decision = routing.parse_triage(answer, case["message"])
    out.triage_intents = [i for i in decision.route() if i != "other"] or ["other"]
    out.triage_safety = decision.safety
    out.triage_source = decision.safety_source


def ask_jev(case: dict, out: Compared, shape: str = "v3") -> None:
    """The four judgements, asked at once about one state."""
    started = time.time()
    questions = SHAPES[shape]
    try:
        answer = typesafe.ask(state_for(case, shape), questions)
    except typesafe.TypeSafeUnavailable as e:
        out.jev_error = str(e)
        out.jev_ms = int((time.time() - started) * 1000)
        return

    out.jev_ms = int((time.time() - started) * 1000)
    out.jev_tokens = (answer.get("usage") or {}).get("input_tokens") or 0
    for name in questions:
        found = typesafe.probability(answer.get("answers") or {}, name)
        if found is not None:
            out.jev[name] = found


def run_case(client, agent_ids: dict, case: dict, timeout: float, with_jev: bool,
             shape: str = "v3") -> Compared:
    out = Compared(id=case["id"], message=case["message"],
                   expect_route=case["expect_route"], expect_safety=case["expect_safety"],
                   note=case.get("note", ""), split=case.get("split", "core"))
    ask_triage(client, agent_ids, case, timeout, out)
    if with_jev:
        ask_jev(case, out, shape)
    else:
        out.jev_error = "TYPESAFE_API_KEY is not set"
    return out


# -------------------------------------------------------------- the report


def sweep(results: list, cut: float) -> dict:
    """Precision and recall for Jev's safety answer at one threshold."""
    hit = miss = false_alarm = 0
    for r in results:
        p = r.jev.get("safety")
        if p is None:
            continue
        said = p >= cut
        if said and r.expect_safety:
            hit += 1
        elif said and not r.expect_safety:
            false_alarm += 1
        elif not said and r.expect_safety:
            miss += 1
    precision = hit / (hit + false_alarm) if (hit + false_alarm) else 0.0
    recall = hit / (hit + miss) if (hit + miss) else 0.0
    return {"threshold": cut, "caught": hit, "missed": miss, "false_alarms": false_alarm,
            "precision": round(precision, 3), "recall": round(recall, 3)}


def today_safety(results: list) -> dict:
    hit = sum(1 for r in results if r.triage_safety and r.expect_safety)
    miss = sum(1 for r in results if r.triage_safety is False and r.expect_safety)
    false_alarm = sum(1 for r in results if r.triage_safety and not r.expect_safety)
    precision = hit / (hit + false_alarm) if (hit + false_alarm) else 0.0
    recall = hit / (hit + miss) if (hit + miss) else 0.0
    return {"caught": hit, "missed": miss, "false_alarms": false_alarm,
            "precision": round(precision, 3), "recall": round(recall, 3)}


def best_cut(results: list) -> float:
    """The SAFETY threshold: fewest misses, then fewest false alarms.

    Missing a safety issue is the expensive direction to be wrong in - the same
    reason SAFETY_WORDS is deliberately blunt - so recall is ranked first.
    """
    scored = [sweep(results, c) for c in THRESHOLDS]
    scored.sort(key=lambda s: (s["missed"], s["false_alarms"]))
    return scored[0]["threshold"]


def best_route_cut(results: list) -> float:
    """The ROUTING threshold, chosen on routing's own terms: most routes right.

    Separate from the safety threshold, and that separation is not a detail.
    Scoring routes at the safety cut is what made the first three runs read as a
    tie at 25/42: the safety question wants a high bar, the intent questions want
    a lower one, and forcing them to share a number threw away six correct routes.
    Two questions with different consequences get two policies - which is the
    whole argument for a probability over a boolean.
    """
    scored = [(sum(1 for r in results if r.jev_right_on_intents(c)), -c) for c in THRESHOLDS]
    return -max(scored)[1]


def tuned_score(results: list) -> int:
    """Routes right with per-intent cuts, which is how the real thing would run."""
    return sum(1 for r in results if set(r.jev_intents(INTENT_CUTS)) == set(r.expect_route))


def median(values: list) -> int:
    return int(statistics.median(values)) if values else 0


def report(results: list, elapsed: float, asked_jev: bool) -> dict:
    cut = best_cut(results) if asked_jev else THRESHOLDS[0]
    route_cut = best_route_cut(results) if asked_jev else THRESHOLDS[0]

    triage_ok = sum(1 for r in results if r.triage_right_on_intents())
    jev_ok = sum(1 for r in results if r.jev_right_on_intents(route_cut)) if asked_jev else 0
    agreed = sum(1 for r in results
                 if set(r.triage_intents) == set(r.jev_intents(route_cut))) if asked_jev else 0

    print(f"\n{'=' * 78}")
    print(f"{len(results)} case(s) in {elapsed:.0f}s\n")

    print("ROUTING (which specialists should run)")
    print(f"  triage agent right : {triage_ok}/{len(results)}")
    if asked_jev:
        print(f"  Jev right          : {jev_ok}/{len(results)}   (one cut, {route_cut})")
        print(f"  Jev, per-intent cuts: {tuned_score(results)}/{len(results)}   "
              f"({', '.join(f'{k[:4]} {v}' for k, v in INTENT_CUTS.items())})")
        print(f"  the two agreed     : {agreed}/{len(results)}")

    by_split = {}
    for r in results:
        by_split.setdefault(r.split, []).append(r)
    if asked_jev and len(by_split) > 1:
        print("\n  by split (the held-out cases were written after the thresholds were chosen):")
        for split in ("core", "held_out"):
            rows = by_split.get(split) or []
            if not rows:
                continue
            t_ok = sum(1 for r in rows if r.triage_right_on_intents())
            j_ok = sum(1 for r in rows if set(r.jev_intents(INTENT_CUTS)) == set(r.expect_route))
            print(f"    {split:<9} triage {t_ok:>2}/{len(rows):<3} jev {j_ok:>2}/{len(rows)}")

    print("\nSAFETY FLAG")
    now = today_safety(results)
    print(f"  today  caught {now['caught']}/{now['caught'] + now['missed']}  "
          f"false alarms {now['false_alarms']}  "
          f"precision {now['precision']}  recall {now['recall']}")
    sweeps = []
    if asked_jev:
        print("  Jev, by threshold:")
        print(f"    {'cut':>6}  {'caught':>6} {'missed':>6} {'false':>6}  {'prec':>5} {'recall':>6}")
        for c in THRESHOLDS:
            s = sweep(results, c)
            sweeps.append(s)
            star = " <-" if c == cut else ""
            print(f"    {c:>6}  {s['caught']:>6} {s['missed']:>6} {s['false_alarms']:>6}  "
                  f"{s['precision']:>5} {s['recall']:>6}{star}")

    print("\nCOST AND LATENCY (per message)")
    t_ms = [r.triage_ms for r in results if not r.triage_error]
    t_tok = [r.triage_tokens for r in results if not r.triage_error]
    print(f"  triage agent : {median(t_ms):>6}ms median   {median(t_tok):>6} tokens   "
          f"${sum(t_tok) / 1e6 * PRICE_PER_MTOK['triage']:.4f} for this run")
    if asked_jev:
        j_ms = [r.jev_ms for r in results if not r.jev_error]
        j_tok = [r.jev_tokens for r in results if not r.jev_error]
        print(f"  Jev          : {median(j_ms):>6}ms median   {median(j_tok):>6} tokens   "
              f"${sum(j_tok) / 1e6 * PRICE_PER_MTOK['jev']:.4f} for this run")

    # Worth printing whether or not Jev was asked: this is where the triage we
    # actually run disagrees with the labels, which is a finding on its own.
    wrong = [r for r in results if not r.triage_error
             and (not r.triage_right_on_intents() or bool(r.triage_safety) != r.expect_safety)]
    if wrong:
        print(f"\nWHERE TODAY'S TRIAGE DIFFERED FROM THE LABEL ({len(wrong)})")
        for r in wrong:
            why = []
            if not r.triage_right_on_intents():
                why.append(f"route {'/'.join(r.triage_intents) or '-'}, wanted {'/'.join(r.expect_route)}")
            if bool(r.triage_safety) != r.expect_safety:
                why.append(f"safety {r.triage_safety} ({r.triage_source}), wanted {r.expect_safety}")
            print(f"  {r.id:<24} {'; '.join(why)}")
            print(f"  {'':<24} {r.message[:64]}")

    disagreements = [r for r in results if asked_jev and not r.jev_error
                     and (set(r.triage_intents) != set(r.jev_intents(route_cut))
                          or bool(r.triage_safety) != ((r.jev.get("safety") or 0) >= cut))]
    if disagreements:
        print(f"\nWHERE THEY DISAGREED ({len(disagreements)})")
        for r in disagreements:
            p = r.jev.get("safety")
            said = "?" if p is None else f"{p:.3f}"
            print(f"\n  {r.id}: {r.message[:70]}")
            print(f"      wanted {'/'.join(r.expect_route)}, safety={r.expect_safety}")
            print(f"      triage {'/'.join(r.triage_intents) or '-'}, safety={r.triage_safety} "
                  f"({r.triage_source})")
            print(f"      jev    {'/'.join(r.jev_intents(route_cut))}, safety={said}")
            if r.note:
                print(f"      note   {r.note[:100]}")

    broken = [r for r in results if r.triage_error]
    if broken:
        print(f"\n  {len(broken)} case(s) the triage agent could not answer: "
              f"{', '.join(r.id + ' (' + r.triage_error[:40] + ')' for r in broken)}")
    if not asked_jev:
        print("\n  Jev was not asked: TYPESAFE_API_KEY is not set. "
              "The triage numbers above still stand.")

    return {
        "cases": len(results),
        "elapsed_s": round(elapsed, 1),
        "threshold": cut,
        "route_threshold": route_cut,
        "routing": {"triage_right": triage_ok, "jev_right": jev_ok, "agreed": agreed,
                    "jev_tuned": tuned_score(results) if asked_jev else 0,
                    "intent_cuts": INTENT_CUTS},
        "safety": {"today": now, "jev_by_threshold": sweeps},
        "latency_ms": {"triage_median": median(t_ms),
                       "jev_median": median([r.jev_ms for r in results if not r.jev_error])},
        "tokens": {"triage_total": sum(r.triage_tokens for r in results),
                   "jev_total": sum(r.jev_tokens for r in results)},
        "per_case": [
            {"id": r.id, "message": r.message, "note": r.note, "split": r.split,
             "expect_route": r.expect_route, "expect_safety": r.expect_safety,
             "triage_intents": r.triage_intents, "triage_safety": r.triage_safety,
             "triage_source": r.triage_source, "triage_ms": r.triage_ms,
             "triage_tokens": r.triage_tokens, "triage_error": r.triage_error,
             "jev": r.jev, "jev_intents": r.jev_intents(route_cut), "jev_ms": r.jev_ms,
             "jev_tokens": r.jev_tokens, "jev_error": r.jev_error}
            for r in results
        ],
    }


# ------------------------------------------------------------------- main


def load_cases(only: str | None, smoke: bool) -> list:
    cases = [json.loads(line) for line in ROUTING_SET.read_text().splitlines() if line.strip()]
    if only:
        cases = [c for c in cases if c["id"] == only]
        if not cases:
            raise SystemExit(f"no case with id {only!r}")
    elif smoke:
        cases = [c for c in cases if c["id"] in SMOKE]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="three cases, to check the wiring")
    parser.add_argument("--split", choices=("core", "held_out", "all"), default="all",
                        help="core is the set the thresholds were tuned on; held_out was written "
                             "afterwards and never used for tuning")
    parser.add_argument("--only", help="run one case by id")
    parser.add_argument("--workers", type=int, default=4, help="cases in parallel")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="v3",
                        help="how Jev is asked: v1 reproduces the first run, v2 follows the docs")
    parser.add_argument("--save", action="store_true", help="write the report to evals/results/")
    args = parser.parse_args()

    cases = load_cases(args.only, args.smoke)
    if args.split != "all" and not args.only and not args.smoke:
        cases = [c for c in cases if c.get("split", "core") == args.split]
    asked_jev = typesafe.configured()

    print(f"{len(cases)} case(s), {args.workers} at a time")
    print(f"triage: the deployed agent  |  jev: {typesafe.model() if asked_jev else 'NOT CONFIGURED'}"
          f"  |  questions: {args.shape}\n")

    started = time.time()
    results = []

    with FoundryAgents(os.environ["PROJECT_ENDPOINT"], DefaultAzureCredential()) as client:
        agent_ids = {a["name"]: a["id"] for a in client.list_agents()}
        if "triage" not in agent_ids:
            raise SystemExit("the triage agent is not deployed; run agents/deploy_agents.py")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, client, agent_ids, c, args.timeout, asked_jev, args.shape): c
                       for c in cases}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                agree = "" if not asked_jev else (
                    "  " if set(r.triage_intents) == set(r.jev_intents(THRESHOLDS[1])) else "!=")
                print(f"  {agree} {r.id:<24} triage {r.triage_ms:>5}ms   jev {r.jev_ms:>5}ms")

    order = [c["id"] for c in cases]
    results.sort(key=lambda r: order.index(r.id))
    summary = report(results, time.time() - started, asked_jev)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"triage-vs-jev-{args.shape}-{stamp}.json"
        path.write_text(json.dumps({"when": stamp, "shape": args.shape, **summary}, indent=2))
        print(f"\n  saved to {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
