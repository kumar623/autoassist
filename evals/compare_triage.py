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
THRESHOLDS = (0.05, 0.10, 0.20, 0.50)

# $ per million input tokens. Output tokens are free on Jev and charged on
# gpt-4.1-mini, so the mini figure is a floor rather than a quote.
PRICE_PER_MTOK = {"triage": 0.40, "jev": 0.042}

SPECIALISTS = ("diagnostics", "booking", "escalation")


# ------------------------------------------------------------ the questions
#
# The wording is lifted from agents/definitions/triage.json so that both sides
# are asked the same thing. Three nouls rather than one choice, because triage is
# genuinely multi-intent - "P0420 is showing, can I come in Saturday?" is both a
# diagnostics question and a booking - and a choice returns exactly one option.

QUESTIONS = {
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


@dataclass
class Compared:
    id: str
    message: str
    expect_route: list
    expect_safety: bool
    note: str = ""

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

    def jev_intents(self, cut: float) -> list:
        """Jev's route at this threshold, in the order the router runs them."""
        chosen = [name for name in SPECIALISTS
                  if (self.jev.get(f"needs_{name}") or 0.0) >= cut]
        return chosen or ["other"]

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
    if not turn.ok:
        out.triage_error = turn.error or turn.status
        return

    # parse_triage, not json.loads: the keyword checks that add booking and
    # force safety are part of today's behaviour and belong in the comparison.
    decision = routing.parse_triage(turn.answer, case["message"])
    out.triage_intents = [i for i in decision.route() if i != "other"] or ["other"]
    out.triage_safety = decision.safety
    out.triage_source = decision.safety_source


def ask_jev(case: dict, out: Compared) -> None:
    """The same input, as a state, with the four questions asked at once."""
    started = time.time()
    state = routing._triage_input(case["message"], case.get("history"))
    try:
        answer = typesafe.ask(state, QUESTIONS)
    except typesafe.TypeSafeUnavailable as e:
        out.jev_error = str(e)
        out.jev_ms = int((time.time() - started) * 1000)
        return

    out.jev_ms = int((time.time() - started) * 1000)
    out.jev_tokens = (answer.get("usage") or {}).get("input_tokens") or 0
    for name in QUESTIONS:
        found = typesafe.probability(answer.get("answers") or {}, name)
        if found is not None:
            out.jev[name] = found


def run_case(client, agent_ids: dict, case: dict, timeout: float, with_jev: bool) -> Compared:
    out = Compared(id=case["id"], message=case["message"],
                   expect_route=case["expect_route"], expect_safety=case["expect_safety"],
                   note=case.get("note", ""))
    ask_triage(client, agent_ids, case, timeout, out)
    if with_jev:
        ask_jev(case, out)
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
    """The threshold with the fewest misses, then the fewest false alarms.

    Missing a safety issue is the expensive direction to be wrong in - the same
    reason SAFETY_WORDS is deliberately blunt - so recall is ranked first.
    """
    scored = [sweep(results, c) for c in THRESHOLDS]
    scored.sort(key=lambda s: (s["missed"], s["false_alarms"]))
    return scored[0]["threshold"]


def median(values: list) -> int:
    return int(statistics.median(values)) if values else 0


def report(results: list, elapsed: float, asked_jev: bool) -> dict:
    cut = best_cut(results) if asked_jev else THRESHOLDS[0]

    triage_ok = sum(1 for r in results if r.triage_right_on_intents())
    jev_ok = sum(1 for r in results if r.jev_right_on_intents(cut)) if asked_jev else 0
    agreed = sum(1 for r in results
                 if set(r.triage_intents) == set(r.jev_intents(cut))) if asked_jev else 0

    print(f"\n{'=' * 78}")
    print(f"{len(results)} case(s) in {elapsed:.0f}s\n")

    print("ROUTING (which specialists should run)")
    print(f"  triage agent right : {triage_ok}/{len(results)}")
    if asked_jev:
        print(f"  Jev right          : {jev_ok}/{len(results)}   (at threshold {cut})")
        print(f"  the two agreed     : {agreed}/{len(results)}")

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
                     and (set(r.triage_intents) != set(r.jev_intents(cut))
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
            print(f"      jev    {'/'.join(r.jev_intents(cut))}, safety={said}")
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
        "routing": {"triage_right": triage_ok, "jev_right": jev_ok, "agreed": agreed},
        "safety": {"today": now, "jev_by_threshold": sweeps},
        "latency_ms": {"triage_median": median(t_ms),
                       "jev_median": median([r.jev_ms for r in results if not r.jev_error])},
        "tokens": {"triage_total": sum(r.triage_tokens for r in results),
                   "jev_total": sum(r.jev_tokens for r in results)},
        "per_case": [
            {"id": r.id, "message": r.message, "note": r.note,
             "expect_route": r.expect_route, "expect_safety": r.expect_safety,
             "triage_intents": r.triage_intents, "triage_safety": r.triage_safety,
             "triage_source": r.triage_source, "triage_ms": r.triage_ms,
             "triage_tokens": r.triage_tokens, "triage_error": r.triage_error,
             "jev": r.jev, "jev_intents": r.jev_intents(cut), "jev_ms": r.jev_ms,
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
    parser.add_argument("--only", help="run one case by id")
    parser.add_argument("--workers", type=int, default=4, help="cases in parallel")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--save", action="store_true", help="write the report to evals/results/")
    args = parser.parse_args()

    cases = load_cases(args.only, args.smoke)
    asked_jev = typesafe.configured()

    print(f"{len(cases)} case(s), {args.workers} at a time")
    print(f"triage: the deployed agent  |  jev: {typesafe.model() if asked_jev else 'NOT CONFIGURED'}\n")

    started = time.time()
    results = []

    with FoundryAgents(os.environ["PROJECT_ENDPOINT"], DefaultAzureCredential()) as client:
        agent_ids = {a["name"]: a["id"] for a in client.list_agents()}
        if "triage" not in agent_ids:
            raise SystemExit("the triage agent is not deployed; run agents/deploy_agents.py")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, client, agent_ids, c, args.timeout, asked_jev): c
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
        path = RESULTS_DIR / f"triage-vs-jev-{stamp}.json"
        path.write_text(json.dumps({"when": stamp, **summary}, indent=2))
        print(f"\n  saved to {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
