"""Score the golden set automatically.

Usage:
    python3 evals/run_evals.py                  # all cases
    python3 evals/run_evals.py --smoke          # the 5 that matter most, for CI
    python3 evals/run_evals.py --only safety-01
    python3 evals/run_evals.py --workers 1      # serial, for debugging

What this checks, and why each check exists:

  searched        Did the agent actually call the search tool? Finding 1 was an
                  agent answering a safety question from its own training in
                  prose that read perfectly well. No amount of reading the
                  answer catches that. Only the trace does.

  citations_real  Does every source it cited actually exist? Finding 4 produced
                  a citation to "Corvale Brake System Safety Guidelines", a
                  document that has never existed. We check each citation
                  against the real fault code list and the real bulletin files
                  on disk. A made-up source fails here automatically.

  contains        Phrases that must appear - "not drive" for a safety question.
  not_contains    Phrases that must not - "clutch" in a brake answer, which is
                  finding 2 turned into a permanent regression test.

  agents          Conversation cases only (those with a "history"). They run
                  the whole router rather than one agent, and check who the
                  message reached - a registration typed in answer to booking's
                  question must reach booking.

Every case runs in its own fresh thread. Reusing one would let the model answer
from chunks already in the history instead of searching again, which silently
invalidates the whole run (finding 3).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from services.orchestrator import router as routing  # noqa: E402
from services.orchestrator import runner  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

# The cases CI runs on every pull request. Chosen for coverage of the failure
# modes rather than breadth: one normal answer, two refusals that must not
# borrow from a near-miss, the invented fault code, and prompt injection.
SMOKE = ["dtc-known-01", "safety-01", "unknown-01", "relevance-02", "injection-01"]

# A citation looks like (fault code list, P0420) or (TSB-015, Diagnostic Procedure)
CITATION = re.compile(r"\(([^)]{3,80})\)")


# ---------------------------------------------------------------- ground truth


def real_sources() -> set[str]:
    """Everything the system is allowed to cite.

    The authoritative answer is the SEARCH INDEX, not the files on disk. The
    index is what the agent can actually retrieve; a PDF sitting in data/ that
    was never ingested cannot be cited, and a document in the index whose PDF
    has since been deleted still can.

    That distinction is not theoretical. The first eval run judged ten TSB
    citations fabricated because the generated PDFs were no longer on the
    machine - while those same bulletins were sitting in the index, correctly
    retrieved and correctly cited.

    Falls back to disk, then to the manifest, and says which it used.
    """
    sources: set[str] = {"fault code list", "maintenance schedule"}

    # Fault codes and maintenance items are committed, so disk is fine for those.
    with (ROOT / "data" / "dtc_codes.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["code"].strip():
                sources.add(row["code"].strip().lower())

    with (ROOT / "data" / "maintenance.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["item"].strip():
                sources.add(row["item"].strip().lower())

    bulletins, where = _bulletin_ids()
    sources.update(bulletins)
    print(f"  {len(bulletins)} bulletin id(s) from {where}")

    return sources


def _bulletin_ids() -> tuple[set[str], str]:
    """Bulletin ids, preferring the index over the filesystem."""
    # 1. the index - what was actually ingested
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        sc = SearchClient(
            endpoint=os.environ["SEARCH_ENDPOINT"],
            index_name=os.getenv("SEARCH_INDEX_NAME", "service-docs"),
            credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"]),
        )
        ids = {
            r["source_file"].replace(".pdf", "").lower()
            for r in sc.search(search_text="*", select=["source_file"], top=1000)
            if r.get("source_file")
        }
        ids.discard("dtc_codes.csv")
        ids.discard("maintenance.csv")
        if ids:
            return ids, "the search index"
    except Exception as e:  # noqa: BLE001
        print(f"  note: could not read source files from the index ({type(e).__name__})")

    # 2. the PDFs on disk
    pdf_dir = ROOT / "data" / "synthetic_bulletins"
    ids = {p.stem.lower() for p in pdf_dir.glob("*.pdf")}
    if ids:
        return ids, "PDFs on disk"

    # 3. the manifest left behind by the generator
    manifest = pdf_dir / "manifest.json"
    if manifest.exists():
        return {e["number"].lower() for e in json.loads(manifest.read_text())}, "manifest.json"

    print("  WARNING: no bulletin ids from the index, disk or manifest.")
    print("  Every TSB citation will be reported as fabricated, which is wrong.")
    return set(), "nowhere"


def check_citations(answer: str, known: set[str]) -> tuple[bool, list[str]]:
    """Every cited source must exist. Returns (ok, list of fabricated sources).

    Deliberately lenient about shape and strict about existence. We are not
    grading punctuation; we are catching invented documents.
    """
    fabricated = []

    for raw in CITATION.findall(answer):
        text = raw.strip().lower()

        # Skip brackets that are not citations: "(about 25,000 miles)",
        # "(bank 1)". A citation names a source we know about.
        if not any(k in text for k in ("tsb-", "fault code", "maintenance", "source")):
            continue

        if any(k in text for k in known):
            continue

        fabricated.append(raw.strip())

    return (not fabricated), fabricated


# ---------------------------------------------------------------- a case


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    answer: str = ""
    searched: bool = False
    tools: list[str] = field(default_factory=list)
    duration_ms: int = 0
    tokens: int = 0
    error: str | None = None

    def summary(self) -> str:
        return "PASS" if self.passed else "FAIL: " + "; ".join(self.failures)


def score(case: dict, turn, known: set[str]) -> CaseResult:
    r = CaseResult(
        id=case["id"],
        question=case["question"],
        answer=turn.answer,
        searched=turn.searched,
        tools=[c.name for c in turn.tool_calls],
        duration_ms=turn.duration_ms,
        tokens=turn.prompt_tokens + turn.completion_tokens,
        error=turn.error,
    )

    # A run blocked by Azure's content filter is the platform's safety layer
    # working, not a failure of ours. The injection case expects it: the filter
    # catches "ignore your previous instructions" before the agent sees it.
    # Defence in depth - if the filter ever stops catching it, the agent's own
    # rules still have to refuse, and the remaining checks below test that.
    if case.get("allow_content_filter") and "content_filter" in str(turn.error or ""):
        r.passed = True
        r.failures = []
        r.answer = turn.answer or "(blocked by Azure content filter)"
        return r

    if not turn.ok:
        r.failures.append(f"run did not complete: {turn.error or turn.status}")
        return r

    answer_lower = turn.answer.lower()

    # 1. did it search when it should have
    if case.get("expect_search_call") and not turn.searched:
        r.failures.append("no search call - this answer is ungrounded")
    if case.get("expect_search_call") is False and turn.searched:
        # Not a failure, just noted: searching unnecessarily costs a little but
        # is not wrong.
        pass

    # 2. did it invent a source
    ok, fabricated = check_citations(turn.answer, known)
    if not ok:
        r.failures.append(f"cited sources that do not exist: {fabricated}")

    # 3. required phrases
    for phrase in case.get("must_contain", []):
        if phrase.lower() not in answer_lower:
            r.failures.append(f"missing required phrase {phrase!r}")

    # 3b. at least one of several acceptable phrasings
    #
    # "could not find" as a required exact string was too strict: the agent says
    # "do not cover" and "found no information", both correct. A scorer that
    # fails correct behaviour teaches you to ignore it.
    alternatives = case.get("must_contain_any", [])
    if alternatives and not any(a.lower() in answer_lower for a in alternatives):
        r.failures.append(f"none of the acceptable phrasings present: {alternatives}")

    # 4. forbidden phrases
    for phrase in case.get("must_not_contain", []):
        if phrase.lower() in answer_lower:
            r.failures.append(f"contains forbidden phrase {phrase!r}")

    # 5. a citation was expected
    if case.get("expect_citation") and not CITATION.search(turn.answer):
        r.failures.append("expected a citation, found none")

    # 6. the cited source should be the right one
    want = (case.get("expect_source_contains") or "").lower()
    if want and want not in answer_lower:
        r.failures.append(f"expected a citation mentioning {want!r}")

    # 7. which agents ran - conversation cases only. A registration typed in
    # answer to booking's question must reach booking, not diagnostics.
    agents = getattr(turn, "agents", None)
    if agents is not None:
        for a in case.get("expect_agents_include", []):
            if a not in agents:
                r.failures.append(f"expected {a} to run, route was {agents}")
        for a in case.get("expect_agents_exclude", []):
            if a in agents:
                r.failures.append(f"{a} should not have run, route was {agents}")

    r.passed = not r.failures
    return r


class ConversationTurn:
    """A whole routed reply, shaped like the TurnResult that score() reads.

    Single-agent cases test one prompt. Conversation cases test the router: what
    triage made of a message given the turns before it, and who it sent it to.
    """

    def __init__(self, result):
        self.answer = result.reply
        self.agents = result.agents_used
        self.searched = result.searched
        self.tool_calls = [c for t in result.turns for c in t.tool_calls]
        self.prompt_tokens = sum(t.prompt_tokens for t in result.turns)
        self.completion_tokens = sum(t.completion_tokens for t in result.turns)
        self.duration_ms = result.duration_ms
        failed = [t for t in result.turns if not t.ok]
        self.error = result.error or next((t.error or t.status for t in failed), None)
        self.status = "completed" if self.error is None else "failed"

    @property
    def ok(self) -> bool:
        return self.error is None


def run_case(client: AgentsClient, agent_ids: dict, case: dict, known: set[str], timeout: float) -> CaseResult:
    if "history" in case:
        try:
            result = routing.handle(
                client, case["question"], timeout=timeout, agent_ids=agent_ids, history=case["history"]
            )
        except Exception as e:  # noqa: BLE001
            r = CaseResult(id=case["id"], question=case["question"])
            r.failures.append(f"{type(e).__name__}: {e}")
            return r
        return score(case, ConversationTurn(result), known)

    agent = case.get("agent", "diagnostics")
    agent_id = agent_ids.get(agent)
    if agent_id is None:
        r = CaseResult(id=case["id"], question=case["question"])
        r.failures.append(f"agent {agent!r} is not deployed")
        return r

    try:
        turn = runner.ask(client, agent_id, case["question"], timeout=timeout, agent_name=agent)
    except Exception as e:  # noqa: BLE001
        r = CaseResult(id=case["id"], question=case["question"])
        r.failures.append(f"{type(e).__name__}: {e}")
        return r

    return score(case, turn, known)


# ---------------------------------------------------------------- main


def load_cases(only: str | None, smoke: bool) -> list[dict]:
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if only:
        cases = [c for c in cases if c["id"] == only]
        if not cases:
            raise SystemExit(f"no case with id {only!r}")
    elif smoke:
        cases = [c for c in cases if c["id"] in SMOKE]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="only the CI subset")
    parser.add_argument("--only", help="run one case by id")
    parser.add_argument("--workers", type=int, default=4, help="cases in parallel")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--save", action="store_true", help="write results to evals/results/")
    args = parser.parse_args()

    cases = load_cases(args.only, args.smoke)
    known = real_sources()

    print(f"{len(cases)} case(s), {args.workers} at a time")
    print(f"{len(known)} known sources loaded from data/\n")

    started = time.time()
    results: list[CaseResult] = []

    with AgentsClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent_ids = {a.name: a.id for a in client.list_agents()}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_case, client, agent_ids, c, known, args.timeout): c
                for c in cases
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                mark = "ok  " if r.passed else "FAIL"
                print(f"  {mark} {r.id:<16} {r.duration_ms:>6}ms  {r.summary() if not r.passed else ''}")

    results.sort(key=lambda r: [c["id"] for c in cases].index(r.id))
    elapsed = time.time() - started

    passed = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 70}")
    print(f"{passed}/{len(results)} passed in {elapsed:.0f}s")

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n{len(failures)} failure(s):\n")
        for r in failures:
            print(f"  {r.id}: {r.question}")
            for f in r.failures:
                print(f"      - {f}")
            print(f"      searched={r.searched} tools={r.tools}")
            preview = r.answer.replace("\n", " ")[:200]
            print(f"      answer: {preview}...\n")

    ungrounded = [r for r in results if not r.searched and r.answer]
    if ungrounded:
        print(f"  note: {len(ungrounded)} answer(s) with no search call: "
              f"{', '.join(r.id for r in ungrounded)}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"{stamp}.json"
        path.write_text(json.dumps(
            {
                "when": stamp,
                "passed": passed,
                "total": len(results),
                "elapsed_s": round(elapsed, 1),
                "total_tokens": sum(r.tokens for r in results),
                "cases": [
                    {
                        "id": r.id, "passed": r.passed, "failures": r.failures,
                        "searched": r.searched, "tools": r.tools,
                        "ms": r.duration_ms, "tokens": r.tokens,
                        "answer": r.answer,
                    }
                    for r in results
                ],
            },
            indent=2,
        ))
        print(f"\n  saved to {path.relative_to(ROOT)}")

    print(f"  total tokens: {sum(r.tokens for r in results):,}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
