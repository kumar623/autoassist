"""Three yes/no judgements the router asks Jev, where a regex was the only judge.

Triage was the first place a typed judgement replaced a chat model (finding 16).
evals/jev_checks.py then measured five more candidates before any was built, on
155 labelled cases (evals/jev_checks_set.jsonl, 21-22 September). Three of them
answer a question the router already asks in code, and does badly:

                     today's code                        Jev at 0.5
    reassures        router.REASSURANCE      15/32       31/32
    maintenance      best regex we had       19/29       29/29
                     (5 faults read as maintenance)
    accepts_call     YES / NOT_YES           24/28       28/28
                     (missed "absolutely", "that works",
                      "haan please call karo"; no false alarms)

About 0.4s and 400-560 input tokens a check. Taking every example out of the
questions cost only two more errors of 155, so the scores are Jev's rather than
the examples describing the test to it.

The questions live here, not in the eval, and the eval imports them: finding 16
was a comparison that measured a prompt the service did not ship. What is
measured is what runs.

Every function here returns None when Jev cannot answer - no key, the service
down, too slow, a question left unanswered - and never raises. None always
means "use the code that was here before": the regex decides, and nothing is
exempted. So a server with no TypeSafe key behaves exactly as it did.

These need only a key (typesafe.configured()), not TRIAGE_BACKEND=jev: they
judge replies and messages whichever classifier routed them. Each can be turned
off on its own - see SWITCHES.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for

from . import jev_triage, limits, typesafe

log = logging.getLogger(__name__)


def _q(question: str, focus: str, true: dict | str, false: dict | str) -> dict:
    return {"type": "noul", "instructions": {"question": question, "focus": focus},
            "criteria": {"true": true, "false": false}}


# Exactly as measured. Changing a word here changes what evals/jev_checks.py
# measures too, which is the point - re-run it before trusting a new wording.
QUESTIONS = {
    "reassures": _q(
        "Does `reply` tell the customer, in any words or language, that they can keep driving, or that "
        "the problem is normal, harmless or nothing to worry about?",
        "Judge `reply` only. `customer_message` is what they asked about.",
        {"covers": "Any statement that carrying on driving is fine, safe or can wait; that the condition is "
                   "normal, expected, common or harmless; that there is no rush or nothing to be concerned "
                   "about.",
         "examples": "'you can carry on as usual', 'it will settle in', 'no rush to have it checked'"},
        {"covers": "A warning not to drive; a statement that it is NOT safe; safety only after a repair "
                   "('once the brakes are bled it will be safe'); a fault code list's own severity line quoted "
                   "as the document's words, such as 'safe to drive with care (fault code list, P0420)'.",
         "other_subjects": "Reassurance about something other than driving or the fault - cost, a booking, "
                           "warranty - does not count."},
    ),
    "maintenance": _q(
        "Is `new_message` only asking when a part is due for service or replacement, without describing "
        "anything wrong with the car now?",
        "Judge `new_message` only.",
        "A schedule or interval question: how often, when is it due, how long does it last.",
        {"covers": "Any sign of a present fault, symptom, damage or event - a noise, a feel, a warning light, a "
                   "leak, a smell, something failing, an accident - even inside a question about intervals."},
    ),
    "accepts_call": _q(
        "Does `new_message` accept the offer of a call from a service advisor made in `assistant_last_asked`?",
        "Judge `new_message` as an answer to `assistant_last_asked`.",
        "It says yes to the call, in any words or language, including a yes with a condition "
        "('yes, after 5pm').",
        "It declines, puts it off ('maybe later'), asks something else, or changes the subject.",
    ),
}

# The environment variable that turns each check off. On by default wherever
# there is a key: "0" turns one off. Read on every call, through limits.setting,
# so a word where a number belongs is a warning in the log and the default, not
# a container that will not start.
SWITCHES = {
    "reassures": "JEV_REASSURANCE_CHECK",
    "maintenance": "JEV_MAINTENANCE_CHECK",
    "accepts_call": "JEV_ACCEPTS_CALL_CHECK",
}

# Where to cut. 0.5 for the two that decide something today, because the
# measured scores are nowhere near it: on the real diagnostics replies, clean
# sentences scored 0.02 at most and reassuring ones 0.89-0.99, so the cut could
# move a long way either side without changing a verdict.
REASSURES_CUT = 0.5
ACCEPTS_CALL_CUT = 0.5
# Higher for the one that would take a safety flag away, and it only does that
# when MAINTENANCE_EXEMPTION_SWITCH is on - see router._with_maintenance_check.
MAINTENANCE_CUT = 0.7
# ...and, when Jev routed the message, only when its own safety score was this
# low as well. Both have to say "nothing is wrong with the car".
MAINTENANCE_SAFETY_CEILING = 0.10
MAINTENANCE_EXEMPTION_SWITCH = "KEYWORD_NET_MAINTENANCE_EXEMPTION"

# The one thing Jev gets consistently wrong and a regex gets right. The fault
# code list's own severity line - "safe to drive the vehicle with care (fault
# code list, P0420)" - is a documented answer, and the reassurance question even
# says so, but Jev scored it 0.66-0.93 on real replies. Scoring whole replies
# was 10/14 for that reason alone; sentence by sentence, with this line taken
# out by code first, it was 14/14. So a sentence carrying it, cited to the list,
# is never sent. The citation has to be there: the words alone are what a
# poisoned bulletin would copy.
DOCUMENTED_SEVERITY = re.compile(
    r"safe to drive[^.]*with (?:care|caution)[^.]*\(fault code list, [PBCU]\d{4}\)",
    re.IGNORECASE,
)

# The same break the router uses to drop sentences about tickets, so "a
# sentence" means one thing in both places.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

# A diagnostics answer is asked for in about 80 words, four to eight sentences.
# One much longer than that is not what was measured, and one request per
# sentence would make it the dearest check here; the regex judges it alone.
MAX_SENTENCES = 16
# Longest a sentence sent to Jev may be. The API takes far more; nothing a
# customer is shown in one sentence needs it.
MAX_SENTENCE_CHARS = 600

# One request per sentence, sent side by side. TypeSafe's own guidance is that
# several candidates in one state shift one another's scores, so they are not
# batched into one request. Bounded, so a burst of safety messages queues checks
# rather than opening a thread each.
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jev-check")
# The whole check, however many sentences. Each request already gives up after
# jev_triage.TIMEOUT; this is for requests still waiting for a thread.
DEADLINE_SECONDS = 6.0

# How often each check was answered, how often Jev could not answer and the
# code that was there before decided instead, and what the answers cost. A
# fallback is silent by design, which is why it is counted: with the key revoked
# every check would quietly go back to the regex. /metrics reports these.
STATS = {name: {"answered": 0, "fell_back": 0, "tokens": 0} for name in QUESTIONS}
_STATS_LOCK = threading.Lock()


def _count(name: str, outcome: str, tokens: int = 0) -> None:
    with _STATS_LOCK:
        STATS[name][outcome] += 1
        STATS[name]["tokens"] += tokens


def enabled(name: str) -> bool:
    """Whether this check will ask Jev at all: a key, and not switched off."""
    return typesafe.configured() and limits.setting(SWITCHES[name], 1) > 0


def maintenance_exemption_on() -> bool:
    """Whether a maintenance question may take the keyword net's flag away.

    Off by default. The safety net is the owner's to change, and shadow mode is
    how the evidence to decide that is collected: the probability is recorded
    on every message the net alone flagged, whether or not this is on.
    """
    return limits.setting(MAINTENANCE_EXEMPTION_SWITCH, 0) > 0


# What the one question in each request is called. "q", because that is what
# the measurement called it: the name is part of what Jev reads, and the point
# of the eval importing the questions is that nothing differs.
ASKED_AS = "q"


def _ask_one(name: str, state: dict) -> tuple[float | None, int]:
    """One question, one request. (probability, input tokens); None when unanswered."""
    try:
        # No retries: the code this falls back to is instant, so a second wait
        # for Jev only ever costs the customer time. Same reasoning, and the same
        # timeout, as triage's (jev_triage.TIMEOUT).
        answer = typesafe.ask(state, {ASKED_AS: QUESTIONS[name]}, timeout=jev_triage.TIMEOUT, retries=0)
    except typesafe.TypeSafeUnavailable as e:
        log.warning("Jev could not answer the %s check, the code decides: %s", name, e)
        return None, 0
    except Exception as e:  # noqa: BLE001 - a check must never cost the customer their answer
        log.warning("the %s check failed (%s), the code decides", name, type(e).__name__)
        return None, 0
    p = typesafe.probability(answer.get("answers") or {}, ASKED_AS)
    return p, (answer.get("usage") or {}).get("input_tokens") or 0


def _single(name: str, state: dict) -> float | None:
    if not enabled(name):
        return None
    p, tokens = _ask_one(name, state)
    _count(name, "fell_back" if p is None else "answered", tokens)
    return None if p is None else round(p, 3)


# ------------------------------------------------------------ reassurance


def sentences(text: str) -> list[str]:
    """`text` as sentences, line by line, empty ones dropped."""
    found = []
    for line in (text or "").split("\n"):
        for s in SENTENCE_BREAK.split(line):
            s = s.strip(" -*\t")
            if any(ch.isalpha() for ch in s):
                found.append(s)
    return found


def to_score(text: str) -> list[str]:
    """The sentences Jev is asked about: every one but the documented severity line."""
    return [s for s in sentences(text) if not DOCUMENTED_SEVERITY.search(s)]


def reassurance_scores(message: str, text: str) -> list[tuple[str, float]] | None:
    """Every sentence of `text` Jev was asked about, with its probability of reassuring.

    None when Jev cannot judge the whole answer - not configured, switched off,
    too long, or any one sentence unanswered. A partly judged answer is not a
    clean one, so it goes to the regex alone. An empty list is an answer with
    nothing to judge (every sentence the documented line), and asks nothing.
    """
    if not enabled("reassures"):
        return None
    candidates = to_score(text)
    if not candidates:
        return []
    if len(candidates) > MAX_SENTENCES:
        log.warning("an answer of %d sentences is past the reassurance check's %d; the regex judges it",
                    len(candidates), MAX_SENTENCES)
        _count("reassures", "fell_back")
        return None

    def one(sentence: str):
        return _ask_one("reassures", {"customer_message": message,
                                      "reply": sentence[:MAX_SENTENCE_CHARS]})

    try:
        futures = [_POOL.submit(one, s) for s in candidates]
    except RuntimeError:  # the pool is shut down: the process is stopping
        _count("reassures", "fell_back")
        return None
    done, late = wait_for(futures, timeout=DEADLINE_SECONDS)
    for f in late:
        f.cancel()
    results = [f.result() if f in done else (None, 0) for f in futures]
    tokens = sum(t for _, t in results)
    if late or any(p is None for p, _ in results):
        _count("reassures", "fell_back", tokens)
        return None
    _count("reassures", "answered", tokens)
    return [(s, round(p, 3)) for s, (p, _) in zip(candidates, results)]


# ------------------------------------------------------------ the other two


def maintenance_only(message: str) -> float | None:
    """How likely `message` is only asking when a part is due, with nothing wrong now."""
    return _single("maintenance", {"new_message": message})


def accepts_call(offer: str, message: str) -> float | None:
    """How likely `message` says yes to the advisor's call offered in `offer`."""
    return _single("accepts_call", {"assistant_last_asked": offer, "new_message": message})


def snapshot() -> dict:
    """What /metrics reports: two counters and the tokens, per check."""
    with _STATS_LOCK:
        return {f"jev_check_{name}_{k}": v for name, counts in STATS.items() for k, v in counts.items()}
