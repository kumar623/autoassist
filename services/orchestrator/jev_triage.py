"""Classify a customer message with Jev instead of with a chat model.

Triage is the one step in this system that is pure classification: read a
message, decide who should handle it, and say whether it looks like a safety
issue. A chat model does that by writing JSON and hoping the JSON is right. Jev
answers four yes/no questions and returns a probability for each, and the policy
- where to cut each probability - lives here, in code, where it can be read and
changed without asking a model to behave differently.

Measured on evals/routing_set.jsonl, 72 labelled messages, 21 September 2026,
after the triage prompt was fixed, with the keyword backstops applied to both
sides as production applies them (docs/evaluation.md, finding 16 - corrected on
22 September; Jev alone, without them, is 63/72 with 2 false alarms):

                        triage agent      Jev
    routes right           52/72         57/72
    held-out only          21/30         22/30
    safety caught          20/20         20/20
    safety false alarms        7             6
    latency median        2,118ms        352ms
    per 1,000 messages     $0.344        $0.056

Not a full replacement, and the gap is worth naming: Jev returns typed answers,
so it cannot do the free-text extraction the agent does for `registration` and
`date`. The registration is taken by a regex below. The date is dropped: booking
already works out "Monday 21 September" for itself and is handed the whole
conversation anyway.

SAFETY_WORDS still runs on top of this, unchanged. Jev scored a routine Hinglish
complaint ("engine se awaaz aa rahi hai") at 0.75 on the safety question, over
the bar - the vendor's own documentation says non-English accuracy is lower, and
a workshop in Vizag will get those. The keyword net is cheap and it does not
care what language it is reading.

Off by default. TRIAGE_BACKEND=jev turns it on; anything else uses the agent, and
so does a message Jev could not answer. A visitor can also pick either one for a
single message from the page, or ask to see both - router.plan_triage decides
what that means on a server with or without a key.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

import httpx

from . import typesafe

log = logging.getLogger(__name__)

SPECIALISTS = ("diagnostics", "booking", "escalation")

# One cut for all three intents, and a higher one for safety.
#
# Both were swept on the routing set and both are plateaus rather than knife
# edges. 0.6 for the intents beat per-intent thresholds tuned on the first 42
# cases: those scored 39/42 on the set they were fitted to and 22/30 on cases
# written afterwards, while a single flat cut scored 37/42 and 26/30. Fitting
# three numbers to 42 examples bought two cases and lost four.
#
# 0.7 for safety separated the labelled set completely on every run: every
# message that is a safety issue scored above it, and all but two that are not
# scored below.
INTENT_CUT = float(os.getenv("JEV_INTENT_CUT", "0.6"))
SAFETY_CUT = float(os.getenv("JEV_SAFETY_CUT", "0.7"))

# How long to wait for Jev before asking the agent instead - and no retries.
#
# Jev answers in about 350ms and the agent it falls back to takes about 2s, so
# the fallback IS the retry: waiting longer for Jev only ever costs the customer
# time. Without this, a stalled connection inherited azure_http's patience - a
# 30s read timeout, four attempts and 3.5s of backoff, about two minutes, longer
# than the 90s the whole request is allowed.
TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# An Indian registration, as customers type it: AP31BD1213, "ap 31 bd 1213".
# Jev cannot hand back a value it was not given options for, so this is code's
# job - and a regex is better at it anyway, being exact.
REGISTRATION = re.compile(r"\b([A-Z]{2})\s?(\d{1,2})\s?([A-Z]{1,3})\s?(\d{1,4})\b", re.IGNORECASE)

# Every question names `new_message` and says what the rest of the state is for.
# Failure mode 1 on TypeSafe's jaggedness page is literal reading: Jev answers
# the question you wrote, so the question has to say which part it is about.
FOCUS = ("Judge `new_message` only. `assistant_last_asked` and `conversation_so_far` are "
         "background, there to resolve what a short reply refers to - they are not the thing "
         "being judged.")

QUESTIONS = {
    "needs_diagnostics": {
        "type": "noul",
        "instructions": {
            "question": "Does `new_message` name a vehicle fault, symptom, warning light, fault "
                        "code, or a part the workshop should say something about?",
            "focus": FOCUS,
        },
        "criteria": {
            "true": {
                "covers": "A fault code, a warning light, a noise, a smell, a symptom, a part that "
                          "needs attention, when a part is due for replacement, or what a service "
                          "includes.",
                "naming_is_enough": "Naming the part or fault is enough. 'book at 2 pm, wiper "
                                    "blades' names a part, so this is true even though the "
                                    "customer only asked to book.",
                "also_true_when": "The customer describes a fault AND asks to come in - both are "
                                  "wanted, and this question is only about the vehicle half.",
            },
            "false": {
                "covers": "Nothing about the vehicle is mentioned at all: only scheduling, only a "
                          "request for a person, or a subject that is not the vehicle.",
                "example": "A bare registration, date, time or yes/no answering a booking question "
                           "names no fault and no part.",
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
                "severity_is_not_escalation": "A fault being serious, dangerous or expensive is NOT "
                                              "a reason for this to be true. Whether the car is "
                                              "unsafe is a different question, asked separately.",
                "needs_a_person": "This is true only when a PERSON is wanted: they asked for one, "
                                  "they are complaining, or it is a dispute, accident or breakdown.",
            },
        },
    },
    "safety": {
        "type": "noul",
        "instructions": {
            "question": "Could `new_message` plausibly describe something that makes the vehicle "
                        "unsafe to drive now?",
            "focus": FOCUS,
        },
        "criteria": {
            "true": {
                "covers": "Brakes, steering, airbags, seat belts, a smell of fuel, smoke, fire, a "
                          "wheel or suspension problem, losing control, a crash, or a warning "
                          "light the customer calls red.",
                "test": "Would a technician tell them to stop driving until it is looked at?",
                # Asking about plausibility rather than certainty is not a thumb on the scale: it
                # is the question worth asking when missing a brake failure costs more than an
                # unnecessary check. Measured - removing it made the model miss a case at 0.5.
                "when_unsure": "If it is genuinely unclear whether the fault affects safety, this "
                               "is true. Missing a brake failure costs more than an extra check.",
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


# How often Jev answered, and how often it could not and the agent did instead.
#
# A fallback is silent by design - the customer is answered either way - and
# that is exactly why it needs counting. With the key revoked or the account's
# allocation spent, every message would quietly go back to the agent and nothing
# would say so: a demo of "Jev" answered by gpt-4.1-mini. /metrics reports these.
STATS = {"answered": 0, "fell_back": 0}


@dataclass
class Classification:
    """What Jev decided, and the numbers it decided it from."""

    intents: list = field(default_factory=list)
    safety: bool = False
    registration: str | None = None
    probabilities: dict = field(default_factory=dict)
    duration_ms: int = 0
    tokens: int = 0

    def reason(self) -> str:
        """A sentence for the trace and the panel, in the same place the agent's
        `reason` goes - so the two backends read alike."""
        scores = " ".join(f"{k.replace('needs_', '')} {v:.2f}" for k, v in self.probabilities.items())
        return f"Jev: {scores}"


def configured() -> bool:
    """Whether Jev is the chosen backend AND has a key to use."""
    return os.getenv("TRIAGE_BACKEND", "agent").strip().lower() == "jev" and available()


def available() -> bool:
    """Whether Jev can be asked at all, whatever TRIAGE_BACKEND says.

    Separate from configured() because the page lets a visitor pick Jev for one
    message on a server whose default is the agent. That needs a key and nothing
    else; without one the router says so rather than pretending.
    """
    return typesafe.configured()


def state_for(message: str, history: list | None) -> dict:
    """Named fields, not a prose block.

    The chat model's own prompt was tried first and is worse for this: it buries
    the message under a preamble, which is failure modes 4 and 5 - indirection,
    and a large state full of irrelevant detail.
    """
    turns = history or []
    said = [t["text"] for t in turns if t.get("role") == "assistant"]
    return {
        "new_message": message,
        "assistant_last_asked": said[-1] if said else None,
        "conversation_so_far": [f"{t.get('role')}: {t.get('text')}" for t in turns],
    }


def registration_in(message: str, history: list | None) -> str | None:
    """The most recent registration anyone typed, newest first."""
    texts = [message] + [str(t.get("text") or "") for t in reversed(history or [])]
    for text in texts:
        found = REGISTRATION.search(text)
        if found:
            return "".join(found.groups()).upper()
    return None


def classify(message: str, history: list | None = None, *, for_routing: bool = True) -> Classification | None:
    """Ask Jev who should handle this. None means "ask the agent instead".

    Never raises. A classifier that is down is a reason to fall back to the
    model that was doing this before, not a reason to fail a customer's message.

    `for_routing=False` is the page's side-by-side comparison, where Jev's
    answer is only shown. It is not counted in STATS, and a failure is not
    logged as a fallback: nothing falls back when a comparison fails, and
    /metrics and the logs are read as what routing did. A comparison's log line
    names the exception type only - it is one card on a page, not worth a
    traceback or an HTTP client's text.
    """
    def count(outcome: str) -> None:
        if for_routing:
            STATS[outcome] += 1

    started = time.time()
    try:
        answer = typesafe.ask(state_for(message, history), QUESTIONS, timeout=TIMEOUT, retries=0)
    except typesafe.TypeSafeUnavailable as e:
        if for_routing:
            log.warning("Jev could not classify this message, falling back to the triage agent: %s", e)
        else:
            log.warning("Jev could not answer a comparison (%s)", type(e).__name__)
        count("fell_back")
        return None
    except Exception as e:  # noqa: BLE001 - a new dependency must not be able to break routing
        if for_routing:
            log.exception("Jev classification failed unexpectedly; falling back to the triage agent")
        else:
            log.warning("Jev could not answer a comparison (%s)", type(e).__name__)
        count("fell_back")
        return None

    answers = answer.get("answers") or {}
    probabilities = {}
    for name in QUESTIONS:
        found = typesafe.probability(answers, name)
        if found is None:
            if for_routing:
                log.warning("Jev did not answer %r; falling back to the triage agent", name)
            else:
                log.warning("Jev did not answer %r in a comparison", name)
            count("fell_back")
            return None
        probabilities[name] = round(found, 3)

    intents = [s for s in SPECIALISTS if probabilities[f"needs_{s}"] >= INTENT_CUT]
    count("answered")
    return Classification(
        # "other" rather than an empty list, so TriageDecision.route() applies its
        # own rule - an unmatched message still gets a diagnostics attempt.
        intents=intents or ["other"],
        safety=probabilities["safety"] >= SAFETY_CUT,
        registration=registration_in(message, history),
        probabilities=probabilities,
        duration_ms=int((time.time() - started) * 1000),
        tokens=(answer.get("usage") or {}).get("input_tokens") or 0,
    )
