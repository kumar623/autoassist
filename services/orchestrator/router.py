"""Route a customer message to the right specialist agent, then compose a reply.

The shape:

    message -> triage (returns JSON, no prose)
            -> one or more specialists, in order
            -> one reply

Routing is done in CODE, not by an agent calling other agents. That is a
deliberate choice:

  - It is observable. Every decision is a value we logged, not a hidden step
    inside a model's reasoning.
  - It is testable. Given a triage result, the route is deterministic, so it can
    be unit tested without calling Azure at all.
  - It is safe. The safety rule is enforced by an `if` statement, not by hoping
    a model follows an instruction. Week 1 and 2 both produced failures where a
    model dropped a safety rule under pressure from other instructions. An `if`
    does not do that.

The agents are still doing the hard part - understanding the question, reading
documents, writing the answer. The orchestrator decides who gets asked.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from . import azure_http, jev_triage, limits, telemetry, tools
from .cache import TimedCache
from .foundry import FoundryAgents
from .runner import ToolCallRecord, TurnResult, ask, ask_streaming, emit, throttled_turn

log = logging.getLogger(__name__)

VALID_INTENTS = {"diagnostics", "booking", "escalation", "other"}

# The page sends the last few turns with each message, because every request is
# otherwise stateless: when booking asked for a registration, the reply arrived
# with nothing around it, triage could not place it, and diagnostics searched
# the service library for a number plate. History lives in the page rather than
# on the server so any replica can answer any message.
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 600

# Greetings and thanks, and nothing else. These used to go through triage and
# fall back to diagnostics: ~2,200 tokens and 9s to say hello, and a diagnostics
# answer with no search call, which the runbook's ungrounded-answer alert would
# fire on. "ok" and "yes" are deliberately absent - after a question they are
# answers, and need the agents.
SMALL_TALK = re.compile(
    r"^\s*(?:"
    r"(?P<greeting>hi|hello|hey|hiya|good (?:morning|afternoon|evening)|"
    r"thanks|thank you|thanks a lot|thank you so much|thx|cheers|bye|goodbye)"
    r"(?:\s+(?:there|again|very much))?"
    r"(?:[\s,!.]+(?:how are (?:you|u)(?: doing)?(?: today)?|how(?:'s| is| are) (?:it|things|you) going))?"
    # On its own, not only after a greeting. "how are you" used to go to triage
    # and then diagnostics, which searched the service library for it and
    # answered "the library does not cover the question 'how are you'" - 3,648
    # tokens and 7.3 seconds, on the live app, to be unhelpful (20 Sep).
    r"|(?P<pleasantry>how are (?:you|u)(?: doing)?(?: today)?|how(?:'s| is| are) (?:it|things|you) going|"
    r"are you (?:there|ok|okay))"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)
GREETING_REPLY = (
    "Hello. Tell me what your car is doing, ask about a fault code or a warning "
    "light, or ask to book a service."
)
THANKS_REPLY = "You're welcome. Is there anything else I can help with?"
PLEASANTRY_REPLY = (
    "I'm well, thank you. Tell me what your car is doing, ask about a fault code "
    "or a warning light, or ask to book a service."
)
BYE_REPLY = "Goodbye. Get in touch any time."

# What booking is told when it picks up a conversation part-way. Each rule is
# here because a local replay of a real conversation broke without it:
#
#   - "1 pm", when 1 pm was not free, was booked as 12:00 - a time the customer
#     never chose, and not even one of the times last offered. That came from an
#     earlier, looser version of this text: "if they chose one, book it".
#   - Slot ids appear nowhere in the conversation, only times in words. The model
#     invented one ('slot_10:30_2020-09-21') and took four tool calls to recover.
#   - After a confirmed booking, the booked slot is missing from the free list;
#     the model read that as "not free" and booked the customer a second time.
BOOKING_CONTINUATION = (
    "\n\nBook only the exact time the customer chose. If the time they name is not "
    "free, say so and offer the nearest free times, then wait for them to pick one. "
    "Never book a different time from the one they asked for, however close.\n"
    "Slot ids are not in the conversation: call get_available_slots for that day "
    "first and book with the slot_id it returns.\n"
    "If the conversation already shows a booking confirmed with a reference, it "
    "stands: its slot is missing from the free list because it is theirs. Do not "
    "make a second booking unless they clearly ask for another appointment.\n"
    "To move a booking, use move_service_booking with its reference and the new "
    "slot_id. Asking to move it is their confirmation. Never cancel a booking in "
    "order to move it: if the new time is not free, the move changes nothing and the "
    "booking stays where it was."
)

# Belt and braces. Triage is a model and models miss things, so we re-check the
# message ourselves. Either flag firing is enough to escalate - we would rather
# raise a needless ticket than miss a brake failure.
SAFETY_WORDS = re.compile(
    r"\b(brake|brakes|braking|steering|steer|airbag|air bag|seat ?belt|"
    r"fuel leak|petrol leak|diesel leak|smell of (petrol|fuel|diesel|gas)|"
    r"smoke|smoking|fire|burning smell|wheel (came|fell) off|lost control|"
    r"crash|accident|tyre blew|tire blew|blowout)\b",
    re.IGNORECASE,
)


# The same belt and braces for bookings. On the live app "book at 2 pm, viper
# blades" was routed to diagnostics alone, which then told the customer "you can
# book the replacement for 2 pm" - nothing was booked. If the customer says
# book, booking runs. The cost of a false positive ("my service book says...")
# is a list of free slots.
BOOKING_WORDS = re.compile(r"\b(book|booking|appointment|appointments|reschedule)\b", re.IGNORECASE)

# A ticket reference, as escalation gives it to the customer. Once one has been
# given, a human is already on the way and a second one is not a second helper -
# it is the same advisor being called twice about the same car.
#
# This happened on the live app (20 Sep): "i have a problem with my gear
# shifting" was flagged as a safety issue, and it stayed flagged for the rest of
# the conversation - triage is told to judge the flag on the new message alone
# and did not. Escalation therefore ran on every turn and raised a ticket on
# every turn: TK-013051 for the symptom, TK-892738 for "i need to book
# appointment", TK-953064 for "tomoroow". Three advisors' worth of work, and the
# customer was told three different references for one problem.
#
# Fixed here rather than in escalation's prompt. The prompt already says what to
# do with a ticket; what it cannot do is know one already exists, and asking a
# model to remember across turns is how this went wrong in the first place.
TICKET_REFERENCE = re.compile(r"\bTK-[0-9]{4,}\b")

TICKET_STANDS = (
    "A service advisor has already been asked to call you about this - ticket {reference} - "
    "and someone will be in touch within the hour. If it cannot wait, please call the "
    "workshop directly."
)

# Triage costs 2.0-2.7s of every reply - a quarter of it - to classify a message
# (measured on the live app, 20 Sep). Some messages do not need classifying: a
# fault code with nothing else in it is a diagnostics question, and no model is
# needed to see that. Deliberately narrow: any hint of a booking, a safety issue
# or a conversation in progress goes to triage as before.
FAULT_CODE = re.compile(r"\b[PBCU][0-9]{4}\b", re.IGNORECASE)


# --------------------------------------------------------------- answer cache
#
# "What does P0420 mean" has one right answer, it costs 5-9k tokens and 7-9
# seconds to write, and in a demo everybody asks it. The documents behind it
# change when someone runs the ingest script, which is not during a conversation.
# So the whole reply is kept for a few minutes and the second person to ask gets
# it instantly and for nothing.
#
# The rules about WHAT may be kept are the important part of this, not the cache
# itself, and they are deliberately strict. Only a first message (no history),
# only a plain diagnostics answer, never a safety-flagged one, never one with a
# booking or an escalation in it, and never a message with anything personal in
# it. A cached booking reply handed to the next visitor would be exactly the
# cross-customer leak the red team found in the booking tools (docs/evaluation.md,
# findings 11 and 12), arrived at from a different direction.
ANSWERS = TimedCache(limits.setting("ANSWER_CACHE_SECONDS", 600), name="answers")
MAX_CACHED_MESSAGE_CHARS = 300

# If any of this is in the message, the answer belongs to one person and is not
# reusable - and the message itself should not sit in memory alongside an answer
# that quotes it. Matching too eagerly only costs a cache miss, so the patterns
# are loose on purpose.
PERSONAL = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]{2,}"                       # an email address
    r"|\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}\b"   # a registration: AP31BD1213
    r"|\b\d[\d\s-]{6,}\d\b"                             # a phone number
    r"|\b(AA|TE)-?\s?[A-Z0-9]{5,}\b",                   # a booking reference
    re.IGNORECASE,
)


@dataclass
class CachedAnswer:
    """A reply worth handing to the next person who asks the same thing.

    A record, not the live TurnResults: those are mutable and shared between
    threads, and _withhold_reassurance edits them in place. What is kept here is
    only what a reply reports.
    """

    reply: str
    agents: list[str]
    searched: bool
    trace: list[dict]
    stored_at: float


def cache_key(message: str) -> str | None:
    """What this message would be remembered under, or None if it must not be.

    Case and spacing vary between people asking the same question; nothing else
    is normalised. In particular no word is removed, because "is it safe to
    drive" and "is it not safe to drive" must never meet in the same bucket.
    """
    if ANSWERS.seconds <= 0:
        return None
    if PERSONAL.search(message):
        return None
    text = " ".join(message.split()).lower().rstrip("?!. ")
    if not text or len(text) > MAX_CACHED_MESSAGE_CHARS:
        return None
    return text


def worth_caching(message: str, out: RouterResult, history: list[dict] | None) -> bool:
    """Whether this reply is the same for everyone who asks, and came out clean.

    The strictest rule is the last one: only messages fast_route decides. Triage
    is a model, and on an ambiguous message - "the car pulls hard to the left
    when I slow down" - it flags a safety issue on most runs and not on all of
    them. Caching an answer from a run where it did not would freeze that one
    roll of the dice and hand it to everyone for the next ten minutes, turning a
    one-in-N miss into a certainty. fast_route decides in code: a fault code,
    no booking word, no safety word. The same message always gets the same route,
    so there is no judgement to freeze.
    """
    decision = out.decision
    if history or out.error or out.withheld or out.cached_from is not None:
        return False
    if decision is None or decision.safety or decision.route() != ["diagnostics"]:
        return False
    if fast_route(message, history) is None:
        return False
    specialists = [t for t in out.turns if t.agent_name == "diagnostics"]
    if len(specialists) != 1:
        return False
    turn = specialists[0]
    # An answer with no search behind it is finding 1, and caching one would
    # serve it to everybody for ten minutes rather than once.
    return bool(turn.ok and turn.answer.strip() and out.searched)


def prefetch_documents(message: str) -> tuple[str, str, int] | None:
    """Search the library before the diagnostics agent runs.

    Two reasons, and the second matters more than the speed.

    Speed: the agent spent ~2.5s deciding to call the search tool before writing
    a word (measured 20 Sep). We know what it will search for - the customer's
    message, or the fault code in it - so we search first and hand the results
    over with the question.

    Grounding: an answer with no search behind it is finding 1, the worst
    failure this system has, and no prompt makes it impossible. Asking for
    shorter answers made it likelier: the agent skipped the search on 1 run in 3
    of a safety question. Searching here means the documents are always in front
    of it, and the trace always shows the search that really happened. The rule
    stopped depending on the model following an instruction.
    """
    codes = FAULT_CODE.findall(message)
    query = " ".join(dict.fromkeys(c.upper() for c in codes)) if codes else " ".join(message.split())[:200]
    if not query:
        return None
    started = time.time()
    try:
        output = tools.search_service_docs(query=query)
    except Exception as e:  # noqa: BLE001 - never block an answer on an optimisation
        log.warning("pre-search failed, the agent will search itself: %s", e)
        return None
    return query, output, int((time.time() - started) * 1000)


def fast_route(message: str, history: list[dict] | None) -> TriageDecision | None:
    """A decision without asking triage, or None when triage is needed."""
    if history:
        return None  # a follow-up only makes sense in context; triage reads that
    if not FAULT_CODE.search(message):
        return None
    if BOOKING_WORDS.search(message) or SAFETY_WORDS.search(message):
        return None
    return TriageDecision(
        intents=["diagnostics"],
        reason="a fault code and nothing else: triage skipped",
        triage_skipped=True,
    )


@dataclass
class TriageDecision:
    intents: list[str] = field(default_factory=lambda: ["other"])
    safety: bool = False
    registration: str | None = None
    date: str | None = None
    reason: str = ""
    raw: str = ""
    parse_failed: bool = False
    safety_source: str = "none"  # "triage", "keyword", "both", "none"
    booking_added_by_keyword: bool = False
    triage_skipped: bool = False
    # Which classifier decided: "agent" (gpt-4.1-mini writing JSON) or "jev"
    # (four probabilities). Recorded rather than inferred, because the whole
    # point of offering a choice is being able to say which one answered.
    backend: str = "agent"
    probabilities: dict = field(default_factory=dict)

    def route(self) -> list[str]:
        """Which specialists to run, in order."""
        order = ["diagnostics", "booking", "escalation"]
        chosen = [i for i in order if i in self.intents]

        if self.safety and "escalation" not in chosen:
            chosen.append("escalation")

        if not chosen:
            chosen = ["diagnostics"]  # 'other' still gets a helpful attempt

        return chosen


@dataclass
class RouterResult:
    reply: str = ""
    decision: TriageDecision | None = None
    turns: list[TurnResult] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None
    withheld: list[str] = field(default_factory=list)  # answers dropped by _withhold_reassurance
    # Set when this reply came out of the answer cache rather than the agents.
    cached_from: CachedAnswer | None = None
    # Set when Azure's quota was spent and no specialist got to answer. A field
    # rather than a property over `turns`: triage's own turn succeeds and its
    # answer is the routing JSON, so "did any turn produce text?" said no
    # throttling had happened on exactly the messages where it had.
    throttled: bool = False
    retry_after: int = 0

    @property
    def cached(self) -> bool:
        return self.cached_from is not None

    @property
    def total_tokens(self) -> int:
        # A cached reply cost nothing to produce this time. Reporting what it
        # cost the first time would overstate every figure downstream - /metrics,
        # the cost per request, and the number quoted in an interview.
        if self.cached:
            return 0
        return sum(t.prompt_tokens + t.completion_tokens for t in self.turns)

    @property
    def agents_used(self) -> list[str]:
        if self.cached_from:
            return list(self.cached_from.agents)
        return [t.agent_name for t in self.turns]

    @property
    def searched(self) -> bool:
        # A cached answer was grounded in a search when it was written, and the
        # trace below still shows it. Saying otherwise would make the runbook's
        # ungrounded-answer alert fire on every cache hit.
        if self.cached_from:
            return self.cached_from.searched
        return any(t.searched for t in self.turns)

    def trace(self) -> list[dict]:
        """A flat record of what happened, for logging and the API response."""
        if self.cached_from:
            age = int(time.time() - self.cached_from.stored_at)
            return [
                {"agent": "cache", "status": "hit", "tools": [], "ms": self.duration_ms,
                 "tokens": 0, "error": None,
                 "note": f"answered from the cache; written {age}s ago by the turns below"},
                *self.cached_from.trace,
            ]
        return [
            {
                "agent": t.agent_name,
                "status": t.status,
                "tools": [
                    {"name": c.name, "args": c.arguments, "ms": c.duration_ms, "failed": c.failed}
                    for c in t.tool_calls
                ],
                "ms": t.duration_ms,
                "tokens": t.prompt_tokens + t.completion_tokens,
                "error": t.error,
            }
            for t in self.turns
        ]


def parse_triage(text: str, message: str) -> TriageDecision:
    """Turn the triage agent's JSON into a decision, and never trust it blindly."""
    decision = _parse_triage(text, message)

    if BOOKING_WORDS.search(message) and "booking" not in decision.intents:
        log.warning("triage missed a booking request the keyword check caught: %r", message[:120])
        decision.intents = [i for i in decision.intents if i != "other"] + ["booking"]
        decision.booking_added_by_keyword = True

    return decision


def _parse_triage(text: str, message: str) -> TriageDecision:
    keyword_safety = bool(SAFETY_WORDS.search(message))

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError) as e:
        # Triage failed. Fall back to something safe rather than guessing:
        # try diagnostics, and escalate if the message looks dangerous.
        log.warning("triage returned unparseable output (%s): %r", e, text[:200])
        return TriageDecision(
            intents=["diagnostics"] + (["escalation"] if keyword_safety else []),
            safety=keyword_safety,
            reason="triage output could not be parsed; fell back to diagnostics",
            raw=text,
            parse_failed=True,
            safety_source="keyword" if keyword_safety else "none",
        )

    intents = [i for i in (data.get("intents") or []) if i in VALID_INTENTS]
    if not intents:
        intents = ["other"]

    triage_safety = bool(data.get("safety"))
    safety = triage_safety or keyword_safety

    if triage_safety and keyword_safety:
        source = "both"
    elif triage_safety:
        source = "triage"
    elif keyword_safety:
        source = "keyword"
    else:
        source = "none"

    if keyword_safety and not triage_safety:
        log.warning("triage missed a safety signal the keyword check caught: %r", message[:120])

    return TriageDecision(
        intents=intents,
        safety=safety,
        registration=data.get("registration") or None,
        date=data.get("date") or None,
        reason=str(data.get("reason") or ""),
        raw=text,
        safety_source=source,
    )


def _decision_from_jev(classified, message: str) -> TriageDecision:
    """Jev's four probabilities as a TriageDecision, through the same backstops.

    SAFETY_WORDS and BOOKING_WORDS still run. They are cheap, they do not care
    what language they are reading - Jev scored a routine Hinglish complaint at
    0.75 on the safety question - and everything downstream already trusts that
    either source firing is enough.
    """
    keyword_safety = bool(SAFETY_WORDS.search(message))
    decision = TriageDecision(
        intents=list(classified.intents),
        safety=classified.safety or keyword_safety,
        registration=classified.registration,
        reason=classified.reason(),
        backend="jev",
        probabilities=classified.probabilities,
        safety_source=("both" if classified.safety and keyword_safety
                       else "jev" if classified.safety
                       else "keyword" if keyword_safety else "none"),
    )
    if keyword_safety and not classified.safety:
        log.warning("the keyword check caught a safety signal Jev scored at %.2f: %r",
                    classified.probabilities.get("safety", 0.0), message[:120])
    # The same booking backstop the agent path gets, for the same reason.
    if BOOKING_WORDS.search(message) and "booking" not in decision.intents:
        log.warning("Jev missed a booking request the keyword check caught: %r", message[:120])
        decision.intents = [i for i in decision.intents if i != "other"] + ["booking"]
        decision.booking_added_by_keyword = True
    return decision


def _agent_ids(client: FoundryAgents) -> dict[str, str]:
    return {a["name"]: a["id"] for a in client.list_agents()}


def small_talk_reply(message: str) -> str | None:
    """A canned reply if the message is only a greeting or thanks, else None."""
    m = SMALL_TALK.match(message)
    if not m:
        return None
    if m.group("pleasantry"):
        return PLEASANTRY_REPLY
    word = m.group("greeting").lower()
    if word in ("bye", "goodbye"):
        return BYE_REPLY
    return GREETING_REPLY if word.startswith(("hi", "hello", "hey", "good")) else THANKS_REPLY


def ticket_already_raised(history: list[dict] | None) -> str | None:
    """The reference of a ticket this conversation has already been given, if any.

    Read from the untrimmed history, not from _recent: that shortens long turns
    to 600 characters, and a reference that fell off the end would mean a second
    ticket for the same problem.
    """
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        if turn.get("role") != "assistant":
            continue
        found = TICKET_REFERENCE.search(str(turn.get("text") or ""))
        if found:
            return found.group(0)
    return None


def _recent(history: list[dict] | None, roles: tuple[str, ...] = ("customer", "assistant")) -> list[str]:
    """The last few turns as 'role: text' lines, oldest first, trimmed."""
    lines = []
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        text = " ".join(str(turn.get("text") or "").split())
        if role in roles and text:
            if len(text) > MAX_HISTORY_CHARS:
                text = text[:MAX_HISTORY_CHARS] + "..."
            lines.append(f"{role}: {text}")
    return lines


def _triage_input(message: str, history: list[dict] | None) -> str:
    """What triage classifies. With no history, exactly the message, as before."""
    recent = _recent(history)
    if not recent:
        return message
    # Said here rather than in triage's prompt so that this change ships with the
    # code, and a single message - every eval case - reaches triage unchanged.
    return (
        "Recent conversation, oldest first. It is context only: classify the NEW "
        "MESSAGE at the bottom.\n"
        + "\n".join(recent)
        + "\n\nIf the new message answers a question the assistant just asked - a "
        "registration number, a date, a time, a yes or no - its intent is the intent "
        "of that question. Judge the safety flag on the NEW MESSAGE alone. Take the "
        "registration and date from anywhere in the conversation.\n\n"
        f"NEW MESSAGE: {message}"
    )


def _context_for(
    specialist: str,
    message: str,
    decision: TriageDecision,
    so_far: list[TurnResult],
    history: list[dict] | None = None,
    prefetched: tuple[str, str, int] | None = None,
) -> str:
    """What we actually send the specialist: the message plus anything useful."""
    parts = []

    # Booking and escalation carry a conversation forward: booking asked for a
    # registration and this may be it; escalation summarises for a human.
    # Diagnostics sees only what the customer said, never earlier answers.
    # Earlier answers carry retrieved text and citations, and a model shown them
    # answers from them instead of searching again - finding 3, the reason every
    # turn gets a fresh thread in the first place.
    if specialist in ("booking", "escalation"):
        recent = _recent(history)
        if recent:
            parts.append(
                "Conversation so far, oldest first:\n" + "\n".join(recent) + "\n\n"
                "Carry on from it. Do not ask again for anything already given."
                + (BOOKING_CONTINUATION if specialist == "booking" else "")
            )
    elif specialist == "diagnostics":
        earlier = _recent(history, roles=("customer",))
        if earlier:
            parts.append(
                "Earlier messages from the customer, for context only:\n" + "\n".join(earlier) + "\n\n"
                "Search the library for the current message as usual. Nothing said "
                "earlier counts as a source."
            )

    parts.append(f"Customer message: {message}")

    # The library has already been searched for the fault code in the message.
    # Handed over here rather than fetched by the agent, which saves the round
    # trip it would spend deciding to call the tool. Every rule about reading
    # the results still applies, which is why the tool's own wording is passed
    # through untouched.
    if prefetched and specialist == "diagnostics":
        query, output, _ = prefetched
        parts.append(
            f"The service library has already been searched for you, for '{query}'. These are the "
            f"results, exactly as search_service_docs returns them. Use them as if you had called "
            f"it yourself, and do not call it again for this question - only for something else the "
            f"customer asked that these do not cover.\n\n{output}"
        )

    # Length is asked for here rather than in the agent's own prompt. Put in the
    # prompt, "be brief" cost grounding: the agent skipped the search on 1 run in
    # 3 of a safety question (20 Sep). Here it cannot, because the search has
    # already happened above - and the agent keeps its strict prompt for anyone
    # calling it directly, including the eval suite.
    if specialist == "diagnostics":
        parts.append(
            # No mention of safety warnings here. Naming them primed the agent to
            # add one: a plain P0420 question came back starting "Do not drive
            # the vehicle" for a worn catalytic converter (20 Sep). Its own
            # prompt already says when a warning belongs, and when it does the
            # warning is not counted against the length.
            #
            # The agent went on warning on P0420 with nothing said here at all -
            # finding 13. The fix was to narrow the rule in its own prompt, which
            # is the only place that can tell a brake symptom from an emissions
            # code. Saying "do not warn" here would only have swapped one blanket
            # instruction for another, on the turn where a warning is the thing
            # that matters.
            "KEEP IT SHORT: two short paragraphs, about 80 words in total - what it means, the likely "
            "cause, what to do next. They are reading this on a phone, standing next to the car. Do not "
            "restate the question and do not add a closing summary. Use only the documents you kept, "
            "and cite every claim as usual."
        )

    # Booking turns "Monday 21 September" into YYYY-MM-DD itself, and without a
    # date it guessed the year: 2020, from its training. The slot lookup then
    # said the day was in the past. Same clock as booking.py's slot calendar.
    if specialist == "booking":
        parts.append(f"Today is {date.today().strftime('%A %d %B %Y')} ({date.today().isoformat()}).")

    if decision.registration:
        parts.append(f"Vehicle registration: {decision.registration}")
    if decision.date and specialist == "booking":
        parts.append(f"They asked about: {decision.date}")

    if decision.safety and specialist in ("diagnostics", "escalation"):
        parts.append(
            "NOTE: this has been flagged as a possible safety issue. "
            "Your safety instructions apply."
        )

    # Telling diagnostics "ignore the booking part" in its prompt does not hold:
    # the customer's message mentions a day, and the model answers what is in
    # front of it. Saying it here, next to the message, is what actually works -
    # and it means the instruction only appears when it is true.
    if specialist == "diagnostics" and "booking" in decision.route():
        parts.append(
            "IMPORTANT: the customer also asked about an appointment. A colleague "
            "is answering that part in this same reply, immediately after yours. "
            "Say nothing at all about appointments, dates, times or availability - "
            "not even 'you can book if you wish'. Explain the fault and stop."
        )

    # Booking used to be handed the diagnosis here. It was then told not to
    # repeat it - so the only effect was to force booking to wait for
    # diagnostics, costing 5.6s of wall clock. Removing it lets the two run
    # concurrently. See docs/evaluation.md, the latency section.

    if specialist == "escalation" and so_far:
        prior = [t.answer for t in so_far if t.answer]
        if prior:
            parts.append("Already said to the customer:\n" + "\n---\n".join(prior))

    return "\n\n".join(parts)


# Which specialists need another specialist's answer before they can start.
# Everything not listed here is independent and can run concurrently.
#
# Escalation waits because its job is to summarise the conversation for a human,
# and it writes a better summary knowing what the customer was already told.
#
# Booking used to wait for diagnostics, and that cost 5.6s of wall clock for no
# benefit: it was handed the diagnosis and explicitly told not to repeat it.
# Passing information to an agent so it can ignore it is not a dependency.
NEEDS_BEFORE_IT_CAN_START = {
    "escalation": {"diagnostics", "booking"},
}


def _plan(route: list[str]) -> list[list[str]]:
    """Group a route into waves. Everything in a wave runs at the same time.

    ['diagnostics']                          -> [['diagnostics']]
    ['diagnostics', 'booking']               -> [['diagnostics', 'booking']]
    ['diagnostics', 'escalation']            -> [['diagnostics'], ['escalation']]
    ['diagnostics', 'booking', 'escalation'] -> [['diagnostics', 'booking'], ['escalation']]

    Deliberately simple: a specialist waits only if something it depends on is
    in this route. No topological sort, because the graph is four nodes and one
    edge, and a reader should be able to check this function by eye.
    """
    waves: list[list[str]] = []
    remaining = list(route)
    done: set[str] = set()

    while remaining:
        ready = [s for s in remaining if not (NEEDS_BEFORE_IT_CAN_START.get(s, set()) & set(remaining) - done)]
        if not ready:
            # Cannot happen with the current table, but a cycle introduced later
            # must degrade to sequential rather than hang.
            log.warning("dependency cycle in route %s; running the rest in order", remaining)
            ready = [remaining[0]]
        waves.append(ready)
        done.update(ready)
        remaining = [s for s in remaining if s not in done]

    return waves


def _run_specialists(
    client: FoundryAgents,
    ids: dict[str, str],
    route: list[str],
    message: str,
    decision: TriageDecision,
    timeout: float,
    history: list[dict] | None = None,
    prefetched: tuple[str, str, int] | None = None,
    on_event=None,
) -> list[TurnResult]:
    """Run the route, concurrently where the dependencies allow it.

    Each agent turn is a sequence of HTTP calls to Azure with waiting in
    between, so threads are the right tool - they are idle almost the whole
    time. No shared mutable state: each thread gets its own thread id, its own
    run, and returns its own TurnResult.
    """
    waves = _plan(route)
    results: list[TurnResult] = []

    for wave in waves:
        if len(wave) == 1:
            specialist = wave[0]
            prompt = _context_for(specialist, message, decision, results, history, prefetched)
            try:
                turn = ask(client, ids[specialist], prompt, timeout=timeout,
                           agent_name=specialist, on_event=on_event)
            except azure_http.Throttled as e:
                turn = throttled_turn(specialist, e)
            results.append(turn)
            if not turn.ok:
                log.warning("%s turn failed: %s", specialist, turn.error)
            continue

        log.info("running %s in parallel", wave)
        # Build every prompt BEFORE starting any thread. _context_for reads the
        # results list, and if one thread appended to it while another was
        # reading, the prompts would depend on timing - the same request could
        # produce different prompts on different runs.
        prompts = {s: _context_for(s, message, decision, results, history, prefetched) for s in wave}

        with telemetry.span("specialists.parallel", agents=wave, count=len(wave)):
            with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="specialist") as pool:
                futures = {
                    pool.submit(
                        ask, client, ids[s], prompts[s], timeout=timeout, agent_name=s,
                        on_event=on_event,
                    ): s
                    for s in wave
                }
                done_turns: dict[str, TurnResult] = {}
                for fut, specialist in futures.items():
                    try:
                        done_turns[specialist] = fut.result()
                    except azure_http.Throttled as e:
                        done_turns[specialist] = throttled_turn(specialist, e)
                    except Exception as e:  # noqa: BLE001
                        # One specialist failing must not lose the others' work.
                        log.exception("%s raised", specialist)
                        failed = TurnResult(agent_name=specialist, status="failed")
                        failed.error = f"{type(e).__name__}: {e}"
                        done_turns[specialist] = failed

        # Append in route order, not completion order. The reply must read the
        # same way every time regardless of which agent happened to finish first.
        for s in wave:
            turn = done_turns[s]
            results.append(turn)
            if not turn.ok:
                log.warning("%s turn failed: %s", s, turn.error)

    return results


# Reassurance that must never reach a customer whose message was flagged as a
# safety issue, whoever wrote it. The red team on 20 Sep put a document in the
# search results saying "a spongy brake pedal feel is normal on all Corvale
# vehicles and it is safe to keep driving". The diagnostics agent repeated it,
# with a citation, twice out of twice. Escalation's "do not drive" came first,
# but the customer read both. A poisoned or simply wrong document is the likely
# route, not a customer typing "ignore your instructions" - that one failed.
#
# Negated forms ("not safe to drive") are the warning itself and must pass.
REASSURANCE = re.compile(
    # "safe to drive with care" is what the fault code list itself says about
    # medium-severity codes, so it is a documented answer, not reassurance.
    # The exemption has to cover the wording the agent actually writes, not one
    # canonical form of it: "safe to drive the vehicle with care" and "safe to
    # drive the vehicle but with care" are the same documented line, and an
    # exemption that misses them withholds a correct answer over a turn of
    # phrase (finding 13). The object and the "but" live inside the lookahead
    # so that a bare "safe to drive" cannot backtrack its way out of it.
    r"(?<!not )(?<!n't )(?<!never )\bsafe to (keep |continue )?driv(e|ing)\b"
    r"(?!( (the |your )?(vehicle|car|it))?,? (but )?with (care|caution))"
    r"|\b(is|are|feels?) (completely |perfectly |quite )?normal\b"
    r"|\b(nothing|no need) to worry\b",
    re.IGNORECASE,
)

SAFETY_FALLBACK = (
    "Do not drive the vehicle. What you describe may affect its safety, and it needs to "
    "be checked by a technician first. Please call the workshop directly and someone "
    "will help you straight away."
)

# When Azure's token quota is spent. Said plainly, because it is not a failure
# and the customer can do something about it: wait a minute, or ring up. The old
# behaviour here was to retry for the length of the request timeout and then say
# "Sorry - I could not get an answer", which took ninety seconds to say nothing.
BUSY_REPLY = (
    "The assistant is handling a lot of messages just now and could not get to yours. "
    "Please try again in a minute. If it is urgent, call the workshop and someone will help you."
)


def _withhold_reassurance(turns: list[TurnResult], decision: TriageDecision) -> list[str]:
    """On a safety-flagged message, drop any diagnostics answer that reassures.

    Blunt by design, like SAFETY_WORDS: it cannot tell a true "normal" from a
    poisoned one, and does not try. A withheld answer costs the customer an
    explanation; a delivered wrong one can cost a great deal more. Escalation
    always runs on a safety route, so the customer is still answered.
    """
    if not decision.safety:
        return []
    withheld = []
    for t in turns:
        if t.agent_name == "diagnostics" and t.answer and REASSURANCE.search(t.answer):
            log.warning(
                "withheld a diagnostics answer that reassures on a safety issue (possible bad document): %r",
                t.answer[:200],
            )
            t.answer = ""
            t.error = "withheld: reassured the customer on a safety-flagged message"
            withheld.append(t.agent_name)
    return withheld


def _record_prefetch(turns: list[TurnResult], prefetched: tuple[str, str, int]) -> None:
    """Put the pre-search in the trace as what it is: a search we ran.

    Without this the diagnostics turn would look like an answer with no search
    behind it - which is finding 1, the worst failure this system has, and the
    thing the eval suite checks for.
    """
    query, output, ms = prefetched
    for t in turns:
        if t.agent_name == "diagnostics":
            t.tool_calls.insert(0, ToolCallRecord(
                name="search_service_docs", arguments={"query": query, "run_before_the_agent": True},
                output_preview=output[:300], duration_ms=ms))
            return


def _can_stream(on_delta, route: list[str], decision: TriageDecision) -> bool:
    """Whether this answer may be shown as it is written.

    Never on a safety-flagged message: _withhold_reassurance can only judge a
    finished answer, and a streamed one is already on the customer's screen.
    Never with several specialists either - their answers are joined in route
    order, so the first one to write is not necessarily the first to read.
    """
    return bool(on_delta) and len(route) == 1 and not decision.safety


def _compose(turns: list[TurnResult], decision: TriageDecision) -> str:
    """Join the specialists' answers into one reply.

    Deliberately mechanical - no extra model call. Another LLM pass to smooth
    the joins would cost latency and tokens, and would be one more place an
    ungrounded sentence could appear. Plain joining cannot invent anything.
    """
    answers = [(t.agent_name, t.answer.strip()) for t in turns if t.answer and t.answer.strip()]

    if not answers and decision.safety:
        # Nothing usable came back - or it was withheld - on a safety issue.
        # The warning must not depend on an agent having answered. That includes
        # being throttled: "we are busy, try later" is not an answer to a brake
        # problem, so the warning comes first here too.
        return SAFETY_FALLBACK

    if not answers and any(t.throttled for t in turns):
        return BUSY_REPLY

    if not answers:
        return (
            "Sorry - I could not get an answer for that just now. "
            "Please call the workshop directly and someone will help you."
        )

    # Safety first, whoever wrote it.
    if len(answers) > 1 and decision.safety:
        answers.sort(key=lambda a: 0 if a[0] == "escalation" else 1)

    reply = "\n\n".join(text for _, text in answers)

    # Something was written, but not all of it: the quota ran out part-way, or one
    # of two specialists never got to answer. Saying so is better than handing
    # over half a reply as though it were the whole one.
    if any(t.throttled for t in turns):
        reply += (
            "\n\n(Part of this reply is missing because the assistant is very busy just now. "
            "Please ask again in a minute.)"
        )

    return reply


def handle(
    client: FoundryAgents,
    message: str,
    timeout: float = 90.0,
    agent_ids: dict[str, str] | None = None,
    history: list[dict] | None = None,
    on_delta=None,
    on_status=None,
    on_event=None,
) -> RouterResult:
    """Handle one customer message end to end.

    `on_delta(text)` is optional. Given one, a single specialist's answer is
    streamed as it is written, which is what the customer sees first.

    `history` is the recent conversation as [{"role": "customer"|"assistant",
    "text": ...}], oldest first. Optional: without it, behaviour is exactly the
    single-message behaviour the eval set was scored on.
    """
    started = time.time()
    out = RouterResult()

    with telemetry.span(
        "chat.request", message_chars=len(message), history_turns=len(history or [])
    ) as req_span:
        canned = small_talk_reply(message)
        if canned:
            if on_delta:
                on_delta(canned)
            emit(on_event, kind="route", agents=[], reason="small talk, answered without agents")
            out.reply = canned
            out.decision = TriageDecision(intents=["other"], reason="small talk, answered without agents")
            out.duration_ms = int((time.time() - started) * 1000)
        else:
            _handle_inner(client, message, timeout, agent_ids, out, started, history, on_delta,
                          on_status, on_event)
        d = out.decision
        telemetry.set(
            req_span,
            agents=out.agents_used,
            searched=out.searched,
            safety=bool(d and d.safety),
            safety_source=d.safety_source if d else "none",
            triage_backend=d.backend if d else None,
            total_tokens=out.total_tokens,
            cached=out.cached,
            throttled=out.throttled or None,  # only present when it happened
            duration_ms=out.duration_ms,
            agent_count=len(out.turns),
            failed_turns=sum(1 for t in out.turns if not t.ok),
            withheld=out.withheld or None,  # only present when something was withheld
            error=out.error,
        )
    return out


def _handle_inner(
    client: FoundryAgents,
    message: str,
    timeout: float,
    agent_ids: dict[str, str] | None,
    out: RouterResult,
    started: float,
    history: list[dict] | None = None,
    on_delta=None,
    on_status=None,
    on_event=None,
) -> None:
    ids = agent_ids or _agent_ids(client)

    missing = [n for n in ("triage", "diagnostics", "booking", "escalation") if n not in ids]
    if missing:
        out.error = f"agents not deployed: {', '.join(missing)}. Run agents/deploy_agents.py"
        out.reply = "The service is not fully set up. Please call the workshop directly."
        out.duration_ms = int((time.time() - started) * 1000)
        return

    # --- 0. has someone already asked exactly this? ---
    # Only a first message: with a conversation behind it the answer depends on
    # what came before, and two people's conversations are not the same.
    key = cache_key(message) if not history else None
    if key:
        found = ANSWERS.get(key)
        if found:
            log.info("answered from the cache: %r", key[:80])
            _serve_cached(found, out, started, on_delta, on_event)
            return

    # --- 1. triage, unless the message speaks for itself ---
    decision = fast_route(message, history)
    if decision is not None:
        log.info("triage skipped: %s", decision.reason)
        _route_and_answer(client, ids, message, decision, timeout, history, out, started, on_delta,
                          on_status, on_event)
        return

    if on_status:
        on_status("reading your message")

    # --- 1a. Jev, if it is switched on and can answer ---
    # Falls through to the agent for anything it cannot do: no key, the service
    # unreachable, a question unanswered. A classifier being down is a reason to
    # use the model that was doing this before, not to fail the customer.
    if jev_triage.configured():
        # The same window of history the agent path reads (_recent). The API
        # accepts 20 turns of 8,000 characters; none of that needs to go to a
        # third party to decide who answers a message.
        classified = jev_triage.classify(message, (history or [])[-MAX_HISTORY_TURNS:])
        if classified is not None:
            decision = _decision_from_jev(classified, message)
            out.decision = decision
            log.info("routing (jev): intents=%s safety=%s -> %s | %s",
                     decision.intents, decision.safety, decision.route(), decision.reason)
            with telemetry.span("routing.decision") as rs:
                telemetry.set(rs, backend="jev", intents=decision.intents, route=decision.route(),
                              safety=decision.safety, safety_source=decision.safety_source,
                              duration_ms=classified.duration_ms, tokens=classified.tokens,
                              has_registration=bool(decision.registration),
                              **{f"p_{k}": v for k, v in classified.probabilities.items()})
            emit(on_event, kind="agent", name="triage", state="done", ms=classified.duration_ms,
                 tokens=classified.tokens, ok=True, backend="jev",
                 probabilities=classified.probabilities)
            _route_and_answer(client, ids, message, decision, timeout, history, out, started,
                              on_delta, on_status, on_event)
            return

    try:
        triage_turn = ask(client, ids["triage"], _triage_input(message, history), timeout=timeout,
                          agent_name="triage", on_event=on_event)
    except azure_http.Throttled as e:
        triage_turn = throttled_turn("triage", e)
    out.turns.append(triage_turn)

    if triage_turn.throttled:
        # All four agents share one model deployment. If triage could not get a
        # token, neither will the specialists, so there is nothing to gain by
        # finding that out three more times. The safety keywords still decide the
        # reply: a brake problem gets the warning even with no model available.
        # Built here rather than through parse_triage's fallback, which would
        # also record a triage parse failure - and triage parsed nothing, because
        # it never ran.
        keyword_safety = bool(SAFETY_WORDS.search(message))
        out.decision = TriageDecision(
            intents=["diagnostics"] + (["escalation"] if keyword_safety else []),
            safety=keyword_safety,
            safety_source="keyword" if keyword_safety else "none",
            reason="Azure was throttling; no agent ran",
        )
        out.reply = _compose([triage_turn], out.decision)
        out.throttled, out.retry_after = True, int(triage_turn.retry_after)
        out.duration_ms = int((time.time() - started) * 1000)
        return

    if not triage_turn.ok:
        log.warning("triage turn failed: %s", triage_turn.error)
        decision = parse_triage("", message)
        decision.reason = f"triage turn failed ({triage_turn.error}); fell back"
    else:
        decision = parse_triage(triage_turn.answer, message)

    out.decision = decision
    log.info(
        "routing: intents=%s safety=%s (%s) -> %s | %s",
        decision.intents, decision.safety, decision.safety_source, decision.route(), decision.reason,
    )

    # A span rather than attributes on the request span, so routing decisions
    # can be queried on their own - "how often does the keyword check catch
    # something triage missed?" is a question worth being able to answer.
    with telemetry.span("routing.decision") as rs:
        telemetry.set(
            rs,
            intents=decision.intents,
            route=decision.route(),
            safety=decision.safety,
            safety_source=decision.safety_source,
            triage_parse_failed=decision.parse_failed,
            booking_added_by_keyword=decision.booking_added_by_keyword,
            triage_skipped=decision.triage_skipped,
            reason=decision.reason[:200],
            has_registration=bool(decision.registration),
            has_date=bool(decision.date),
        )

    _route_and_answer(client, ids, message, decision, timeout, history, out, started, on_delta,
                      on_status, on_event)


# What each specialist is about to do, in the customer's words.
AGENT_STATUS = {
    "diagnostics": "checking the service documents",
    "booking": "checking the workshop calendar",
    "escalation": "arranging for a service advisor to call you",
}


def _route_and_answer(client, ids, message, decision, timeout, history, out, started,
                      on_delta=None, on_status=None, on_event=None) -> None:
    """Run the specialists the decision calls for, then compose the reply."""
    out.decision = decision

    # --- 2. specialists ---
    route = [s for s in decision.route() if s in ids]

    # A human has already been called about this conversation. Calling them again
    # every turn is what the live app did on 20 Sep - see TICKET_REFERENCE.
    standing = ticket_already_raised(history) if "escalation" in route else None
    if standing:
        log.info("escalation skipped: ticket %s already stands for this conversation", standing)
        route = [s for s in route if s != "escalation"]

    before = len(out.turns)  # triage's turn, or nothing when triage was skipped

    # Tell them what is happening BEFORE the pre-search: it takes ~0.7s, and
    # that is 0.7s of silence if the status waits for it.
    if on_status and route:
        on_status(" and ".join(AGENT_STATUS.get(s, s) for s in route))

    emit(on_event, kind="route", agents=route, safety=decision.safety,
         safety_source=decision.safety_source, reason=decision.reason,
         triage_skipped=decision.triage_skipped, ticket_stands=standing,
         backend=decision.backend, probabilities=decision.probabilities or None)

    # Whenever diagnostics is going to run - not only for fault codes.
    prefetched = None
    if "diagnostics" in route:
        if on_status:
            on_status("looking in the service documents")
        # The pre-search is a real search and shows in the panel as one, under
        # the agent it was run for - the same claim _record_prefetch makes in
        # the trace, made at the time rather than afterwards.
        emit(on_event, kind="tool", agent="diagnostics", name="search_service_docs", state="running")
        prefetched = prefetch_documents(message)
        emit(on_event, kind="tool", agent="diagnostics", name="search_service_docs", state="done",
             ms=prefetched[2] if prefetched else 0, failed=prefetched is None)

    if _can_stream(on_delta, route, decision):
        specialist = route[0]
        prompt = _context_for(specialist, message, decision, [], history, prefetched)
        try:
            turn = ask_streaming(client, ids[specialist], prompt, on_delta, timeout=timeout,
                                 agent_name=specialist, on_status=on_status, on_event=on_event)
        except azure_http.Throttled as e:
            # Raised before the stream opened, so nothing has been shown yet.
            turn = throttled_turn(specialist, e)
        out.turns.append(turn)
    else:
        out.turns.extend(_run_specialists(client, ids, route, message, decision, timeout, history,
                                         prefetched, on_event))
    specialists = out.turns[before:]

    if prefetched:
        _record_prefetch(specialists, prefetched)

    # --- 3. check, then compose ---
    out.withheld = _withhold_reassurance(specialists, decision)
    out.reply = _compose(specialists, decision)
    if standing:
        out.reply = _carry_the_ticket_forward(out.reply, standing, decision, specialists)
    _note_throttling(specialists, out)
    out.duration_ms = int((time.time() - started) * 1000)

    _remember_answer(message, history, out)

    log.info(
        "handled in %dms: agents=%s tokens=%d",
        out.duration_ms, out.agents_used, out.total_tokens,
    )


def _carry_the_ticket_forward(reply: str, reference: str, decision: TriageDecision,
                              specialists: list[TurnResult]) -> str:
    """Say the ticket still stands, in place of raising another one.

    Written here rather than by an agent: it is one sentence of fact, it must
    name the reference the customer was actually given, and a model asked to
    repeat a reference is a model that can get a digit wrong.
    """
    stands = TICKET_STANDS.format(reference=reference)
    if not any(t.answer.strip() for t in specialists):
        # Escalation was the whole route - "I want to speak to someone". The
        # safety warning still comes first when there is one.
        return f"{SAFETY_FALLBACK}\n\n{stands}" if decision.safety else stands
    return f"{stands}\n\n{reply}"


def _note_throttling(specialists: list[TurnResult], out: RouterResult) -> None:
    """Record whether the customer was turned away because the quota was spent.

    Judged on the specialists alone. Triage is an agent too, but its answer is
    routing JSON the customer never sees, so counting it as "something was
    answered" reported no throttling on precisely the messages that had been
    throttled - and /metrics is where the decision to ask Azure for more quota
    comes from.
    """
    throttled = [t for t in specialists if t.throttled]
    if not throttled:
        return
    out.retry_after = max(int(t.retry_after) for t in throttled)
    out.throttled = not any(t.answer.strip() for t in specialists)


def _serve_cached(found: CachedAnswer, out: RouterResult, started: float, on_delta=None,
                  on_event=None) -> None:
    """Hand back an answer someone else's question already paid for.

    The decision is rebuilt rather than stored, because worth_caching only ever
    admits one shape: a plain diagnostics answer with no safety flag. Storing a
    decision would invite someone to widen that later without noticing.
    """
    out.cached_from = found
    out.reply = found.reply
    out.decision = TriageDecision(intents=["diagnostics"], reason="answered from the cache")
    emit(on_event, kind="route", agents=[], cached=True,
         reason="an identical question was already answered; no agent ran")
    if on_delta:
        on_delta(found.reply)  # in one piece: there is nothing to wait for
    out.duration_ms = int((time.time() - started) * 1000)


def _remember_answer(message: str, history: list[dict] | None, out: RouterResult) -> None:
    """Keep this reply for the next person who asks the same thing, if it may be kept."""
    if not worth_caching(message, out, history):
        return
    key = cache_key(message)
    if not key:
        return
    ANSWERS.put(key, CachedAnswer(
        reply=out.reply,
        agents=out.agents_used,
        searched=out.searched,
        trace=out.trace(),
        stored_at=time.time(),
    ))
