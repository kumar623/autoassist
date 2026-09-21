"""Route a customer message to the right specialist agent, then compose a reply.

The shape:

    message -> answered at once? (small talk, the answer cache)
            -> who should answer: the message itself (a bare fault code),
               else Jev's probabilities (TRIAGE_BACKEND=jev, or the visitor
               picked Jev), else the triage agent's JSON - with the keyword
               backstops on top whichever it was
            -> one or more specialists, concurrently where they can be
            -> one reply

A visitor can pick the classifier for one message, and ask to see the other one
read it too. The other one is display only: see plan_triage and _compare_with.

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

import contextvars
import json
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import date

from . import azure_http, jev_triage, limits, telemetry, tools
from .cache import TimedCache
from .foundry import FoundryAgents
from .runner import ToolCallRecord, TurnResult, ask, ask_streaming, emit, throttled_turn

log = logging.getLogger(__name__)

# The specialists, in the order they run and are shown. With triage in front,
# the four agents /ready and handle() insist are deployed. roster.ORDER and
# jev_triage.SPECIALISTS say the same, and a test holds them to it.
SPECIALISTS = ("diagnostics", "booking", "escalation")
AGENTS = ("triage", *SPECIALISTS)
VALID_INTENTS = {*SPECIALISTS, "other"}

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
#
# Skipping escalation was not the whole of it (live app, 21 Sep). Diagnostics had
# raise_ticket too, so "How often should brake fluid be changed?" raised two
# tickets in one reply, one from each agent, and the customer was told one of
# them. On the next message escalation was skipped as above - and diagnostics
# raised a third ticket and said "I have raised a safety ticket", just below
# the line saying the first one still stood. The rule is now held in three places:
#
#   - tools.OneTicket, which every raise_ticket call for a message goes through:
#     at most one ticket per message, none while one stands, and none from
#     anyone but escalation when escalation is on the route
#   - _leave_the_ticket_to, which drops what any other agent says about a ticket
#     when escalation or the router is the one telling the customer
#   - diagnostics no longer has raise_ticket at all (agents/definitions). It
#     never sees its own earlier answers, so it could never see the customer
#     accept the ticket it offered; every call it made was one nobody asked for.
#
# "One stands" means the page told us so. The server keeps no conversation, so
# the ticket is found in what the page sends: the reference it was last given
# (handle's `ticket`), or failing that one read back out of the history. The
# history alone was not enough - it is six turns, and three exchanges after the
# ticket was named it had gone, and the next brake message raised a second one.
TICKET_REFERENCE = re.compile(r"\bTK-[0-9]{4,}\b")
# Anything shaped like a reference, including one a digit short. Used only to
# put the real reference in place of a wrong one (_name_the_new_ticket).
ANY_REFERENCE = re.compile(r"\bTK-[0-9]+\b")

# No callback time. booking.create_ticket promises one by urgency - within the
# hour for safety, one working day otherwise - and the router does not know the
# urgency of a ticket raised on an earlier message, so any time said here would
# be wrong for some tickets. What the ticket promised was said when it was raised.
TICKET_STANDS = (
    "A service advisor has already been asked to call you about this - ticket {reference}. "
    "If it cannot wait, please call the workshop directly."
)

# What diagnostics is told when the ticket is someone else's to raise and to
# talk about. Next to the message, for the reason the appointment note in
# _context_for gives: said only in its prompt, the prompt's own "offer the
# customer an advisor's call" is what the model follows.
TICKET_STANDS_NOTE = (
    "IMPORTANT: a service advisor has already been asked to call the customer about this "
    "conversation, and this reply tells them so in a sentence of its own. Do not raise a ticket, "
    "do not offer one, and say nothing about tickets or advisors."
)
TICKET_OWNED_NOTE = (
    "IMPORTANT: a colleague is arranging a call from a service advisor in this same reply, "
    "straight after yours, and gives the customer the ticket reference. Do not raise a ticket, "
    "do not offer one, and say nothing about tickets or advisors."
)

# A sentence about a ticket: the word, a reference, or the advisor's call that a
# ticket is. Diagnostics is now told to offer "a call from a service advisor" and
# never to say "ticket", so the word alone let that offer through - and a
# customer whose call was already arranged was asked whether they wanted one.
# An advisor with no call in the sentence ("have a service advisor check the
# pads") is advice, and stays.
TICKET_TALK = re.compile(
    r"\bticket|\bTK-[0-9]+"
    r"|\badvis[eo]rs?\b[^.!?]*\b(?:call|phone|ring|contact|in touch)"
    r"|\b(?:call|phone|ring|contact)\b[^.!?]*\badvis[eo]rs?\b"
    r"|\b(?:call|phone|ring) you\b|\bcall(?:ed)? back\b",
    re.IGNORECASE,
)
# The warning, in the ways the agents write it (typographic apostrophes are made
# plain first, see _warns). It answers one question: does a reply that lost
# sentences to the rule above still tell the customer not to drive? A phrasing
# it misses costs a second warning, never a missing one - see _keep_the_warning.
DO_NOT_DRIVE = re.compile(
    r"\b(?:do not|don't|dont|should not|shouldn't|must not|mustn't|cannot|can't|never|stop|avoid)"
    r"(?: be)? driv"
    r"|(?:\bnot |n't |\bun)safe to (?:keep |continue )?driv"
    r"|\b(?:do not|don't|stop) us(?:e|ing) (?:the|your|this) (?:car|vehicle)",
    re.IGNORECASE,
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

# The customer saying yes to diagnostics' offer of an advisor's call. Only
# escalation can make that call, and neither classifier is asked about
# accepting an offer: Jev's booking question counts a yes that answers an
# appointment question, escalation's has no such clause, and a bare "yes please"
# that matches nothing goes to diagnostics - which is told never to say that
# someone will call. So, like BOOKING_WORDS, in code: see _with_accepted_offer.
ADVISOR_OFFER = re.compile(r"[^.!?\n]*\badvis[eo]rs?\b[^.!?\n]*\?", re.IGNORECASE)
YES = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|please|go ahead|(?:ok|okay)\b[\s,]*(?:please|go ahead|do it|sure))\b",
    re.IGNORECASE,
)
NOT_YES = re.compile(r"\b(?:no|not|don't|dont|never|later)\b", re.IGNORECASE)
MAX_YES_WORDS = 8

# Triage costs 2.0-2.7s of every reply - a quarter of it - to classify a message
# (measured on the live app, 20 Sep). Some messages do not need classifying: a
# fault code with nothing else in it is a diagnostics question, and no model is
# needed to see that. Deliberately narrow: any hint of a booking, a safety issue
# or a conversation in progress goes to triage as before.
FAULT_CODE = re.compile(r"\b[PBCU][0-9]{4}\b", re.IGNORECASE)


# ------------------------------------------------ which classifier, per message
#
# What the page may ask for: whatever this server does (TRIAGE_BACKEND), Jev, or
# the triage agent. "auto" is what every caller got before there was a choice,
# and still gets without asking - the eval runner and the red team included.
TRIAGE_CHOICES = ("auto", "jev", "agent")

# Said on the page and in the reply when the classifier that read a message is
# not the one the visitor picked. Fixed wording on purpose: these go to a public
# page, and exception text from an HTTP client does not.
NO_JEV_KEY = "no TypeSafe key is set on this server"
JEV_DID_NOT_ANSWER = "Jev could not answer"

# $ per million tokens, as the routing eval prices the two classifiers
# (evals/compare_triage.py, docs/evaluation.md finding 16), so the cost the page
# shows is worked out the way the finding's $0.056 and $0.344 per 1,000
# messages were. A test holds the two tables to each other.
PRICE_PER_MTOK = {"agent": 0.40, "jev": 0.042}
COST_BASIS = {
    "jev": "input tokens at $0.042 per million; Jev's output is free",
    "agent": "all tokens at gpt-4.1-mini's $0.40 per million input rate, as the routing eval prices "
             "it - a floor rather than a quote, since output is charged too",
}

# The compared classifier runs beside the answer, never in front of it. Bounded,
# so a page full of visitors ticking "compare" queues comparisons rather than
# starting a thread each. The agent gets a shorter leash than a real triage turn:
# nothing waits on it but a card on the page.
_COMPARE = ThreadPoolExecutor(max_workers=4, thread_name_prefix="compare")
COMPARE_TIMEOUT = 20.0
# How long a finished answer waits for its comparison. It is normally done
# first - Jev takes ~350ms, the agent ~2s, the specialists 5-8s - and when it
# is not, the answer goes without it and the page says why.
COMPARE_GRACE_SECONDS = 5.0

# What comparisons have spent since this process started, per classifier, for
# /metrics. Kept apart because a Jev token and an agent token are priced about
# ten times apart, so one sum would move more for the cheap one than the dear
# one. Counted when each comparison finishes, not when its answer goes: one
# still running at the grace deadline is still billed, and is counted then.
COMPARE_SPENT = {"jev": 0, "agent": 0}
_COMPARE_SPENT_LOCK = threading.Lock()   # comparisons finish on several threads at once


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
# are loose on purpose. The registration pattern is jev_triage's, so what counts
# as a registration here cannot drift from what Jev's path extracts as one.
PERSONAL = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]{2,}"                       # an email address
    rf"|{jev_triage.REGISTRATION.pattern}"             # a registration: AP31BD1213
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


def cache_key(message: str, decided_by: str | None = None) -> str | None:
    """What this message would be remembered under, or None if it must not be.

    Case and spacing vary between people asking the same question; nothing else
    is normalised. In particular no word is removed, because "is it safe to
    drive" and "is it not safe to drive" must never meet in the same bucket.

    `decided_by` names the classifier that routed the answer, when that is not
    this server's default (TriagePlan.variant_for). An answer one classifier
    routed is then never handed to someone who asked for the other, which would
    make the page's toggle show no difference at all. An answer no classifier
    routed - a bare fault code - has none, and is shared by every choice. It
    sits on a line of its own: the text has had its whitespace collapsed, so no
    message can reach that key.
    """
    if ANSWERS.seconds <= 0:
        return None
    if PERSONAL.search(message):
        return None
    text = " ".join(message.split()).lower().rstrip("?!. ")
    if not text or len(text) > MAX_CACHED_MESSAGE_CHARS:
        return None
    return f"{decided_by}\n{text}" if decided_by else text


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
    # A reply that raised a ticket names it, and the next person to be handed
    # that reply would have a ticket of someone else's in their conversation -
    # which ticket_already_raised would then find, and skip escalation for them.
    if any(c.name == "raise_ticket" for t in out.turns for c in t.tool_calls):
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
    parse_failed: bool = False
    # Who flagged safety: the classifier ("triage" for the agent, "jev"), the
    # keyword check, "both", or "none".
    safety_source: str = "none"
    booking_added_by_keyword: bool = False
    # Escalation added because the message said yes to an advisor's call that
    # the last reply offered (_with_accepted_offer).
    advisor_call_accepted: bool = False
    triage_skipped: bool = False
    # Which classifier decided: "agent" (gpt-4.1-mini writing JSON) or "jev"
    # (four probabilities). Recorded rather than inferred, because the whole
    # point of offering a choice is being able to say which one answered.
    backend: str = "agent"
    probabilities: dict = field(default_factory=dict)
    # What the classifier said on its own, before the keyword net, and which
    # words the net matched. None when no classifier answered (the fast route,
    # the fallback). Nothing decides from these - `intents` and `safety` are
    # still the decision. They are kept so the page can show a Jev safety score
    # of 0.05 next to the "brake" that escalated the message anyway.
    classifier_intents: list[str] | None = None
    classifier_safety: bool | None = None
    keyword_match: str | None = None
    booking_match: str | None = None

    def route(self) -> list[str]:
        """Which specialists to run, in order."""
        chosen = [i for i in SPECIALISTS if i in self.intents]

        if self.safety and "escalation" not in chosen:
            chosen.append("escalation")

        if not chosen:
            chosen = ["diagnostics"]  # 'other' still gets a helpful attempt

        return chosen

    def classifier_route(self) -> list[str] | None:
        """The route the classifier's own answer gave, before the keyword net.

        Through route() itself, so "the net changed the outcome" means the same
        rules applied to two inputs, not two sets of rules.
        """
        if self.classifier_intents is None or self.classifier_safety is None:
            return None
        return TriageDecision(intents=list(self.classifier_intents), safety=self.classifier_safety).route()


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
    # What the visitor asked to classify this message with, which classifier
    # did ("none" when none did), why that differs, and what it made of the
    # message - see _choice. Set on every path that reaches a reply, so the page
    # can say "triage was skipped" as plainly as "Jev read it".
    triage: dict | None = None
    # The other classifier's reading of the same message, when the visitor
    # asked to compare. Display only: nothing in this result is decided from it.
    comparison: dict | None = None
    # How many times an agent called raise_ticket for this message, and whether
    # a ticket was actually raised. More than one call and at most one ticket is
    # tools.OneTicket doing its job; the runbook's query reads both.
    ticket_requests: int = 0
    ticket_raised: bool = False
    # The conversation's ticket once this message has been answered: the one that
    # stood, or the one raised for it. The page keeps it and sends it back with
    # every message (handle's `ticket`), because its six turns of history forget
    # a reference three exchanges after it was given.
    ticket: str | None = None

    @property
    def cached(self) -> bool:
        return self.cached_from is not None

    @property
    def comparison_tokens(self) -> int:
        """What the compared classifier spent, as far as this answer saw it: 0
        for one still running when the answer went. Kept out of total_tokens,
        which is what the answer cost. /metrics reads COMPARE_SPENT instead,
        which has the late ones too, per classifier."""
        other = (self.comparison or {}).get("other") or {}
        return int(other.get("tokens") or 0)

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
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("triage returned unparseable output (%s): %r", e, text[:200])
        return _keyword_fallback(message, "triage output could not be parsed; fell back to diagnostics",
                                 parse_failed=True)

    intents = [i for i in (data.get("intents") or []) if i in VALID_INTENTS]
    decision = TriageDecision(
        intents=intents or ["other"],
        registration=data.get("registration") or None,
        date=data.get("date") or None,
        reason=str(data.get("reason") or ""),
    )
    return _with_backstops(decision, message, "triage", classifier_safety=bool(data.get("safety")))


def _decision_from_jev(classified, message: str) -> TriageDecision:
    """Jev's four probabilities as a TriageDecision, through the same backstops.

    They matter as much here: the keyword check does not care what language it
    is reading, and Jev scored a routine Hinglish complaint at 0.75 on the
    safety question.
    """
    decision = TriageDecision(
        intents=list(classified.intents),
        registration=classified.registration,
        reason=classified.reason(),
        backend="jev",
        probabilities=classified.probabilities,
    )
    return _with_backstops(decision, message, "jev", classifier_safety=classified.safety)


def _with_backstops(decision: TriageDecision, message: str, who: str, classifier_safety: bool) -> TriageDecision:
    """SAFETY_WORDS and BOOKING_WORDS on top of whichever classifier decided.

    One function for both classifiers, so "the keyword net runs whatever made
    the decision" is true by construction rather than by keeping two copies
    alike. Either source flagging safety is enough, and everything downstream
    already trusts that. `who` names the classifier in safety_source and logs.
    """
    found = SAFETY_WORDS.search(message)
    keyword_safety = found is not None
    # Recorded before the net changes anything, so what the classifier said and
    # what the net did can be shown apart. Nothing below reads them.
    decision.classifier_intents = list(decision.intents)
    decision.classifier_safety = classifier_safety
    decision.keyword_match = found.group(0).lower() if found else None
    decision.safety = classifier_safety or keyword_safety
    decision.safety_source = ("both" if classifier_safety and keyword_safety
                              else who if classifier_safety
                              else "keyword" if keyword_safety else "none")
    if keyword_safety and not classifier_safety:
        scored = decision.probabilities.get("safety")
        log.warning("%s missed a safety signal the keyword check caught%s: %r", who,
                    f" (scored {scored:.2f})" if scored is not None else "", message[:120])
    return _with_booking_backstop(decision, message, who)


def _with_booking_backstop(decision: TriageDecision, message: str, who: str) -> TriageDecision:
    """Add booking when the customer plainly asked for an appointment and the
    classifier did not say so. `other` goes: it meant "none of the above"."""
    found = BOOKING_WORDS.search(message)
    decision.booking_match = found.group(0).lower() if found else None
    if found and "booking" not in decision.intents:
        log.warning("%s missed a booking request the keyword check caught: %r", who, message[:120])
        decision.intents = [i for i in decision.intents if i != "other"] + ["booking"]
        decision.booking_added_by_keyword = True
    return decision


def _keyword_fallback(message: str, reason: str, parse_failed: bool = False) -> TriageDecision:
    """The decision when no classifier answered: its output was unparseable, its
    turn failed, or Azure had no quota for it.

    Something safe rather than a guess: try diagnostics, escalate if the message
    looks dangerous, book if it asks to. The keywords need no model, so a brake
    problem still gets the warning when nothing else is working.
    """
    found = SAFETY_WORDS.search(message)
    keyword_safety = found is not None
    decision = TriageDecision(
        intents=["diagnostics"] + (["escalation"] if keyword_safety else []),
        safety=keyword_safety,
        safety_source="keyword" if keyword_safety else "none",
        reason=reason,
        parse_failed=parse_failed,
        keyword_match=found.group(0).lower() if found else None,
    )
    return _with_booking_backstop(decision, message, "the fallback")


@dataclass
class TriagePlan:
    """Which classifier reads a message: as the visitor asked, and as it can be done."""

    requested: str = "auto"
    first: str = "agent"        # the classifier asked first: "jev" or "agent"
    why: str | None = None      # set when `first` is not what was requested
    default: str = "agent"      # what "auto" means on this server

    @property
    def other(self) -> str:
        """The classifier a comparison runs beside the chosen one. When Jev was
        chosen and could not answer, the agent routes and there is no second
        one to ask - see _handle_inner."""
        return "agent" if self.first == "jev" else "jev"

    def variant_for(self, backend: str | None) -> str | None:
        """The answer-cache variant for an answer `backend` routed: None for this
        server's default classifier, or when no classifier read the message.

        Picking the classifier the server already uses changes nothing about the
        answer, so it shares the default's cache entries; the other one does not.
        """
        return None if backend in (None, "none", self.default) else backend

    @property
    def cache_variant(self) -> str | None:
        """Where to look for an answer the classifier asked first would route.

        Only where to look. An answer is kept under the classifier that actually
        routed it (_remember_answer), which is not this one when Jev could not
        answer.
        """
        return self.variant_for(self.first)


def plan_triage(requested: str = "auto") -> TriagePlan:
    """Resolve the visitor's choice of classifier against this server.

    "auto" is exactly what happened before the choice existed. "jev" needs only a
    TypeSafe key, not TRIAGE_BACKEND=jev. Without a key the agent reads the
    message and the plan says why, so the page never shows the agent's answer
    under Jev's name.
    """
    if requested not in TRIAGE_CHOICES:
        raise ValueError(f"triage must be one of {', '.join(TRIAGE_CHOICES)}")
    default = "jev" if jev_triage.configured() else "agent"
    if requested == "auto":
        return TriagePlan(requested, default, None, default)
    if requested == "jev" and not jev_triage.available():
        return TriagePlan(requested, "agent", NO_JEV_KEY, default)
    return TriagePlan(requested, requested, None, default)


def _choice(plan: TriagePlan, used: str, why: str | None = None, reading: dict | None = None) -> dict:
    """What the visitor asked for, which classifier read the message ("none"
    when none did), why that is not what they asked for, and what it made of it."""
    return {"requested": plan.requested, "used": used, "why": why, "reading": reading}


def _reading(decision: TriageDecision | None, backend: str, ms: int = 0, prompt_tokens: int = 0,
             completion_tokens: int = 0, failed: str | None = None) -> dict:
    """What one classifier made of a message, and what the keyword net then did.

    One shape for the classifier that routed (in the route event) and the one
    only compared (in the compare event), so the page draws both with the same
    code and recomputes nothing. `own_*` is the classifier alone; `route` and
    `safety` are after the net; `keyword_changed` says whether the net made the
    difference - "how often should brake fluid be changed" at Jev 0.05,
    escalated because of "brake".

    `decision` is None when a compared classifier did not answer, and `failed`
    then says why in a few fixed words. Never exception text: this is shown on a
    public page.
    """
    tokens = prompt_tokens + completion_tokens
    own_route = decision.classifier_route() if decision else None
    answered = own_route is not None and not failed
    return {
        "backend": backend,
        "ok": answered,
        "error": None if answered else (failed or "no classifier answered"),
        "own_intents": decision.classifier_intents if answered else None,
        "own_safety": decision.classifier_safety if answered else None,
        "own_route": own_route if answered else None,
        "probabilities": (decision.probabilities or None) if answered else None,
        "reason": decision.reason[:200] if answered else None,
        "keyword_match": decision.keyword_match if decision else None,
        "booking_match": decision.booking_match if decision else None,
        "keyword_changed": bool(answered and (own_route != decision.route()
                                              or decision.classifier_safety != decision.safety)),
        # Which of the net's two rules made that difference, so the page can say
        # "brake" added safety rather than only that something changed.
        "keyword_added": [rule for rule, added in (
            ("safety", decision.safety and not decision.classifier_safety),
            ("booking", decision.booking_added_by_keyword),
        ) if added] if answered else [],
        # Jev's bar for its safety probability, so 0.05 can be read against it.
        "safety_cut": jev_triage.SAFETY_CUT if backend == "jev" else None,
        # What actually happened, which for the classifier that routed is the
        # keyword fallback when it did not answer: worth showing, it is the route.
        "route": decision.route() if decision else None,
        "safety": decision.safety if decision else None,
        "safety_source": decision.safety_source if decision else None,
        "ms": int(ms),
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        # Ten places: one Jev call costs about $0.00006, and fewer would round
        # away the figure being shown.
        "cost_usd": round(tokens * PRICE_PER_MTOK[backend] / 1e6, 10),
        "cost_basis": COST_BASIS[backend],
    }


def _differences(chosen: dict | None, other: dict | None) -> list[str]:
    """Where two readings disagree: the classifiers' own answers (own_route,
    own_safety) and the outcome after the keyword net (route, safety).

    Both levels, because they tell different stories. The net often makes the
    outcomes agree when the classifiers did not - that is its job - and a card
    that only compared outcomes would hide the disagreement worth seeing.
    """
    if not (chosen and other and chosen["ok"] and other["ok"]):
        return []
    return [k for k in ("own_route", "own_safety", "route", "safety") if chosen[k] != other[k]]


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

    The latest reference wins. There is only ever one ticket, but a streamed
    reply can carry a reference an agent mis-copied, with the real one said
    after it (_name_the_new_ticket) - and the real one is the last.

    Six turns at most, the window the page sends. The reference it was last
    given travels beside the history rather than in it, and handle() prefers
    that; this is for callers that send only history.
    """
    for turn in reversed((history or [])[-MAX_HISTORY_TURNS:]):
        if turn.get("role") != "assistant":
            continue
        found = TICKET_REFERENCE.findall(str(turn.get("text") or ""))
        if found:
            return found[-1]
    return None


def _with_accepted_offer(decision: TriageDecision, message: str, history: list[dict] | None) -> TriageDecision:
    """Add escalation when the message says yes to an advisor's call the last reply offered.

    Diagnostics offers the call and cannot make it (it has no raise_ticket);
    escalation makes it. Left to the classifiers, a bare "yes please" is
    whatever they make of it, and when that is nothing it goes to diagnostics,
    which is told never to say that someone will call - so a customer who
    accepted a call would get none. Narrow on purpose: the last reply has to
    have asked a question about an advisor, and the message has to be a short
    yes with no "not" in it. A needless escalation is one ticket for a
    conversation that has none; a missed one is a customer left waiting.
    """
    last = next((t for t in reversed(history or []) if t.get("role") == "assistant"), None)
    if last is None or "escalation" in decision.intents:
        return decision
    if not ADVISOR_OFFER.search(str(last.get("text") or "")):
        return decision
    if len(message.split()) > MAX_YES_WORDS or not YES.search(message) or NOT_YES.search(message):
        return decision
    log.info("escalation added: %r says yes to the advisor's call the last reply offered", message[:80])
    decision.intents = [i for i in decision.intents if i != "other"] + ["escalation"]
    decision.advisor_call_accepted = True
    return decision


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
    tickets: tools.OneTicket | None = None,
) -> str:
    """What we actually send the specialist: the message plus anything useful.

    `tickets` is the message's OneTicket, which says whose ticket it is.
    """
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

    # After the safety note, because diagnostics' safety instructions end by
    # offering the customer an advisor's call, and this is what says whether
    # that still applies. The OneTicket turns a raise_ticket away whatever the
    # answer says; this is so the answer does not then promise what was turned
    # away, or offer a second advisor when one is already on the way.
    if specialist == "diagnostics" and tickets is not None and tickets.speaker:
        parts.append(TICKET_STANDS_NOTE if tickets.standing else TICKET_OWNED_NOTE)

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
        # What the customer will actually read. Another agent's word on a ticket
        # is dropped from the reply (_leave_the_ticket_to), and escalation shown
        # "I have raised a ticket" could take it as done and not raise the real one.
        prior = [_without_ticket_talk(t.answer) for t in so_far if t.answer]
        prior = [p for p in prior if p.strip()]
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
    in this route. No topological sort, because the graph is three nodes and two
    edges (escalation waits for the other two), and a reader should be able to
    check this function by eye.
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
    tickets: tools.OneTicket | None = None,
) -> list[TurnResult]:
    """Run the route, concurrently where the dependencies allow it.

    Each agent turn is a sequence of HTTP calls to Azure with waiting in
    between, so threads are the right tool - they are idle almost the whole
    time. Each thread gets its own thread id, its own run, and returns its own
    TurnResult. The one thing they share is `tickets`, the message's OneTicket,
    and sharing it is the point: it is how two threads end up with one ticket.
    It is passed to each thread as an argument, and it locks.
    """
    waves = _plan(route)
    results: list[TurnResult] = []
    if tickets is None:
        tickets = _ticket_desk(route, standing=None)

    for wave in waves:
        if len(wave) == 1:
            specialist = wave[0]
            prompt = _context_for(specialist, message, decision, results, history, prefetched, tickets=tickets)
            turn = _ask_or_throttled(ask, specialist, client, ids[specialist], prompt, timeout=timeout,
                                     on_event=on_event, tickets=tickets)
            results.append(turn)
            if not turn.ok:
                log.warning("%s turn failed: %s", specialist, turn.error)
            continue

        log.info("running %s in parallel", wave)
        # Build every prompt BEFORE starting any thread. _context_for reads the
        # results list, and if one thread appended to it while another was
        # reading, the prompts would depend on timing - the same request could
        # produce different prompts on different runs.
        prompts = {s: _context_for(s, message, decision, results, history, prefetched, tickets=tickets)
                   for s in wave}

        with telemetry.span("specialists.parallel", agents=wave, count=len(wave)):
            with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="specialist") as pool:
                futures = {
                    pool.submit(
                        _ask_or_throttled, ask, s, client, ids[s], prompts[s], timeout=timeout,
                        on_event=on_event, tickets=tickets,
                    ): s
                    for s in wave
                }
                done_turns: dict[str, TurnResult] = {}
                for fut, specialist in futures.items():
                    try:
                        done_turns[specialist] = fut.result()
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

# The warning in the router's own words, for when an agent's was lost. The
# first half of SAFETY_FALLBACK, so the customer reads the same sentence
# whichever of the two it came from.
SAFETY_WARNING = (
    "Do not drive the vehicle. What you describe may affect its safety, and it needs to "
    "be checked by a technician first."
)
SAFETY_FALLBACK = (
    f"{SAFETY_WARNING} Please call the workshop directly and someone "
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


def _can_stream(on_delta, route: list[str], decision: TriageDecision, standing: str | None = None) -> bool:
    """Whether this answer may be shown as it is written.

    Never on a safety-flagged message: _withhold_reassurance can only judge a
    finished answer, and a streamed one is already on the customer's screen.
    Never with several specialists either - their answers are joined in route
    order, so the first one to write is not necessarily the first to read.
    Never while a ticket stands: what an agent says about a ticket then is
    dropped in favour of the router's own sentence (_leave_the_ticket_to), and
    that too needs the finished answer.
    """
    return bool(on_delta) and len(route) == 1 and not decision.safety and not standing


def _ticket_desk(route: list[str], standing: str | None) -> tools.OneTicket:
    """The OneTicket for one message. Escalation owns the ticket whenever it runs:
    writing a summary a human can act on is its whole job, and nobody else's."""
    return tools.OneTicket(standing=standing, owner="escalation" if "escalation" in route else None)


def _without_ticket_talk(text: str) -> str:
    """`text` without its sentences about a ticket, or `text` itself if it has none.

    Sentences, not the whole answer: the rest is a grounded explanation the
    customer still needs. Every such sentence goes, including one that also
    says not to drive: "Do not drive the vehicle - I have raised a safety
    ticket" kept whole is the contradiction this exists to remove. The warning
    is put back, in the router's own words, by _keep_the_warning.
    """
    if not TICKET_TALK.search(text):
        return text
    lines, dropped = [], False
    for line in text.split("\n"):
        sentences = SENTENCE_BREAK.split(line)
        kept = [s for s in sentences if not TICKET_TALK.search(s)]
        if len(kept) == len(sentences):
            lines.append(line)  # untouched, spacing and all
            continue
        dropped = True
        if any(s.strip() for s in kept):
            lines.append(" ".join(kept))
    if not dropped:
        return text
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _leave_the_ticket_to(speaker: str | None, turns: list[TurnResult]) -> list[str]:
    """Drop what other specialists say about a ticket when it is `speaker`'s to say.

    `speaker` is OneTicket.speaker: "router" when a ticket already stands, whose
    one sentence of fact _carry_the_ticket_forward writes; escalation when it is
    on the route and raises the ticket. Anyone else talking about a ticket then
    is either wrong - "I have raised a safety ticket" under the router's line
    that one already stood, on the live app (21 Sep) - or repeating what the
    customer is told anyway. Returns who was edited.

    Blunt, like _withhold_reassurance, and for the same reason: it does not try
    to tell a true sentence about a ticket from a false one.
    """
    if not speaker:
        return []
    edited = []
    for t in turns:
        if t.agent_name == speaker or not t.answer:
            continue
        kept = _without_ticket_talk(t.answer)
        if kept != t.answer:
            log.info("dropped what %s said about a ticket: it is for %s to say", t.agent_name, speaker)
            t.answer = kept
            edited.append(t.agent_name)
    return edited


def _warns(text: str) -> bool:
    """Whether `text` tells the customer not to drive. "Don\u2019t" is written with a
    typographic apostrophe as often as a plain one."""
    return bool(DO_NOT_DRIVE.search(text.replace("\u2019", "'")))


def _keep_the_warning(reply: str, decision: TriageDecision, edited: list[str]) -> str:
    """Put the do-not-drive warning back at the top if dropping ticket talk took it out.

    While a ticket stands, escalation is skipped, and diagnostics is the only
    agent left to warn. It often warns in the same sentence as it mentions the
    ticket - "You should not be driving the car, and your safety ticket
    TK-519169 is with an advisor" - and _leave_the_ticket_to drops that
    sentence whole. Nothing else would put the warning back: the fallback only
    covers a reply with no answer left in it at all.

    Only on a safety-flagged message, and only when something was dropped: the
    warning is not the router's to add to an answer it did not touch. When in
    doubt it is added - DO_NOT_DRIVE missing a phrasing costs the customer a
    second warning, and a missing one can cost a great deal more.
    """
    if not (decision.safety and edited) or _warns(reply):
        return reply
    log.warning("a safety reply lost its warning with what %s said about a ticket; the router's is added",
                ", ".join(edited))
    return f"{SAFETY_WARNING}\n\n{reply}"


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
    triage: str = "auto",
    compare: bool = False,
    ticket: str | None = None,
) -> RouterResult:
    """Handle one customer message end to end.

    `on_delta(text)` is optional. Given one, a single specialist's answer is
    streamed as it is written, which is what the customer sees first.

    `history` is the recent conversation as [{"role": "customer"|"assistant",
    "text": ...}], oldest first. Optional: without it, behaviour is exactly the
    single-message behaviour the eval set was scored on.

    `triage` is the visitor's choice of classifier: "auto" (this server's
    TRIAGE_BACKEND, and what every caller got before there was a choice), "jev"
    or "agent" - see plan_triage. `compare` also runs the other one on the same
    message, for the page to show beside it. Nothing is decided from that.

    `ticket` is the conversation's ticket reference as an earlier reply reported
    it (RouterResult.ticket), kept by the page beside the history. Like the
    history it is the page's word: a caller who forges one turns escalation off
    for their own conversation only, as a forged assistant turn already could.
    """
    plan = plan_triage(triage)
    started = time.time()
    out = RouterResult()

    with telemetry.span(
        "chat.request", message_chars=len(message), history_turns=len(history or [])
    ) as req_span:
        canned = small_talk_reply(message)
        if canned:
            if on_delta:
                on_delta(canned)
            out.triage = _choice(plan, "none", "small talk, answered without agents")
            emit(on_event, kind="route", agents=[], reason="small talk, answered without agents",
                 triage=out.triage)
            out.reply = canned
            out.decision = TriageDecision(intents=["other"], reason="small talk, answered without agents")
            out.duration_ms = int((time.time() - started) * 1000)
        else:
            _handle_inner(client, message, timeout, agent_ids, out, started, history, on_delta,
                          on_status, on_event, plan=plan, compare=compare, ticket=ticket)
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
            # Only present when an agent asked for a ticket: more requests than
            # tickets raised is OneTicket turning the extras away.
            ticket_requests=out.ticket_requests or None,
            ticket_raised=out.ticket_raised or None,
            error=out.error,
            triage_choice=plan.requested,
            compared=out.comparison is not None or None,  # only present when it happened
            comparison_tokens=out.comparison_tokens or None,
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
    plan: TriagePlan | None = None,
    compare: bool = False,
    ticket: str | None = None,
) -> None:
    plan = plan or plan_triage()
    ids = agent_ids or _agent_ids(client)

    missing = [n for n in AGENTS if n not in ids]
    if missing:
        out.error = f"agents not deployed: {', '.join(missing)}. Run agents/deploy_agents.py"
        out.reply = "The service is not fully set up. Please call the workshop directly."
        out.duration_ms = int((time.time() - started) * 1000)
        return

    # --- 0. has someone already asked exactly this? ---
    # Only a first message: with a conversation behind it the answer depends on
    # what came before, and two people's conversations are not the same.
    #
    # Never for a comparison, in either direction. The visitor asked to watch
    # two classifiers read the message, and a cached answer was read by neither;
    # and a comparison must have no way to change what the next visitor is served.
    #
    # Looked up under the classifier that will read the message - or under the
    # plain question when none will, because a bare fault code gets the same
    # answer whichever one the visitor picked.
    #
    # Nor with a ticket, even one sent with no history: a ticket is a
    # conversation behind the message. While it stands the reply can carry its
    # reference (_carry_the_ticket_forward), and kept, that reference would be
    # handed to the next visitor - whose next message would read it back as
    # their own ticket and skip escalation (see worth_caching).
    cache_for = plan if not (history or compare or ticket) else None
    decision = fast_route(message, history)
    key = cache_key(message, None if decision else plan.cache_variant) if cache_for else None
    if key:
        found = ANSWERS.get(key)
        if found:
            log.info("answered from the cache: %r", key[:80])
            out.triage = _choice(plan, "none", "an identical question was already answered")
            _serve_cached(found, out, started, on_delta, on_event)
            return

    # --- 1. who should answer: the message itself, then the chosen classifier ---
    if decision is not None:
        # No classifier reads a bare fault code, so there is nothing to choose
        # between and nothing to compare. The page says the toggle did not apply.
        out.triage = _choice(plan, "none", decision.reason)
        # A no-op today - fast_route only decides a first message, and a yes
        # needs an offer before it - but every path goes through it, so no
        # path can be the one that forgot.
        decision = _with_accepted_offer(decision, message, history)
        _record_decision(decision, choice=out.triage)
        _route_and_answer(client, ids, message, decision, timeout, history, out, started, on_delta,
                          on_status, on_event, cache_for=cache_for, ticket=ticket)
        return

    # The comparison runs beside the chosen classifier, never in front of it.
    # With the agent chosen, Jev is asked at the same moment: it takes ~350ms
    # and is done long before the answer. With Jev chosen, the agent waits for
    # Jev's turn. If Jev cannot answer, the agent routes the message, and an
    # agent turn started beside it would be the LLM compared with itself - and
    # paid for twice. The card then shows Jev as the one that did not answer.
    comparing = None
    if compare and plan.first != "jev":
        comparing = _start_comparison(plan.other, client, ids, message, history, timeout)
    try:
        classified = _classify(client, ids, message, history, timeout, out, started, plan, on_status, on_event)
        if compare and plan.first == "jev":
            comparing = _compare_after_jev(out, client, ids, message, history, timeout)
        if classified is None:
            return  # Azure was throttling; the reply is already written
        decision, measured = classified
        # After whichever classifier read it, Jev or the agent, and after their
        # keyword backstops or the keyword fallback, which it does not replace:
        # it only ever adds escalation. Not part of out.triage's reading, which
        # is what the classifier and the net made of the message - and never
        # applied to the compared one, which routes nothing.
        decision = _with_accepted_offer(decision, message, history)
        _record_decision(decision, choice=out.triage, compared=compare, **measured)
        _route_and_answer(client, ids, message, decision, timeout, history, out, started, on_delta,
                          on_status, on_event, cache_for=cache_for, ticket=ticket)
    finally:
        if comparing is not None:
            _finish_comparison(comparing, out, on_event)


def _classify(client, ids, message, history, timeout, out, started, plan, on_status=None, on_event=None):
    """The chosen classifier's decision and what was measured of it - or None
    when Azure had no quota, and the reply has already been written.

    Jev when the plan says so, the agent when it does not or when Jev could not
    answer. Either way out.triage says which one read the message, and why when
    it is not the one the visitor asked for.
    """
    if on_status:
        on_status("reading your message")

    why = plan.why
    if plan.first == "jev":
        classified = _classify_with_jev(message, history)
        if classified is not None:
            decision = _decision_from_jev(classified, message)
            emit(on_event, kind="agent", name="triage", state="done", ms=classified.duration_ms,
                 tokens=classified.tokens, ok=True, backend="jev",
                 probabilities=classified.probabilities)
            out.triage = _choice(plan, "jev", why,
                                 _reading(decision, "jev", classified.duration_ms, classified.tokens))
            return decision, {"duration_ms": classified.duration_ms, "tokens": classified.tokens,
                              **{f"p_{k}": v for k, v in classified.probabilities.items()}}
        # Said out loud, not only counted in STATS: a reply the agent routed
        # must not be shown on the page as Jev's.
        why = JEV_DID_NOT_ANSWER

    triage_turn = _ask_or_throttled(ask, "triage", client, ids["triage"], _triage_input(message, history),
                                    timeout=timeout, on_event=on_event)
    out.turns.append(triage_turn)
    spent = {"ms": triage_turn.duration_ms, "prompt_tokens": triage_turn.prompt_tokens,
             "completion_tokens": triage_turn.completion_tokens}

    if triage_turn.throttled:
        # All four agents share one model deployment. If triage could not get
        # a token, neither will the specialists, so there is nothing to gain
        # by finding that out three more times. The safety keywords still
        # decide the reply: a brake problem gets the warning with no model.
        out.decision = _keyword_fallback(message, "Azure was throttling; no agent ran")
        out.triage = _choice(plan, "agent", why,
                             _reading(out.decision, "agent", **spent, failed="Azure had no quota for it"))
        out.reply = _compose([triage_turn], out.decision)
        out.throttled, out.retry_after = True, int(triage_turn.retry_after)
        out.duration_ms = int((time.time() - started) * 1000)
        return None

    if triage_turn.ok:
        decision = parse_triage(triage_turn.answer, message)
        failed = "its JSON could not be read" if decision.parse_failed else None
    else:
        # Not a parse failure - triage parsed nothing, because its turn
        # failed - so not counted as one in /metrics.
        log.warning("triage turn failed: %s", triage_turn.error)
        decision = _keyword_fallback(message, f"triage turn failed ({triage_turn.error}); fell back")
        failed = "the triage agent's turn failed"
    out.triage = _choice(plan, "agent", why, _reading(decision, "agent", **spent, failed=failed))
    return decision, {}


def _classify_with_jev(message: str, history: list[dict] | None, for_routing: bool = True):
    """Jev's classification, or None when it could not answer.

    None falls through to the agent - the service unreachable, a question
    unanswered. A classifier being down is a reason to use the model that was
    doing this before, not to fail the customer. Whether Jev is asked at all is
    plan_triage's decision, made before this is called.
    """
    # The same window of history the agent path reads (_recent). The API
    # accepts 20 turns of 8,000 characters; none of that needs to go to a third
    # party to decide who answers a message.
    return jev_triage.classify(message, (history or [])[-MAX_HISTORY_TURNS:], for_routing=for_routing)


def _start_comparison(backend: str, client, ids, message, history, timeout) -> tuple[str, Future]:
    """Ask the other classifier in the background. See _compare_with."""
    try:
        # A copy of this request's context, so the comparison's spans (the
        # agent's turn, the call to Jev) sit under this request's trace in App
        # Insights rather than each starting a trace of its own.
        run = contextvars.copy_context().run
        return backend, _COMPARE.submit(run, _compare_and_count, backend, client, ids, message, history,
                                        timeout)
    except RuntimeError:  # the pool is shut down: the process is stopping
        return backend, _settled(_reading(None, backend, failed="the comparison could not start"))


def _compare_after_jev(out: RouterResult, client, ids, message, history, timeout) -> tuple[str, Future]:
    """The comparison when Jev was chosen, once Jev has had its turn.

    When Jev answered, the agent is asked now, and reads the message while the
    specialists write. When Jev did not, the agent has already read it to route
    it, so there is no second opinion to ask for: the other column is Jev, not
    answering, and nothing more is spent.
    """
    if (out.triage or {}).get("used") == "jev":
        return _start_comparison("agent", client, ids, message, history, timeout)
    return "jev", _settled(_reading(None, "jev", failed=JEV_DID_NOT_ANSWER))


def _settled(reading: dict) -> Future:
    """A comparison that is already over, in the shape of one still running."""
    done = Future()
    done.set_result(reading)
    return done


def _compare_and_count(backend: str, client, ids, message, history, timeout) -> dict:
    """_compare_with, and what it spent added to COMPARE_SPENT when it finishes -
    whether or not its answer waited for it."""
    reading = _compare_with(backend, client, ids, message, history, timeout)
    with _COMPARE_SPENT_LOCK:
        COMPARE_SPENT[backend] += reading["tokens"]
    return reading


def _compare_with(backend: str, client, ids, message, history, timeout) -> dict:
    """The other classifier's reading of the same message, for the page only.

    Nothing routing does reads what this returns: the route, the reply, the
    tickets and the answer cache are the chosen classifier's alone. It never
    raises, and it runs through the same backstops, so the page can show what
    the keyword net would have done to this classifier's answer too.
    """
    started = time.time()
    try:
        if backend == "jev":
            if not jev_triage.available():
                return _reading(None, "jev", failed=NO_JEV_KEY)
            classified = _classify_with_jev(message, history, for_routing=False)
            if classified is None:
                return _reading(None, "jev", int((time.time() - started) * 1000), failed=JEV_DID_NOT_ANSWER)
            return _reading(_decision_from_jev(classified, message), "jev", classified.duration_ms,
                            classified.tokens)

        # Its own thread and run, like every turn. No on_event: the panel's
        # triage card belongs to the classifier that routed.
        turn = _ask_or_throttled(ask, "triage", client, ids["triage"], _triage_input(message, history),
                                 timeout=min(timeout, COMPARE_TIMEOUT))
        spent = {"ms": turn.duration_ms, "prompt_tokens": turn.prompt_tokens,
                 "completion_tokens": turn.completion_tokens}
        if turn.throttled:
            return _reading(None, "agent", **spent, failed="Azure had no quota for it")
        if not turn.ok:
            return _reading(None, "agent", **spent, failed="the triage agent's turn failed")
        decision = parse_triage(turn.answer, message)
        if decision.parse_failed:
            return _reading(None, "agent", **spent, failed="its JSON could not be read")
        return _reading(decision, "agent", **spent)
    except Exception as e:  # noqa: BLE001 - a comparison must never cost the customer their answer
        # The type only. The text of an HTTP client's exception can carry a URL,
        # and a card on a demo page is not worth that risk.
        log.warning("the %s comparison failed (%s); the answer is unaffected", backend, type(e).__name__)
        return _reading(None, backend, int((time.time() - started) * 1000), failed="the comparison failed")


def _finish_comparison(pending: tuple[str, Future], out: RouterResult, on_event=None) -> None:
    """Collect the comparison, put it beside the answer, and tell the page.

    Called once the answer is ready, and waits a little rather than for ever:
    when the comparison is still going, the answer goes without it and the page
    says why. One still queued is cancelled; one already running finishes and
    is dropped, though what it spent still reaches COMPARE_SPENT.
    """
    backend, future = pending
    try:
        other = future.result(timeout=COMPARE_GRACE_SECONDS)
    except FutureTimeout:
        future.cancel()
        other = _reading(None, backend, failed="still running when the answer was ready")
    except Exception as e:  # noqa: BLE001 - _compare_with does not raise; this keeps that true if it ever does
        log.warning("the %s comparison failed (%s); the answer is unaffected", backend, type(e).__name__)
        other = _reading(None, backend, failed="the comparison failed")

    chosen = (out.triage or {}).get("reading")
    differences = _differences(chosen, other)
    out.comparison = {"chosen": chosen, "other": other, "differences": differences}
    emit(on_event, kind="compare", **out.comparison)

    # Its own span, like routing.decision, so "how often do they disagree, and
    # on what?" can be asked of App Insights without reading anyone's messages.
    with telemetry.span("routing.compare") as cs:
        both = bool(chosen and chosen["ok"] and other["ok"])
        telemetry.set(
            cs,
            chosen=chosen["backend"] if chosen else None,
            other=backend,
            other_ok=other["ok"],
            agrees=not differences if both else None,
            differences=differences or None,
            other_ms=other["ms"],
            other_tokens=other["tokens"],
        )


def _record_decision(decision: TriageDecision, choice: dict | None = None, compared: bool = False,
                     **measured) -> None:
    """One log line and one routing.decision span, whichever path decided.

    A span rather than attributes on the request span, so routing decisions can
    be queried on their own - "how often does the keyword check catch something
    the classifier missed?" is a question worth being able to answer, and it has
    the same answer shape whether the agent, Jev or the fast route decided.

    Only the classifier that routed is recorded here. A comparison has its own
    span, routing.compare, so the counts the runbook reads from this one still
    mean one decision per message.
    """
    backend = "none" if decision.triage_skipped else decision.backend
    choice = choice or {}
    requested = choice.get("requested", "auto")
    # Why the classifier that decided is not the one asked for. Not set when no
    # classifier ran: that reason is already the decision's own.
    fell_back = choice.get("why") if choice.get("used") not in (None, "none") else None
    log.info("routing (%s%s): intents=%s safety=%s (%s) -> %s | %s", backend,
             f"; asked for {requested}: {fell_back}" if fell_back else "", decision.intents,
             decision.safety, decision.safety_source, decision.route(), decision.reason)
    with telemetry.span("routing.decision") as rs:
        telemetry.set(
            rs,
            backend=backend,
            intents=decision.intents,
            route=decision.route(),
            safety=decision.safety,
            safety_source=decision.safety_source,
            triage_parse_failed=decision.parse_failed,
            booking_added_by_keyword=decision.booking_added_by_keyword,
            advisor_call_accepted=decision.advisor_call_accepted,
            triage_skipped=decision.triage_skipped,
            reason=decision.reason[:200],
            has_registration=bool(decision.registration),
            has_date=bool(decision.date),
            triage_choice=requested,
            triage_fallback=fell_back,
            classifier_safety=decision.classifier_safety,
            keyword_match=decision.keyword_match,
            booking_match=decision.booking_match,
            compared=compared or None,  # only present when it happened
            **measured,
        )


def _ask_or_throttled(fn, agent_name: str, *args, **kwargs) -> TurnResult:
    """fn(...) for one agent, or a throttled TurnResult if Azure had no quota.

    Throttled is the one exception every caller answers the same way: nothing is
    broken, this agent did not get to answer, and the reply will say we are busy.
    """
    try:
        return fn(*args, agent_name=agent_name, **kwargs)
    except azure_http.Throttled as e:
        return throttled_turn(agent_name, e)


# What each specialist is about to do, in the customer's words.
AGENT_STATUS = {
    "diagnostics": "checking the service documents",
    "booking": "checking the workshop calendar",
    "escalation": "arranging for a service advisor to call you",
}


def _route_and_answer(client, ids, message, decision, timeout, history, out, started,
                      on_delta=None, on_status=None, on_event=None,
                      cache_for: TriagePlan | None = None, ticket: str | None = None) -> None:
    """Run the specialists the decision calls for, then compose the reply.

    `cache_for` is the plan the reply may be kept for, or None when it must not
    be kept at all - a conversation, or a comparison. `ticket` is the reference
    the page kept for this conversation - see handle().
    """
    out.decision = decision

    # --- 2. specialists ---
    route = [s for s in decision.route() if s in ids]

    # A human has already been called about this conversation. Calling them again
    # every turn is what the live app did on 20 Sep - see TICKET_REFERENCE.
    # Looked for on every message, not only when escalation is routed: any
    # agent with raise_ticket has to be turned away while it stands. The
    # reference the page kept comes first: it is the one this server reported,
    # and it is still there when the history has moved past the reply that gave it.
    known = ticket if ticket and TICKET_REFERENCE.fullmatch(ticket) else None
    standing = known or ticket_already_raised(history)
    escalation_skipped = bool(standing) and "escalation" in route
    if escalation_skipped:
        log.info("escalation skipped: ticket %s already stands for this conversation", standing)
        route = [s for s in route if s != "escalation"]

    # Made here, once per message, and handed to every turn that answers it.
    tickets = _ticket_desk(route, standing)

    before = len(out.turns)  # triage's turn, or nothing when triage was skipped

    # Tell them what is happening BEFORE the pre-search: it takes ~0.7s, and
    # that is 0.7s of silence if the status waits for it.
    if on_status and route:
        on_status(" and ".join(AGENT_STATUS.get(s, s) for s in route))

    # ticket_stands only when it cost escalation its turn: that is what the
    # panel explains, and a ticket standing on a booking question explains nothing.
    emit(on_event, kind="route", agents=route, safety=decision.safety,
         safety_source=decision.safety_source, reason=decision.reason,
         triage_skipped=decision.triage_skipped, ticket_stands=standing if escalation_skipped else None,
         backend=decision.backend, probabilities=decision.probabilities or None,
         triage=out.triage)

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

    streamed = _can_stream(on_delta, route, decision, standing)
    if streamed:
        specialist = route[0]
        prompt = _context_for(specialist, message, decision, [], history, prefetched, tickets=tickets)
        # Throttled is raised before the stream opens, so nothing has been shown.
        turn = _ask_or_throttled(ask_streaming, specialist, client, ids[specialist], prompt, on_delta,
                                 timeout=timeout, on_status=on_status, on_event=on_event, tickets=tickets)
        out.turns.append(turn)
    else:
        out.turns.extend(_run_specialists(client, ids, route, message, decision, timeout, history,
                                         prefetched, on_event, tickets))
    specialists = out.turns[before:]

    if prefetched:
        _record_prefetch(specialists, prefetched)

    # --- 3. check, then compose ---
    out.withheld = _withhold_reassurance(specialists, decision)
    edited = _leave_the_ticket_to(tickets.speaker, specialists)
    out.reply = _compose(specialists, decision)
    # The standing ticket is said whenever it came up: escalation would have
    # said it, an agent asked for a ticket and was given this one, or an
    # agent's own words about a ticket were just dropped in favour of these.
    if standing and (escalation_skipped or tickets.asked_by or edited):
        out.reply = _carry_the_ticket_forward(out.reply, standing, decision, specialists)
    out.reply = _keep_the_warning(out.reply, decision, edited)
    _name_the_new_ticket(out, tickets, on_delta if streamed else None)
    out.ticket_requests, out.ticket_raised = len(tickets.asked_by), bool(tickets.raised)
    out.ticket = tickets.reference
    _note_throttling(specialists, out)
    out.duration_ms = int((time.time() - started) * 1000)

    _remember_answer(message, history, out, cache_for)

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


def _name_the_new_ticket(out: RouterResult, tickets: tools.OneTicket, on_delta=None) -> None:
    """Make sure a ticket raised for this message is named in the reply.

    The reply is the only memory the next message has: ticket_already_raised
    finds the ticket by reading its reference back out of the conversation. A
    ticket the customer was never told about cannot be found there, and the next
    message raises another. On 21 Sep the live app held one like that, TK-709113,
    because two had been raised and the reply named one; OneTicket stops the
    second, and this stops the only one going unnamed.

    Escalation is asked to give the reference and nearly always does, so this
    rarely adds anything. When it does, the words are the ticket's own, from
    booking.create_ticket. A streamed reply is already on the screen, so the
    sentence is streamed after it rather than only added to the record.

    Named means named correctly. A model asked to repeat a reference can get a
    digit wrong (TICKET_REFERENCE), and the customer would then hold a reference
    that does not exist - which the next message would also read back as the
    one that stands. There is one ticket, so any other reference in the reply
    is that one mis-copied, and the real one is put in its place. A streamed
    reply cannot be corrected on the screen, so the real one is said after it,
    last, where ticket_already_raised looks first.
    """
    real = tickets.raised
    if not real:
        return
    wrong = sorted({r for r in ANY_REFERENCE.findall(out.reply) if r != real})
    if wrong:
        log.warning("the reply named %s for ticket %s", ", ".join(wrong), real)
        if not on_delta:
            out.reply = ANY_REFERENCE.sub(real, out.reply)
            wrong = []
    if real in out.reply and not wrong:
        return
    log.warning("ticket %s was raised but the reply did not name it correctly; adding it", real)
    said = tickets.confirmation
    out.reply = f"{out.reply}\n\n{said}" if out.reply.strip() else said
    if on_delta:
        on_delta(f"\n\n{said}")


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
         reason="an identical question was already answered; no agent ran", triage=out.triage)
    if on_delta:
        on_delta(found.reply)  # in one piece: there is nothing to wait for
    out.duration_ms = int((time.time() - started) * 1000)


def _remember_answer(message: str, history: list[dict] | None, out: RouterResult,
                     plan: TriagePlan | None) -> None:
    """Keep this reply for the next person who asks the same thing, if it may be kept.

    Under the classifier that actually routed it, worked out now rather than
    from what the visitor asked for: when Jev could not answer, the agent routed
    it, and a Jev chooser must not be handed that as Jev's. Under the plain
    question when no classifier read the message - see cache_key.
    """
    if plan is None or not worth_caching(message, out, history):
        return
    decision = out.decision
    key = cache_key(message, plan.variant_for("none" if decision.triage_skipped else decision.backend))
    if not key:
        return
    ANSWERS.put(key, CachedAnswer(
        reply=out.reply,
        agents=out.agents_used,
        searched=out.searched,
        trace=out.trace(),
        stored_at=time.time(),
    ))
