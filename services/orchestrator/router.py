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

from . import telemetry
from .foundry import FoundryAgents
from .runner import TurnResult, ask

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
    r"^\s*(hi|hello|hey|hiya|good (morning|afternoon|evening)|"
    r"thanks|thank you|thanks a lot|thank you so much|thx|cheers|bye|goodbye)"
    r"(\s+(there|again|very much))?"
    r"([\s,!.]+how are (you|u)( doing)?( today)?)?[\s!.,?]*$",
    re.IGNORECASE,
)
GREETING_REPLY = (
    "Hello. Tell me what your car is doing, ask about a fault code or a warning "
    "light, or ask to book a service."
)
THANKS_REPLY = "You're welcome. Is there anything else I can help with?"
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

# Triage costs 2.0-2.7s of every reply - a quarter of it - to classify a message
# (measured on the live app, 20 Sep). Some messages do not need classifying: a
# fault code with nothing else in it is a diagnostics question, and no model is
# needed to see that. Deliberately narrow: any hint of a booking, a safety issue
# or a conversation in progress goes to triage as before.
FAULT_CODE = re.compile(r"\b[PBCU][0-9]{4}\b", re.IGNORECASE)


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

    @property
    def total_tokens(self) -> int:
        return sum(t.prompt_tokens + t.completion_tokens for t in self.turns)

    @property
    def agents_used(self) -> list[str]:
        return [t.agent_name for t in self.turns]

    @property
    def searched(self) -> bool:
        return any(t.searched for t in self.turns)

    def trace(self) -> list[dict]:
        """A flat record of what happened, for logging and the API response."""
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


def _agent_ids(client: FoundryAgents) -> dict[str, str]:
    return {a["name"]: a["id"] for a in client.list_agents()}


def small_talk_reply(message: str) -> str | None:
    """A canned reply if the message is only a greeting or thanks, else None."""
    m = SMALL_TALK.match(message)
    if not m:
        return None
    word = m.group(1).lower()
    if word in ("bye", "goodbye"):
        return BYE_REPLY
    return GREETING_REPLY if word.startswith(("hi", "hello", "hey", "good")) else THANKS_REPLY


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
            prompt = _context_for(specialist, message, decision, results, history)
            turn = ask(client, ids[specialist], prompt, timeout=timeout, agent_name=specialist)
            results.append(turn)
            if not turn.ok:
                log.warning("%s turn failed: %s", specialist, turn.error)
            continue

        log.info("running %s in parallel", wave)
        # Build every prompt BEFORE starting any thread. _context_for reads the
        # results list, and if one thread appended to it while another was
        # reading, the prompts would depend on timing - the same request could
        # produce different prompts on different runs.
        prompts = {s: _context_for(s, message, decision, results, history) for s in wave}

        with telemetry.span("specialists.parallel", agents=wave, count=len(wave)):
            with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="specialist") as pool:
                futures = {
                    pool.submit(
                        ask, client, ids[s], prompts[s], timeout=timeout, agent_name=s
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
    r"(?<!not )(?<!n't )(?<!never )\bsafe to (keep |continue )?driv(e|ing)\b"
    r"|\b(is|are|feels?) (completely |perfectly |quite )?normal\b"
    r"|\b(nothing|no need) to worry\b",
    re.IGNORECASE,
)

SAFETY_FALLBACK = (
    "Do not drive the vehicle. What you describe may affect its safety, and it needs to "
    "be checked by a technician first. Please call the workshop directly and someone "
    "will help you straight away."
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


def _compose(turns: list[TurnResult], decision: TriageDecision) -> str:
    """Join the specialists' answers into one reply.

    Deliberately mechanical - no extra model call. Another LLM pass to smooth
    the joins would cost latency and tokens, and would be one more place an
    ungrounded sentence could appear. Plain joining cannot invent anything.
    """
    answers = [(t.agent_name, t.answer.strip()) for t in turns if t.answer and t.answer.strip()]

    if not answers and decision.safety:
        # Nothing usable came back - or it was withheld - on a safety issue.
        # The warning must not depend on an agent having answered.
        return SAFETY_FALLBACK

    if not answers:
        return (
            "Sorry - I could not get an answer for that just now. "
            "Please call the workshop directly and someone will help you."
        )

    if len(answers) == 1:
        return answers[0][1]

    # Safety first, whoever wrote it.
    if decision.safety:
        answers.sort(key=lambda a: 0 if a[0] == "escalation" else 1)

    return "\n\n".join(text for _, text in answers)


def handle(
    client: FoundryAgents,
    message: str,
    timeout: float = 90.0,
    agent_ids: dict[str, str] | None = None,
    history: list[dict] | None = None,
) -> RouterResult:
    """Handle one customer message end to end.

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
            out.reply = canned
            out.decision = TriageDecision(intents=["other"], reason="small talk, answered without agents")
            out.duration_ms = int((time.time() - started) * 1000)
        else:
            _handle_inner(client, message, timeout, agent_ids, out, started, history)
        d = out.decision
        telemetry.set(
            req_span,
            agents=out.agents_used,
            searched=out.searched,
            safety=bool(d and d.safety),
            safety_source=d.safety_source if d else "none",
            total_tokens=out.total_tokens,
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
) -> None:
    ids = agent_ids or _agent_ids(client)

    missing = [n for n in ("triage", "diagnostics", "booking", "escalation") if n not in ids]
    if missing:
        out.error = f"agents not deployed: {', '.join(missing)}. Run agents/deploy_agents.py"
        out.reply = "The service is not fully set up. Please call the workshop directly."
        out.duration_ms = int((time.time() - started) * 1000)
        return

    # --- 1. triage, unless the message speaks for itself ---
    decision = fast_route(message, history)
    if decision is not None:
        log.info("triage skipped: %s", decision.reason)
        _route_and_answer(client, ids, message, decision, timeout, history, out, started)
        return

    triage_turn = ask(client, ids["triage"], _triage_input(message, history), timeout=timeout, agent_name="triage")
    out.turns.append(triage_turn)

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

    _route_and_answer(client, ids, message, decision, timeout, history, out, started)


def _route_and_answer(client, ids, message, decision, timeout, history, out, started) -> None:
    """Run the specialists the decision calls for, then compose the reply."""
    out.decision = decision

    # --- 2. specialists ---
    route = [s for s in decision.route() if s in ids]
    before = len(out.turns)  # triage's turn, or nothing when triage was skipped
    out.turns.extend(_run_specialists(client, ids, route, message, decision, timeout, history))
    specialists = out.turns[before:]

    # --- 3. check, then compose ---
    out.withheld = _withhold_reassurance(specialists, decision)
    out.reply = _compose(specialists, decision)
    out.duration_ms = int((time.time() - started) * 1000)

    log.info(
        "handled in %dms: agents=%s tokens=%d",
        out.duration_ms, out.agents_used, out.total_tokens,
    )
