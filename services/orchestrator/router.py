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
from dataclasses import dataclass, field

from azure.ai.agents import AgentsClient

from .runner import TurnResult, ask

log = logging.getLogger(__name__)

VALID_INTENTS = {"diagnostics", "booking", "escalation", "other"}

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


def _agent_ids(client: AgentsClient) -> dict[str, str]:
    return {a.name: a.id for a in client.list_agents()}


def _context_for(specialist: str, message: str, decision: TriageDecision, so_far: list[TurnResult]) -> str:
    """What we actually send the specialist: the message plus anything useful."""
    parts = [f"Customer message: {message}"]

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

    if specialist == "booking" and so_far:
        diag = next((t for t in so_far if t.agent_name == "diagnostics" and t.answer), None)
        if diag:
            parts.append(
                "A colleague has already looked at the technical side. "
                "Do not repeat it - just handle the appointment.\n"
                f"What they said:\n{diag.answer}"
            )

    if specialist == "escalation" and so_far:
        prior = [t.answer for t in so_far if t.answer]
        if prior:
            parts.append("Already said to the customer:\n" + "\n---\n".join(prior))

    return "\n\n".join(parts)


def _compose(turns: list[TurnResult], decision: TriageDecision) -> str:
    """Join the specialists' answers into one reply.

    Deliberately mechanical - no extra model call. Another LLM pass to smooth
    the joins would cost latency and tokens, and would be one more place an
    ungrounded sentence could appear. Plain joining cannot invent anything.
    """
    answers = [(t.agent_name, t.answer.strip()) for t in turns if t.answer and t.answer.strip()]

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
    client: AgentsClient,
    message: str,
    timeout: float = 90.0,
    agent_ids: dict[str, str] | None = None,
) -> RouterResult:
    """Handle one customer message end to end."""
    started = time.time()
    out = RouterResult()

    ids = agent_ids or _agent_ids(client)

    missing = [n for n in ("triage", "diagnostics", "booking", "escalation") if n not in ids]
    if missing:
        out.error = f"agents not deployed: {', '.join(missing)}. Run agents/deploy_agents.py"
        out.reply = "The service is not fully set up. Please call the workshop directly."
        out.duration_ms = int((time.time() - started) * 1000)
        return out

    # --- 1. triage ---
    triage_turn = ask(client, ids["triage"], message, timeout=timeout, agent_name="triage")
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

    # --- 2. specialists ---
    for specialist in decision.route():
        agent_id = ids.get(specialist)
        if agent_id is None:
            continue

        prompt = _context_for(specialist, message, decision, out.turns[1:])
        turn = ask(client, agent_id, prompt, timeout=timeout, agent_name=specialist)
        out.turns.append(turn)

        if not turn.ok:
            log.warning("%s turn failed: %s", specialist, turn.error)

    # --- 3. compose ---
    out.reply = _compose(out.turns[1:], decision)
    out.duration_ms = int((time.time() - started) * 1000)

    log.info(
        "handled in %dms: agents=%s tokens=%d",
        out.duration_ms, out.agents_used, out.total_tokens,
    )
    return out