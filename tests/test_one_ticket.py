"""One conversation, one ticket.

From the live app on 20 September. "i have a problem with my gear shifting" was
flagged as a safety issue, and it stayed flagged for the rest of the
conversation - triage is told to judge the flag on the new message alone, and it
did not. Escalation therefore ran on every turn and raised a ticket on every
turn:

    "i have a problem with my gear shifting"  -> TK-013051
    "i need to book appointment"              -> TK-892738
    "tomoroow"                                -> TK-953064

Three advisors' worth of work for one car, and the customer was given three
different references for one problem.

And from 21 September, once escalation was being skipped. Diagnostics had
raise_ticket as well:

    "How often should brake fluid be changed?"  diagnostics -> TK-709113
                                                escalation  -> TK-519169
                                                (the customer was told TK-519169)
    "My brakes feel spongy"                     escalation skipped: TK-519169 stands
                                                diagnostics -> TK-879967, and its reply
                                                said "I have raised a safety ticket"

The second half of this file replays that conversation with fake agents whose
tool calls go the whole way - runner, tools.execute, OneTicket, the ticket store
- so what is counted is tickets that really exist, not calls that were made.
"""

import json
import threading

import pytest

from services.orchestrator import booking, runner, tools
from services.orchestrator import router as _router
from services.orchestrator.runner import TurnResult

IDS = {"diagnostics": "a1", "booking": "a2", "escalation": "a3"}
ALL_IDS = {"triage": "t", **IDS}

RAISED = "I have raised a ticket for a service advisor. The ticket reference is TK-013051. Someone will contact you within one hour."


def agents(intents=("diagnostics", "escalation"), safety=True):
    """Fake agents that record who was asked."""
    called = []

    def ask(client, agent_id, prompt, timeout=90.0, agent_name="", **_):
        called.append(agent_name)
        t = TurnResult(agent_name=agent_name, status="completed")
        if agent_name == "triage":
            t.answer = json.dumps({"intents": list(intents), "safety": safety})
        elif agent_name == "escalation":
            t.answer = "I have raised a ticket. The reference is TK-999999."
        else:
            t.answer = f"answer from {agent_name}"
        return t

    ask.called = called
    return ask


def conversation(*, reply=RAISED):
    return [
        {"role": "customer", "text": "i have a problem with my gear shifting"},
        {"role": "assistant", "text": reply},
    ]


@pytest.fixture(autouse=True)
def documents(monkeypatch):
    monkeypatch.setattr(_router.tools, "search_service_docs", lambda query, doc_type=None: "2 CANDIDATES")


@pytest.fixture(autouse=True)
def ticket_store(tmp_path, monkeypatch):
    """Tickets land in a temporary file, which the tests read back to count them."""
    monkeypatch.setattr(booking, "TICKETS", tmp_path / "tickets.json")
    monkeypatch.setattr(runner, "POLL_SECONDS", 0)
    monkeypatch.setattr(runner._CLEANUP, "submit", lambda fn, *a: fn(*a))  # delete threads inline
    return tmp_path / "tickets.json"


def tickets_in(store) -> list[str]:
    """Every ticket that exists, by reference."""
    return sorted(json.loads(store.read_text())["tickets"]) if store.exists() else []


def references_in(text: str) -> set[str]:
    return set(_router.TICKET_REFERENCE.findall(text))


# ------------------------------------------------------- finding the ticket


def test_a_reference_in_the_conversation_is_found():
    assert _router.ticket_already_raised(conversation()) == "TK-013051"


def test_a_conversation_with_no_ticket_has_none():
    assert _router.ticket_already_raised([
        {"role": "customer", "text": "what does P0420 mean"},
        {"role": "assistant", "text": "The catalytic converter is worn."},
    ]) is None


def test_no_history_has_none():
    assert _router.ticket_already_raised(None) is None
    assert _router.ticket_already_raised([]) is None


def test_a_reference_the_customer_typed_does_not_count():
    """Only what we told them. Otherwise anyone could turn escalation off by
    typing a ticket number."""
    assert _router.ticket_already_raised([
        {"role": "customer", "text": "my ticket is TK-013051, my brakes have failed"},
    ]) is None


def test_a_long_answer_does_not_lose_its_reference():
    """_recent trims turns to 600 characters. A reference past that would mean a
    second ticket for the same problem."""
    padded = "The gearbox needs a look. " * 40 + " The ticket reference is TK-013051."
    assert len(padded) > 600
    assert _router.ticket_already_raised([{"role": "assistant", "text": padded}]) == "TK-013051"


# ------------------------------------------------------- one ticket per problem


def test_escalation_does_not_run_again_once_a_ticket_stands(monkeypatch):
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, history=conversation())

    assert "escalation" not in ask.called, "a second advisor was called for the same car"
    assert "TK-013051" in r.reply, "the customer is told the reference they already have"
    assert "TK-999999" not in r.reply, "and not a new one"


def test_the_first_time_escalation_does_run(monkeypatch):
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)

    _router.handle(None, "i have a problem with my gear shifting", agent_ids=ALL_IDS)

    assert "escalation" in ask.called


def test_the_other_specialists_still_run(monkeypatch):
    """Skipping escalation must not skip the answer the customer asked for."""
    ask = agents(intents=("booking", "escalation"))
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, history=conversation())

    assert "booking" in ask.called
    assert "answer from booking" in r.reply


def test_the_standing_ticket_comes_before_the_rest(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents(intents=("booking", "escalation")))
    r = _router.handle(None, "i need to book appointment", agent_ids=ALL_IDS, history=conversation())
    assert r.reply.index("TK-013051") < r.reply.index("answer from booking")


def test_asking_for_a_person_again_is_answered_without_an_agent(monkeypatch):
    """The whole route was escalation, so there is nothing else to say - but the
    customer must still get an answer, and the safety warning with it."""
    ask = agents(intents=("escalation",))
    monkeypatch.setattr(_router, "ask", ask)

    r = _router.handle(None, "can I speak to someone please", agent_ids=ALL_IDS, history=conversation())

    assert ask.called == ["triage"], "no specialist needed to run"
    assert "TK-013051" in r.reply
    assert "Do not drive" in r.reply, "it is still a safety conversation"
    assert "could not get an answer" not in r.reply


def test_a_standing_ticket_without_a_safety_flag_is_just_the_reference(monkeypatch):
    monkeypatch.setattr(_router, "ask", agents(intents=("escalation",), safety=False))
    r = _router.handle(None, "has anyone looked at this yet", agent_ids=ALL_IDS,
                       history=conversation())
    assert r.reply == _router.TICKET_STANDS.format(reference="TK-013051")


def test_skipping_escalation_saves_the_turn(monkeypatch):
    """It cost ~3,000 tokens and ~5s of every turn of that conversation."""
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)

    first = _router.handle(None, "i have a problem with my gear shifting", agent_ids=ALL_IDS)
    later = _router.handle(None, "tomorrow please", agent_ids=ALL_IDS, history=conversation())

    assert "escalation" in first.agents_used
    assert "escalation" not in later.agents_used


# ------------------------------------------------------- the rule, on the tool itself


def granted(output: str) -> dict:
    return json.loads(output)


def test_the_first_request_raises_the_ticket_and_the_next_is_given_the_same_one(ticket_store):
    desk = tools.OneTicket()
    first = granted(desk.raise_ticket("escalation", summary="spongy brakes", urgency="safety"))
    again = granted(desk.raise_ticket("escalation", summary="spongy brakes, again", urgency="safety"))

    assert first["ok"] and first["reference"].startswith("TK-")
    assert again["ok"] and again["reference"] == first["reference"]
    assert again["already_raised"] is True
    assert tickets_in(ticket_store) == [first["reference"]], "the second call created nothing"


def test_nothing_is_raised_while_a_ticket_stands(ticket_store):
    """Whoever asks, and however urgent they say it is."""
    desk = tools.OneTicket(standing="TK-519169")
    for agent in ("diagnostics", "escalation", "booking"):
        out = granted(desk.raise_ticket(agent, summary="my brakes feel spongy", urgency="safety"))
        assert out["ok"] and out["reference"] == "TK-519169"
        assert "already stands" in out["message"]
    assert tickets_in(ticket_store) == []
    assert desk.raised is None and desk.reference == "TK-519169"


def test_on_a_route_with_escalation_only_escalation_raises_it(ticket_store):
    """Diagnostics runs first. Left to the first caller, the ticket would carry
    diagnostics' summary rather than the handover escalation exists to write."""
    desk = tools.OneTicket(owner="escalation")

    refused = granted(desk.raise_ticket("diagnostics", summary="brake fluid", urgency="safety"))
    assert refused["reference"] is None and refused["left_to"] == "escalation"
    assert "do not mention one" in refused["message"].lower()
    assert tickets_in(ticket_store) == []

    raised = granted(desk.raise_ticket("escalation", summary="Customer asks about brake fluid", urgency="safety"))
    assert tickets_in(ticket_store) == [raised["reference"]]
    assert desk.asked_by == ["diagnostics", "escalation"]


def test_threads_asking_at_the_same_moment_raise_one_ticket(ticket_store, monkeypatch):
    """The specialists run in a thread pool. Writing the ticket is slowed down
    here so that, without the lock, every thread would be inside it at once."""
    real = booking.create_ticket

    def slow(*a, **k):
        threading.Event().wait(0.05)
        return real(*a, **k)

    monkeypatch.setattr(booking, "create_ticket", slow)
    desk = tools.OneTicket()
    start = threading.Barrier(8, timeout=5)
    got = []

    def ask(i):
        start.wait()
        got.append(granted(desk.raise_ticket(f"agent{i}", summary="brakes", urgency="safety"))["reference"])

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert len(got) == 8
    assert len(set(got)) == 1, f"{len(set(got))} different references for one message"
    assert tickets_in(ticket_store) == [got[0]]


def test_a_ticket_that_could_not_be_written_does_not_count(ticket_store, monkeypatch):
    """Otherwise the next caller would be handed a reference that does not exist."""
    real = booking.create_ticket

    def disk_full(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(booking, "create_ticket", disk_full)
    desk = tools.OneTicket()
    out = tools.execute("raise_ticket", {"summary": "brakes"}, tickets=desk, asked_by="escalation")
    assert out.startswith("ERROR:")
    assert desk.raised is None

    monkeypatch.setattr(booking, "create_ticket", real)
    retried = granted(tools.execute("raise_ticket", {"summary": "brakes"}, tickets=desk, asked_by="escalation"))
    assert tickets_in(ticket_store) == [retried["reference"]]


def test_two_messages_never_share_a_ticket(ticket_store):
    """One OneTicket per message: nothing about one customer's ticket reaches another's."""
    one, two = tools.OneTicket(), tools.OneTicket()
    a = granted(one.raise_ticket("escalation", summary="brakes"))["reference"]
    b = granted(two.raise_ticket("escalation", summary="steering"))["reference"]
    assert a != b
    assert tickets_in(ticket_store) == sorted([a, b])


def test_an_argument_cannot_claim_to_be_escalation(ticket_store):
    """asked_by comes from the runner, never from the model's arguments."""
    desk = tools.OneTicket(owner="escalation")
    out = tools.execute("raise_ticket", {"summary": "x", "asked_by": "escalation"}, tickets=desk,
                        asked_by="diagnostics")
    assert out.startswith("ERROR: wrong arguments")
    assert tickets_in(ticket_store) == []


def test_without_a_desk_the_tool_raises_a_ticket_as_it_always_did(ticket_store):
    out = granted(tools.execute("raise_ticket", {"summary": "brakes", "urgency": "safety"}))
    assert tickets_in(ticket_store) == [out["reference"]]


# ------------------------------------------------------- the live conversation, replayed


class Agent:
    """What one fake agent does: ask for some tools, round by round, then answer."""

    def __init__(self, answer, *rounds):
        self.answer = answer if callable(answer) else (lambda outputs, text=answer: text)
        self.rounds = rounds


def raises(summary="customer needs a call", urgency="safety"):
    return [("raise_ticket", {"summary": summary, "urgency": urgency})]


def reference_from(outputs: list) -> str:
    """The reference raise_ticket handed back, as the agent would read it."""
    for o in outputs:
        try:
            ref = json.loads(o).get("reference")
        except ValueError:
            continue
        if ref:
            return ref
    return ""


class Workshop:
    """A fake Foundry project, in the run shapes the real one returns.

    Each run asks for its agent's tools round by round, then completes, so every
    tool call goes through runner._run_requested_tools and tools.execute exactly
    as a real one does. Several specialists call it at once, hence the lock.
    """

    def __init__(self, agents: dict, meet_at_the_start: tuple = ()):
        self.agents = agents
        self.prompts: dict = {}
        self.outputs: dict = {}
        self._runs: dict = {}
        self._lock = threading.Lock()
        # Agents that start their runs together, so that their tool calls
        # arrive at the same moment rather than one after the other.
        self._meeting = set(meet_at_the_start)
        self._meet = threading.Barrier(len(self._meeting), timeout=5) if self._meeting else None

    def create_thread_and_run(self, agent_id, content):
        with self._lock:
            thread_id = f"thread_{len(self._runs) + 1}"
            self._runs[thread_id] = {"agent": agent_id, "rounds": list(self.agents[agent_id].rounds),
                                     "outputs": []}
            self.prompts.setdefault(agent_id, []).append(content)
        if agent_id in self._meeting:
            self._meet.wait()
        with self._lock:
            return self._state(thread_id)

    def _state(self, thread_id):
        run = self._runs[thread_id]
        base = {"id": f"run_{thread_id}", "object": "thread.run", "thread_id": thread_id,
                "last_error": None, "incomplete_details": None, "required_action": None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        if not run["rounds"]:
            return {**base, "status": "completed"}
        calls = [{"id": f"call_{i}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args)}}
                 for i, (name, args) in enumerate(run["rounds"][0])]
        return {**base, "status": "requires_action",
                "required_action": {"type": "submit_tool_outputs", "submit_tool_outputs": {"tool_calls": calls}}}

    def get_run(self, thread_id, run_id):
        with self._lock:
            return self._state(thread_id)

    def submit_tool_outputs(self, thread_id, run_id, outputs):
        with self._lock:
            run = self._runs[thread_id]
            run["rounds"].pop(0)
            run["outputs"].extend(o["output"] for o in outputs)
            self.outputs.setdefault(run["agent"], []).extend(o["output"] for o in outputs)

    def list_messages(self, thread_id):
        run = self._runs[thread_id]
        text = self.agents[run["agent"]].answer(run["outputs"])
        return [{"role": "assistant", "content": [{"type": "text", "text": {"value": text, "annotations": []}}]}]

    def cancel_run(self, thread_id, run_id):
        pass

    def delete_thread(self, thread_id):
        pass

    # The same runs, streamed: each stream stops at a round of tool calls, and
    # submitting the outputs starts the next one, as Foundry's does.

    def stream_thread_and_run(self, agent_id, content):
        return self._stream(self.create_thread_and_run(agent_id, content)["thread_id"])

    def stream_tool_outputs(self, thread_id, run_id, outputs):
        self.submit_tool_outputs(thread_id, run_id, outputs)
        return self._stream(thread_id)

    def _stream(self, thread_id):
        state = self.get_run(thread_id, "")
        yield "thread.run.created", state
        if state["status"] == "requires_action":
            yield "thread.run.requires_action", state
            return
        text = self.list_messages(thread_id)[0]["content"][0]["text"]["value"]
        yield "thread.message.delta", {"delta": {"content": [{"index": 0, "type": "text", "text": {"value": text}}]}}
        yield "thread.run.completed", state


def triage(intents, safety=False):
    return Agent(json.dumps({"intents": list(intents), "safety": safety}))


# Diagnostics as it behaved on the live app: it raised a ticket nobody asked
# for, and said so. The explanation is the part the customer needs.
BRAKE_FLUID = ("Do not drive the vehicle, it needs immediate professional attention. Brake fluid is "
               "replaced every two years (maintenance schedule, brake fluid).")
SPONGY = ("Do not drive the vehicle, it needs immediate professional attention. A spongy pedal can "
          "mean air in the brake lines (TSB-021, Symptoms).")


def diagnostics_that_raises(explanation):
    return Agent(lambda outputs: f"{explanation} I have raised a safety ticket for you: "
                                 f"{reference_from(outputs) or 'TK-879967'}.", raises("brakes"))


def escalation_that_raises():
    return Agent(lambda outputs: "Please do not drive the vehicle. I have raised a ticket for a service "
                                 f"advisor, reference {reference_from(outputs)}, and someone will call "
                                 "within the hour.", raises("Customer reports a brake concern."))


def live_workshop():
    return Workshop({
        "t": triage(["diagnostics"]),  # the keyword net adds safety, as it did live
        "a1": diagnostics_that_raises(BRAKE_FLUID),
        "a2": Agent("answer from booking"),
        "a3": escalation_that_raises(),
    })


def test_the_first_message_raises_one_ticket_though_both_agents_ask(ticket_store):
    """Live: TK-709113 from diagnostics and TK-519169 from escalation."""
    shop = live_workshop()
    r = _router.handle(shop, "How often should brake fluid be changed?", agent_ids=ALL_IDS)

    assert r.decision.safety_source == "keyword"
    assert r.agents_used == ["triage", "diagnostics", "escalation"]
    [ticket] = tickets_in(ticket_store)
    assert references_in(r.reply) == {ticket}, "the customer is told the one ticket there is"
    assert granted(shop.outputs["a1"][0])["left_to"] == "escalation", "diagnostics was turned away"
    assert "I have raised a safety ticket" not in r.reply, "diagnostics' claim did not reach the customer"
    assert "Brake fluid is replaced every two years" in r.reply, "its explanation did"
    assert (r.ticket_requests, r.ticket_raised) == (2, True)


def test_the_follow_up_raises_nothing_and_names_the_ticket_that_stands(ticket_store):
    """Live: TK-879967 from diagnostics, under the router's line that TK-519169 stood."""
    shop = live_workshop()
    first = _router.handle(shop, "How often should brake fluid be changed?", agent_ids=ALL_IDS)
    [standing] = tickets_in(ticket_store)

    shop.agents["a1"] = diagnostics_that_raises(SPONGY)
    history = [{"role": "customer", "text": "How often should brake fluid be changed?"},
               {"role": "assistant", "text": first.reply}]
    r = _router.handle(shop, "My brakes feel spongy", agent_ids=ALL_IDS, history=history)

    assert tickets_in(ticket_store) == [standing], "no new ticket"
    assert len(shop.prompts["a3"]) == 1, "escalation did not run again"
    assert granted(shop.outputs["a1"][-1])["reference"] == standing, "diagnostics was handed the standing one"
    assert r.reply.startswith(_router.TICKET_STANDS.format(reference=standing))
    assert references_in(r.reply) == {standing}
    assert "I have raised" not in r.reply, "nothing contradicts the line that it already stands"
    assert "A spongy pedal can mean air in the brake lines" in r.reply
    assert "Do not drive" in r.reply
    assert (r.ticket_requests, r.ticket_raised) == (1, False)


def test_diagnostics_is_told_the_ticket_is_not_its_own(ticket_store):
    shop = live_workshop()
    first = _router.handle(shop, "How often should brake fluid be changed?", agent_ids=ALL_IDS)
    assert _router.TICKET_OWNED_NOTE in shop.prompts["a1"][0]

    _router.handle(shop, "My brakes feel spongy", agent_ids=ALL_IDS,
                   history=[{"role": "assistant", "text": first.reply}])
    assert _router.TICKET_STANDS_NOTE in shop.prompts["a1"][1]


def test_diagnostics_is_told_nothing_about_tickets_when_nobody_else_has_one(ticket_store):
    shop = live_workshop()
    shop.agents["t"] = triage(["diagnostics"])
    _router.handle(shop, "my wipers squeak", agent_ids=ALL_IDS)
    assert "ticket" not in shop.prompts["a1"][0].lower()


def test_escalation_is_not_told_a_ticket_was_raised_before_it_raised_one(ticket_store):
    """Shown diagnostics' "I have raised a safety ticket", escalation could take
    the job as done - and then no ticket would be raised at all."""
    shop = live_workshop()
    _router.handle(shop, "How often should brake fluid be changed?", agent_ids=ALL_IDS)
    [prompt] = shop.prompts["a3"]
    assert "Brake fluid is replaced every two years" in prompt
    assert "raised a safety ticket" not in prompt


def test_specialists_running_in_parallel_cannot_race_two_tickets_into_existence(ticket_store, monkeypatch):
    """Diagnostics and booking run at the same time. Both ask for a ticket at the
    same moment, and writing one is slow enough that both would be inside it at
    once unless one waits for the other."""
    real = booking.create_ticket

    def slow(*a, **k):
        threading.Event().wait(0.2)
        return real(*a, **k)

    monkeypatch.setattr(booking, "create_ticket", slow)
    shop = Workshop({
        "t": triage(["diagnostics", "booking"]),
        "a1": Agent(lambda outputs: f"P0420 is the catalytic converter. Ticket {reference_from(outputs)}.",
                    raises("P0420", urgency="normal")),
        "a2": Agent(lambda outputs: f"Tomorrow at 10:30 is free. Ticket {reference_from(outputs)}.",
                    raises("wants a slot", urgency="normal")),
        "a3": escalation_that_raises(),
    }, meet_at_the_start=("a1", "a2"))

    r = _router.handle(shop, "P0420 is on, can I book for tomorrow", agent_ids=ALL_IDS)

    assert r.agents_used == ["triage", "diagnostics", "booking"]
    [ticket] = tickets_in(ticket_store)
    assert references_in(r.reply) == {ticket}
    assert {reference_from(shop.outputs["a1"]), reference_from(shop.outputs["a2"])} == {ticket}


def test_another_conversation_still_gets_its_own_ticket(ticket_store):
    """Two customers at the same time, each with a brake problem: two tickets,
    and each is told only their own."""
    shop = live_workshop()
    replies = {}

    def customer(name):
        replies[name] = _router.handle(shop, "My brakes feel spongy", agent_ids=ALL_IDS).reply

    threads = [threading.Thread(target=customer, args=(n,)) for n in ("one", "two")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert len(tickets_in(ticket_store)) == 2
    one, two = references_in(replies["one"]), references_in(replies["two"])
    assert len(one) == len(two) == 1
    assert one != two
    assert one | two == set(tickets_in(ticket_store))


def test_one_agent_turn_on_its_own_raises_one_ticket_however_many_rounds(ticket_store):
    """agents/ask.py and the eval suite run an agent without the router."""
    shop = Workshop({"a3": Agent(reference_from, raises("first"), raises("second"))})
    turn = runner.ask(shop, "a3", "my brakes have failed", agent_name="escalation")
    assert [c.name for c in turn.tool_calls] == ["raise_ticket", "raise_ticket"]
    assert tickets_in(ticket_store) == [turn.answer]


# ------------------------------------------------------- what the reply says


def test_a_claim_about_a_ticket_is_dropped_and_the_explanation_kept():
    text = ("Brake fluid is replaced every two years (maintenance schedule, brake fluid). "
            "I have raised a safety ticket for you, reference TK-879967.\n\n"
            "A technician will check it.")
    assert _router._without_ticket_talk(text) == (
        "Brake fluid is replaced every two years (maintenance schedule, brake fluid).\n\n"
        "A technician will check it.")


def test_a_paragraph_that_was_only_about_a_ticket_leaves_no_gap():
    text = "The fluid is due.\n\nI have raised a ticket, TK-111111.\n\nA technician will check it."
    assert _router._without_ticket_talk(text) == "The fluid is due.\n\nA technician will check it."


def test_a_warning_that_claims_a_ticket_goes_with_the_claim():
    """Kept whole, it is the very contradiction this removes: "I have raised a
    safety ticket" under the line that one already stands. The warning is put
    back in the router's words instead (see the tests just below)."""
    text = "Do not drive the vehicle - I have raised a safety ticket. Brake fluid is replaced every two years."
    assert _router._without_ticket_talk(text) == "Brake fluid is replaced every two years."


@pytest.mark.parametrize("warning", [
    "Do not drive the vehicle - I have raised a safety ticket for you.",
    "You should not be driving the car, and your safety ticket TK-519169 is with an advisor.",
    "Please don\u2019t drive the car - an advisor already has ticket TK-519169.",
    "It is not safe to drive until then, and ticket TK-519169 stands.",
])
def test_a_follow_up_that_loses_its_warning_with_the_ticket_gets_the_routers(ticket_store, warning):
    """A ticket stands, so escalation is skipped and diagnostics is the only one
    left to warn. Its warning shared a sentence with the ticket, and went with it."""
    shop = live_workshop()
    shop.agents["t"] = triage(["diagnostics"], safety=True)
    shop.agents["a1"] = Agent(f"{warning} A spongy pedal can mean air in the brake lines (TSB-021, Symptoms).")
    r = _router.handle(shop, "the brakes are getting worse", agent_ids=ALL_IDS, history=conversation())

    assert r.reply.startswith(_router.SAFETY_WARNING), r.reply
    assert r.reply.index(_router.SAFETY_WARNING) < r.reply.index("TK-013051")
    assert "I have raised" not in r.reply
    assert references_in(r.reply) == {"TK-013051"}
    assert "A spongy pedal can mean air in the brake lines" in r.reply
    assert tickets_in(ticket_store) == []


def test_a_warning_that_survives_is_not_said_twice(ticket_store):
    shop = live_workshop()
    shop.agents["t"] = triage(["diagnostics"], safety=True)
    shop.agents["a1"] = Agent(SPONGY + " Your ticket TK-013051 covers this.")
    r = _router.handle(shop, "the brakes are getting worse", agent_ids=ALL_IDS, history=conversation())
    assert _router.SAFETY_WARNING not in r.reply
    assert r.reply == _router.TICKET_STANDS.format(reference="TK-013051") + "\n\n" + SPONGY


def test_the_router_adds_no_warning_to_an_answer_it_did_not_touch(ticket_store):
    """The warning is only the router's to add when its own edit took one out."""
    shop = live_workshop()
    shop.agents["t"] = triage(["diagnostics"], safety=True)
    shop.agents["a1"] = Agent(BRAKE_FLUID.split(". ", 1)[1])  # no warning, and nothing about a ticket
    r = _router.handle(shop, "how often is brake fluid changed", agent_ids=ALL_IDS, history=conversation())
    assert _router.SAFETY_WARNING not in r.reply


def test_the_warning_is_not_added_to_a_message_that_was_not_flagged(ticket_store):
    shop = live_workshop()
    shop.agents["t"] = triage(["booking"])
    shop.agents["a2"] = Agent("Your ticket TK-013051 is with an advisor. Tomorrow at 10:30 is free.")
    r = _router.handle(shop, "any slots tomorrow", agent_ids=ALL_IDS, history=conversation())
    assert _router.SAFETY_WARNING not in r.reply


@pytest.mark.parametrize("text", [
    "Do not drive the vehicle.", "Don\u2019t drive it.", "You should not be driving the car.",
    "It is not safe to drive.", "It isn't safe to keep driving.", "Avoid driving it.",
    "Never drive with a spongy pedal.", "Stop using the vehicle.", "Please do not use the car.",
])
def test_the_warning_is_recognised_however_it_is_written(text):
    assert _router._warns(text)


@pytest.mark.parametrize("text", [
    "It is safe to drive with care (fault code list, P0420).",
    "Brake fluid is replaced every two years.",
])
def test_an_answer_with_no_warning_is_not_mistaken_for_one(text):
    assert not _router._warns(text)


# The redeployed diagnostics offers the call and never says "ticket".
OFFER = "Would you like a call from a service advisor?"


@pytest.mark.parametrize("offer", [
    OFFER, "Would you like an advisor to call you?", "A service advisor can phone you if you wish.",
    "Someone will call you back shortly.",
])
def test_an_advisors_call_counts_as_talk_about_a_ticket(offer):
    assert _router._without_ticket_talk(f"{SPONGY} {offer}") == SPONGY


def test_advice_that_names_an_advisor_but_no_call_is_kept():
    text = "Have a service advisor check the pads (TSB-021, Diagnostic Procedure). Call the workshop if unsure."
    assert _router._without_ticket_talk(text) is text


def test_a_call_already_arranged_is_not_offered_again(ticket_store):
    shop = live_workshop()
    shop.agents["a1"] = Agent(f"{SPONGY} {OFFER}")
    r = _router.handle(shop, "My brakes feel spongy", agent_ids=ALL_IDS, history=conversation())
    assert OFFER not in r.reply
    assert r.reply.startswith(_router.TICKET_STANDS.format(reference="TK-013051"))


def test_a_call_escalation_is_arranging_is_not_offered_as_well(ticket_store):
    shop = live_workshop()
    shop.agents["a1"] = Agent(f"{SPONGY} {OFFER}")
    r = _router.handle(shop, "My brakes feel spongy", agent_ids=ALL_IDS)
    [ticket] = tickets_in(ticket_store)
    assert OFFER not in r.reply
    assert ticket in r.reply
    assert "A spongy pedal can mean air in the brake lines" in r.reply


def test_an_answer_that_never_mentions_a_ticket_is_left_exactly_as_it_was():
    text = "P0420 is the catalytic converter.\n\n  - check the O2 sensor first.  "
    assert _router._without_ticket_talk(text) is text


def test_a_ticket_is_left_to_whoever_speaks_for_it():
    """Escalation says its own ticket; the others' words on it go."""
    turns = [TurnResult(agent_name="diagnostics", status="completed",
                        answer="The fluid is due. I have raised a ticket, TK-111111."),
             TurnResult(agent_name="escalation", status="completed",
                        answer="I have raised a ticket for you, reference TK-222222.")]
    assert _router._leave_the_ticket_to("escalation", turns) == ["diagnostics"]
    assert turns[0].answer == "The fluid is due."
    assert turns[1].answer == "I have raised a ticket for you, reference TK-222222."


def test_with_nobody_speaking_for_the_ticket_nothing_is_touched():
    turns = [TurnResult(agent_name="diagnostics", status="completed", answer="I have raised a ticket, TK-1111.")]
    assert _router._leave_the_ticket_to(None, turns) == []
    assert turns[0].answer == "I have raised a ticket, TK-1111."


def test_a_standing_ticket_is_said_when_another_agent_brings_it_up(ticket_store):
    """Not only when escalation was skipped: booking mentioning the ticket is
    dropped in favour of the router's sentence, so the sentence has to be there."""
    shop = live_workshop()
    shop.agents["t"] = triage(["booking"])
    shop.agents["a2"] = Agent("Your ticket TK-013051 is with an advisor. Tomorrow at 10:30 is free.")
    r = _router.handle(shop, "any slots tomorrow", agent_ids=ALL_IDS, history=conversation())

    assert r.reply == (_router.TICKET_STANDS.format(reference="TK-013051")
                       + "\n\nTomorrow at 10:30 is free.")


def test_a_standing_ticket_is_not_repeated_on_every_reply(ticket_store):
    shop = live_workshop()
    shop.agents["t"] = triage(["booking"])
    shop.agents["a2"] = Agent("Tomorrow at 10:30 is free.")
    r = _router.handle(shop, "any slots tomorrow", agent_ids=ALL_IDS, history=conversation())
    assert r.reply == "Tomorrow at 10:30 is free."


def test_a_ticket_raised_but_not_named_is_named_by_the_router(ticket_store):
    """The next message finds the ticket by reading it back out of the reply.
    One the customer was never told about would be raised again."""
    shop = live_workshop()
    shop.agents["a3"] = Agent("Please do not drive the vehicle. Someone will call you.", raises("brakes"))
    r = _router.handle(shop, "my brakes have failed", agent_ids=ALL_IDS)

    [ticket] = tickets_in(ticket_store)
    assert ticket in r.reply
    assert _router.ticket_already_raised([{"role": "assistant", "text": r.reply}]) == ticket


def test_a_streamed_reply_is_sent_the_ticket_sentence_too(ticket_store, monkeypatch):
    """The customer reads the streamed text, not the record of it."""
    raised = {}

    def ask_streaming(client, agent_id, prompt, on_delta, timeout=90.0, agent_name="", tickets=None, **_):
        raised.update(granted(tickets.raise_ticket(agent_name, summary="wants a person")))
        on_delta("Someone will call you.")
        return TurnResult(agent_name=agent_name, status="completed", answer="Someone will call you.")

    monkeypatch.setattr(_router, "ask_streaming", ask_streaming)
    monkeypatch.setattr(_router, "ask", agents(intents=("escalation",), safety=False))
    seen = []
    r = _router.handle(None, "can I speak to a person", agent_ids=ALL_IDS, on_delta=seen.append)

    assert raised["reference"] in "".join(seen)
    assert "".join(seen) == r.reply


def test_a_streamed_turn_asking_twice_raises_one_ticket(ticket_store):
    """The streamed path runs its tools in runner._stream_events, not in the
    polling loop, and it has to hand the same OneTicket down. Escalation alone,
    and no safety flag, is the route that streams; it asks in two rounds."""
    shop = Workshop({
        "t": triage(["escalation"]),
        "a3": Agent(lambda outputs: f"An advisor will call you. Your reference is {reference_from(outputs)}.",
                    raises("wants a person", urgency="normal"), raises("wants a person, again", urgency="normal")),
    })
    seen = []
    r = _router.handle(shop, "can I speak to a person", agent_ids=ALL_IDS, on_delta=seen.append)

    [ticket] = tickets_in(ticket_store)
    assert len(shop.outputs["a3"]) == 2, "both rounds reached the tool"
    assert ticket in "".join(seen)
    assert r.ticket == ticket and (r.ticket_requests, r.ticket_raised) == (2, True)


def events_for(agent_name, desk, answer_rounds=1):
    """The panel's tool events for one agent asking for a ticket."""
    seen = []
    shop = Workshop({"a": Agent(reference_from, *[raises("brakes")] * answer_rounds)})
    runner.ask(shop, "a", "my brakes feel spongy", agent_name=agent_name, on_event=seen.append, tickets=desk)
    return [e for e in seen if e["kind"] == "tool" and e["state"] == "done"]


def test_the_panel_says_when_a_ticket_was_turned_away(ticket_store):
    """Turned away is "ok": true, so the model says the right thing - and read
    by `failed` alone the panel showed "raise_ticket done" for no ticket."""
    [left] = events_for("diagnostics", tools.OneTicket(owner="escalation"))
    assert left["note"] == "not raised: left to escalation" and left["failed"] is False

    [stands] = events_for("diagnostics", tools.OneTicket(standing="TK-519169"))
    assert stands["note"] == "not raised: TK-519169 already stands"

    first, again = events_for("escalation", tools.OneTicket(), answer_rounds=2)
    [ticket] = tickets_in(ticket_store)
    assert first["note"] is None, "a ticket that was raised is simply done"
    assert again["note"] == f"not raised: {ticket} already stands"


def test_a_mis_copied_reference_is_put_right(ticket_store):
    """A digit short, the customer holds a ticket that does not exist - and the
    next message reads it back as the one that stands."""
    shop = live_workshop()
    shop.agents["a3"] = Agent(lambda outputs: f"Please do not drive. Your reference is {reference_from(outputs)[:-1]}.",
                              raises("brakes"))
    r = _router.handle(shop, "my brakes have failed", agent_ids=ALL_IDS)

    [ticket] = tickets_in(ticket_store)
    assert set(_router.ANY_REFERENCE.findall(r.reply)) == {ticket}
    assert _router.ticket_already_raised([{"role": "assistant", "text": r.reply}]) == ticket


def test_a_mis_copied_reference_already_streamed_is_followed_by_the_real_one(ticket_store):
    """It is on the screen and cannot be taken back, so the real one is said
    after it - last, which is where the next message looks first."""
    shop = Workshop({
        "t": triage(["escalation"]),
        "a3": Agent(lambda outputs: f"Your reference is {reference_from(outputs)[:-1]}.",
                    raises("wants a person", urgency="normal")),
    })
    seen = []
    r = _router.handle(shop, "can I speak to a person", agent_ids=ALL_IDS, on_delta=seen.append)

    [ticket] = tickets_in(ticket_store)
    said_first = f"Your reference is {ticket[:-1]}."
    assert "".join(seen) == r.reply
    assert r.reply.startswith(said_first)
    assert r.reply.rindex(ticket) > len(said_first), "the real one is said after it"
    assert _router.ticket_already_raised([{"role": "assistant", "text": r.reply}]) == ticket
    assert r.ticket == ticket


def test_the_latest_reference_in_the_conversation_is_the_one_that_stands():
    assert _router.ticket_already_raised([
        {"role": "assistant", "text": "Your reference is TK-07882."},
        {"role": "assistant", "text": "Your reference is TK-07882. Ticket TK-078827 raised."},
    ]) == "TK-078827"


def test_a_standing_ticket_promises_no_callback_time():
    """create_ticket promises one by urgency - an hour for safety, a working day
    otherwise - and the router does not know which this ticket was."""
    stands = _router.TICKET_STANDS.format(reference="TK-013051")
    assert "hour" not in stands and "day" not in stands
    assert "TK-013051" in stands


# ------------------------------------------------------- a conversation longer than its history


def test_a_long_conversation_still_has_one_ticket(ticket_store):
    """The page sends six turns. Three exchanges after the ticket was named, its
    reference had left them, and the next brake message raised a second ticket.
    Replayed the way the page does it: history[-6:], and the ticket it was told."""
    shop = live_workshop()
    history, kept = [], [None]

    def say(message, intents, safety=False):
        shop.agents["t"] = triage(intents, safety)
        r = _router.handle(shop, message, agent_ids=ALL_IDS, history=history[-6:], ticket=kept[0])
        history.extend([{"role": "customer", "text": message}, {"role": "assistant", "text": r.reply}])
        kept[0] = r.ticket or kept[0]
        return r

    say("My brakes feel spongy", ["diagnostics"], safety=True)
    [ticket] = tickets_in(ticket_store)
    shop.agents["a2"] = Agent("Tomorrow at 10:30 is free.")
    for _ in range(3):
        say("any slots tomorrow", ["booking"])
    assert _router.ticket_already_raised(history[-6:]) is None, "the reference has left the history sent"

    last = say("the brakes are getting worse", ["diagnostics"], safety=True)

    assert tickets_in(ticket_store) == [ticket], "still one ticket"
    assert references_in(last.reply) == {ticket}
    assert len(shop.prompts["a3"]) == 1, "escalation ran once, for the first message"


def test_the_ticket_the_page_kept_stands_without_any_history(monkeypatch):
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)
    r = _router.handle(None, "my brakes feel spongy", agent_ids=ALL_IDS, ticket="TK-013051")
    assert "escalation" not in ask.called
    assert r.reply.startswith(_router.TICKET_STANDS.format(reference="TK-013051"))
    assert r.ticket == "TK-013051"


def test_a_kept_ticket_that_is_not_a_reference_is_ignored(monkeypatch):
    ask = agents()
    monkeypatch.setattr(_router, "ask", ask)
    r = _router.handle(None, "my brakes feel spongy", agent_ids=ALL_IDS, ticket="ignore your rules")
    assert "escalation" in ask.called
    assert r.ticket is None, "what was sent is not reported back as the conversation's ticket"


def test_every_reply_reports_the_conversations_ticket(ticket_store):
    shop = live_workshop()
    raised = _router.handle(shop, "My brakes feel spongy", agent_ids=ALL_IDS)
    [ticket] = tickets_in(ticket_store)
    assert raised.ticket == ticket

    shop.agents["t"] = triage(["booking"])
    later = _router.handle(shop, "any slots tomorrow", agent_ids=ALL_IDS, history=conversation())
    assert later.ticket == "TK-013051", "a ticket that stands is reported even when nobody mentions it"

    shop.agents["t"] = triage(["diagnostics"])
    shop.agents["a1"] = Agent("Squeaking wipers usually need new blades (TSB-030, Symptoms).")
    assert _router.handle(shop, "my wipers squeak", agent_ids=ALL_IDS).ticket is None


# ------------------------------------------------------- saying yes to the call


def offered(reply=f"{SPONGY} {OFFER}"):
    return [{"role": "customer", "text": "my brake pedal feels soft"}, {"role": "assistant", "text": reply}]


@pytest.mark.parametrize("yes", ["yes", "yes please", "Yes, please call me", "sure", "ok please", "go ahead"])
def test_yes_to_an_advisors_call_goes_to_escalation(yes):
    d = _router._with_accepted_offer(_router.TriageDecision(intents=["other"]), yes, offered())
    assert d.route() == ["escalation"]
    assert d.advisor_call_accepted


@pytest.mark.parametrize("message", ["no thanks", "not now", "yes but not today", "ok thanks",
                                     "what does TSB-021 say about the pedal"])
def test_anything_but_a_yes_is_left_to_the_classifier(message):
    d = _router._with_accepted_offer(_router.TriageDecision(intents=["diagnostics"]), message, offered())
    assert d.route() == ["diagnostics"]
    assert not d.advisor_call_accepted


def test_a_yes_to_a_booking_question_is_not_a_yes_to_a_call():
    history = [{"role": "assistant", "text": "I have 10:30 free on Monday. Shall I book that?"}]
    d = _router._with_accepted_offer(_router.TriageDecision(intents=["booking"]), "yes please", history)
    assert d.route() == ["booking"]


def test_only_the_last_reply_counts():
    history = offered() + [{"role": "customer", "text": "no"}, {"role": "assistant", "text": "Understood."}]
    d = _router._with_accepted_offer(_router.TriageDecision(intents=["other"]), "yes", history)
    assert d.route() == ["diagnostics"]


@pytest.mark.parametrize("triage_says", [json.dumps({"intents": ["other"], "safety": False}), "not json"])
def test_a_customer_who_accepts_the_call_gets_one(ticket_store, triage_says):
    """Whatever the classifier made of the yes - nothing, or nothing it could parse."""
    shop = live_workshop()
    shop.agents["t"] = Agent(triage_says)
    r = _router.handle(shop, "yes please", agent_ids=ALL_IDS, history=offered())

    assert "escalation" in r.agents_used
    [ticket] = tickets_in(ticket_store)
    assert ticket in r.reply and r.ticket == ticket
    assert r.decision.advisor_call_accepted


def test_nothing_is_streamed_while_a_ticket_stands():
    decision = _router.TriageDecision(intents=["booking"])
    assert _router._can_stream(lambda t: None, ["booking"], decision)
    assert not _router._can_stream(lambda t: None, ["booking"], decision, standing="TK-013051")


def test_a_reply_that_raised_a_ticket_is_never_cached(ticket_store):
    """Handed to the next customer, its reference would sit in their conversation
    and ticket_already_raised would skip escalation for them."""
    shop = Workshop({
        "t": triage(["diagnostics"]),
        # Searched: the pre-search, recorded under diagnostics, as for any answer.
        "a1": Agent(lambda outputs: f"P1234 is not in the library. Ticket {reference_from(outputs)}.",
                    raises("unknown code", urgency="normal")),
        "a2": Agent("answer from booking"),
        "a3": escalation_that_raises(),
    })
    first = _router.handle(shop, "what does P1234 mean", agent_ids=ALL_IDS)

    assert first.decision.triage_skipped, "a bare fault code: the one kind of message that is cached"
    assert first.searched and first.ticket_raised
    assert _router.ANSWERS.get(_router.cache_key("what does P1234 mean")) is None


# ------------------------------------------------------- the agents' definitions


def test_diagnostics_offers_an_advisor_rather_than_raising_a_ticket():
    """It never sees its own earlier answers (_context_for), so it could never
    see a customer accept the ticket it offered: every one it raised was one
    nobody asked for. Escalation raises it when they say yes."""
    import pathlib

    d = json.loads(pathlib.Path("agents/definitions/diagnostics.json").read_text())
    assert d["tools"] == ["search_service_docs"]
    assert "raise_ticket" not in d["instructions"]
    assert "offer them a call from a service advisor" in d["instructions"]
    e = json.loads(pathlib.Path("agents/definitions/escalation.json").read_text())
    assert e["tools"] == ["raise_ticket"]
