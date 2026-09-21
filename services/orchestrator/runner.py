"""Run an agent turn, executing tool calls ourselves.

Why not runs.create_and_process()? Two reasons:

  1. It polls with no timeout. A run stuck in 'in_progress' hangs the caller
     forever - which is exactly what happened with the built-in search tool
     (docs/evaluation.md, finding 5).

  2. It hides the tool calls. We want every call, its arguments, its timing and
     its output recorded, because week 1 showed that whether a tool was called
     matters more than what the answer looks like.

The loop: create the run, poll it, and when Azure says 'requires_action', run
the requested functions here and submit the outputs back.

Runs are the JSON the Foundry API returns (see foundry.py), not SDK objects.
The fields read here:

    status             queued | in_progress | requires_action | completed | failed |
                       cancelling | cancelled | expired | incomplete
    required_action    {"type": "submit_tool_outputs",
                        "submit_tool_outputs": {"tool_calls": [
                            {"id", "type": "function", "function": {"name", "arguments"}}]}}
    last_error         {"code", "message"}          when failed
    incomplete_details {"reason": ...}              when incomplete
    usage              {"prompt_tokens", "completion_tokens", ...}
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import azure_http, telemetry, tools
from .foundry import FoundryAgents

log = logging.getLogger(__name__)

# Every state a run can stop in. Missing one means polling a finished run until
# the timeout - which is what happened with "incomplete", a status Azure returns
# when a run stops early (max tokens, content filter, or a truncated response).
# The run was over in seconds; we waited 90 and reported a timeout.
TERMINAL = {
    "completed",
    "failed",
    "cancelled",
    "cancelling",
    "expired",
    "incomplete",
}

# Measured: at 0.8s we waited an average of 0.4s after a run had already
# finished, several times per request. 0.25s costs a few more cheap GETs and
# gives most of that back. Below ~0.15s the polling itself starts to matter.
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "0.25"))

# A run that asks for tools round after round without ever answering is stuck,
# not thorough. Both paths, polled and streamed, stop it here rather than let it
# spend the whole timeout.
MAX_TOOL_ROUNDS = 8

# Thread cleanup runs in the background. Deleting a thread is housekeeping - the
# answer is already in hand - so blocking the customer's response on it buys
# nothing. Two workers are plenty; deletes are ~200ms and never urgent.
_CLEANUP = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thread-cleanup")


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    output_preview: str
    duration_ms: int
    failed: bool = False


@dataclass
class TurnResult:
    """Everything that happened in one agent turn."""

    answer: str = ""
    status: str = ""
    agent_name: str = ""
    thread_id: str = ""
    run_id: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None
    # Set when the turn ended because Azure's token quota was spent rather than
    # because anything went wrong. The reply the customer gets is different, so
    # the difference has to survive as far as _compose.
    throttled: bool = False
    retry_after: float = 0.0

    @property
    def searched(self) -> bool:
        return any(c.name == "search_service_docs" for c in self.tool_calls)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and not self.error

    def tool_summary(self) -> str:
        if not self.tool_calls:
            return "(no tools called)"
        return ", ".join(f"{c.name}({c.duration_ms}ms)" for c in self.tool_calls)


def emit(on_event, **event) -> None:
    """Tell the page something happened, if anyone is listening.

    Never raises. This exists to draw a panel; a request must not fail because
    the drawing went wrong, and it is called from inside the agent threads.
    """
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:  # noqa: BLE001
        log.debug("could not report %s", event.get("kind"), exc_info=True)


def _status_of(run: dict) -> str:
    return str(run.get("status") or "").lower()


def _error_text(err) -> str:
    if isinstance(err, dict):
        return f"{err.get('code') or 'error'}: {err.get('message') or ''}".strip(": ")
    return str(err or "run failed")


def run_turn(
    client: FoundryAgents,
    run: dict,
    timeout: float = 90.0,
    agent_name: str = "",
    on_event=None,
    tickets: tools.OneTicket | None = None,
) -> TurnResult:
    """Poll a run that has already started until it ends, running its tools.

    `run` comes from ask(), which starts it in the same call that creates the
    thread, so it arrives carrying its thread_id.

    `tickets` is the customer message's OneTicket, shared with every other turn
    answering it. A turn run on its own - agents/ask.py, an eval case - gets one
    of its own, so it too raises one ticket at most, however many rounds it takes.
    """
    started = time.time()
    thread_id = run.get("thread_id", "")
    out = TurnResult(thread_id=thread_id, agent_name=agent_name)
    tickets = tickets if tickets is not None else tools.OneTicket()

    with telemetry.span("agent.turn", agent=agent_name, thread_id=thread_id) as turn_span:
        _run_turn_inner(client, run, timeout, out, started, on_event, tickets)
        _record_turn(turn_span, out)
    return out


def _run_turn_inner(
    client: FoundryAgents,
    run: dict,
    timeout: float,
    out: TurnResult,
    started: float,
    on_event=None,
    tickets: tools.OneTicket | None = None,
) -> None:
    thread_id, run_id = out.thread_id, run["id"]
    out.run_id = run_id

    rounds = 0
    last_logged = None

    while True:
        elapsed = time.time() - started
        if elapsed > timeout:
            out.status = _status_of(run)
            out.error = f"timed out after {timeout:.0f}s in state '{out.status}'"
            _cancel(client, thread_id, run_id)
            break

        status = _status_of(run)
        if status != last_logged:
            log.info("run %s: %s (%.1fs)", run_id, status, elapsed)
            last_logged = status

        if status in TERMINAL:
            _record_ending(out, run)
            break

        if status == "requires_action":
            rounds += 1
            if _too_many_rounds(out, rounds):
                _cancel(client, thread_id, run_id)
                break

            outputs = _run_requested_tools(run, out, on_event, tickets)
            if outputs is None:
                action_type = (run.get("required_action") or {}).get("type")
                out.status = status
                out.error = f"run needs an action we do not handle: {action_type or 'none given'}"
                break

            client.submit_tool_outputs(thread_id, run_id, outputs)

        time.sleep(POLL_SECONDS)
        run = client.get_run(thread_id, run_id)

    _record_usage(out, run)

    if out.status == "completed":
        out.answer = _latest_assistant_text(client, thread_id)

    out.duration_ms = int((time.time() - started) * 1000)


# ------------------------------------------------- shared by polling and streaming


def _run_requested_tools(run: dict, out: TurnResult, on_event=None,
                         tickets: tools.OneTicket | None = None) -> list[dict] | None:
    """Execute the tools a run asked for, recording each one, and return the
    outputs to submit - or None when the run wants an action we do not handle.

    Both paths come through here, so a tool call leaves the same record, span
    and panel events whether the answer was polled or streamed - and every
    raise_ticket meets the same OneTicket, whichever path and thread it came from.
    """
    action = run.get("required_action") or {}
    if action.get("type") != "submit_tool_outputs":
        return None

    agent_name = out.agent_name
    outputs: list[dict] = []
    for call in (action.get("submit_tool_outputs") or {}).get("tool_calls") or []:
        if call.get("type") != "function":
            outputs.append({"tool_call_id": call.get("id"), "output": "ERROR: unsupported tool type"})
            continue

        name = call["function"]["name"]
        raw_args = call["function"].get("arguments") or ""

        emit(on_event, kind="tool", agent=agent_name, name=name, state="running")
        with telemetry.span("tool.call", tool=name, agent=agent_name) as ts:
            t0 = time.time()
            result = tools.execute(name, raw_args, tickets=tickets, asked_by=agent_name)
            ms = int((time.time() - t0) * 1000)
            failed = result.startswith("ERROR:")
            emit(on_event, kind="tool", agent=agent_name, name=name, state="done", ms=ms, failed=failed,
                 note=_not_raised(name, result))
            telemetry.set(
                ts,
                duration_ms=ms,
                failed=failed,
                args=raw_args[:500] if raw_args else "",
                # Retrieval reports its own filtering in the first line
                # of its output; keep it so a bad search is visible in
                # the trace without opening the whole result.
                result_head=result[:200],
            )

        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed = {"_raw": raw_args}

        out.tool_calls.append(
            ToolCallRecord(
                name=name,
                arguments=parsed if isinstance(parsed, dict) else {"_raw": raw_args},
                output_preview=result[:300],
                duration_ms=ms,
                failed=failed,
            )
        )
        log.info("  tool %s(%s) -> %dms%s", name, parsed, ms, " FAILED" if failed else "")

        outputs.append({"tool_call_id": call["id"], "output": result})
    return outputs


def _not_raised(name: str, result: str) -> str | None:
    """Why a raise_ticket call raised nothing, for the panel - None when it raised one.

    tools.OneTicket answers a call it turns away with "ok": true, so that the
    model tells the customer what is actually happening rather than that
    something failed. Judged on `failed` alone, the panel showed "raise_ticket
    done" for a ticket that was never raised - under an agent that, once
    redeployed, does not even have the tool.
    """
    if name != "raise_ticket" or result.startswith("ERROR:"):
        return None
    try:
        said = json.loads(result)
    except ValueError:
        return None
    if not isinstance(said, dict):
        return None
    if said.get("left_to"):
        return f"not raised: left to {said['left_to']}"
    if said.get("already_raised"):
        return f"not raised: {said.get('reference')} already stands"
    return None


def _too_many_rounds(out: TurnResult, rounds: int) -> bool:
    """Whether this round of tool calls is one too many; if so, the turn says why."""
    if rounds <= MAX_TOOL_ROUNDS:
        return False
    out.status = "requires_action"
    out.error = f"stopped after {MAX_TOOL_ROUNDS} rounds of tool calls (possible loop)"
    return True


def _record_ending(out: TurnResult, run: dict) -> None:
    """How a run that has stopped ended, and why when it was not a success."""
    out.status = _status_of(run)
    if out.status == "failed":
        out.error = _error_text(run.get("last_error"))
    elif out.status == "incomplete":
        # The run stopped early. Azure says why in incomplete_details -
        # usually max tokens or a content filter. Worth surfacing: the
        # answer may be truncated mid-sentence.
        out.error = f"run ended early: {run.get('incomplete_details') or 'no reason given'}"
    elif out.status in ("cancelled", "cancelling", "expired"):
        out.error = f"run {out.status}"


def _record_usage(out: TurnResult, run: dict) -> None:
    usage = run.get("usage") or {}
    out.prompt_tokens = usage.get("prompt_tokens") or 0
    out.completion_tokens = usage.get("completion_tokens") or 0


def _record_turn(span, out: TurnResult) -> None:
    """The agent.turn span's attributes, from one list: a streamed turn has to
    look the same in the trace as a polled one."""
    telemetry.set(
        span,
        run_id=out.run_id,
        status=out.status,
        searched=out.searched,
        tool_count=len(out.tool_calls),
        tools=[c.name for c in out.tool_calls],
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
        duration_ms=out.duration_ms,
        error=out.error,
    )


def _cancel(client: FoundryAgents, thread_id: str, run_id: str) -> None:
    """Best effort: the turn has already been given up on, and failing to stop
    the run must not turn that into an exception."""
    try:
        client.cancel_run(thread_id, run_id)
    except Exception:  # noqa: BLE001 - cancelling is best effort
        pass


def _latest_assistant_text(client: FoundryAgents, thread_id: str) -> str:
    for m in client.list_messages(thread_id):
        if m.get("role") == "assistant":
            parts = [c["text"]["value"] for c in m.get("content") or [] if c.get("type") == "text"]
            return "\n".join(parts).strip()
    return ""


def ask(
    client: FoundryAgents,
    agent_id: str,
    question: str,
    timeout: float = 90.0,
    agent_name: str = "",
    on_event=None,
    tickets: tools.OneTicket | None = None,
) -> TurnResult:
    """One question, one fresh thread.

    A fresh thread every time is not tidiness. Reusing a thread lets the model
    answer from chunks already in its history instead of searching again, which
    silently invalidates any test you run that way (finding 3).

    `on_event(dict)` is optional and is how the page shows what is happening
    while it happens: which agent is working, which tool it just called, how
    long it took. It may be called from several threads at once - the specialists
    run concurrently - so whatever is behind it has to cope with that.

    `tickets` is the message's OneTicket; see run_turn.
    """
    emit(on_event, kind="agent", name=agent_name, state="working")
    run = client.create_thread_and_run(agent_id, question)
    thread_id = run.get("thread_id", "")
    try:
        turn = run_turn(client, run, timeout=timeout, agent_name=agent_name, on_event=on_event,
                        tickets=tickets)
        emit(on_event, kind="agent", name=agent_name, state="done", ms=turn.duration_ms,
             tokens=turn.prompt_tokens + turn.completion_tokens, ok=turn.ok)
        return turn
    except BaseException:
        emit(on_event, kind="agent", name=agent_name, state="failed")
        raise
    finally:
        if thread_id:
            _CLEANUP.submit(_delete_thread, client, thread_id)


THROTTLED_ERROR = "throttled: Azure had no token quota left"


def throttled_turn(agent_name: str, e: azure_http.Throttled) -> TurnResult:
    """A turn that never ran, because Azure's token quota was spent.

    Returned rather than raised so the caller can carry on: on a safety-flagged
    message the warning still has to reach the customer, and it does not come
    from a model. See router._compose.
    """
    log.warning("%s: Azure is throttling, not retrying (%s)", agent_name or "agent", e.message)
    out = TurnResult(agent_name=agent_name, status="failed")
    # Azure's own wording, not repeated here: it names the deployment and the
    # pricing tier, and the trace this ends up in is returned in the body of a
    # public, unauthenticated endpoint. The full text is in the log above.
    out.error = THROTTLED_ERROR
    out.throttled = True
    out.retry_after = e.retry_after
    return out


def _delete_thread(client: FoundryAgents, thread_id: str) -> None:
    """Best effort. A thread we failed to delete costs nothing but clutter."""
    try:
        client.delete_thread(thread_id)
    except Exception as e:  # noqa: BLE001
        log.debug("could not delete thread %s: %s", thread_id, e)


# ---------------------------------------------------------------- streaming


def ask_streaming(
    client: FoundryAgents,
    agent_id: str,
    question: str,
    on_delta,
    timeout: float = 90.0,
    agent_name: str = "",
    on_status=None,
    on_event=None,
    tickets: tools.OneTicket | None = None,
) -> TurnResult:
    """One question, one fresh thread, with the answer delivered as it is written.

    `on_delta(text)` is called with each fragment. Everything else matches ask():
    the same tool execution, the same TurnResult, the same spans - so the trace,
    the `searched` flag and the eval suite do not know the difference.

    Tool calls are run here as usual: the stream stops with requires_action, we
    execute, and submitting the outputs starts the stream again on the same run.
    """
    started = time.time()
    out = TurnResult(agent_name=agent_name)
    tickets = tickets if tickets is not None else tools.OneTicket()  # as in run_turn

    emit(on_event, kind="agent", name=agent_name, state="working")
    with telemetry.span("agent.turn", agent=agent_name, streamed=True) as turn_span:
        try:
            _stream_turn(client, agent_id, question, on_delta, timeout, out, started, on_status, on_event,
                         tickets)
        finally:
            emit(on_event, kind="agent", name=agent_name, state="done" if out.status else "failed",
                 ms=out.duration_ms, tokens=out.prompt_tokens + out.completion_tokens, ok=out.ok)
            _record_turn(turn_span, out)
            if out.thread_id:
                _CLEANUP.submit(_delete_thread, client, out.thread_id)
    return out


# What each tool is doing, in words a customer understands. Shown while they
# wait: an agent searches before it writes, so the first words of an answer can
# be 5s away, and an empty screen for 5s is most of what "slow" means here.
TOOL_STATUS = {
    "search_service_docs": "looking in the service documents",
    "get_available_slots": "checking the calendar",
    "book_service_slot": "making the booking",
    "move_service_booking": "moving the booking",
    "cancel_service_booking": "cancelling the booking",
    "look_up_booking": "looking up the booking",
    "raise_ticket": "raising a ticket for a service advisor",
}


def _stream_turn(client, agent_id, question, on_delta, timeout, out: TurnResult, started: float,
                 on_status=None, on_event=None, tickets: tools.OneTicket | None = None) -> None:
    answer: list[str] = []
    try:
        _stream_events(client, agent_id, question, on_delta, timeout, out, started, on_status, answer, on_event,
                       tickets)
    except azure_http.Throttled as e:
        # Mid-answer throttling: the run started, so some of the answer may
        # already be on the customer's screen. Whatever arrived is kept - it is
        # theirs and it was grounded - and the reply says it was cut short.
        log.warning("%s: throttled while streaming (%s)", out.agent_name or "agent", e.message)
        out.throttled, out.retry_after = True, e.retry_after
        out.status = out.status or "failed"
        out.error = THROTTLED_ERROR

    out.answer = "".join(answer).strip()
    out.duration_ms = int((time.time() - started) * 1000)
    if not out.status:
        out.status, out.error = "failed", out.error or "the stream ended without finishing the run"


def _stream_events(client, agent_id, question, on_delta, timeout, out: TurnResult, started: float,
                   on_status, answer: list[str], on_event=None, tickets: tools.OneTicket | None = None) -> None:
    events = client.stream_thread_and_run(agent_id, question)
    rounds = 0

    while events is not None:
        next_events = None
        for event, data in events:
            if time.time() - started > timeout:
                out.status, out.error = "in_progress", f"timed out after {timeout:.0f}s while streaming"
                break

            if event == "thread.run.created":
                out.run_id, out.thread_id = data.get("id", ""), data.get("thread_id", "")

            elif event == "thread.message.delta":
                for part in (data.get("delta") or {}).get("content") or []:
                    text = ((part or {}).get("text") or {}).get("value") or ""
                    if text:
                        answer.append(text)
                        on_delta(text)

            elif event == "thread.run.requires_action":
                rounds += 1
                if _too_many_rounds(out, rounds):
                    break
                if on_status:
                    for call in ((data.get("required_action") or {}).get("submit_tool_outputs") or {}).get("tool_calls") or []:
                        said = TOOL_STATUS.get((call.get("function") or {}).get("name", ""))
                        if said:
                            on_status(said)
                outputs = _run_requested_tools(data, out, on_event, tickets)
                if outputs is None:  # an action we do not handle
                    out.status = "requires_action"
                    out.error = "run needs an action we do not handle"
                    break
                next_events = client.stream_tool_outputs(out.thread_id, out.run_id, outputs)
                break  # this stream is finished; the next one continues the run

            elif event in ("thread.run.completed", "thread.run.failed", "thread.run.incomplete",
                           "thread.run.cancelled", "thread.run.expired"):
                _record_ending(out, data)
                _record_usage(out, data)
        events = next_events
