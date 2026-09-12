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
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import RequiredFunctionToolCall, SubmitToolOutputsAction, ToolOutput

from . import telemetry, tools

log = logging.getLogger(__name__)

TERMINAL = {"completed", "failed", "cancelled", "expired"}

# Measured: at 0.8s we waited an average of 0.4s after a run had already
# finished, several times per request. 0.25s costs a few more cheap GETs and
# gives most of that back. Below ~0.15s the polling itself starts to matter.
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "0.25"))

# Thread cleanup runs in the background. Deleting a thread is housekeeping - the
# answer is already in hand - so blocking the customer's response on it buys
# nothing. One worker is enough; deletes are ~200ms and never urgent.
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


def _status_of(run) -> str:
    return str(run.status).lower().split(".")[-1]


def run_turn(
    client: AgentsClient,
    agent_id: str,
    thread_id: str,
    timeout: float = 90.0,
    max_tool_rounds: int = 8,
    agent_name: str = "",
) -> TurnResult:
    """Run one turn on an existing thread that already has the user message."""
    started = time.time()
    out = TurnResult(thread_id=thread_id, agent_name=agent_name)

    with telemetry.span("agent.turn", agent=agent_name, thread_id=thread_id) as turn_span:
        _run_turn_inner(client, agent_id, thread_id, timeout, max_tool_rounds, agent_name, out, started)
        telemetry.set(
            turn_span,
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
    return out


def _run_turn_inner(
    client: AgentsClient,
    agent_id: str,
    thread_id: str,
    timeout: float,
    max_tool_rounds: int,
    agent_name: str,
    out: TurnResult,
    started: float,
) -> None:
    run = client.runs.create(thread_id=thread_id, agent_id=agent_id)
    out.run_id = run.id

    rounds = 0
    last_logged = None

    while True:
        elapsed = time.time() - started
        if elapsed > timeout:
            out.status = _status_of(run)
            out.error = f"timed out after {timeout:.0f}s in state '{out.status}'"
            try:
                client.runs.cancel(thread_id=thread_id, run_id=run.id)
            except Exception:  # noqa: BLE001 - cancelling is best effort
                pass
            break

        status = _status_of(run)
        if status != last_logged:
            log.info("run %s: %s (%.1fs)", run.id, status, elapsed)
            last_logged = status

        if status in TERMINAL:
            out.status = status
            if status == "failed":
                out.error = str(getattr(run, "last_error", "run failed"))
            break

        if status == "requires_action":
            rounds += 1
            if rounds > max_tool_rounds:
                out.status = status
                out.error = f"stopped after {max_tool_rounds} rounds of tool calls (possible loop)"
                try:
                    client.runs.cancel(thread_id=thread_id, run_id=run.id)
                except Exception:  # noqa: BLE001
                    pass
                break

            action = run.required_action
            if not isinstance(action, SubmitToolOutputsAction):
                out.status = status
                out.error = f"run needs an action we do not handle: {type(action).__name__}"
                break

            outputs: list[ToolOutput] = []
            for call in action.submit_tool_outputs.tool_calls:
                if not isinstance(call, RequiredFunctionToolCall):
                    outputs.append(ToolOutput(tool_call_id=call.id, output="ERROR: unsupported tool type"))
                    continue

                name = call.function.name
                raw_args = call.function.arguments

                with telemetry.span("tool.call", tool=name, agent=agent_name) as ts:
                    t0 = time.time()
                    result = tools.execute(name, raw_args)
                    ms = int((time.time() - t0) * 1000)
                    telemetry.set(
                        ts,
                        duration_ms=ms,
                        failed=result.startswith("ERROR:"),
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
                        failed=result.startswith("ERROR:"),
                    )
                )
                log.info("  tool %s(%s) -> %dms%s", name, parsed, ms,
                         " FAILED" if result.startswith("ERROR:") else "")

                outputs.append(ToolOutput(tool_call_id=call.id, output=result))

            client.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=outputs)

        time.sleep(POLL_SECONDS)
        run = client.runs.get(thread_id=thread_id, run_id=run.id)

    usage = getattr(run, "usage", None)
    if usage:
        out.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        out.completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    if out.status == "completed":
        out.answer = _latest_assistant_text(client, thread_id)

    out.duration_ms = int((time.time() - started) * 1000)


def _latest_assistant_text(client: AgentsClient, thread_id: str) -> str:
    for m in client.messages.list(thread_id=thread_id):
        if m.role == "assistant":
            return "\n".join(p.text.value for p in m.text_messages).strip()
    return ""


def ask(
    client: AgentsClient,
    agent_id: str,
    question: str,
    timeout: float = 90.0,
    agent_name: str = "",
) -> TurnResult:
    """One question, one fresh thread.

    A fresh thread every time is not tidiness. Reusing a thread lets the model
    answer from chunks already in its history instead of searching again, which
    silently invalidates any test you run that way (finding 3).
    """
    thread = client.threads.create()
    try:
        client.messages.create(thread_id=thread.id, role="user", content=question)
        return run_turn(client, agent_id, thread.id, timeout=timeout, agent_name=agent_name)
    finally:
        _CLEANUP.submit(_delete_thread, client, thread.id)


def _delete_thread(client: AgentsClient, thread_id: str) -> None:
    """Best effort. A thread we failed to delete costs nothing but clutter."""
    try:
        client.threads.delete(thread_id)
    except Exception as e:  # noqa: BLE001
        log.debug("could not delete thread %s: %s", thread_id, e)
