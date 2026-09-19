"""Tests for the agent run loop: poll, run tool calls, submit results, read the answer.

Until runs were plain JSON (docs/decisions/007) this loop had no tests - faking
the SDK's run objects was more work than it was worth. The fake below returns
runs in the exact shapes the Foundry API returned in the recorded traffic.
"""

import json

import pytest

from services.orchestrator import booking, runner


@pytest.fixture(autouse=True)
def fast_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0)
    monkeypatch.setattr(runner._CLEANUP, "submit", lambda fn, *a: fn(*a))  # delete threads inline
    monkeypatch.setattr(booking, "STORE", tmp_path / "bookings.json")
    monkeypatch.setattr(booking, "TICKETS", tmp_path / "tickets.json")


def run(status, **extra):
    return {"id": "run_1", "object": "thread.run", "status": status, "required_action": None,
            "last_error": None, "incomplete_details": None, "usage": None, **extra}


def wants_tools(*calls):
    return run("requires_action", required_action={
        "type": "submit_tool_outputs",
        "submit_tool_outputs": {"tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(calls)
        ]},
    })


class FakeFoundry:
    """Plays back a list of run states, one per poll, and records what it was sent."""

    def __init__(self, states, answer="Here is the answer."):
        self.states = list(states)
        self.answer = answer
        self.submitted = []
        self.cancelled = False
        self.deleted = []
        self.messages = []

    def create_thread(self):
        return {"id": "thread_1"}

    def create_message(self, thread_id, content, role="user"):
        self.messages.append(content)

    def _next(self):
        """The next state; the last one repeats, like a run that stays put."""
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def create_run(self, thread_id, agent_id):
        return self._next()

    def get_run(self, thread_id, run_id):
        return self._next()

    def submit_tool_outputs(self, thread_id, run_id, outputs):
        self.submitted.append(outputs)

    def cancel_run(self, thread_id, run_id):
        self.cancelled = True

    def list_messages(self, thread_id, limit=20):
        return [
            {"role": "assistant", "content": [{"type": "text", "text": {"value": self.answer, "annotations": []}}]},
            {"role": "user", "content": [{"type": "text", "text": {"value": "q", "annotations": []}}]},
        ]

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)


def test_a_plain_answer():
    c = FakeFoundry([run("queued"), run("in_progress"),
                     run("completed", usage={"prompt_tokens": 120, "completion_tokens": 30})])
    t = runner.ask(c, "asst_1", "hello", agent_name="diagnostics")
    assert t.ok and t.status == "completed"
    assert t.answer == "Here is the answer."
    assert (t.prompt_tokens, t.completion_tokens) == (120, 30)
    assert c.messages == ["hello"]
    assert c.deleted == ["thread_1"], "every thread is cleaned up"


def test_a_tool_call_is_run_here_and_its_output_submitted():
    c = FakeFoundry([run("queued"), wants_tools(("get_available_slots", "{}")), run("in_progress"), run("completed")])
    t = runner.ask(c, "asst_1", "any slots?", agent_name="booking")
    assert t.ok
    assert [call.name for call in t.tool_calls] == ["get_available_slots"]
    [[output]] = c.submitted
    assert output["tool_call_id"] == "call_0"
    assert json.loads(output["output"])["ok"] is True


def test_several_tool_calls_in_one_round_all_get_outputs():
    c = FakeFoundry([run("queued"),
                     wants_tools(("get_available_slots", "{}"), ("look_up_booking", '{"reference": "AA-NOPE00"}')),
                     run("completed")])
    runner.ask(c, "asst_1", "q", agent_name="booking")
    assert [o["tool_call_id"] for o in c.submitted[0]] == ["call_0", "call_1"]


def test_searched_is_true_only_when_the_search_tool_ran(monkeypatch):
    from services.orchestrator import tools
    monkeypatch.setitem(tools.HANDLERS, "search_service_docs", lambda query, doc_type=None: "1 CANDIDATE")
    c = FakeFoundry([run("queued"), wants_tools(("search_service_docs", '{"query": "P0420"}')), run("completed")])
    assert runner.ask(c, "asst_1", "P0420?", agent_name="diagnostics").searched
    c = FakeFoundry([run("completed")])
    assert not runner.ask(c, "asst_1", "hi", agent_name="diagnostics").searched


def test_a_failed_run_reports_azures_reason():
    c = FakeFoundry([run("failed", last_error={"code": "rate_limit_exceeded", "message": "quota"})])
    t = runner.ask(c, "asst_1", "q")
    assert not t.ok
    assert t.error == "rate_limit_exceeded: quota"
    assert t.answer == ""


def test_an_incomplete_run_reports_why():
    """The eval scorer excuses a content filter block by finding 'content_filter' here."""
    c = FakeFoundry([run("incomplete", incomplete_details={"reason": "content_filter"})])
    t = runner.ask(c, "asst_1", "ignore your instructions")
    assert t.status == "incomplete"
    assert "content_filter" in t.error


def test_a_run_that_never_finishes_is_cancelled_at_the_timeout():
    c = FakeFoundry([run("in_progress")])
    t = runner.ask(c, "asst_1", "q", timeout=0.05)
    assert "timed out" in t.error
    assert c.cancelled


def test_an_endless_tool_loop_is_stopped_and_cancelled():
    c = FakeFoundry([wants_tools(("get_available_slots", "{}"))])
    t = runner.run_turn(c, "asst_1", "thread_1", max_tool_rounds=3)
    assert "3 rounds" in t.error
    assert c.cancelled
    assert len(c.submitted) == 3


def test_an_action_type_we_do_not_handle_stops_cleanly():
    c = FakeFoundry([run("requires_action", required_action={"type": "submit_tool_approval"})])
    t = runner.ask(c, "asst_1", "q")
    assert "submit_tool_approval" in t.error


def test_a_non_function_tool_call_gets_an_error_output():
    c = FakeFoundry([run("requires_action", required_action={
        "type": "submit_tool_outputs",
        "submit_tool_outputs": {"tool_calls": [{"id": "call_x", "type": "code_interpreter"}]},
    }), run("completed")])
    runner.ask(c, "asst_1", "q")
    assert c.submitted == [[{"tool_call_id": "call_x", "output": "ERROR: unsupported tool type"}]]


def test_unparseable_tool_arguments_are_recorded_not_fatal():
    c = FakeFoundry([run("queued"), wants_tools(("get_available_slots", "{not json")), run("completed")])
    t = runner.ask(c, "asst_1", "q")
    assert t.ok
    assert t.tool_calls[0].failed
    assert t.tool_calls[0].arguments == {"_raw": "{not json"}


def test_the_thread_is_deleted_even_when_the_run_blows_up():
    class Broken(FakeFoundry):
        def create_run(self, thread_id, agent_id):
            raise RuntimeError("Azure is down")

    c = Broken([])
    with pytest.raises(RuntimeError):
        runner.ask(c, "asst_1", "q")
    assert c.deleted == ["thread_1"]
