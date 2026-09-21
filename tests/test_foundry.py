"""Tests for the Foundry agents client.

The expected paths, API version and bodies are what the azure-ai-agents SDK
actually sent, recorded from its traffic against the live project on 19 Sep
2026. If these pass, the plain calls say exactly what the SDK said.

Starting a run in one call, and streaming, came the day after that recording,
so there is nothing of the SDK's to compare them with. Their tests pin what the
service sends now.
"""

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from services.orchestrator import foundry

ENDPOINT = "https://rg-autoassist.services.ai.azure.com/api/projects/autoassist"


class FakeCredential:
    def __init__(self, lifetime=3600):
        self.calls = []
        self.lifetime = lifetime

    def get_token(self, *scopes):
        self.calls.append(scopes)
        return SimpleNamespace(token=f"token-{len(self.calls)}", expires_on=time.time() + self.lifetime)


def make(responder=None, credential=None):
    seen = []

    def handler(request):
        seen.append(request)
        answer = responder(request) if responder else {}
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)

    c = foundry.FoundryAgents(
        ENDPOINT + "/", credential or FakeCredential(), http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return c, seen


def body(request):
    return json.loads(request.content) if request.content else None


# ---------------------------------------------------------------- the recorded requests


@pytest.mark.parametrize("call,method,path,sent", [
    (lambda c: c.get_run("thread_1", "run_2"), "GET", "/threads/thread_1/runs/run_2", None),
    (lambda c: c.submit_tool_outputs("thread_1", "run_2", [{"tool_call_id": "call_3", "output": "x"}]),
     "POST", "/threads/thread_1/runs/run_2/submit_tool_outputs",
     {"tool_outputs": [{"tool_call_id": "call_3", "output": "x"}]}),
    (lambda c: c.cancel_run("thread_1", "run_2"), "POST", "/threads/thread_1/runs/run_2/cancel", None),
    (lambda c: c.list_messages("thread_1"), "GET", "/threads/thread_1/messages", None),
    (lambda c: c.delete_thread("thread_1"), "DELETE", "/threads/thread_1", None),
])
def test_each_call_sends_what_the_sdk_sent(call, method, path, sent):
    c, seen = make()
    call(c)
    r = seen[0]
    assert r.method == method
    assert r.url.path == "/api/projects/autoassist" + path
    assert r.url.params["api-version"] == "v1"
    assert body(r) == sent


# ---------------------------------------------------------------- starting and streaming a run


QUESTION = {"messages": [{"role": "user", "content": "what does P0420 mean"}]}


@pytest.mark.parametrize("call,path,sent", [
    (lambda c: c.create_thread_and_run("asst_9", "what does P0420 mean"),
     "/threads/runs", {"assistant_id": "asst_9", "thread": QUESTION}),
    (lambda c: list(c.stream_thread_and_run("asst_9", "what does P0420 mean")),
     "/threads/runs", {"assistant_id": "asst_9", "stream": True, "thread": QUESTION}),
    (lambda c: list(c.stream_tool_outputs("thread_1", "run_2", [{"tool_call_id": "call_3", "output": "x"}])),
     "/threads/thread_1/runs/run_2/submit_tool_outputs",
     {"tool_outputs": [{"tool_call_id": "call_3", "output": "x"}], "stream": True}),
])
def test_a_run_is_started_or_continued_in_one_request(call, path, sent):
    c, seen = make()
    call(c)
    [r] = seen
    assert r.method == "POST"
    assert r.url.path == "/api/projects/autoassist" + path
    assert r.url.params["api-version"] == "v1"
    assert body(r) == sent


def sse(text):
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


def test_a_stream_yields_each_event_until_done():
    c, seen = make(lambda request: sse(
        'event: thread.run.created\ndata: {"id": "run_1"}\n\n'
        "event: thread.message.delta\ndata: not json\n\n"
        'event: thread.message.delta\ndata: {"delta": {}}\n\n'
        "event: done\ndata: [DONE]\n\n"
        'event: thread.run.created\ndata: {"id": "after the end"}\n\n'
    ))
    assert list(c.stream_thread_and_run("asst_9", "hi")) == [
        ("thread.run.created", {"id": "run_1"}),
        ("thread.message.delta", {"delta": {}}),
    ]
    assert seen[0].headers["accept"] == "text/event-stream"


def test_a_rejected_token_is_refreshed_once_before_a_stream_starts():
    answers = iter([httpx.Response(401, text="token expired"), sse('data: {"ok": true}\n\n')])
    c, seen = make(lambda request: next(answers))
    assert list(c.stream_thread_and_run("asst_9", "hi")) == [("", {"ok": True})]
    assert [r.headers["authorization"] for r in seen] == ["Bearer token-1", "Bearer token-2"]


# ---------------------------------------------------------------- signing in


def test_every_call_is_signed_in_with_a_foundry_token():
    cred = FakeCredential()
    c, seen = make(credential=cred)
    c.get_run("thread_1", "run_2")
    assert seen[0].headers["authorization"] == "Bearer token-1"
    assert cred.calls[0] == ("https://ai.azure.com/.default",)


def test_the_token_is_reused_until_it_is_near_expiry():
    cred = FakeCredential()
    c, _ = make(credential=cred)
    c.get_run("thread_1", "run_2")
    c.get_run("thread_1", "run_2")
    assert len(cred.calls) == 1


def test_a_token_close_to_expiry_is_refreshed():
    cred = FakeCredential(lifetime=foundry.TOKEN_REFRESH_SECONDS - 10)
    c, seen = make(credential=cred)
    c.get_run("thread_1", "run_2")
    c.get_run("thread_1", "run_2")
    assert len(cred.calls) == 2
    assert seen[1].headers["authorization"] == "Bearer token-2"


def test_listing_agents_follows_every_page():
    pages = {
        None: {"data": [{"id": "a1", "name": "triage"}], "has_more": True, "last_id": "a1"},
        "a1": {"data": [{"id": "a2", "name": "booking"}], "has_more": False, "last_id": "a2"},
    }
    c, seen = make(lambda r: pages[r.url.params.get("after")])
    assert [a["name"] for a in c.list_agents()] == ["triage", "booking"]
    assert [r.url.params.get("after") for r in seen] == [None, "a1"]
    assert all(r.url.path.endswith("/assistants") for r in seen)


def test_messages_are_read_newest_first():
    c, seen = make(lambda r: {"data": [{"role": "assistant"}, {"role": "user"}]})
    assert [m["role"] for m in c.list_messages("thread_1")] == ["assistant", "user"]
    assert seen[0].url.params["order"] == "desc"
    assert seen[0].url.params["limit"] == "20"


def test_it_closes_its_connections():
    c, _ = make()
    with c:
        pass
    assert c._http.is_closed
