"""Tests for the Foundry agents client.

The expected paths, API version and bodies are what the azure-ai-agents SDK
actually sent, recorded from its traffic against the live project on 19 Sep
2026. If these pass, the plain calls say exactly what the SDK said.
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
        body = responder(request) if responder else {}
        return httpx.Response(200, json=body)

    c = foundry.FoundryAgents(
        ENDPOINT + "/", credential or FakeCredential(), http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return c, seen


def body(request):
    return json.loads(request.content) if request.content else None


# ---------------------------------------------------------------- the recorded requests


@pytest.mark.parametrize("call,method,path,sent", [
    (lambda c: c.create_thread(), "POST", "/threads", {}),
    (lambda c: c.create_message("thread_1", "what does P0420 mean"), "POST", "/threads/thread_1/messages",
     {"role": "user", "content": "what does P0420 mean"}),
    (lambda c: c.create_run("thread_1", "asst_9"), "POST", "/threads/thread_1/runs", {"assistant_id": "asst_9"}),
    (lambda c: c.get_run("thread_1", "run_2"), "GET", "/threads/thread_1/runs/run_2", None),
    (lambda c: c.submit_tool_outputs("thread_1", "run_2", [{"tool_call_id": "call_3", "output": "x"}]),
     "POST", "/threads/thread_1/runs/run_2/submit_tool_outputs",
     {"tool_outputs": [{"tool_call_id": "call_3", "output": "x"}]}),
    (lambda c: c.cancel_run("thread_1", "run_2"), "POST", "/threads/thread_1/runs/run_2/cancel", None),
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


def test_every_call_is_signed_in_with_a_foundry_token():
    cred = FakeCredential()
    c, seen = make(credential=cred)
    c.create_thread()
    assert seen[0].headers["authorization"] == "Bearer token-1"
    assert cred.calls[0] == ("https://ai.azure.com/.default",)


def test_the_token_is_reused_until_it_is_near_expiry():
    cred = FakeCredential()
    c, _ = make(credential=cred)
    c.create_thread()
    c.create_thread()
    assert len(cred.calls) == 1


def test_a_token_close_to_expiry_is_refreshed():
    cred = FakeCredential(lifetime=foundry.TOKEN_REFRESH_SECONDS - 10)
    c, seen = make(credential=cred)
    c.create_thread()
    c.create_thread()
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


def test_it_closes_its_connections():
    c, _ = make()
    with c:
        pass
    assert c._http.is_closed
