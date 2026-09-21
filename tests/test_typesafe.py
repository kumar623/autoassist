"""Tests for the Jev client. TypeSafe is replaced by httpx.MockTransport - no network.

The key is the thing to be careful about here. It travels in a header rather
than in the URL, so it is less exposed than Zoho's MCP address, but an exception
that quotes the request would still put it somewhere it does not belong.
"""

import json

import httpx
import pytest

from services.orchestrator import azure_http, typesafe


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-s3cr3tkey0123456789")
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(azure_http, "_pause", waits.append)
    return waits


def client(*responses):
    """A client whose server answers with each response in turn."""
    seen = []
    queue = list(responses)

    def handler(request):
        seen.append(request)
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    c = httpx.Client(transport=httpx.MockTransport(handler))
    c.seen = seen
    return c


def answered(**answers):
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                     "usage": {"input_tokens": 304, "output_tokens": 18}})


NOUL = {"type": "noul", "noul": 0.93}


# ------------------------------------------------------------ the request


def test_the_request_carries_the_state_the_model_and_the_questions():
    c = client(answered(is_urgent=NOUL))
    typesafe.ask("my brakes have failed", {"is_urgent": typesafe.noul("Is this urgent?")}, http=c)

    body = json.loads(c.seen[0].content)
    assert body["state"] == "my brakes have failed"
    assert body["model"] == typesafe.DEFAULT_MODEL
    assert body["questions"]["is_urgent"] == {"type": "noul", "instructions": "Is this urgent?"}


def test_the_key_is_sent_as_a_bearer_token():
    c = client(answered(q=NOUL))
    typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert c.seen[0].headers["authorization"] == "Bearer ts-s3cr3tkey0123456789"


def test_the_model_can_be_pinned(monkeypatch):
    """jev-latest moves. A measurement worth repeating names the version."""
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-1.13.0")
    c = client(answered(q=NOUL))
    typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert json.loads(c.seen[0].content)["model"] == "jev-1.13.0"


def test_a_structured_state_is_allowed():
    """The API takes an object or array, which is how a conversation goes in."""
    c = client(answered(q=NOUL))
    state = [{"role": "customer", "text": "hi"}, {"role": "assistant", "text": "hello"}]
    typesafe.ask(state, {"q": typesafe.noul("?")}, http=c)
    assert json.loads(c.seen[0].content)["state"] == state


def test_criteria_are_sent_when_given():
    q = typesafe.noul("Is it a safety issue?", true_means="brakes, steering", false_means="anything else")
    assert q["criteria"] == {"true": "brakes, steering", "false": "anything else"}


def test_a_question_with_no_criteria_does_not_send_an_empty_one():
    assert "criteria" not in typesafe.noul("Is this urgent?")


# ------------------------------------------------------------ the answer


def test_a_noul_answer_is_a_probability():
    c = client(answered(safety=NOUL))
    out = typesafe.ask("x", {"safety": typesafe.noul("?")}, http=c)
    assert typesafe.probability(out["answers"], "safety") == 0.93
    assert out["usage"]["input_tokens"] == 304


def test_a_missing_answer_is_none_not_zero():
    """"Jev did not answer" and "Jev said almost certainly not" are different, and
    a caller treating them the same would read a missing safety answer as safe."""
    assert typesafe.probability({}, "safety") is None
    assert typesafe.probability({"safety": {"type": "noul"}}, "safety") is None


def test_a_reply_without_answers_is_an_error():
    c = client(httpx.Response(200, json={"model": "jev", "usage": {}}))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="answers"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)


def test_a_reply_that_is_not_json_is_an_error():
    c = client(httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="not JSON"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)


# ------------------------------------------------------------ going wrong


def test_being_overloaded_is_retried(no_waiting):
    """529 is TypeSafe's 'come back shortly'."""
    c = client(httpx.Response(typesafe.OVERLOADED), answered(q=NOUL))
    assert typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)["answers"]
    assert len(c.seen) == 2


def test_a_brief_rate_limit_is_waited_out(no_waiting):
    c = client(httpx.Response(429, headers={"retry-after": "1"}), answered(q=NOUL))
    typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert no_waiting == [1.0] and len(c.seen) == 2


def test_a_long_rate_limit_is_not_waited_out(no_waiting):
    c = client(httpx.Response(429, headers={"retry-after": "60"}))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="rate limiting"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert len(c.seen) == 1 and no_waiting == []


def test_a_bad_key_is_not_retried():
    """401 will not fix itself; retrying only delays the real answer."""
    c = client(httpx.Response(401, json={"error": {"message": "invalid api key"}}))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="invalid api key"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert len(c.seen) == 1


def test_a_validation_error_explains_itself():
    c = client(httpx.Response(422, json={"detail": "questions.safety.instructions is required"}))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="instructions is required"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)


def test_a_server_that_cannot_be_reached_is_an_error(no_waiting):
    c = client(*[httpx.ConnectError("down")] * (azure_http.MAX_RETRIES + 1))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="could not reach"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)


def test_a_short_throttle_on_the_last_attempt_is_reported_not_crashed(no_waiting):
    """Three 529s then a brief 429 used to fall out of the loop into
    AssertionError('unreachable'), which jev_triage then logged as an
    unexpected failure rather than as Jev being busy."""
    c = client(*[httpx.Response(typesafe.OVERLOADED)] * azure_http.MAX_RETRIES,
               httpx.Response(429, headers={"retry-after": "1"}))
    with pytest.raises(typesafe.TypeSafeUnavailable, match="rate limiting"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)


def test_no_retries_means_one_attempt(no_waiting):
    """What jev_triage asks for: its fallback is the retry."""
    c = client(httpx.ConnectError("down"))
    with pytest.raises(typesafe.TypeSafeUnavailable):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c, retries=0)
    assert len(c.seen) == 1 and no_waiting == []


def test_a_request_httpx_will_not_send_is_not_retried(no_waiting):
    """LocalProtocolError quotes the header it refuses, which is the key."""
    c = client(httpx.LocalProtocolError("Illegal header value b'Bearer ts-s3cr3tkey0123456789 '"))
    with pytest.raises(typesafe.TypeSafeUnavailable) as e:
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert len(c.seen) == 1
    assert "s3cr3tkey" not in str(e.value) and e.value.__cause__ is None


# --------------------------------------------------------- keeping the key


@pytest.mark.parametrize("response", [
    httpx.Response(401, json={"error": {"message": "invalid api key"}}),
    httpx.Response(422, json={"detail": "bad question"}),
    httpx.Response(429, headers={"retry-after": "60"}),
    httpx.Response(500, text="boom"),
])
def test_the_key_never_appears_in_an_error(response):
    c = client(response)
    with pytest.raises(typesafe.TypeSafeUnavailable) as e:
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert "s3cr3tkey" not in str(e.value)


def test_a_transport_failure_does_not_chain_the_original(no_waiting):
    """httpx's exception repeats the request; nothing from it is passed on."""
    c = client(*[httpx.ConnectError("no route to host")] * (azure_http.MAX_RETRIES + 1))
    with pytest.raises(typesafe.TypeSafeUnavailable) as e:
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert e.value.__cause__ is None
    assert "s3cr3tkey" not in str(e.value)


# ------------------------------------------------------------ not set up


def test_without_a_key_it_says_so_rather_than_calling(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert not typesafe.configured()
    with pytest.raises(typesafe.TypeSafeUnavailable, match="TYPESAFE_API_KEY"):
        typesafe.ask("x", {"q": typesafe.noul("?")}, http=client())


def test_a_blank_key_counts_as_unset(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    assert not typesafe.configured()


def test_asking_nothing_is_refused():
    with pytest.raises(typesafe.TypeSafeUnavailable, match="no questions"):
        typesafe.ask("x", {}, http=client())


def test_importing_never_raises_however_it_is_configured(monkeypatch):
    """The service imports this module (through jev_triage) whether or not Jev
    is switched on, so a bad setting must not stop it starting - the rule
    limits.setting() exists for."""
    import importlib

    monkeypatch.setenv("TYPESAFE_MODEL", "")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    importlib.reload(typesafe)
    assert typesafe.model() == typesafe.DEFAULT_MODEL


# ------------------------------------------------- one connection, reused


def test_the_client_is_shared_between_calls():
    """A fresh connection per call was 1,115ms from Vizag; a reused one 391ms.
    Almost two thirds of the latency was a TCP and TLS handshake being thrown
    away and paid again."""
    typesafe.close()
    first = typesafe._shared_client()
    assert typesafe._shared_client() is first
    typesafe.close()
    assert typesafe._shared_client() is not first


def test_closing_twice_is_harmless():
    typesafe.close()
    typesafe.close()


def test_a_caller_supplied_client_is_used_instead():
    """Tests pass their own, and so could a caller wanting its own pool."""
    typesafe.close()
    c = client(answered(q=NOUL))
    typesafe.ask("x", {"q": typesafe.noul("?")}, http=c)
    assert len(c.seen) == 1
    assert typesafe._CLIENT is None, "the shared one was never created"
