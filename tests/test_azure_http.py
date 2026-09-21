"""Tests for the plain-HTTPS layer: the retry, timeout and error rules the SDKs
used to supply. Azure is replaced by httpx.MockTransport - no network."""

import httpx
import pytest

from services.orchestrator import azure_http


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


def test_a_good_response_is_returned_as_json():
    c = client(httpx.Response(200, json={"ok": True}))
    assert azure_http.request(c, "GET", "https://x/thing") == {"ok": True}


def test_an_empty_body_is_an_empty_dict():
    c = client(httpx.Response(200))
    assert azure_http.request(c, "DELETE", "https://x/thing") == {}


def test_a_brief_burst_of_throttling_is_waited_out(no_waiting):
    """Azure saying 'one second' is worth a second. It clears bursts."""
    c = client(httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json={"ok": True}))
    assert azure_http.request(c, "POST", "https://x/runs", json={})["ok"]
    assert len(c.seen) == 2
    assert no_waiting == [1.0]


def test_a_long_wait_is_not_waited_out(no_waiting):
    """The quota is spent. Queueing behind it wastes the customer's time."""
    c = client(httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "quota"}}))
    with pytest.raises(azure_http.Throttled) as e:
        azure_http.request(c, "POST", "https://x/runs", json={})
    assert e.value.retry_after == 7.0
    assert len(c.seen) == 1, "no retry at all"
    assert no_waiting == []


def test_throttling_stops_after_one_quick_retry(no_waiting):
    """One short wait, then the truth - not three waits and then a timeout."""
    c = client(*[httpx.Response(429)] * 4)
    with pytest.raises(azure_http.Throttled):
        azure_http.request(c, "POST", "https://x/runs", json={})
    assert len(c.seen) == azure_http.THROTTLE_RETRIES + 1 == 2
    assert no_waiting == [0.5]


def test_throttling_is_its_own_kind_of_error():
    """Callers tell 'we are busy' from 'it broke'; Throttled is still an AzureError."""
    c = client(httpx.Response(429, headers={"retry-after": "30"}))
    with pytest.raises(azure_http.AzureError) as e:
        azure_http.request(c, "GET", "https://x/run")
    assert isinstance(e.value, azure_http.Throttled)
    assert e.value.status == 429


def test_advice_for_the_customer_is_not_half_a_second():
    """Azure OpenAI often omits Retry-After when a quota window is full."""
    c = client(httpx.Response(429), httpx.Response(429))
    with pytest.raises(azure_http.Throttled) as e:
        azure_http.request(c, "GET", "https://x/run")
    assert e.value.retry_after == 20.0


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_short_azure_outages_are_retried(status):
    c = client(httpx.Response(status), httpx.Response(200, json={"ok": True}))
    assert azure_http.request(c, "GET", "https://x/run")["ok"]


def test_backoff_doubles_without_retry_after(no_waiting):
    c = client(httpx.Response(503), httpx.Response(503), httpx.Response(503), httpx.Response(200, json={}))
    azure_http.request(c, "GET", "https://x/run")
    assert no_waiting == [0.5, 1.0, 2.0]


def test_a_huge_retry_after_is_capped(no_waiting):
    c = client(httpx.Response(503, headers={"retry-after": "3600"}), httpx.Response(200, json={}))
    azure_http.request(c, "GET", "https://x/run")
    assert no_waiting == [azure_http.MAX_WAIT_SECONDS]


def test_it_gives_up_after_max_retries_with_azures_message():
    failures = [httpx.Response(503, json={"error": {"code": "ServiceUnavailable", "message": "busy"}})]
    c = client(*failures * (azure_http.MAX_RETRIES + 1))
    with pytest.raises(azure_http.AzureError) as e:
        azure_http.request(c, "GET", "https://x/run?api-version=v1")
    assert e.value.status == 503
    assert "ServiceUnavailable: busy" in str(e.value)
    assert len(c.seen) == azure_http.MAX_RETRIES + 1


def test_a_client_error_is_not_retried():
    """A 400 or 401 will not fix itself; retrying only delays the real error."""
    c = client(httpx.Response(401, json={"error": {"code": "PermissionDenied", "message": "no role"}}))
    with pytest.raises(azure_http.AzureError) as e:
        azure_http.request(c, "GET", "https://x/assistants")
    assert e.value.status == 401
    assert "PermissionDenied" in e.value.message
    assert len(c.seen) == 1


def test_a_dropped_connection_is_retried():
    c = client(httpx.ConnectError("reset"), httpx.Response(200, json={"ok": True}))
    assert azure_http.request(c, "GET", "https://x/run")["ok"]


def test_a_connection_that_keeps_failing_raises():
    c = client(*[httpx.ConnectError("down")] * (azure_http.MAX_RETRIES + 1))
    with pytest.raises(azure_http.Unreachable, match="GET https://x/run -> no answer: ConnectError"):
        azure_http.request(c, "GET", "https://x/run")
    assert len(c.seen) == azure_http.MAX_RETRIES + 1


def test_a_request_httpx_will_not_send_is_not_retried_and_quotes_nothing():
    """21 September: a key pasted with page text around it. httpx refused the
    header and said so in full - LocalProtocolError quotes the value, and the
    value was the api-key header. With a real key and a stray space, that is
    the key in Log Analytics and in the tool output the model reads."""
    refusal = httpx.LocalProtocolError("Illegal header value b' s3cr3tkey0123456789 '")
    c = client(refusal)
    with pytest.raises(azure_http.Unreachable) as e:
        azure_http.request(c, "POST", "https://x/openai/deployments/emb/embeddings?api-version=1",
                           headers={"api-key": "irrelevant"})
    assert len(c.seen) == 1, "the same malformed request fails the same way every time"
    assert "s3cr3tkey" not in str(e.value)
    assert "LocalProtocolError" in str(e.value)
    # A log.exception further up prints the chain; there must not be one.
    assert e.value.__cause__ is None and e.value.__suppress_context__


def test_a_short_throttle_on_the_last_attempt_is_reported_not_crashed():
    """Three outages then a brief 429 used to 'retry' past the end of the loop
    and raise AssertionError('unreachable') - a 500 where the customer should
    have been told the service is busy."""
    c = client(*[httpx.Response(503)] * azure_http.MAX_RETRIES,
               httpx.Response(429, headers={"retry-after": "1"}))
    with pytest.raises(azure_http.Throttled):
        azure_http.request(c, "GET", "https://x/run")


def test_errors_do_not_include_the_query_string():
    c = client(httpx.Response(404, text="not here"))
    with pytest.raises(azure_http.AzureError) as e:
        azure_http.request(c, "GET", "https://x/threads/abc?api-version=v1&secret=1")
    assert "secret" not in str(e.value)


def test_every_client_has_a_timeout():
    t = azure_http.new_client().timeout
    assert t.read and t.connect, "a call with no timeout is how a request hangs forever"
