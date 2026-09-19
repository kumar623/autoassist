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


def test_throttling_is_retried_and_retry_after_is_honoured(no_waiting):
    c = client(httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json={"ok": True}))
    assert azure_http.request(c, "POST", "https://x/runs", json={})["ok"]
    assert len(c.seen) == 2
    assert no_waiting == [7.0]


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_short_azure_outages_are_retried(status):
    c = client(httpx.Response(status), httpx.Response(200, json={"ok": True}))
    assert azure_http.request(c, "GET", "https://x/run")["ok"]


def test_backoff_doubles_without_retry_after(no_waiting):
    c = client(httpx.Response(503), httpx.Response(503), httpx.Response(503), httpx.Response(200, json={}))
    azure_http.request(c, "GET", "https://x/run")
    assert no_waiting == [0.5, 1.0, 2.0]


def test_a_huge_retry_after_is_capped(no_waiting):
    c = client(httpx.Response(429, headers={"retry-after": "3600"}), httpx.Response(200, json={}))
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
    with pytest.raises(httpx.ConnectError):
        azure_http.request(c, "GET", "https://x/run")


def test_errors_do_not_include_the_query_string():
    c = client(httpx.Response(404, text="not here"))
    with pytest.raises(azure_http.AzureError) as e:
        azure_http.request(c, "GET", "https://x/threads/abc?api-version=v1&secret=1")
    assert "secret" not in str(e.value)


def test_every_client_has_a_timeout():
    t = azure_http.new_client().timeout
    assert t.read and t.connect, "a call with no timeout is how a request hangs forever"
