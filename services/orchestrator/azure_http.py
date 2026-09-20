"""Plain HTTPS calls to Azure, with the rules the SDKs used to supply.

The service calls three Azure APIs - Foundry agents, AI Search and Azure OpenAI
embeddings - directly, without their SDKs (docs/decisions/007). The SDKs did
three things for free that every call still needs, so they live here, once:

  - Retries on throttling and short Azure outages: 408, 429, 500, 502, 503, 504
    and dropped connections, the same set azure-core retried. Azure's
    Retry-After header is honoured, otherwise backoff doubles from 0.5s.
  - A timeout on every call. A call with no timeout is how a request hangs
    forever (docs/evaluation.md, the side note under finding 5).
  - Error responses become an exception carrying Azure's own message, rather
    than a bare status code.

Like azure-core, this retries writes too. A 5xx or dropped connection on
"create run" could in principle mean the run was created and the reply lost; a
duplicate run in a throwaway thread costs a few tokens, while not retrying
fails the customer's request.
"""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 20.0

# Throttling is not an outage and is not retried like one. A 500 means something
# broke and trying again usually works; a 429 from Azure OpenAI means the minute's
# token quota is spent, and every retry queues behind the same exhausted quota.
#
# The old rule treated them the same: up to three waits of up to 20s, on each of
# the several calls one customer message makes. The customer waited out the 90s
# request timeout and was then told "Sorry - I could not get an answer", which is
# true but says nothing. Now: one quick retry if Azure says the wait is short - a
# burst clears in a second - and otherwise stop at once and say we are busy.
THROTTLE_RETRIES = 1
THROTTLE_PATIENCE_SECONDS = 2.0

# Connecting should be quick; a response can take a while (a search over the
# index, a run being created). The run loop has its own overall limit.
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# Indirection so tests can skip the waiting.
_pause = time.sleep


class AzureError(Exception):
    """Azure answered with an error that retrying will not fix."""

    def __init__(self, status: int, message: str, method: str, url: str):
        super().__init__(f"{method} {_path(url)} -> HTTP {status}: {message}")
        self.status = status
        self.message = message


class Throttled(AzureError):
    """Azure refused because the quota is spent, not because anything is broken.

    Its own class because the answer is different: nothing is wrong, waiting will
    not help within this request, and the customer deserves to be told that the
    service is busy rather than that it failed. `retry_after` is Azure's own
    advice where it gave any, in seconds.
    """

    def __init__(self, status: int, message: str, method: str, url: str, retry_after: float):
        super().__init__(status, message, method, url)
        self.retry_after = retry_after


def new_client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT)


def request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    json: dict | None = None,
) -> dict:
    """Send one request, retrying what is worth retrying. Returns parsed JSON."""
    throttles = 0
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = client.request(method, url, headers=headers, params=params, json=json)
        except httpx.TransportError as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning("%s %s failed (%s), retrying", method, _path(url), type(e).__name__)
            _pause(_backoff(attempt, None))
            continue

        if r.status_code == 429:
            advised = r.headers.get("retry-after")
            wait = _backoff(throttles, advised)
            if throttles < THROTTLE_RETRIES and wait <= THROTTLE_PATIENCE_SECONDS:
                throttles += 1
                log.warning("%s %s -> throttled, waiting %.1fs", method, _path(url), wait)
                _pause(wait)
                continue
            log.warning("%s %s -> throttled, giving up (Azure advised %ss)", method, _path(url), advised or "nothing")
            raise Throttled(429, _message(r), method, url, retry_after=advised_wait(advised))

        if r.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
            log.warning("%s %s -> HTTP %d, retrying", method, _path(url), r.status_code)
            _pause(_backoff(attempt, r.headers.get("retry-after")))
            continue

        if r.status_code >= 400:
            raise AzureError(r.status_code, _message(r), method, url)

        return r.json() if r.content else {}

    raise AssertionError("unreachable")  # the loop always returns or raises


def advised_wait(retry_after: str | None, default: float = 20.0) -> float:
    """How long to tell the customer to wait, from Azure's Retry-After header.

    Separate from _backoff on purpose. _backoff decides how long WE pause before
    trying again, and half a second is a fine answer to that. This decides what
    to tell a person, and "try again in half a second" is not advice. Azure OpenAI
    usually omits the header when a quota window is exhausted, so the default is
    the length of the window that is full.
    """
    try:
        wait = float(retry_after) if retry_after else default
    except ValueError:
        wait = default
    return min(max(wait, 1.0), 300.0)


def _backoff(attempt: int, retry_after: str | None) -> float:
    try:
        wait = float(retry_after) if retry_after else 0.5 * 2**attempt
    except ValueError:
        wait = 0.5 * 2**attempt
    return min(max(wait, 0.0), MAX_WAIT_SECONDS)


def _message(r: httpx.Response) -> str:
    """Azure's own explanation, from {"error": {"message": ...}} where there is one."""
    try:
        body = r.json()
    except ValueError:
        return r.text[:300] or r.reason_phrase
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return f"{err.get('code') or ''}: {err.get('message') or ''}".strip(": ")
    return str(body)[:300]


def _path(url: str) -> str:
    """The URL without its query string, for logs and errors. Keys never go in
    query strings here, but there is no reason to log the rest either."""
    return url.split("?", 1)[0]
