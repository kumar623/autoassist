"""Ask TypeSafe's Jev a typed question about some text.

Jev is a "System One" model: it does not write prose. You give it a state and a
set of named questions, and it returns a typed answer per question - a yes/no as
a probability ("noul"), a choice with a probability per option, or a score. Every
question in one request is evaluated against the state in parallel.

That shape fits exactly one part of this service: triage, which reads a message
and returns JSON saying who should handle it and whether it looks like a safety
issue. Nothing here is wired into the request path. It exists so that
evals/compare_triage.py can measure Jev against the triage agent we actually run,
the same way gpt-4.1-nano was measured and rejected (docs/evaluation.md).

No vendor SDK, for the reasons in docs/decisions/007: it is one JSON POST, and
`typesafe-sdk` needs Python 3.10+ while the venv these evals run in is 3.9 -
the same constraint that kept the MCP SDK out (mcp_client.py).

THE API KEY IS A SECRET. It travels in the Authorization header, so unlike
Zoho's MCP URL it is not part of any address - but it must still never reach a
log line or an exception. Errors here name the host and the status, nothing else.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import httpx

from . import azure_http

log = logging.getLogger(__name__)

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# 529 is TypeSafe's "we are overloaded, come back shortly". It is not in
# azure_http.RETRY_STATUSES because Azure does not use it, and it is exactly the
# kind of thing worth retrying.
OVERLOADED = 529

# Throttling is answered the same way as in azure_http: one quick retry if the
# server says the wait is short, and otherwise stop rather than queue behind an
# exhausted quota. Jev's published limits (1,200 requests a minute) are far above
# anything this project sends, so a 429 here almost certainly means something is
# wrong with the account rather than with the traffic.
THROTTLE_PATIENCE_SECONDS = 2.0


# One client for the process, so the TCP and TLS handshake is paid once.
#
# Measured from Vizag on 21 September: a fresh connection per call is 1,115ms
# median, a reused one 391ms. Of the ~725ms saved, curl's own breakdown puts
# ~260ms in the TCP connect and ~390ms more in the TLS handshake - the endpoint
# is in the US, and every round trip crosses an ocean. The model's own work is
# only about 340ms of it.
#
# This is why published latencies for Jev (70-500ms) looked unreachable at
# first: the first measurement here opened a new connection for every request.
_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()


def _shared_client() -> httpx.Client:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = azure_http.new_client()
        return _CLIENT


def close() -> None:
    """Drop the shared connection. For tests, and for a clean shutdown."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None


class TypeSafeUnavailable(Exception):
    """Jev could not be asked. The caller decides what to do without it.

    Its own type, like ZohoUnavailable, so that httpx's exceptions - which carry
    URLs and sometimes headers - never escape this module.
    """


def configured() -> bool:
    return bool(os.getenv("TYPESAFE_API_KEY", "").strip())


def model() -> str:
    return os.getenv("TYPESAFE_MODEL", "").strip() or DEFAULT_MODEL


def noul(instructions: str, true_means: str = "", false_means: str = "") -> dict:
    """A yes/no question. The answer comes back as a probability, not a boolean.

    `criteria` is optional in the API but worth writing: it is where "what counts
    as yes" is said, and for the safety question that wording is lifted from the
    triage agent's own prompt so both are asked the same thing.
    """
    question: dict = {"type": "noul", "instructions": instructions}
    if true_means or false_means:
        question["criteria"] = {"true": true_means, "false": false_means}
    return question


def choice(instructions: str, options: dict) -> dict:
    """Pick one of `options` - a map of option name to what that option means."""
    return {"type": "choice", "instructions": instructions, "criteria": options}


def ask(state, questions: dict, http: httpx.Client | None = None, timeout: float | None = None) -> dict:
    """Put `questions` to Jev about `state`. Returns {"answers": ..., "usage": ...}.

    `state` may be a string or any JSON-serialisable structure; the API takes
    both. Raises TypeSafeUnavailable for anything that stops an answer coming
    back, with no key and no URL in the message.
    """
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise TypeSafeUnavailable("TYPESAFE_API_KEY is not set")
    if not questions:
        raise TypeSafeUnavailable("no questions to ask")

    client = http or _shared_client()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"state": state, "model": model(), "questions": questions}
    host = httpx.URL(ENDPOINT).host

    throttled = False
    for attempt in range(azure_http.MAX_RETRIES + 1):
        try:
            r = client.post(ENDPOINT, headers=headers, content=json.dumps(body),
                            timeout=timeout or azure_http.TIMEOUT)
        except httpx.TransportError as e:
            if attempt == azure_http.MAX_RETRIES:
                # `from None`: httpx's exception repeats the URL, and one day
                # that URL may carry something worth not repeating.
                raise TypeSafeUnavailable(f"could not reach {host}: {type(e).__name__}") from None
            azure_http._pause(azure_http._backoff(attempt, None))
            continue

        if r.status_code == 429:
            wait = azure_http._backoff(0, r.headers.get("retry-after"))
            if not throttled and wait <= THROTTLE_PATIENCE_SECONDS:
                throttled = True
                azure_http._pause(wait)
                continue
            raise TypeSafeUnavailable(f"{host} is rate limiting this key (HTTP 429)")

        if r.status_code == OVERLOADED and attempt < azure_http.MAX_RETRIES:
            azure_http._pause(azure_http._backoff(attempt, r.headers.get("retry-after")))
            continue

        if r.status_code >= 400:
            raise TypeSafeUnavailable(f"{host} answered HTTP {r.status_code}: {_why(r)}")

        try:
            answer = r.json()
        except ValueError:
            raise TypeSafeUnavailable(f"{host} answered HTTP 200 with something that is not JSON") from None
        if not isinstance(answer, dict) or "answers" not in answer:
            raise TypeSafeUnavailable(f"{host} answered without an 'answers' map")
        return answer

    raise AssertionError("unreachable")  # the loop always returns or raises


def _why(r: httpx.Response) -> str:
    """The server's own explanation, trimmed. Never the request, which has the key."""
    try:
        body = r.json()
    except ValueError:
        return (r.text or r.reason_phrase)[:200]
    if isinstance(body, dict):
        for field in ("error", "message", "detail"):
            found = body.get(field)
            if isinstance(found, str):
                return found[:200]
            if isinstance(found, dict) and isinstance(found.get("message"), str):
                return found["message"][:200]
    return str(body)[:200]


# ------------------------------------------------------- reading the answers


def probability(answers: dict, name: str) -> float | None:
    """The yes-probability of a noul answer, or None if it is not there.

    None rather than 0.0 on purpose: "Jev did not answer" and "Jev said almost
    certainly not" are different, and a caller that treats them the same would
    read a missing safety answer as safe.
    """
    found = (answers or {}).get(name)
    if not isinstance(found, dict):
        return None
    value = found.get("noul")
    return float(value) if isinstance(value, (int, float)) else None


def chosen(answers: dict, name: str) -> tuple[str | None, dict]:
    """A choice answer as (option, probabilities)."""
    found = (answers or {}).get(name)
    if not isinstance(found, dict):
        return None, {}
    options = found.get("probabilities")
    return found.get("choice"), options if isinstance(options, dict) else {}
