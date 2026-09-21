"""FastAPI service in front of the agents.

Endpoints:
    GET  /            the chat page
    POST /chat        {"message": "...", "history": [...]} -> {"reply": "...", "trace": [...]}
                      optionally "triage": "auto"|"jev"|"agent" and "compare": true
    POST /chat/stream same, but the answer arrives as it is written (SSE)
    GET  /agents      the four agents, their tools, and where each tool call goes
    GET  /library     every document the assistant can cite, from the index
    GET  /health      liveness  - is the process up?
    GET  /ready       readiness - can it actually serve? (checks Azure)
    GET  /metrics     this replica's counters since it started; the full picture,
                      per request, is in Application Insights (telemetry.py)

Health vs ready matters in Kubernetes and Container Apps. Liveness failing gets
the container restarted. Readiness failing takes it out of the load balancer but
leaves it running. Wiring a dependency check into liveness means a brief Azure
blip restarts every container at once, which turns a small outage into a large
one.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import jev_triage, library, limits, roster, telemetry, typesafe
from . import router as routing
from .foundry import FoundryAgents

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# The Azure SDK logs every request and every response header at INFO. That is
# perhaps 40 lines per agent turn, which buries our own logging completely and
# would cost real money in Log Analytics ingestion. Turn it down; set
# AZURE_LOG_LEVEL=INFO in .env when you actually need to see the HTTP traffic.
_azure_level = os.getenv("AZURE_LOG_LEVEL", "WARNING").upper()
for noisy in (
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.identity._credentials.chained",
    "azure.identity._credentials.environment",
    "azure.identity._credentials.managed_identity",
    "httpx",
    "openai",
    # The exporter logs every batch it ships at INFO. Useful when telemetry
    # itself is misbehaving, noise the rest of the time.
    "azure.monitor.opentelemetry",
    "azure.monitor.opentelemetry.exporter",
    "opentelemetry",
):
    logging.getLogger(noisy).setLevel(_azure_level)

log = logging.getLogger("autoassist")

STATIC = pathlib.Path(__file__).resolve().parent / "static"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))
MAX_MESSAGE_CHARS = 2000

# Set by the deploy pipeline to the git SHA it built. Reported by /health so a
# smoke test can prove the NEW revision is answering, not the old one still
# holding traffic while the new one crash-loops quietly behind it.
BUILD_SHA = os.getenv("GIT_SHA", "dev")

STATE: dict = {"client": None, "agent_ids": {}, "started_at": time.time()}
METRICS = {
    "requests": 0,
    "failures": 0,
    "safety_flagged": 0,
    "triage_parse_failures": 0,
    "total_tokens": 0,
    "total_ms": 0,
    "refused": 0,       # rate limited or too many at once, never reached an agent
    "throttled": 0,     # Azure had no quota left
    "cache_hits": 0,    # answered from an identical earlier question
    # The page's triage toggle: how often a visitor picked a classifier rather
    # than taking the server's default, and how often they asked to see both.
    "triage_choice_jev": 0,
    "triage_choice_agent": 0,
    "triage_compare": 0,
}

# /chat and /chat/stream are public. LIMITS is what stops one visitor - or one
# curl loop - from being the whole load and spending the month's quota in an
# afternoon. Nothing else is limited: the deploy smoke test polls /health every
# five seconds, and /library is a static read.
LIMITS = limits.Limits()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the client once at startup, not per request.

    DefaultAzureCredential fetches and caches a token; creating one per request
    means a token exchange on every call and a lot of wasted latency.
    """
    STATE["telemetry"] = telemetry.setup()

    try:
        client = FoundryAgents(os.environ["PROJECT_ENDPOINT"], DefaultAzureCredential())
        STATE["client"] = client
        STATE["agent_ids"] = {a["name"]: a["id"] for a in client.list_agents()}
        log.info("ready, agents: %s", list(STATE["agent_ids"]))
    except Exception as e:  # noqa: BLE001
        # Start anyway. /ready will report unhealthy, /health stays up, and the
        # logs say why - better than a crash loop that hides the reason.
        log.exception("could not reach Azure at startup: %s", e)
        STATE["startup_error"] = str(e)

    yield

    if STATE.get("client"):
        STATE["client"].close()
    typesafe.close()


app = FastAPI(title="AutoAssist", version="0.2.0", lifespan=lifespan)


class HistoryTurn(BaseModel):
    role: Literal["customer", "assistant"]
    text: str = Field(..., max_length=MAX_MESSAGE_CHARS * 4)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    # The last few turns, oldest first, kept by the page. Optional, so a plain
    # {"message": ...} call behaves exactly as it always did. The router uses
    # only the most recent routing.MAX_HISTORY_TURNS; the cap here just keeps
    # request bodies bounded.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    # Which classifier reads this message. "auto" is whatever TRIAGE_BACKEND
    # says, and is what every existing caller gets without asking; "jev" and
    # "agent" are the page's toggle. See router.plan_triage.
    triage: Literal["auto", "jev", "agent"] = "auto"
    # Also run the other classifier on the same message and report what it made
    # of it. Display only: the route, the reply and the tickets do not change.
    compare: bool = False


class ChatResponse(BaseModel):
    reply: str
    agents: list[str]
    searched: bool
    safety: bool
    tokens: int
    ms: int
    trace: list[dict]
    # An identical question had already been answered, so this one cost nothing.
    cached: bool = False
    # Azure had no token quota left. The reply says so; this lets a caller tell
    # that apart from an ordinary answer without reading the text.
    throttled: bool = False
    retry_after: int = 0
    # The triage choice as asked for and as it happened: {"requested", "used",
    # "why", "reading"}. "used" is "none" when no classifier read the message.
    triage: Optional[dict] = None
    # The other classifier's reading, when `compare` was asked for:
    # {"chosen", "other", "differences"}.
    comparison: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    """Liveness. Deliberately checks nothing external."""
    return {
        "status": "ok",
        "build": BUILD_SHA,
        "uptime_s": int(time.time() - STATE["started_at"]),
    }


@app.get("/ready")
def ready() -> dict:
    """Readiness. Can this instance actually serve a request?"""
    if STATE.get("client") is None:
        raise HTTPException(503, detail=f"no Azure client: {STATE.get('startup_error', 'unknown')}")

    missing = [n for n in routing.AGENTS if n not in STATE["agent_ids"]]
    if missing:
        raise HTTPException(503, detail=f"agents not deployed: {', '.join(missing)}")

    return {
        "status": "ready",
        "agents": sorted(STATE["agent_ids"]),
        "telemetry": bool(STATE.get("telemetry")),
    }


@app.get("/metrics")
def metrics() -> dict:
    m = dict(METRICS)
    n = m["requests"]
    m["avg_ms"] = round(m["total_ms"] / n, 1) if n else 0
    m["avg_tokens"] = round(m["total_tokens"] / n, 1) if n else 0
    m.update(LIMITS.snapshot())
    # Which classifier is live, and whether Jev is actually answering or quietly
    # handing everything back to the agent.
    m["triage_backend"] = roster.triage_backend()
    m["jev_answered"] = jev_triage.STATS["answered"]
    m["jev_fell_back"] = jev_triage.STATS["fell_back"]
    # Spent on the compared classifier, which answers nobody. Not in
    # total_tokens - that is what the answers cost - but not hidden either. One
    # figure per classifier, because their tokens are priced about ten times
    # apart, and counted as each comparison finishes, so one that outlived its
    # answer is in it too.
    m["compare_tokens_jev"] = routing.COMPARE_SPENT["jev"]
    m["compare_tokens_agent"] = routing.COMPARE_SPENT["agent"]
    return m


def _admit(request: Request) -> None:
    """Decide whether to handle this message at all. Every caller must release().

    Refusing is not an error: the reply says what happened in words, and
    Retry-After says when to come back. Counted separately from failures, which
    are ours.
    """
    key = limits.visitor_key(request.headers, request.client.host if request.client else None)
    try:
        LIMITS.admit(key)
    except limits.Refused as e:
        METRICS["refused"] += 1
        # A refused message never reaches routing.handle, so it opens no
        # chat.request span - and without this one a script being turned away
        # five thousand times an hour looks in App Insights exactly like a quiet
        # afternoon: tokens flat, latency unchanged, nothing failing, fewer
        # requests. The visitor is recorded as a short digest, so two refusals
        # can be tied together without this service writing addresses into Log
        # Analytics.
        with telemetry.span("chat.refused") as s:
            telemetry.set(s, reason=e.reason, retry_after=e.retry_after,
                          visitor=limits.digest(key), in_flight=LIMITS.in_flight.count)
        raise HTTPException(
            e.status, detail=e.customer_message, headers={"Retry-After": str(e.retry_after)}
        ) from None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    client = STATE.get("client")
    if client is None:
        raise HTTPException(503, detail="service is not connected to Azure")

    _admit(request)
    METRICS["requests"] += 1
    _count_choice(req)
    started = time.time()

    try:
        result = routing.handle(
            client,
            req.message.strip(),
            timeout=REQUEST_TIMEOUT,
            agent_ids=STATE["agent_ids"],
            history=[h.model_dump() for h in req.history],
            triage=req.triage,
            compare=req.compare,
        )
    except Exception as e:  # noqa: BLE001
        METRICS["failures"] += 1
        log.exception("chat failed")
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}") from e
    finally:
        LIMITS.release()

    return _chat_response(result, started)


def _count_choice(req: ChatRequest) -> None:
    """Count the page's triage toggle, for /metrics. Used by both endpoints.

    Counted on the request, not on what happened: a visitor who picks Jev on a
    server with no key has still picked Jev, and the reply says what answered.
    """
    if req.triage != "auto":
        METRICS[f"triage_choice_{req.triage}"] += 1
    if req.compare:
        METRICS["triage_compare"] += 1


def _chat_response(result, started: float) -> ChatResponse:
    """Count one handled message and shape the reply. Used by both endpoints."""
    decision = result.decision
    METRICS["total_tokens"] += result.total_tokens
    METRICS["total_ms"] += int((time.time() - started) * 1000)
    if decision and decision.safety:
        METRICS["safety_flagged"] += 1
    if decision and decision.parse_failed:
        METRICS["triage_parse_failures"] += 1
    if result.error:
        METRICS["failures"] += 1
    if result.cached:
        METRICS["cache_hits"] += 1
    if result.throttled:
        # Not a failure. Nothing broke; there was no quota left. The runbook's
        # throttling alert reads this, and lumping it in with failures would hide
        # the one number that says "ask Azure for more quota".
        METRICS["throttled"] += 1

    return ChatResponse(
        reply=result.reply,
        agents=result.agents_used,
        searched=result.searched,
        safety=bool(decision and decision.safety),
        tokens=result.total_tokens,
        ms=result.duration_ms,
        trace=result.trace(),
        cached=result.cached,
        throttled=result.throttled,
        retry_after=result.retry_after,
        triage=result.triage,
        comparison=result.comparison,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """The same answer as /chat, sent as it is written.

    A specialist takes 5-8s to write; waiting for all of it before showing
    anything is most of what makes the app feel slow. The events:

        {"type": "status", "text": "..."}  what is happening while they wait
        {"type": "activity", "kind": ...}  for the agent panel: route, agent, tool,
                                           and compare when both classifiers were asked
        {"type": "delta", "text": "..."}   a fragment of the answer
        {"type": "done",  ...}             the whole ChatResponse, with the trace
        {"type": "error", "detail": "..."}

    Safety-flagged messages are not streamed - see router._can_stream - so the
    reply still arrives in one piece there, after the checks have run.
    """
    client = STATE.get("client")
    if client is None:
        raise HTTPException(503, detail="service is not connected to Azure")

    # Admitted here so a refusal is an HTTP status the page can read, before any
    # of the body has been written. The slot is held until the generator below
    # finishes, which is long after this function has returned, so it is
    # released there rather than here.
    _admit(request)
    METRICS["requests"] += 1
    _count_choice(req)
    started = time.time()
    fragments: queue.Queue = queue.Queue()
    DONE = object()

    def work():
        try:
            result = routing.handle(
                client, req.message.strip(), timeout=REQUEST_TIMEOUT, agent_ids=STATE["agent_ids"],
                history=[h.model_dump() for h in req.history], triage=req.triage, compare=req.compare,
                on_delta=fragments.put, on_status=lambda text: fragments.put(("status", text)),
                # Called from inside the agent threads, several at once when the
                # specialists run in parallel. Queue.put is what makes that safe.
                on_event=lambda event: fragments.put(("activity", event)),
            )
            fragments.put(("done", result))
        except Exception as e:  # noqa: BLE001
            log.exception("streamed chat failed")
            fragments.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            fragments.put(DONE)

    threading.Thread(target=work, name="chat-stream", daemon=True).start()

    def events():
        streamed = ""
        try:
            while True:
                item = fragments.get()
                if item is DONE:
                    return
                if isinstance(item, str):  # a fragment of the answer
                    streamed += item
                    yield _sse({"type": "delta", "text": item})
                    continue
                kind, payload = item
                if kind == "status":
                    yield _sse({"type": "status", "text": payload})
                    continue
                if kind == "activity":
                    # What the panel beside the chat draws: which agent is
                    # working, which tool it just called, how long it took.
                    yield _sse({"type": "activity", **payload})
                    continue
                if kind == "error":
                    METRICS["failures"] += 1
                    yield _sse({"type": "error", "detail": payload})
                    continue
                body = _chat_response(payload, started)
                # The customer has already read the streamed text; sending it again
                # would duplicate it, so `reply` is only included when nothing was
                # streamed (several agents, or a safety answer held back for checks).
                yield _sse({"type": "done", **body.model_dump(),
                            "reply": "" if streamed.strip() else body.reply})
        finally:
            # However this ends - finished, failed, or the customer closing the
            # tab part-way - the slot goes back. A leaked slot is permanent: the
            # replica would answer one fewer message for the rest of its life.
            # The worker thread may still be running when a customer leaves; it
            # is not cancellable, but it holds no slot of its own.
            LIMITS.release()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/agents")
def agent_roster() -> dict:
    """The four agents, what each is for, and which tools each may call.

    Read from the definitions the deploy script uses, so the panel beside the
    chat cannot drift from what is actually deployed.
    """
    return roster.load(STATE.get("agent_ids"))


@app.get("/library")
def document_library() -> dict:
    """Every document the assistant can cite, for the page's Library tab."""
    try:
        return library.load()
    except Exception as e:  # noqa: BLE001
        log.exception("could not read the document library")
        raise HTTPException(503, detail=f"could not read the document library: {type(e).__name__}") from e


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.exists():
        return {"service": "autoassist", "hint": "POST /chat with {\"message\": \"...\"}"}
    return FileResponse(page)
