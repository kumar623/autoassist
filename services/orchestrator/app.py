"""FastAPI service in front of the agents.

Endpoints:
    GET  /            the chat page
    POST /chat        {"message": "...", "history": [...]} -> {"reply": "...", "trace": [...]}
    POST /chat/stream same, but the answer arrives as it is written (SSE)
    GET  /library     every document the assistant can cite, from the index
    GET  /health      liveness  - is the process up?
    GET  /ready       readiness - can it actually serve? (checks Azure)
    GET  /metrics     counters, in lieu of App Insights until week 3

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
from typing import Literal

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import library, telemetry
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
}


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


class ChatResponse(BaseModel):
    reply: str
    agents: list[str]
    searched: bool
    safety: bool
    tokens: int
    ms: int
    trace: list[dict]


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

    missing = [n for n in ("triage", "diagnostics", "booking", "escalation")
               if n not in STATE["agent_ids"]]
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
    m["avg_ms"] = round(m["total_ms"] / m["requests"], 1) if m["requests"] else 0
    m["avg_tokens"] = round(m["total_tokens"] / m["requests"], 1) if m["requests"] else 0
    return m


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    client = STATE.get("client")
    if client is None:
        raise HTTPException(503, detail="service is not connected to Azure")

    METRICS["requests"] += 1
    started = time.time()

    try:
        result = routing.handle(
            client,
            req.message.strip(),
            timeout=REQUEST_TIMEOUT,
            agent_ids=STATE["agent_ids"],
            history=[h.model_dump() for h in req.history],
        )
    except Exception as e:  # noqa: BLE001
        METRICS["failures"] += 1
        log.exception("chat failed")
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}") from e

    return _chat_response(result, started)


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

    return ChatResponse(
        reply=result.reply,
        agents=result.agents_used,
        searched=result.searched,
        safety=bool(decision and decision.safety),
        tokens=result.total_tokens,
        ms=result.duration_ms,
        trace=result.trace(),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """The same answer as /chat, sent as it is written.

    A specialist takes 5-8s to write; waiting for all of it before showing
    anything is most of what makes the app feel slow. The events:

        {"type": "status", "text": "..."}  what is happening while they wait
        {"type": "delta", "text": "..."}   a fragment of the answer
        {"type": "done",  ...}             the whole ChatResponse, with the trace
        {"type": "error", "detail": "..."}

    Safety-flagged messages are not streamed - see router._can_stream - so the
    reply still arrives in one piece there, after the checks have run.
    """
    client = STATE.get("client")
    if client is None:
        raise HTTPException(503, detail="service is not connected to Azure")

    METRICS["requests"] += 1
    started = time.time()
    fragments: queue.Queue = queue.Queue()
    DONE = object()

    def work():
        try:
            result = routing.handle(
                client, req.message.strip(), timeout=REQUEST_TIMEOUT, agent_ids=STATE["agent_ids"],
                history=[h.model_dump() for h in req.history],
                on_delta=fragments.put, on_status=lambda text: fragments.put(("status", text)),
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

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


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
