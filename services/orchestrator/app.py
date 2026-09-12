"""FastAPI service in front of the agents.

Endpoints:
    GET  /            the chat page
    POST /chat        {"message": "..."} -> {"reply": "...", "trace": [...]}
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

import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager

from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import router as routing
from . import telemetry

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
        client = AgentsClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
        STATE["client"] = client
        STATE["agent_ids"] = {a.name: a.id for a in client.list_agents()}
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


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


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
    return {"status": "ok", "uptime_s": int(time.time() - STATE["started_at"])}


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
        )
    except Exception as e:  # noqa: BLE001
        METRICS["failures"] += 1
        log.exception("chat failed")
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}") from e

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


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.exists():
        return {"service": "autoassist", "hint": "POST /chat with {\"message\": \"...\"}"}
    return FileResponse(page)
