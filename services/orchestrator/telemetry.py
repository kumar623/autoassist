"""OpenTelemetry tracing into Application Insights.

Set APPLICATIONINSIGHTS_CONNECTION_STRING and traces go to Azure. Leave it unset
and everything here becomes a no-op - the app runs exactly the same, just
without telemetry. That matters: tests, CI and local development must not need
an Azure resource, and a monitoring outage must never take the service down.

What we record, and why:

  span per request      -> end to end latency, which agents ran, total tokens
  span per agent turn   -> which agent is slow, which one fails
  span per tool call    -> whether it searched, how long retrieval took
  searched=true/false   -> the single most useful field in the whole system

That last one is worth explaining. Week 1's first bug was an agent answering a
safety question from its own training with no search call, in prose that looked
perfectly good. Reading the answer would never catch it. Filtering for
`searched == false` on diagnostics answers catches every instance instantly.
Instrument the thing that is hard to see, not the thing that is easy to read.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

log = logging.getLogger(__name__)

_ENABLED = False
_tracer = None


def setup() -> bool:
    """Wire up Azure Monitor if a connection string is present.

    Returns True if telemetry is on. Never raises - a monitoring problem must
    not stop the service starting.
    """
    global _ENABLED, _tracer

    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if not conn:
        log.info("telemetry off (APPLICATIONINSIGHTS_CONNECTION_STRING not set)")
        return False

    # Azure Monitor instruments only the libraries on its own list - azure_sdk,
    # django, fastapi, flask, psycopg2, requests, urllib and urllib3 in 1.6.13 -
    # and ignores any other name set here. The service uses FastAPI, and
    # azure-identity fetches its tokens through the Azure SDK and requests. Our
    # own calls go through httpx, which is not on the list at all (decision
    # 007), so our spans are what record them. django, flask and psycopg2 are
    # not installed: psycopg2 logs a stack trace on every start when it finds
    # that out, and the other two a debug line each.
    os.environ.setdefault("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "psycopg2,django,flask")

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace

        configure_azure_monitor(
            connection_string=conn,
            logger_name="autoassist",
            # Sample everything. At this volume it costs almost nothing, and a
            # sampled-out trace is exactly the one you wanted during an incident.
            # Revisit if traffic ever justifies it.
        )
        _tracer = trace.get_tracer("autoassist-orchestrator")
        _ENABLED = True
        log.info("telemetry on -> Application Insights")
        return True

    except ImportError:
        log.warning(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is set but azure-monitor-opentelemetry "
            "is not installed. Run: pip install -r requirements-service.txt"
        )
        return False
    except Exception as e:  # noqa: BLE001
        log.exception("telemetry setup failed, carrying on without it: %s", e)
        return False


@contextmanager
def span(name: str, **attributes):
    """Start a span, or do nothing if telemetry is off.

    Used as a context manager everywhere, so the call sites read the same
    whether or not Azure is configured:

        with telemetry.span("agent.turn", agent="diagnostics") as s:
            ...
            telemetry.set(s, tokens=1234)
    """
    if not _ENABLED or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(_key(k), _value(v))
        try:
            yield s
        except Exception as e:
            s.record_exception(e)
            s.set_attribute("error", True)
            raise


def set(s, **attributes) -> None:  # noqa: A001 - reads well at the call site
    """Add attributes to a span that may be None."""
    if s is None:
        return
    for k, v in attributes.items():
        if v is not None:
            try:
                s.set_attribute(_key(k), _value(v))
            except Exception:  # noqa: BLE001
                pass


def _key(k: str) -> str:
    """Namespace our attributes so they are easy to find in KQL.

    Everything lands under customDimensions; prefixing means you can write
    `where customDimensions.["autoassist.agent"] == "diagnostics"` without
    wondering whether some library set a field called `agent`.
    """
    return k if k.startswith("autoassist.") else f"autoassist.{k}"


def _value(v):
    """OpenTelemetry accepts str, bool, int, float and sequences of those."""
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return str(v)
