# Multi-stage: wheels get built in a throwaway stage so the shipped image has no
# compilers and nothing from pip's cache. Smaller image, smaller attack surface.

FROM python:3.12-slim AS build

WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-service.txt constraints.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
      -c constraints.txt \
      -r requirements.txt -r requirements-service.txt


FROM python:3.12-slim AS runtime

# Never run as root. If something gets in, it lands as a user that owns nothing.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY --from=build /wheels /wheels
COPY requirements.txt requirements-service.txt constraints.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
      -c constraints.txt \
      -r requirements.txt -r requirements-service.txt \
 && rm -rf /wheels

COPY --chown=app:app services/ ./services/
COPY --chown=app:app agents/ ./agents/

# Writable home for the booking store. Week 3 moves that to Table Storage, at
# which point the filesystem can be read-only.
RUN mkdir -p /app/data && chown app:app /app/data

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    BOOKING_STORE=/app/data/bookings.json \
    TICKET_STORE=/app/data/tickets.json

EXPOSE 8000

# Hits /health (liveness), never /ready. A readiness failure means "do not send
# me traffic"; using it here would restart the container over a brief Azure blip
# and turn a small problem into an outage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

# One worker: each holds its own Azure credential and token cache, and the work
# is IO-bound waiting on Azure rather than CPU-bound. Scale with more replicas,
# not more workers in one container.
CMD ["uvicorn", "services.orchestrator.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
