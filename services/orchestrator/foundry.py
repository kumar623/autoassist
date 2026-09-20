"""Azure AI Foundry Agent Service, over plain HTTPS.

Replaces the azure-ai-agents SDK in the running service. It needs nine
operations, and each one here sends what the SDK sent - recorded from the SDK's
own traffic on 19 Sep 2026 against the live project, then reproduced:

    GET    /assistants                                  list agents (paged)
    POST   /threads                                     {}
    POST   /threads/{thread}/messages                   {"role": "user", "content": ...}
    POST   /threads/{thread}/runs                       {"assistant_id": ...}
    GET    /threads/{thread}/runs/{run}
    POST   /threads/{thread}/runs/{run}/submit_tool_outputs   {"tool_outputs": [...]}
    POST   /threads/{thread}/runs/{run}/cancel
    GET    /threads/{thread}/messages                   newest first
    DELETE /threads/{thread}

Every path is under PROJECT_ENDPOINT and carries api-version=v1, the version
the SDK used. Responses are plain dicts in the shapes documented in
services/orchestrator/runner.py.

Sign-in stays with azure-identity: DefaultAzureCredential gives a token from
`az login` on a laptop and from the managed identity in Azure. Writing that by
hand is security code with nothing to gain.
"""

from __future__ import annotations

import json
import os
import threading
import time

import httpx

from . import azure_http

API_VERSION = os.getenv("FOUNDRY_API_VERSION", "v1")

# The token audience the SDK asked for.
SCOPE = "https://ai.azure.com/.default"

# Refresh a token this long before it expires, so a request never goes out
# with one that lapses mid-flight.
TOKEN_REFRESH_SECONDS = 300


class FoundryAgents:
    """The agent operations the orchestrator uses. Safe to share between threads."""

    def __init__(self, endpoint: str, credential, http_client: httpx.Client | None = None):
        self.endpoint = endpoint.rstrip("/")
        self._credential = credential
        self._http = http_client or azure_http.new_client()
        self._token = None
        self._token_lock = threading.Lock()

    # ------------------------------------------------------------ plumbing

    def _auth(self) -> dict:
        with self._token_lock:
            if self._token is None or self._token.expires_on - TOKEN_REFRESH_SECONDS < time.time():
                self._token = self._credential.get_token(SCOPE)
            return {"Authorization": f"Bearer {self._token.token}"}

    def _call(self, method: str, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        return azure_http.request(
            self._http,
            method,
            f"{self.endpoint}{path}",
            headers=self._auth(),
            params={"api-version": API_VERSION, **(params or {})},
            json=json,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FoundryAgents:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ agents

    def list_agents(self) -> list[dict]:
        """Every agent in the project, following pages until there are no more."""
        agents: list[dict] = []
        params: dict = {"limit": 100}
        while True:
            page = self._call("GET", "/assistants", params=params)
            agents.extend(page.get("data") or [])
            if not page.get("has_more") or not page.get("last_id"):
                return agents
            params = {"limit": 100, "after": page["last_id"]}

    # ------------------------------------------------------------ threads and runs

    def create_thread(self) -> dict:
        return self._call("POST", "/threads", json={})

    def delete_thread(self, thread_id: str) -> dict:
        return self._call("DELETE", f"/threads/{thread_id}")

    def create_message(self, thread_id: str, content: str, role: str = "user") -> dict:
        return self._call("POST", f"/threads/{thread_id}/messages", json={"role": role, "content": content})

    def list_messages(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Newest first, which is also the API's default."""
        page = self._call("GET", f"/threads/{thread_id}/messages", params={"order": "desc", "limit": limit})
        return page.get("data") or []

    def create_run(self, thread_id: str, agent_id: str) -> dict:
        return self._call("POST", f"/threads/{thread_id}/runs", json={"assistant_id": agent_id})

    def create_thread_and_run(self, agent_id: str, content: str, role: str = "user") -> dict:
        """Thread, message and run in one request instead of three.

        Three round trips to South India cost roughly 0.5s of every reply, and
        the thread is thrown away after the turn anyway. The run comes back
        carrying its thread_id.
        """
        return self._call("POST", "/threads/runs", json={
            "assistant_id": agent_id,
            "thread": {"messages": [{"role": role, "content": content}]},
        })

    def get_run(self, thread_id: str, run_id: str) -> dict:
        return self._call("GET", f"/threads/{thread_id}/runs/{run_id}")

    def submit_tool_outputs(self, thread_id: str, run_id: str, outputs: list[dict]) -> dict:
        """outputs: [{"tool_call_id": ..., "output": "..."}]"""
        return self._call(
            "POST", f"/threads/{thread_id}/runs/{run_id}/submit_tool_outputs", json={"tool_outputs": outputs}
        )

    def cancel_run(self, thread_id: str, run_id: str) -> dict:
        return self._call("POST", f"/threads/{thread_id}/runs/{run_id}/cancel")

    # ------------------------------------------------------------ streaming

    def stream_thread_and_run(self, agent_id: str, content: str):
        """Start a run and yield (event, data) as Azure sends them.

        The customer waits 5-8s for a specialist to write an answer they cannot
        see until it is finished (measured 20 Sep). Streaming shows the words as
        they arrive - the first of them in about 2s.
        """
        return self._stream(
            "/threads/runs",
            {"assistant_id": agent_id, "stream": True,
             "thread": {"messages": [{"role": "user", "content": content}]}},
        )

    def stream_tool_outputs(self, thread_id: str, run_id: str, outputs: list[dict]):
        """Hand back tool results and keep streaming the same run."""
        return self._stream(
            f"/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
            {"tool_outputs": outputs, "stream": True},
        )

    def _stream(self, path: str, body: dict):
        """Server-sent events from one POST, as (event name, parsed data).

        No retries: a half-written answer cannot be replayed, and the caller
        already has a timeout. A rejected token is refreshed once, before the
        stream starts.
        """
        url = f"{self.endpoint}{path}"
        params = {"api-version": API_VERSION}
        for attempt in (1, 2):
            headers = {**self._auth(), "Accept": "text/event-stream", "Content-Type": "application/json"}
            with self._http.stream("POST", url, params=params, headers=headers, json=body) as r:
                if r.status_code == 401 and attempt == 1:
                    self._token = None  # force a new one, then try again
                    r.close()
                    continue
                if r.status_code >= 400:
                    r.read()
                    raise azure_http.AzureError(r.status_code, r.text[:300], "POST", url)
                event = ""
                for line in r.iter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        raw = line.split(":", 1)[1].strip()
                        if raw == "[DONE]":
                            return
                        try:
                            yield event, json.loads(raw)
                        except ValueError:
                            continue
                return
