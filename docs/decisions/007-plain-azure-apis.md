# 007 - The service calls Azure's APIs directly, not through SDKs

## Status
Accepted (week 4). The running service is done; the scripts follow.

## Context
The orchestrator used three SDKs to reach three Azure services:

| Service | SDK | What we actually used |
|---|---|---|
| Foundry Agent Service | `azure-ai-agents` 1.1.0 (preview) | 9 operations: list agents, create thread, add message, create run, get run, submit tool outputs, cancel run, list messages, delete thread |
| AI Search | `azure-search-documents` | 1 operation: search (hybrid, or `*` for the Library tab) |
| Azure OpenAI | `openai` | 1 operation: embed the question |

Eleven operations, all JSON over HTTPS. The SDKs were most of the dependency
tree, and `azure-ai-agents` is a preview whose method names changed between
minor versions - the reason every Azure package is pinned exactly in
`requirements.txt`.

## Decision
The service sends those eleven requests itself, with `httpx`:

- `services/orchestrator/azure_http.py` - retries, timeouts, readable errors
- `services/orchestrator/foundry.py` - the nine agent operations
- `services/orchestrator/retrieval.py` - embed, then search

Two things stay:

- **`azure-identity`**, for sign-in tokens. It gets them from `az login` on a
  laptop and from the managed identity in Azure, and refreshes them. That is
  security code; writing it by hand gains nothing.
- **`azure-monitor-opentelemetry`**, for tracing. It is not a client for any
  API the app uses.

## How it was made safe
**Record, then reproduce.** The SDKs' traffic was recorded at the HTTP level -
method, URL, API version, body, response - during one real agent turn with a
tool call: 17 requests, 10 distinct. The plain calls reproduce them. Replaying
the same question through the new code produced the same 10 calls, the same
API versions (`v1` for Foundry, `2026-04-01` for Search, the configured
version for embeddings), all 200, and the same token count to the unit
(3,884 in, 123 out) - the search returned the same documents.

One deliberate difference: the `openai` SDK asked for embeddings as base64.
The plain call asks for plain numbers. Same values, easier to read.

**Tests the SDK made impractical.** The run loop - poll, run tool calls,
submit, read the answer, time out, cancel - had no tests, because faking the
SDK's run objects was more work than it was worth. Runs are now plain JSON, so
`tests/test_runner.py` drives the whole loop with a fake that returns the
recorded shapes. `tests/test_foundry.py` checks each request against the
recording. `tests/test_azure_http.py` covers the retry rules.

**Live checks.**
- The full golden set: 17/18. The one failure, `relevance-01`, fails 2 runs in
  3 on the SDK code too.
- The retry rule fired for real during that run: a dropped connection while
  deleting a thread, retried, and it succeeded.
- A booking conversation, the Library tab, and `/ready`, all on the new code.

## What we took on
What the SDKs did for free is now ours:

- **Retries.** Same statuses as azure-core (408, 429, 500, 502, 503, 504,
  dropped connections), honouring `Retry-After`, 3 retries. Like azure-core,
  writes are retried too (see `azure_http.py` for why).
- **Timeouts.** 30s per call, 5s to connect.
- **API versions.** Pinned in code: `FOUNDRY_API_VERSION`,
  `SEARCH_API_VERSION`, `AZURE_OPENAI_API_VERSION`. Moving to a newer one is a
  deliberate change with a test run, not a side effect of a package upgrade.
- **Paging.** Only listing agents pages. A search returns at most 1000 results,
  and `library.py` warns if it gets that many.

## Consequences
- **The service's code no longer depends on any SDK's method names.** The
  preview renames that forced the pins cannot break it.
- **Every request the app makes can be read in two files.** That is useful when
  something fails, and easy to explain.
- **One loss in tracing.** Application Insights used to record each SDK HTTP
  call as its own dependency span, through Azure Monitor's `azure_sdk`
  instrumentation. Those spans are gone for these calls. Our own spans
  (`chat.request`, `agent.turn`, `tool.call`, `routing.decision`) are
  unchanged, and every query in `docs/runbook.md` uses those. Adding
  `opentelemetry-instrumentation-httpx` would bring the HTTP spans back, at the
  cost of another pinned dependency.
- **The image is not smaller yet.** `scripts/ingest.py`, `agents/deploy_agents.py`,
  `scripts/search_test.py`, `scripts/generate_bulletins.py` and the eval
  scorer's ground-truth lookup still use the SDKs, and the image installs the
  same requirements. When the scripts move, the three SDKs leave
  `requirements.txt`.

## Revisit when
- Foundry's agent API has a GA SDK with stable names, and we need something
  from it that is more than a few requests (streaming runs, file search).
- The number of operations grows well past a dozen. At that point a thin
  hand-written client stops being thin.
