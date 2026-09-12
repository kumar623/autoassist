# 005 - No orchestration framework (LangGraph, CrewAI, Semantic Kernel)

## Status
Accepted (week 3)

## Context
This system coordinates four agents: triage classifies, then one or more
specialists run, then their answers are joined. That is a classify → fan out →
join shape, run per request, finishing in well under a minute.

LangGraph is the obvious candidate and was considered seriously. CrewAI and
Semantic Kernel solve a similar problem.

## What a framework would actually give us

Worth naming honestly, because these are real capabilities and not marketing:

- **Checkpointing.** Persist graph state so a long workflow survives a crash or
  a restart and resumes where it stopped.
- **Human-in-the-loop interrupts.** Pause at a node, wait for a person to
  approve or edit, continue hours later.
- **Cyclic graphs.** Agents that loop - a critic sending work back to a writer,
  repeated until some condition holds.
- **Streaming intermediate state.** Push each node's progress to a UI as it
  happens.
- **A shared vocabulary.** Nodes and edges are a common language for describing
  a flow to other engineers.

## Decision
Plain Python. Routing is an ordinary function; parallelism is
`concurrent.futures.ThreadPoolExecutor`.

## Why

**We need none of the above.** Our flow is acyclic and completes in one request.
There is nothing to resume, nothing to interrupt, no loop, and the UI shows the
trace after the fact rather than during. We would be paying a framework's cost
for features we do not use.

**The routing logic is small and we own it.** `router.py` is about 200 lines.
It is covered by 26 tests that run with no Azure, no mocks and no network - see
`tests/test_router.py`. Expressing the same logic as graph configuration would
make those tests exercise the framework's traversal rather than our decisions.

**Safety is enforced by an `if` statement.** `docs/evaluation.md` findings 1, 6
and 7 are all cases of a model dropping a safety rule under pressure from
competing instructions. The rule "a safety issue always reaches escalation" is
now a line of Python with a test asserting it, plus an independent keyword check
on the raw message. Both are things a reviewer can read in ten seconds. A
conditional edge in a graph is the same logic, one layer further from the
reader.

**Tracing stays ours.** Every span in App Insights is one we chose to emit, with
attributes we chose: `searched`, `safety_source`, `triage_parse_failed`. A
framework brings its own instrumentation, which is useful but generic - it
would tell us a node took 3.5s, not that triage cost 26% of the request for 44
tokens of output. That observation came from our own spans and is what drove the
latency work.

**The parallelism we need is fifteen lines.** Running two independent
specialists at once is submit-both-wait-for-both. Adopting a framework to get
`ThreadPoolExecutor` is not a trade worth making.

## What this costs us

Being fair to the alternative:

- If this grows cycles - a verifier sending an answer back for revision - we
  would be hand-rolling something LangGraph does well. That is the point at
  which to revisit.
- Long-running or resumable workflows would mean building checkpointing
  ourselves, which is real work and easy to get wrong.
- "We use LangGraph" is instantly legible to another engineer. "We wrote our own
  router" needs this document, which is part of why it exists.

## Revisit when

- The flow needs a cycle (agents revising each other)
- A step needs human approval mid-flow, with a gap of minutes or hours
- Workflows outlive a single request and must survive restarts
- Routing logic grows past roughly 500 lines or stops being readable in one sitting

Until then, the cost is real and the benefit is hypothetical.

## Note
This is not a judgement about LangGraph, which is good at what it does. It is a
judgement about *this* system: a short acyclic flow where the valuable parts are
grounding discipline and observability, and where every layer between the code
and the model is a layer to debug through.
