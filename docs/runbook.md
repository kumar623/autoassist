# Runbook

Queries and checks for when something is wrong. Written to be usable at 2am by
someone who did not build this.

Run KQL in **Azure portal → Application Insights → Logs**.

All custom fields are namespaced `autoassist.*` inside `customDimensions`.

---

## First: is it up?

```bash
curl -s https://<host>/health    # process alive?
curl -s https://<host>/ready     # can it actually serve?
curl -s https://<host>/metrics   # counters since start
```

`/health` failing means the process is gone — the container restarts itself.
`/ready` failing means it is running but cannot reach Azure or the agents are
not deployed; the body says which. **A ready failure does not restart anything**,
and that is deliberate: restarting every replica because Azure blipped turns a
small outage into a large one.

---

## The one that matters most: ungrounded answers

Every diagnostics answer must come from a document. `searched == false` means
it answered from the model's own training, which is finding 1 and the worst
failure this system has.

```kql
traces
| where timestamp > ago(24h)
| where customDimensions["autoassist.agent"] == "diagnostics"
| extend searched = tobool(customDimensions["autoassist.searched"])
| summarize total = count(), ungrounded = countif(searched == false)
| extend pct = round(100.0 * ungrounded / total, 1)
```

Anything above zero needs investigating the same day. To see them:

```kql
dependencies
| where timestamp > ago(24h)
| where name == "agent.turn"
| where customDimensions["autoassist.agent"] == "diagnostics"
| where tobool(customDimensions["autoassist.searched"]) == false
| project timestamp, operation_Id, duration,
          status = customDimensions["autoassist.status"],
          error  = customDimensions["autoassist.error"]
| order by timestamp desc
```

Take an `operation_Id` and paste it into Transaction Search to see the whole
request — routing decision, every agent, every tool call.

---

## Safety routing

Did anything flagged as a safety issue fail to reach escalation? This should
always be empty; if it is not, the routing code has a bug, not the model.

```kql
dependencies
| where timestamp > ago(7d)
| where name == "chat.request"
| where tobool(customDimensions["autoassist.safety"]) == true
| extend agents = tostring(customDimensions["autoassist.agents"])
| where agents !contains "escalation"
| project timestamp, operation_Id, agents
```

How often does the keyword check catch something triage missed? A rising number
means the triage prompt needs work.

```kql
dependencies
| where timestamp > ago(7d)
| where name == "routing.decision"
| summarize count() by source = tostring(customDimensions["autoassist.safety_source"])
```

`keyword` = triage missed it, the regex saved us.
`triage` = the model caught something the regex has no word for — good.
`both` = agreement.

---

## Latency

Where does the time actually go?

```kql
dependencies
| where timestamp > ago(24h)
| where name == "agent.turn"
| summarize
    calls = count(),
    p50 = percentile(duration, 50),
    p95 = percentile(duration, 95),
    max = max(duration)
  by agent = tostring(customDimensions["autoassist.agent"])
| order by p95 desc
```

End to end, split by how many agents ran:

```kql
dependencies
| where timestamp > ago(24h)
| where name == "chat.request"
| extend n = toint(customDimensions["autoassist.agent_count"])
| summarize count(), p50 = percentile(duration, 50), p95 = percentile(duration, 95) by n
| order by n asc
```

Known baseline as of week 3: about 21s for a three-agent reply, roughly 7s each
plus overhead. Sequential. See README limitations.

Tool call timings:

```kql
dependencies
| where timestamp > ago(24h)
| where name == "tool.call"
| summarize count(), avg_ms = avg(duration), p95 = percentile(duration, 95)
  by tool = tostring(customDimensions["autoassist.tool"])
| order by p95 desc
```

Retrieval should sit near 1s (an embedding call plus a search). Booking tools
should be a few milliseconds — they are a local file. A booking tool taking
seconds means the storage layer changed and nobody updated this line.

---

## Retrieval quality

The tool output's first line reports what the filters did. It is captured in
`result_head`, so filtering can be watched without re-running searches.

```kql
dependencies
| where timestamp > ago(24h)
| where name == "tool.call"
| where customDimensions["autoassist.tool"] == "search_service_docs"
| extend head = tostring(customDimensions["autoassist.result_head"])
| extend nothing_found = head startswith "NOTHING USABLE" or head startswith "NO DOCUMENTS"
| summarize searches = count(), found_nothing = countif(nothing_found)
| extend pct = round(100.0 * found_nothing / searches, 1)
```

A rising "found nothing" rate means people are asking about things the library
does not cover. That is a **content** problem, not a code problem — the fix is
more documents, not more prompt engineering.

What are they asking that we cannot answer?

```kql
dependencies
| where timestamp > ago(7d)
| where name == "tool.call"
| where customDimensions["autoassist.tool"] == "search_service_docs"
| extend head = tostring(customDimensions["autoassist.result_head"])
| where head startswith "NOTHING USABLE" or head startswith "NO DOCUMENTS"
| extend args = tostring(customDimensions["autoassist.args"])
| project timestamp, args
| order by timestamp desc
| take 50
```

---

## Cost

```kql
dependencies
| where timestamp > ago(30d)
| where name == "chat.request"
| extend tokens = toint(customDimensions["autoassist.total_tokens"])
| summarize requests = count(), tokens = sum(tokens) by bin(timestamp, 1d)
| extend est_usd = round(tokens / 1000000.0 * 0.10, 3)   // gpt-4.1-mini input rate
| order by timestamp desc
```

Rough only — it does not separate input from output tokens, which are priced
differently. Good enough to spot a jump. Azure Cost Management has the real
number.

Which agent spends the most:

```kql
dependencies
| where timestamp > ago(7d)
| where name == "agent.turn"
| summarize
    turns = count(),
    prompt = sum(toint(customDimensions["autoassist.prompt_tokens"])),
    completion = sum(toint(customDimensions["autoassist.completion_tokens"]))
  by agent = tostring(customDimensions["autoassist.agent"])
| extend total = prompt + completion
| order by total desc
```

Expect diagnostics to dominate: it carries retrieved documents in its prompt.

---

## Failures

```kql
dependencies
| where timestamp > ago(24h)
| where name in ("agent.turn", "tool.call", "chat.request")
| where isnotempty(customDimensions["autoassist.error"])
   or tobool(customDimensions["autoassist.failed"]) == true
| project timestamp, name, operation_Id,
          agent = customDimensions["autoassist.agent"],
          tool  = customDimensions["autoassist.tool"],
          error = customDimensions["autoassist.error"]
| order by timestamp desc
```

Triage returning something that is not JSON:

```kql
dependencies
| where timestamp > ago(7d)
| where name == "routing.decision"
| summarize total = count(),
            parse_failures = countif(tobool(customDimensions["autoassist.triage_parse_failed"]))
```

Should be zero — triage runs with `response_format: json_object`. Anything here
means that setting was lost, or the model produced malformed JSON anyway. The
fallback routes to diagnostics and keeps the keyword safety check, so it fails
safe, but it should not be happening.

---

## Alerts to configure

| Alert | Condition | Why |
|---|---|---|
| Ungrounded answers | any diagnostics turn with `searched == false` in 1h | worst failure mode; should never happen |
| Safety not escalated | any safety request without escalation in 1h | a routing bug |
| p95 latency | > 40s over 15 min | roughly double baseline |
| Failure rate | > 5% of requests in 15 min | |
| AOAI throttling | any 429 in 5 min | quota exhausted; requests are failing |
| Not ready | `/ready` failing 5 min | deployment or Azure problem |

---

## Common problems

**Every request fails, `/ready` says no Azure client.** Credentials. Locally:
`az account show`. In Azure: the container's managed identity needs the
**Azure AI Developer** role on the Foundry project.

**Runs sit in `in_progress` then time out.** This happened in week 2 with the
built-in Azure AI Search tool on vector query types (evaluation.md, finding 5).
If it returns, check whether the tool configuration changed. The run loop gives
up at 90s rather than hanging; that limit is `REQUEST_TIMEOUT`.

**Answers are correct but have no citations.** Check `tool.call` spans for that
operation. No `search_service_docs` call means the prompt changed, or the tool
failed to attach during deploy. `deploy_agents.py --list` shows what each agent
actually has.

**Everything is slow, tool calls are fast.** The time is in the model, not in
us. Check Azure OpenAI quota and for 429s. Sequential agents are the design; see
README limitations.

**Bookings vanished.** The store is `data/bookings.json` inside the container.
Container restarts lose it. That is a known limitation, not a bug — Table
Storage is the fix.
