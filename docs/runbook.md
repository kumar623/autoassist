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

### Answers withheld on a safety issue

`router._withhold_reassurance` drops a diagnostics answer that reassures ("safe
to keep driving", "is normal") on a safety-flagged message. Each one is either a
false alarm or a document telling customers something dangerous. Look at every
one.

```kql
dependencies
| where timestamp > ago(7d)
| where name == "chat.request"
| where isnotempty(customDimensions["autoassist.withheld"])
| project timestamp, operation_Id, withheld = customDimensions["autoassist.withheld"]
```

Then search the service logs for `withheld a diagnostics answer` to see the
text and find the document it came from.

---

## Too busy: refusals, throttling and the cache

Three different things look like "it is slow or it is not answering", and they
have three different answers. Start with `/metrics` - but read the caveat at the
bottom of this section first.

```bash
curl -s "$URL/metrics" | jq '{requests, refused, throttled, cache_hits,
                              failures, in_flight, in_flight_peak, in_flight_limit,
                              refused_rate, refused_busy}'
```

| What you see | What it means | What to do |
|---|---|---|
| `refused_rate` climbing | one visitor, or a script, is over 10 a minute or 60 an hour | see the query below - one visitor or many? |
| `refused_busy` climbing | the replica is full (20 in flight) | it should be scaling out; check replica count |
| `throttled` climbing | Azure's token quota is spent | ask for more quota, or wait it out |
| `cache_hits` near 0 in a demo | nobody is repeating a question, or the replica is cold | expected after a scale to zero |

**Is it one visitor or a crowd?** A refused message never reaches the router, so
it has no `chat.request` span. It has this one:

```kql
dependencies
| where timestamp > ago(1h)
| where name == "chat.refused"
| summarize refusals = count() by visitor = tostring(customDimensions["autoassist.visitor"]),
                                  reason = tostring(customDimensions["autoassist.reason"])
| order by refusals desc
```

One digest with hundreds of refusals is a script: leave the limit where it is.
Many digests with a few each means the limit is too tight for real use - raise
`RATE_LIMIT_PER_MINUTE` rather than removing it. The digest is a hash, not an
address; it is stable per visitor so the rows can be grouped.

**How often is the quota running out?**

```kql
dependencies
| where timestamp > ago(24h)
| where name == "chat.request"
| summarize total = count(), busy = countif(tobool(customDimensions["autoassist.throttled"]) == true)
| extend pct = round(100.0 * busy / total, 1)
```

Anything above zero for more than a burst means 100,000 tokens a minute is not
enough for the traffic. Raise the quota in the Azure OpenAI resource; nothing in
the code will fix it. The service logs `throttled, giving up` with Azure's own
wording, which is kept out of the customer's trace on purpose.

**How much is the cache saving?**

```kql
dependencies
| where timestamp > ago(24h)
| where name == "chat.request"
| summarize total = count(), cached = countif(tobool(customDimensions["autoassist.cached"]) == true)
| extend saved_tokens = cached * 7000
```

A cache hit still reports `searched = true`, because the answer was grounded when
it was written - so the ungrounded-answer query above is unaffected by it.

**The caveat.** All of these counters are per replica and `/metrics` answers for
whichever replica the ingress picked, so two curls a second apart can disagree.
The KQL queries do not have this problem: use them for anything that matters.
Setting `RATE_LIMIT_PER_MINUTE` to a number based on one `/metrics` read is how
a limit gets raised for the wrong reason.

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
| AOAI throttling | any 429 in 5 min | quota exhausted; customers are told we are busy |
| Refusing visitors | `chat.refused` rising over 15 min | a script, or the limit is too tight - group by visitor first |
| Replica full | `refused_busy` > 0 | check it scaled out; the cap is 20, ingress adds a replica past 10 |
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

**Customers are told we are busy, but traffic is low.** Check `throttled` before
`refused`. Being throttled is Azure's quota, not our limit, and our limit does
not cause it. If both are zero and people still see the busy reply, look for a
throttle on a *different* deployment sharing the quota.

**"It answered instantly and skipped the agents."** That is the answer cache
doing its job; the trace opens with a `cache` entry saying how old the answer is.
It only ever holds plain fault-code answers - never a booking, an escalation or
anything safety-flagged. To turn it off in the live app:
`az containerapp update -g "$RG" -n "$APP" --set-env-vars ANSWER_CACHE_SECONDS=0`.

**Bookings vanished.** The store is `data/bookings.json` inside the container.
Container restarts lose it. That is a known limitation, not a bug — Table
Storage is the fix.
