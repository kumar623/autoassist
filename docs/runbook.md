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

Every diagnostics answer must come from a document. A reply with no search
behind it was written from the model's own training, which is finding 1 and the
worst failure this system has.

Read `searched` from the **`chat.request`** span, not from `agent.turn`. The
router searches the library itself before the diagnostics agent runs
(`router.prefetch_documents`) and tells the agent not to search again for the
same question. So on a well-grounded answer the agent's own turn usually made
no search call, and its `agent.turn` span says `searched = false`.
`chat.request` sets `searched` after the pre-search has been added to the
diagnostics turn, so it counts both kinds.

```kql
dependencies
| where timestamp > ago(24h)
| where name == "chat.request"
| where tostring(customDimensions["autoassist.agents"]) contains "diagnostics"
| extend searched = tobool(customDimensions["autoassist.searched"])
| summarize total = count(), ungrounded = countif(searched == false)
| extend pct = round(100.0 * ungrounded / total, 1)
```

Anything above zero needs investigating the same day. To see them:

```kql
dependencies
| where timestamp > ago(24h)
| where name == "chat.request"
| where tostring(customDimensions["autoassist.agents"]) contains "diagnostics"
| where tobool(customDimensions["autoassist.searched"]) == false
| project timestamp, operation_Id, duration,
          agents = customDimensions["autoassist.agents"],
          failed_turns = customDimensions["autoassist.failed_turns"],
          error  = customDimensions["autoassist.error"]
| order by timestamp desc
```

One way it happens: the pre-search fails - the service log says `pre-search
failed, the agent will search itself` - and the agent does not search either.

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

`keyword` = the classifier missed it, the regex saved us.
`triage` = the triage agent caught something the regex has no word for — good.
`jev` = the same, when Jev is routing (`TRIAGE_BACKEND=jev`): its safety
probability was at or above `JEV_SAFETY_CUT`.
`both` = agreement.
`none` = nobody flagged it — most messages.

On the Jev path the span also carries `backend = jev` and each probability
(`p_safety`, `p_needs_diagnostics`, ...), so a false alarm can be read as a
number rather than a guess.

Every path also records `triage_choice` (`auto`, `jev` or `agent` - the page's
toggle), `triage_fallback` when the classifier asked for could not be used (no
TypeSafe key, or Jev did not answer), the classifier's own `classifier_safety`
before the regex, and `keyword_match`, the word the regex caught. When a visitor
ticks "Compare both", the other classifier's reading goes in a separate
`routing.compare` span (`chosen`, `other`, `agrees`, `differences`), never in
`routing.decision`, so the counts above are still one per message. `chosen` and
`other` are never the same classifier: when Jev was chosen and could not answer,
the agent routes and `other` is Jev, not answering, rather than a second agent
turn. `agrees` is only set when both answered. How often do the two disagree,
and on what?

```kql
dependencies
| where timestamp > ago(7d)
| where name == "routing.compare"
| where isnotempty(customDimensions["autoassist.agrees"])   // both answered
| summarize compared = count(), disagreed = countif(tobool(customDimensions["autoassist.agrees"]) == false)
  by differences = tostring(customDimensions["autoassist.differences"])
```

`/metrics` counts the toggle: `triage_choice_jev`, `triage_choice_agent`,
`triage_compare`, and what the compared classifier spent, per classifier:
`compare_tokens_jev` (input tokens) and `compare_tokens_agent` (prompt plus
completion). Two figures, not one, because they are priced about ten times apart
(`PRICE_PER_MTOK` in the router). Each comparison is counted when it finishes,
so one still running when its answer went - billed all the same - is in them
too. They are kept out of `total_tokens`, which is what the answers cost. A
compared Jev call is not counted in `jev_answered` or `jev_fell_back`, and a
compared Jev that fails is not logged as a fallback: those stay a record of what
routing did.

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

### One ticket per conversation

A conversation gets one ticket, however many agents ask. On 21 September one
brake conversation raised three: diagnostics and escalation each raised one for
the first message, and diagnostics raised another for the next, under the
reply's own line that the first still stood. Four things now hold the rule:

- **`tools.OneTicket`.** Every `raise_ticket` call for one message goes through
  the same one, including calls from specialists running at the same time. While
  a ticket stands in the conversation, any call is given that reference and
  nothing is raised. When escalation is on the route, only escalation may raise
  one. Otherwise the first call raises it and every later call gets the same
  reference.
- **The page remembers the ticket.** The server keeps no conversation, so "a
  ticket stands" means the page said so. Every reply reports the conversation's
  ticket (`ticket` in the response); the page keeps it and sends it back with
  each message (`ticket` in the request, beside `triage` and `compare`), next to
  the six turns of history rather than in them. Read from the history alone
  (`router.ticket_already_raised`), the reference was gone three exchanges after
  it was given, and the next brake message raised a second ticket. The history
  is still read, for callers that send nothing else. A message sent with a
  ticket is never answered from the answer cache or kept in it, even with no
  history: the reply can name the ticket, and the next visitor handed it would
  hold someone else's.
- **One voice for the ticket.** When a ticket stands, the router says so in its
  own sentence (`router.TICKET_STANDS`). When escalation raised it, escalation
  says it. Whatever any other agent says about a ticket - or about an advisor's
  call - is dropped from the reply (`router._leave_the_ticket_to`). If that took
  the do-not-drive warning with it on a safety message, the router puts its own
  warning back at the top (`router._keep_the_warning`). A ticket the reply forgot
  to name, or named with a wrong digit, is named correctly, so the next message
  can find it.
- **Only escalation has `raise_ticket`.** Diagnostics offers an advisor's call
  instead. A short yes to that offer is sent to escalation by code
  (`router._with_accepted_offer`), because neither classifier is asked about
  accepting one; escalation then raises the ticket. It runs after whichever
  classifier routed the message - Jev, the LLM, or the keyword fallback - and
  never on the one only compared. On the comparison card "Route decided" is the
  classifier and the keyword net, so escalation added for a yes shows in the
  route line above the card, not in it. The offer lives in
  `agents/definitions/diagnostics.json`, so it only takes effect when the agents
  are redeployed (`make agents`, which the app deploy does not do); until then
  the others hold the rule on their own. The routing eval set has no case for
  this yes yet, so it is covered by unit tests only.

Where it can still fail. A reply that never reaches the page - the connection
dropped before the answer finished - leaves the page without the reference,
because the server finishes the message regardless and the page only keeps what
arrives; the next message can then raise a second ticket. A reloaded page is
a new conversation, with no ticket of its own until it raises one.

Who calls `raise_ticket`. Once diagnostics is redeployed, only `escalation`
should appear here.

```kql
dependencies
| where timestamp > ago(7d)
| where name == "tool.call"
| where tostring(customDimensions["autoassist.tool"]) == "raise_ticket"
| summarize calls = count() by agent = tostring(customDimensions["autoassist.agent"])
```

Messages where an agent asked for a ticket, and whether one was raised. More
requests than tickets is `OneTicket` turning the extras away. A message with
`ticket_raised` true and a ticket reference already earlier in the same
conversation would be a bug; the service logs say `nothing raised` for every
call that was turned away, with the reference it was given instead.

```kql
dependencies
| where timestamp > ago(7d)
| where name == "chat.request"
| where isnotempty(customDimensions["autoassist.ticket_requests"])
| summarize messages = count(),
            raised = countif(tobool(customDimensions["autoassist.ticket_raised"]))
  by requests = toint(customDimensions["autoassist.ticket_requests"])
```

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

Known baseline, measured in week 3: about 12s for a three-agent reply, down from
20s once independent specialists ran in parallel (`docs/evaluation.md`, "Week 3
— latency"). Booking's turn is hidden inside diagnostics'; triage and
diagnostics are what remain. With Jev routing, triage is about 350ms at the
median instead of about 2s (finding 16).

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
call Zoho Bookings over MCP (decision 008), so they take network round trips,
not the 2ms the old local file did; slot lookups are cached for
`ZOHO_CACHE_SECONDS` (60 by default).

These are only the tools the agents called. The router's own pre-search runs
before the diagnostics agent and opens no `tool.call` span, so it is not here.

---

## Retrieval quality

The tool output's first line reports what the filters did. It is captured in
`result_head`, so filtering can be watched without re-running searches.

**These queries see only the searches an agent made itself.** The pre-search the
router runs before diagnostics - now the usual one - opens no `tool.call` span.
Its line, `retrieval query=... -> N kept of M candidates`, is in the container's
console log, not in Application Insights.

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
| Ungrounded answers | any `chat.request` that ran diagnostics with `searched == false` in 1h | worst failure mode; should never happen. Not `agent.turn` - see above |
| Safety not escalated | any safety request without escalation in 1h | a routing bug |
| p95 latency | > 25s over 15 min | roughly double the ~12s baseline |
| Failure rate | > 5% of requests in 15 min | |
| AOAI throttling | any 429 in 5 min | quota exhausted; customers are told we are busy |
| Refusing visitors | `chat.refused` rising over 15 min | a script, or the limit is too tight - group by visitor first |
| Replica full | `refused_busy` > 0 | check it scaled out; the cap is 20, ingress adds a replica past 10 |
| Not ready | `/ready` failing 5 min | deployment or Azure problem |

---

## Common problems

**Every request fails, `/ready` says no Azure client.** Credentials. Locally:
`az account show`. In Azure: the app's managed identity, `id-autoassist`, needs
**Foundry User** on the Foundry account `rg-autoassist`. With a weaker role such
as Cognitive Services User it signs in and gets back an empty list of agents,
which `/ready` reports as `agents not deployed` - the 12 September rollback in
`docs/deploy.md`.

**Runs sit in `in_progress` then time out.** This happened in week 2 with the
built-in Azure AI Search tool on vector query types (evaluation.md, finding 5).
If it returns, check whether the tool configuration changed. The run loop gives
up at 90s rather than hanging; that limit is `REQUEST_TIMEOUT`.

**Answers are correct but have no citations.** Check `searched` on that
operation's `chat.request` span, and the trace in the API response: the
pre-search appears there as a `search_service_docs` call marked
`run_before_the_agent`. `tool.call` spans show only searches the agent made
itself, so their absence alone proves nothing. If there was no search at all,
the prompt changed or the tool failed to attach during deploy;
`deploy_agents.py --list` shows what each agent actually has.

**Everything is slow, tool calls are fast.** The time is in the model, not in
us. Check Azure OpenAI quota and for 429s. Independent specialists run in
parallel, so a slow reply is triage plus the slowest specialist, not the sum.

**Customers are told we are busy, but traffic is low.** Check `throttled` before
`refused`. Being throttled is Azure's quota, not our limit, and our limit does
not cause it. If both are zero and people still see the busy reply, look for a
throttle on a *different* deployment sharing the quota.

**"It answered instantly and skipped the agents."** That is the answer cache
doing its job; the trace opens with a `cache` entry saying how old the answer is.
It only ever holds plain fault-code answers - never a booking, an escalation or
anything safety-flagged. To turn it off in the live app:
`az containerapp update -g "$RG" -n "$APP" --set-env-vars ANSWER_CACHE_SECONDS=0`.

**Tickets vanished.** Tickets are a JSON file inside the container
(`TICKET_STORE`, `/app/data/tickets.json`), and a restart or a deploy loses
them. A known limitation, not a bug.

**A booking is missing.** Bookings are in Zoho Bookings (decision 008), which
survives restarts - look there first. Then check the app has
`BOOKING_BACKEND=zoho`: with `file` it books into a JSON file in the container,
which the next restart loses.

**Searches fail on every message, but `/ready` is green.** `/ready` checks the
agents, not the keys, so a bad search or OpenAI key does not show there. See
"Keys" below: on 21 September this was page text pasted into `openai-key`.

---

## Keys

The app's six keys are in Key Vault `kv-autoassist-kk`, and the container app
holds only references to them, resolved by its identity `id-autoassist`
(`docs/decisions/010-keys-in-key-vault.md`). Neither procedure below needs a
deploy.

### Rotating a key

1. **Add the new value as a new version**, from a terminal (`openai-key` here;
   the other five work the same way). On 21 September a portal paste put page
   text into `openai-key`; httpx refused to send it as a header and search
   failed on every message until a correct version was added.

   ```bash
   read -rs NEW_VALUE     # paste, press Enter; nothing is echoed
   az keyvault secret set --vault-name kv-autoassist-kk -n openai-key \
     --value "$NEW_VALUE" -o none
   unset NEW_VALUE
   ```

   `-o none` matters: the command's output includes the value it stored.

2. **Check its shape without printing it.** A key has no spaces, and its length
   is the length of what you meant to copy. Page text fails both.

   ```bash
   az keyvault secret show --vault-name kv-autoassist-kk -n openai-key \
     --query "{length: length(value), has_space: contains(value, ' ')}"
   ```

3. **Wait.** The references have no version in them, so Container Apps picks up
   the new version by itself within 30 minutes - 12 minutes on 21 September -
   and restarts the revision. Then ask the app a fault-code question and check
   the trace shows a search that found something.

4. **Only then retire the old key.** Azure OpenAI and AI Search each have two
   keys: put the other one in the vault (steps 1 to 3), then regenerate the one
   the app stopped using. Revoke first and the app is down until the new
   version reaches it.

### Key Vault reference not resolving

A revision that will not start, or one that starts and then fails every search,
after anything changed in the vault or the app's identity. Check, in order:

```bash
# What the app points at: names and vault URLs, never values.
az containerapp show -g Ai_solution -n ca-autoassist \
  --query "properties.configuration.secrets[].{name:name, vault:keyVaultUrl}" -o table

# That each secret exists. Names only; `show` would return the value.
az keyvault secret list --vault-name kv-autoassist-kk --query "[].name" -o tsv

# That the app's identity may read them. Expect "Key Vault Secrets User".
az role assignment list \
  --assignee "$(az identity show -g Ai_solution -n id-autoassist --query principalId -o tsv)" \
  --scope "$(az keyvault show -n kv-autoassist-kk --query id -o tsv)" \
  --query "[].roleDefinitionName" -o tsv
```

A secret deleted by mistake is recoverable, because soft delete is on:
`az keyvault secret list-deleted --vault-name kv-autoassist-kk`, then
`az keyvault secret recover`. A deploy cannot fix any of this: the GitHub
deploy identity has no role on the vault, by design.
