# 009 - Protecting a public chat endpoint: a limit, an honest refusal, and a cache

## Status
Accepted (week 4). 20 September 2026.

## Context
`/chat` and `/chat/stream` are public. No sign-in, no key, no limit. The sums
behind that are unkind:

| | |
|---|---|
| One customer message | 5,000-9,000 tokens across triage and the specialists |
| `gpt-4.1-mini` quota | 100,000 tokens a minute (`GlobalStandard`, capacity 100) |
| So, sustained | roughly **12-20 messages a minute**, about 900 an hour |
| Container app | `minReplicas 0`, `maxReplicas 3`, 0.5 vCPU, **no scale rule** |
| Azure AI Search | Free tier, 1 replica, no SLA |
| Zoho Bookings | trial, one staff member, 60-minute slots: about 8 bookings a day |

A `while true; do curl ...; done` on a laptop empties an hour of quota in about
four minutes and spends real money doing it. The workshop's own customers then
get nothing for the rest of the hour. 10,000 requests would take about eleven
hours to serve and cost roughly £30 in tokens - most of it to whoever sent them.

What happened when the quota did run out was worse than the running out.
`azure_http` treated Azure's `429` exactly like a `500`: up to three retries of
up to 20 seconds, on each of the several calls a message makes. The customer sat
through the 90-second request timeout and was then told **"Sorry - I could not
get an answer for that just now"** - true, and useless, and ninety seconds late.

## Decision
Three changes, in the order they matter.

### 1. A rate limit per visitor and a concurrency cap per replica
`services/orchestrator/limits.py`. Ten messages a minute and sixty an hour per
visitor; twenty messages in flight per replica. Any of them set to `0` is off.
Refusals answer `429` (rate) or `503` (full) with `Retry-After` and a sentence
written for a customer, not a stack trace. `/health`, `/ready`, `/metrics` and
`/library` are not limited - the deploy smoke test polls `/health` every five
seconds.

The concurrency cap is deliberately **above** the ingress's own scale-out
threshold. Container Apps is created with no scale rule, so Envoy's default
applies: another replica past 10 concurrent requests each. A cap of 4 or 8 would
refuse the eleventh customer instantly, hold measured concurrency under 10, and
so quietly prevent the scaling out that is the real answer to a crowd. The cap
is the backstop behind the autoscaler, not a substitute for it.

### 2. Fail fast, and say the true thing
`azure_http` now raises `Throttled` for a `429` rather than retrying it like an
outage. One quick retry if Azure advises a short wait - a burst clears in a
second - and otherwise it stops at once. The customer gets: *"The assistant is
handling a lot of messages just now and could not get to yours. Please try again
in a minute."* That is the truth, it arrives in about a second instead of ninety,
and it tells them what to do.

Triage being throttled stops the request there: all four agents share one model
deployment, so discovering it three more times costs only the customer's time.

**A safety warning never depends on a model being available.** If the message
trips the safety keywords and no agent could run, the reply is still
`SAFETY_FALLBACK` - "Do not drive the vehicle" - because that sentence comes from
an `if`, not from a model. "We are busy, try later" is not an answer to a brake
problem.

### 3. Cache whole answers to identical questions
`router.ANSWERS`, ten minutes. In a demo everyone asks the same handful of
things; the second person to ask gets the answer instantly and for nothing.

The cache is four lines. The rules about what may go into it are the part worth
reading:

- **Only a first message.** With a conversation behind it, "yes" does not mean
  what it meant last time.
- **Only a plain diagnostics answer** that a search actually grounded. Never a
  booking (it names slots, times and references), never an escalation, never a
  safety-flagged message, never a withheld or failed one.
- **Never anything personal** in the message: a registration, an email, a phone
  number, a booking reference.
- **Only questions `fast_route` decides.** This is the strict one. Triage is a
  model, and on an ambiguous symptom - "the car pulls hard to the left when I
  slow down" - it flags a safety issue on most runs and not on all of them.
  Caching the run where it did not would freeze one roll of the dice and hand it
  to everyone for ten minutes, turning a one-in-N miss into a certainty.
  `fast_route` decides in code: a fault code, no booking word, no safety word.
  The same message always takes the same route, so there is no judgement to
  freeze. Of the page's five example questions exactly one qualifies - the P0420
  one, which is the question a demo actually repeats.

A cached reply reports `tokens: 0` and `cached: true`, and its trace opens with
a `cache` entry and then shows the turns and the search that originally produced
it. It still reports `searched: true`, because it was grounded - saying otherwise
would fire the runbook's ungrounded-answer alert on every hit.

## What this does not do, said plainly
- **The counters are per replica.** There is no Redis. At three replicas a
  determined visitor gets up to three times the configured limit, and `/metrics`
  answers for whichever replica the ingress picked. The honest claim is "one
  visitor cannot be the whole load", not "the limit is exact".
- **The cache dies with its replica**, and `minReplicas` is 0. It helps inside a
  busy period, not across one.
- **The rate limit counts messages, not tokens.** Ten long messages a minute
  from one visitor can still outrun the quota; that is what change 2 is for.
- **A visitor who forges `X-Forwarded-For`** cannot buy a new identity - the
  rightmost entry is the one Azure's ingress wrote - but a visitor with a real
  IPv6 /64 has plenty of genuine addresses. The bounded visitor table then puts
  newcomers in one shared bucket, so the rotating addresses throttle each other
  rather than the regulars.
- **`/metrics` is public** and now also reports how full the replica is. It
  already reported traffic and token spend. Worth closing, with everything else
  that wants authentication.

## Alternatives considered
- **A global requests-per-minute cap** sized to the quota. Rejected: it would
  refuse legitimate traffic whenever messages were cheaper than the average, and
  Azure already knows the real number. Change 2 uses Azure's own answer instead
  of guessing it.
- **Keep retrying on 429, for longer.** That is what the ninety seconds already
  was. Under a sustained shortage every retry queues behind the same exhausted
  quota and the customer pays for it in silence.
- **Cache the retrieved documents only.** Already done (`retrieval._SEARCHES`,
  120s). It saves about 0.7s of a 7-9s answer. The expensive part is the model.
- **Authentication in front of `/chat`.** The right answer, and still open. This
  is the protection that costs nothing and does not stop a stranger trying the
  demo, which is what the app is for.

## Configuration
| Variable | Default | |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | 10 | per visitor; 0 is off |
| `RATE_LIMIT_PER_HOUR` | 60 | per visitor; 0 is off |
| `MAX_CONCURRENT_CHATS` | 20 | per replica; must stay above 10 |
| `ANSWER_CACHE_SECONDS` | 600 | 0 is off, which is what the eval runners set |

A value that is not a number is logged and ignored rather than raising at
import - this is read in a module `app.py` imports, so a typo would otherwise be
a container that dies before it serves anything.

## Consequences
- The eval runner and the red team set `ANSWER_CACHE_SECONDS=0`. A case answered
  out of an earlier case's cache measures nothing, and two runs an hour apart
  would not be comparable.
- `tests/conftest.py` clears the caches around every test, for the same reason:
  adding the cache turned six router tests green for the wrong reason.
- `/metrics` gains `refused`, `throttled`, `cache_hits` and the limiter's state.
  Throttling is counted apart from failures on purpose: it is the one number
  that says "ask Azure for more quota", and burying it in failures hides it.
- A refused message never reaches `routing.handle`, so it opens no
  `chat.request` span. `chat.refused` is emitted instead, with a short digest of
  the visitor - otherwise a script being turned away five thousand times an hour
  looks in App Insights exactly like a quiet afternoon.
