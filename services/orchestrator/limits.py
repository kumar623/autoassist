"""How much of the service one visitor may use, and how much runs at once.

/chat and /chat/stream are public: no sign-in, no key. Until now there was also
no limit, and the sums are unkind. One message costs 5,000-9,000 tokens across
the agents, and the gpt-4.1-mini deployment has 100,000 tokens a minute. A
`while true; do curl ...; done` on a laptop empties an hour of quota in about
four minutes, and spends real money doing it. The workshop's own customers then
get "the assistant is very busy" for the rest of the hour.

So two limits, both deliberately plain:

  - a rate limit per visitor: how many messages one person may send in a minute
    and in an hour. Sized for a real conversation, not for a script.
  - a concurrency cap per replica: how many messages may be in flight at once.
    Each one holds a worker thread and several open HTTPS calls on half a vCPU.

Neither protects the token quota exactly - a burst of long messages can still
outrun it - and neither tries to. That is what the fail-fast in azure_http does.
These stop one visitor from being the whole load.

WHAT THIS IS NOT: there is no shared store, so the counters live in the process.
With Container Apps scaled to three replicas a determined visitor gets up to
three times the configured limit, because the replica they land on is the only
one counting. A real deployment would put the counters in Redis and the limit at
the front door. Said plainly here because the honest version of this claim is
"one visitor cannot flood us", not "the limit is exact".
"""

from __future__ import annotations

import collections
import hashlib
import logging
import os
import threading
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

# Sized for a live demo, not for a script. The longest real conversation the app
# has had - book a service, a day, a time, a registration, a confirmation - is
# five messages in about ninety seconds, so a minute limit below about eight
# would interrupt a customer who types quickly. A script does hundreds.
def setting(name: str, default: int) -> int:
    """A whole number from the environment, which can never stop the app starting.

    Read at import, in a module app.py imports, so a ValueError here is a
    container that dies before it serves anything - and the likeliest way to get
    one is an operator turning a limit off with a word rather than a zero.
    Negative values mean the same as zero: off.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(float(raw)))
    except ValueError:
        log.warning("%s is set to %r, which is not a number; using %d", name, raw, default)
        return default


PER_MINUTE = setting("RATE_LIMIT_PER_MINUTE", 10)
PER_HOUR = setting("RATE_LIMIT_PER_HOUR", 60)

# Per replica, and deliberately ABOVE the ingress's own scale-out threshold.
# Container Apps is created with no scale rule (scripts/setup_deploy_target.sh),
# which means Envoy's default: add a replica past 10 concurrent requests each, up
# to three. A cap below 10 would refuse the eleventh customer instantly, keep
# measured concurrency under the threshold, and so quietly prevent the scaling
# out that is the actual answer to a crowd. This is the backstop behind that: at
# 20 apiece the app has already asked for all three replicas.
#
# Each message in flight holds a worker thread, and a route with two specialists
# holds two more, against 0.5 vCPU - but they are waiting on Azure almost the
# whole time, so the cost is threads and memory rather than processor. A slot is
# held for at most REQUEST_TIMEOUT (90s).
MAX_IN_FLIGHT = setting("MAX_CONCURRENT_CHATS", 20)

# Any limit set to 0 is off. The eval runner and the tests rely on this.
OFF = 0

# How many visitors are counted separately. Past this, new arrivals share one
# bucket - see Visitors._bucket_for. The number is a memory bound, nothing more:
# 5,000 deques of at most PER_HOUR timestamps is a few megabytes.
MAX_VISITORS = 5000
SHARED_BUCKET = "(everyone else)"

MINUTE = 60.0
HOUR = 3600.0

TOO_MANY = (
    "You have sent a lot of messages in a short time. Please wait a moment and try again."
)
BUSY = (
    "The assistant is busy with other customers just now. Please try again in a few seconds."
)


class Refused(Exception):
    """This message will not be handled. The customer is told why, politely."""

    def __init__(self, message: str, retry_after: int, reason: str, status: int):
        super().__init__(message)
        self.customer_message = message
        self.retry_after = retry_after
        self.reason = reason  # "rate" or "busy", for metrics and logs
        self.status = status  # HTTP status the endpoint should answer with


def visitor_key(headers, peer: str | None) -> str:
    """Who is asking, as well as we can tell from behind the ingress.

    Container Apps runs an Envoy ingress, which APPENDS the address it accepted
    the connection from to X-Forwarded-For. So the rightmost entry is the one
    Azure wrote and the only one worth trusting: anything to the left of it was
    sent by the caller and can say whatever they like.

    Taking the leftmost entry - which is the common example on the internet, and
    is right when you run your own chain of proxies - would mean a visitor could
    pick a new identity per request by setting the header themselves, and the
    rate limit would count nothing. Hence the right-hand end.

    This assumes exactly one proxy in front, which is true of Container Apps and
    of nothing else here. Run the app with no proxy - `make serve` on a laptop -
    and the header is whatever the caller typed; that is fine on a laptop and
    would not be behind a CDN, which would need the second entry from the right.
    """
    forwarded = ""
    if headers is not None:
        forwarded = headers.get("x-forwarded-for") or ""
    if forwarded:
        last = forwarded.split(",")[-1].strip()
        if last:
            return _without_port(last)
    return _without_port(peer or "") or "unknown"


def digest(key: str) -> str:
    """A short, stable stand-in for a visitor, for logs and telemetry.

    Enough to tell one script sending four hundred messages from four hundred
    customers sending one each, which is the question an operator actually has.
    Not enough to be an address in Log Analytics, which is the thing worth not
    writing there.
    """
    return hashlib.blake2s(key.encode(), digest_size=4).hexdigest()


def _without_port(address: str) -> str:
    """'20.1.2.3:41234' -> '20.1.2.3'. IPv6 in brackets keeps its colons."""
    address = address.strip()
    if address.startswith("["):  # [2001:db8::1]:443
        return address.split("]")[0].lstrip("[")
    if address.count(":") == 1:  # host:port; a bare IPv6 has more
        return address.split(":")[0]
    return address


class Visitors:
    """Sliding-window counters, one window per visitor.

    A deque of arrival times per visitor, pruned on use. No background sweeper:
    a visitor who never comes back is dropped the next time the table is full,
    which is the only moment their memory matters.
    """

    def __init__(self, per_minute: int = PER_MINUTE, per_hour: int = PER_HOUR):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self._seen: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_minute > OFF or self.per_hour > OFF

    def record(self, key: str, now: float | None = None) -> int:
        """Count one message from `key`. Returns 0, or seconds until they may retry.

        Counted on arrival, not on success. A request that then fails still cost
        the quota it was going to cost, so refunding it would let a visitor whose
        messages all fail keep the service busy for free.
        """
        if not self.enabled:
            return 0
        now = time.time() if now is None else now

        with self._lock:
            times = self._bucket_for(key, now)
            _prune(times, now - HOUR)

            wait = self._wait(times, now)
            if wait:
                return wait

            times.append(now)
            return 0

    def _wait(self, times: collections.deque, now: float) -> int:
        """Seconds until this visitor is under both limits, or 0 if they are now."""
        waits = []
        if self.per_hour > OFF and len(times) >= self.per_hour:
            waits.append(times[-self.per_hour] + HOUR - now)
        if self.per_minute > OFF:
            in_last_minute = sum(1 for t in times if t > now - MINUTE)
            if in_last_minute >= self.per_minute:
                recent = [t for t in times if t > now - MINUTE]
                waits.append(recent[-self.per_minute] + MINUTE - now)
        return max(1, int(max(waits) + 0.999)) if waits else 0

    def _bucket_for(self, key: str, now: float) -> collections.deque:
        """This visitor's arrival times, or the shared bucket when the table is full.

        The table is bounded because the visitor key comes from a header, and a
        visitor who forges a new one per request would otherwise grow it without
        end. When it is full, expired visitors are dropped first; if everyone
        present is still active, new arrivals share one bucket and so throttle
        each other rather than the regulars. An attacker rotating addresses ends
        up rate-limiting themselves, and nobody already being counted loses their
        own budget.
        """
        found = self._seen.get(key)
        if found is not None:
            return found

        if len(self._seen) >= MAX_VISITORS:
            for stale in [k for k, t in self._seen.items() if not t or t[-1] <= now - HOUR]:
                del self._seen[stale]
        if len(self._seen) >= MAX_VISITORS:
            shared = self._seen.get(SHARED_BUCKET)
            if shared is None:  # said once, not on every request while it lasts
                log.warning("rate limiter is full at %d visitors; new ones now share one bucket", len(self._seen))
                shared = self._seen[SHARED_BUCKET] = collections.deque()
            return shared

        self._seen[key] = collections.deque()
        return self._seen[key]

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


def _prune(times: collections.deque, before: float) -> None:
    while times and times[0] <= before:
        times.popleft()


class InFlight:
    """How many messages this replica is handling at once."""

    def __init__(self, limit: int = MAX_IN_FLIGHT):
        self.limit = limit
        self._count = 0
        self._lock = threading.Lock()
        self.peak = 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def take(self) -> bool:
        if self.limit <= OFF:
            return True
        with self._lock:
            if self._count >= self.limit:
                return False
            self._count += 1
            self.peak = max(self.peak, self._count)
            return True

    def release(self) -> None:
        if self.limit <= OFF:
            return
        with self._lock:
            self._count = max(0, self._count - 1)


class Limits:
    """Both limits together, as one context manager around handling a message."""

    def __init__(self, visitors: Visitors | None = None, in_flight: InFlight | None = None):
        self.visitors = visitors or Visitors()
        self.in_flight = in_flight or InFlight()
        self.refused = {"rate": 0, "busy": 0}

    def admit(self, key: str) -> None:
        """Let one message from `key` through, or raise Refused.

        Every admit() needs a matching release(), which is what hold() is for.
        The streaming endpoint cannot use hold(), because it has to answer with
        a status code before it starts writing the body and only finishes long
        after the handler returns.

        The concurrency slot is taken first and the visitor is counted second, so
        that a message refused because the replica is full is not also counted
        against the person who sent it. They did not get an answer; charging them
        for it would push them towards the rate limit for our shortage.
        """
        if not self.in_flight.take():
            self.refused["busy"] += 1
            log.warning("refused a message: %d already in flight", self.in_flight.limit)
            raise Refused(BUSY, retry_after=5, reason="busy", status=503)

        try:
            wait = self.visitors.record(key)
        except Exception:  # noqa: BLE001 - a limiter must never break the service
            log.exception("rate limiter failed; letting the message through")
            wait = 0
        if wait:
            self.in_flight.release()
            self.refused["rate"] += 1
            log.info("rate limited visitor %s for %ds", digest(key), wait)
            raise Refused(TOO_MANY, retry_after=wait, reason="rate", status=429)

    def release(self) -> None:
        self.in_flight.release()

    @contextmanager
    def hold(self, key: str):
        """admit(), then release() however the request ends."""
        self.admit(key)
        try:
            yield
        finally:
            self.release()

    def snapshot(self) -> dict:
        """What /metrics reports about the limits."""
        return {
            "in_flight": self.in_flight.count,
            "in_flight_peak": self.in_flight.peak,
            "in_flight_limit": self.in_flight.limit,
            "visitors_tracked": len(self.visitors._seen),
            "refused_rate": self.refused["rate"],
            "refused_busy": self.refused["busy"],
        }
