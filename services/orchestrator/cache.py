"""A small time-limited cache, for answers that are the same for everyone.

Without it the service pays for the same answer twice: the booking agent asked
Zoho for the same day's availability twice inside one reply (1.2s each, measured
20 Sep), and a customer who asks about the same fault code twice pays for the
same search twice. Four places use it - Zoho availability (zoho_bookings.py),
document searches (retrieval.py), whole answers (router.py) and the agent
roster (roster.py).

Deliberately plain: a dict, a lock, and an age. No eviction beyond expiry, since
what is cached here is small and shortlived. Nothing customer-specific goes in -
each of those is the same whoever is asking.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

# A tiny cache cannot become a leak. Past this many entries the expired ones are
# dropped, and if every one is still live the cache starts again from empty.
MAX_ENTRIES = 256

# The clock, as a name the tests can replace, so that expiry is tested by moving
# time rather than by sleeping through it (the same trick as azure_http._pause).
_now = time.time


class TimedCache:
    """Values that stay usable for `seconds`, then are fetched again.

    `name` says what is cached, for the log line when the cache fills up.
    """

    def __init__(self, seconds: float, name: str = ""):
        self.seconds = seconds
        self.name = name
        self._values: dict = {}
        self._lock = threading.Lock()

    def get_or_call(self, key, produce):
        """The cached value for `key`, or `produce()` remembered under it.

        `produce` runs outside the lock: it is a network call, and holding a
        lock across one would serialise every request in the process. So two
        callers arriving together on a miss both produce, and the later value is
        the one kept.

        A None is never a hit (see get()), so a `produce` that returned None
        would run every time. No caller's does.
        """
        value = self.get(key)
        if value is None:
            value = produce()
            self.put(key, value)
        return value

    def get(self, key):
        """The cached value for `key`, or None.

        For values that are only worth keeping sometimes - a whole answer is
        cached only if it turns out to be the kind of answer that is the same for
        everyone, which is not known until it has been written. get_or_call
        cannot express that, because it decides before the work happens.

        A cached None is indistinguishable from a miss. Nothing stores one.
        """
        now = _now()
        with self._lock:
            found = self._values.get(key)
        if found is not None and found[0] > now:
            return found[1]
        return None

    def put(self, key, value) -> None:
        now = _now()
        with self._lock:
            self._values[key] = (now + self.seconds, value)
            if len(self._values) > MAX_ENTRIES:
                self._prune(now)

    def clear(self) -> None:
        """Forget everything. Called when the thing cached has just changed."""
        with self._lock:
            self._values.clear()

    def _prune(self, now: float) -> None:
        for key in [k for k, (expires, _) in self._values.items() if expires <= now]:
            del self._values[key]
        if len(self._values) > MAX_ENTRIES:  # still full of live entries: start again
            log.info("cache %r held %d live entries; starting again", self.name, len(self._values))
            self._values.clear()
