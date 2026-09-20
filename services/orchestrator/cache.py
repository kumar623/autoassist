"""A small time-limited cache, for answers that are the same for everyone.

Two places pay for the same answer twice: the booking agent asked Zoho for the
same day's availability twice inside one reply (1.2s each, measured 20 Sep), and
a customer who asks about the same fault code twice pays for the same search
twice.

Deliberately plain: a dict, a lock, and an age. No eviction beyond expiry, since
what is cached here is small and shortlived. Nothing customer-specific goes in -
availability and document searches are the same whoever is asking.
"""

from __future__ import annotations

import threading
import time


class TimedCache:
    """Values that stay usable for `seconds`, then are fetched again."""

    def __init__(self, seconds: float, name: str = ""):
        self.seconds = seconds
        self.name = name
        self._values: dict = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_call(self, key, produce):
        """The cached value for `key`, or `produce()` remembered under it.

        `produce` runs outside the lock: it is a network call, and holding a
        lock across one would serialise every request in the process.
        """
        now = time.time()
        with self._lock:
            found = self._values.get(key)
            if found and found[0] > now:
                self.hits += 1
                return found[1]
            self.misses += 1

        value = produce()

        with self._lock:
            self._values[key] = (time.time() + self.seconds, value)
            if len(self._values) > 256:  # a tiny cache cannot become a leak
                self._prune(time.time())
        return value

    def clear(self) -> None:
        """Forget everything. Called when the thing cached has just changed."""
        with self._lock:
            self._values.clear()

    def _prune(self, now: float) -> None:
        for key in [k for k, (expires, _) in self._values.items() if expires <= now]:
            del self._values[key]
        if len(self._values) > 256:  # still full of live entries: start again
            self._values.clear()
