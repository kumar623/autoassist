"""Tests for the small time-limited cache."""

import time

from services.orchestrator.cache import TimedCache


def test_the_second_ask_does_not_call_again():
    calls = []
    c = TimedCache(60)
    for _ in range(3):
        assert c.get_or_call("monday", lambda: calls.append(1) or "09:00, 10:30") == "09:00, 10:30"
    assert len(calls) == 1
    assert (c.hits, c.misses) == (2, 1)


def test_different_keys_are_different_answers():
    c = TimedCache(60)
    assert c.get_or_call("monday", lambda: "a") == "a"
    assert c.get_or_call("tuesday", lambda: "b") == "b"


def test_a_stale_value_is_fetched_again():
    c = TimedCache(0.05)
    assert c.get_or_call("k", lambda: "old") == "old"
    time.sleep(0.06)
    assert c.get_or_call("k", lambda: "new") == "new"


def test_clearing_forgets_everything():
    """Called after a booking: the slot just taken must not still look free."""
    c = TimedCache(60)
    c.get_or_call("monday", lambda: "09:00, 10:30")
    c.clear()
    assert c.get_or_call("monday", lambda: "10:30 only") == "10:30 only"


def test_it_cannot_grow_without_limit():
    c = TimedCache(60)
    for i in range(400):
        c.get_or_call(i, lambda n=i: n)
    assert len(c._values) <= 256
