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


# get/put, for values that are only worth keeping once they exist - a whole
# answer is cached only if it turns out to be the kind that is the same for
# everyone, which is not known until it has been written.


def test_a_missing_value_is_none():
    assert TimedCache(60).get("never stored") is None


def test_what_was_put_comes_back():
    c = TimedCache(60)
    c.put("what does p0420 mean", "the catalytic converter is worn")
    assert c.get("what does p0420 mean") == "the catalytic converter is worn"


def test_a_stale_put_value_is_gone():
    c = TimedCache(0.05)
    c.put("k", "old")
    time.sleep(0.06)
    assert c.get("k") is None


def test_putting_again_replaces():
    c = TimedCache(60)
    c.put("k", "first")
    c.put("k", "second")
    assert c.get("k") == "second"


def test_hits_and_misses_are_counted_for_both_ways_in():
    c = TimedCache(60)
    c.get("k")
    c.put("k", "v")
    c.get("k")
    assert (c.hits, c.misses) == (1, 1)


def test_put_cannot_grow_without_limit():
    c = TimedCache(60)
    for i in range(400):
        c.put(i, i)
    assert len(c._values) <= 256
