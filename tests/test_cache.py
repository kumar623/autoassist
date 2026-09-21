"""Tests for the small time-limited cache."""

import pytest

from services.orchestrator import cache
from services.orchestrator.cache import TimedCache


@pytest.fixture
def clock(monkeypatch):
    """The cache's clock, moved by hand rather than slept through."""
    now = [1000.0]
    monkeypatch.setattr(cache, "_now", lambda: now[0])
    return now


def test_the_second_ask_does_not_call_again():
    calls = []
    c = TimedCache(60)
    for _ in range(3):
        assert c.get_or_call("monday", lambda: calls.append(1) or "09:00, 10:30") == "09:00, 10:30"
    assert len(calls) == 1


def test_different_keys_are_different_answers():
    c = TimedCache(60)
    assert c.get_or_call("monday", lambda: "a") == "a"
    assert c.get_or_call("tuesday", lambda: "b") == "b"


def test_a_stale_value_is_fetched_again(clock):
    c = TimedCache(60)
    assert c.get_or_call("k", lambda: "old") == "old"
    clock[0] += 59
    assert c.get_or_call("k", lambda: "new") == "old", "still fresh a second before it expires"
    clock[0] += 2
    assert c.get_or_call("k", lambda: "new") == "new"


def test_an_empty_value_is_still_a_hit():
    """An empty slot list is an answer - "nothing free that day" - not a miss."""
    calls = []
    c = TimedCache(60)
    for _ in range(3):
        assert c.get_or_call("sunday", lambda: calls.append(1) or []) == []
    assert len(calls) == 1


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
    assert len(c._values) <= cache.MAX_ENTRIES


# get/put, for values that are only worth keeping once they exist - a whole
# answer is cached only if it turns out to be the kind that is the same for
# everyone, which is not known until it has been written.


def test_a_missing_value_is_none():
    assert TimedCache(60).get("never stored") is None


def test_what_was_put_comes_back():
    c = TimedCache(60)
    c.put("what does p0420 mean", "the catalytic converter is worn")
    assert c.get("what does p0420 mean") == "the catalytic converter is worn"


def test_a_stale_put_value_is_gone(clock):
    c = TimedCache(60)
    c.put("k", "old")
    clock[0] += 61
    assert c.get("k") is None


def test_putting_again_replaces():
    c = TimedCache(60)
    c.put("k", "first")
    c.put("k", "second")
    assert c.get("k") == "second"


def test_put_cannot_grow_without_limit():
    c = TimedCache(60)
    for i in range(400):
        c.put(i, i)
    assert len(c._values) <= cache.MAX_ENTRIES
