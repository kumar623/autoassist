"""Tests for the rate limit and the concurrency cap.

No clock is patched: the windows are driven by passing `now` in, which is what
`Visitors.record` takes it for. A test that sleeps for an hour is not a test.
"""

import threading

import pytest

from services.orchestrator import limits

# ------------------------------------------------------------- settings


def test_a_setting_that_is_not_a_number_does_not_stop_the_app_starting(monkeypatch):
    """This is read at import, in a module app.py imports. A ValueError here is
    a container that dies before it serves anything."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "off")
    assert limits.setting("RATE_LIMIT_PER_MINUTE", 10) == 10


def test_a_negative_setting_means_off_not_nonsense(monkeypatch):
    """The obvious way to guess 'no limit' is -1, and it must not be an error."""
    monkeypatch.setenv("MAX_CONCURRENT_CHATS", "-1")
    assert limits.setting("MAX_CONCURRENT_CHATS", 20) == 0


@pytest.mark.parametrize("raw,expected", [("0", 0), ("5", 5), (" 7 ", 7), ("", 10), ("3.9", 3)])
def test_settings_are_read_as_whole_numbers(monkeypatch, raw, expected):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", raw)
    assert limits.setting("RATE_LIMIT_PER_MINUTE", 10) == expected


def test_the_concurrency_cap_sits_above_the_ingress_scale_threshold():
    """Container Apps has no scale rule, so Envoy's default applies: a new
    replica past 10 concurrent requests each. A cap below that would refuse the
    eleventh customer and keep measured concurrency under the threshold, so the
    app would never scale out - the cap would silence the autoscaler."""
    assert limits.MAX_IN_FLIGHT > 10


# ------------------------------------------------------- naming the visitor


def test_the_same_visitor_gets_the_same_digest():
    assert limits.digest("20.1.2.3") == limits.digest("20.1.2.3")


def test_the_digest_is_not_the_address():
    """Enough to group refusals in the logs; not an address in Log Analytics."""
    assert "20.1.2.3" not in limits.digest("20.1.2.3")
    assert limits.digest("20.1.2.3") != limits.digest("20.1.2.4")


# ----------------------------------------------------------- who is asking


class Headers(dict):
    """Just enough of Starlette's headers: case-insensitive get."""

    def get(self, key, default=None):
        return super().get(key.lower(), default)


def test_the_visitor_is_the_address_azure_added():
    """Envoy appends the address it accepted; everything left of it is hearsay."""
    key = limits.visitor_key(Headers({"x-forwarded-for": "9.9.9.9, 20.1.2.3"}), "10.0.0.1")
    assert key == "20.1.2.3"


def test_a_forged_header_does_not_buy_a_new_identity():
    """Taking the leftmost entry - the common example - would count nothing: a
    visitor could send a different one on every request."""
    first = limits.visitor_key(Headers({"x-forwarded-for": "1.1.1.1, 20.1.2.3"}), None)
    second = limits.visitor_key(Headers({"x-forwarded-for": "2.2.2.2, 20.1.2.3"}), None)
    assert first == second == "20.1.2.3"


def test_without_the_header_the_peer_is_used():
    assert limits.visitor_key(Headers(), "10.0.0.7") == "10.0.0.7"


def test_an_unknown_caller_still_has_a_key():
    """They all share one, which is the safe direction: they share one budget."""
    assert limits.visitor_key(Headers(), None) == "unknown"


@pytest.mark.parametrize("raw,expected", [
    ("20.1.2.3:41234", "20.1.2.3"),
    ("[2001:db8::1]:443", "2001:db8::1"),
    ("2001:db8::1", "2001:db8::1"),
])
def test_the_port_is_not_part_of_the_visitor(raw, expected):
    assert limits.visitor_key(Headers({"x-forwarded-for": raw}), None) == expected


# ------------------------------------------------------------- the windows


def test_messages_under_the_limit_are_let_through():
    v = limits.Visitors(per_minute=3, per_hour=100)
    assert [v.record("a", now=1000 + i) for i in range(3)] == [0, 0, 0]


def test_the_minute_limit_says_how_long_to_wait():
    v = limits.Visitors(per_minute=2, per_hour=100)
    v.record("a", now=1000)
    v.record("a", now=1001)
    wait = v.record("a", now=1002)
    assert wait == 58, "the older of the two leaves the window 60s after it arrived"


def test_the_window_slides():
    v = limits.Visitors(per_minute=2, per_hour=100)
    v.record("a", now=1000)
    v.record("a", now=1001)
    assert v.record("a", now=1061) == 0, "the first one is over a minute old"


def test_the_hour_limit_holds_when_the_minute_limit_does_not():
    """Five a minute for twelve minutes is sixty, and then that is the hour gone."""
    v = limits.Visitors(per_minute=10, per_hour=5)
    for i in range(5):
        assert v.record("a", now=1000 + i * 120) == 0
    assert v.record("a", now=1000 + 5 * 120) > 0


def test_visitors_are_counted_separately():
    v = limits.Visitors(per_minute=1, per_hour=10)
    assert v.record("a", now=1000) == 0
    assert v.record("b", now=1000) == 0, "b is not a"
    assert v.record("a", now=1000) > 0


def test_a_limit_of_zero_is_off():
    v = limits.Visitors(per_minute=0, per_hour=0)
    assert not v.enabled
    assert [v.record("a", now=1000) for _ in range(100)] == [0] * 100


def test_only_one_of_the_two_limits_need_be_set():
    v = limits.Visitors(per_minute=0, per_hour=2)
    assert v.record("a", now=1000) == 0
    assert v.record("a", now=1001) == 0
    assert v.record("a", now=1002) > 0


def test_a_refused_message_is_not_counted_again():
    """Otherwise a visitor who keeps retrying pushes their own window out for ever."""
    v = limits.Visitors(per_minute=1, per_hour=10)
    v.record("a", now=1000)
    for _ in range(5):
        v.record("a", now=1010)
    assert v.record("a", now=1061) == 0, "still just the one message in the hour"


def test_the_table_of_visitors_is_bounded(monkeypatch):
    """The key comes from a header. A visitor forging a new one per request must
    not be able to grow this without end."""
    monkeypatch.setattr(limits, "MAX_VISITORS", 10)
    v = limits.Visitors(per_minute=100, per_hour=100)
    for i in range(50):
        v.record(f"forged-{i}", now=1000)
    assert len(v._seen) <= limits.MAX_VISITORS + 1, "the shared bucket is the one extra"


def test_overflow_visitors_share_a_budget_and_regulars_keep_theirs(monkeypatch):
    monkeypatch.setattr(limits, "MAX_VISITORS", 2)
    v = limits.Visitors(per_minute=2, per_hour=100)
    v.record("regular", now=1000)          # tracked first, keeps its own window
    v.record("someone-else", now=1000)     # table is now full
    assert v.record("forged-1", now=1000) == 0
    assert v.record("forged-2", now=1000) == 0
    assert v.record("forged-3", now=1000) > 0, "the forgers throttle each other"
    assert v.record("regular", now=1000) == 0, "and not the visitor already counted"


def test_expired_visitors_are_dropped_before_the_shared_bucket(monkeypatch):
    monkeypatch.setattr(limits, "MAX_VISITORS", 2)
    v = limits.Visitors(per_minute=10, per_hour=10)
    v.record("yesterday", now=1000)
    v.record("also-yesterday", now=1000)
    v.record("today", now=1000 + 2 * limits.HOUR)
    assert "today" in v._seen
    assert limits.SHARED_BUCKET not in v._seen


# ------------------------------------------------------------- in flight


def test_in_flight_is_capped_and_released():
    f = limits.InFlight(limit=2)
    assert f.take() and f.take()
    assert not f.take()
    f.release()
    assert f.take()


def test_in_flight_zero_is_off():
    f = limits.InFlight(limit=0)
    assert all(f.take() for _ in range(100))


def test_release_never_goes_below_zero():
    f = limits.InFlight(limit=2)
    f.release()
    f.release()
    assert f.count == 0
    assert f.take()


def test_in_flight_counts_correctly_under_threads():
    f = limits.InFlight(limit=1000)
    def churn():
        for _ in range(200):
            f.take()
            f.release()
    threads = [threading.Thread(target=churn) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert f.count == 0


# ------------------------------------------------------------ both together


def test_hold_lets_a_message_through_and_gives_the_slot_back():
    lim = limits.Limits(limits.Visitors(5, 50), limits.InFlight(1))
    with lim.hold("a"):
        assert lim.in_flight.count == 1
    assert lim.in_flight.count == 0


def test_a_rate_limited_visitor_is_told_when_to_come_back():
    lim = limits.Limits(limits.Visitors(1, 50), limits.InFlight(4))
    with lim.hold("a"):
        pass
    with pytest.raises(limits.Refused) as e:
        with lim.hold("a"):
            pass
    assert e.value.status == 429
    assert e.value.reason == "rate"
    assert 0 < e.value.retry_after <= 60
    assert "wait a moment" in e.value.customer_message


def test_a_full_replica_answers_busy_not_rate_limited():
    lim = limits.Limits(limits.Visitors(50, 500), limits.InFlight(1))
    with lim.hold("a"):
        with pytest.raises(limits.Refused) as e:
            with lim.hold("b"):
                pass
    assert e.value.status == 503
    assert e.value.reason == "busy"


def test_being_refused_for_our_shortage_does_not_spend_their_budget():
    """They got nothing. Charging them for it would push them towards the rate
    limit because we were full, which is our problem and not theirs."""
    lim = limits.Limits(limits.Visitors(2, 50), limits.InFlight(1))
    with lim.hold("a"):
        for _ in range(5):
            with pytest.raises(limits.Refused):
                with lim.hold("b"):
                    pass
    with lim.hold("b"):  # b has spent nothing yet
        pass
    with lim.hold("b"):
        pass


def test_the_slot_comes_back_even_when_the_request_fails():
    lim = limits.Limits(limits.Visitors(50, 500), limits.InFlight(1))
    with pytest.raises(ValueError):
        with lim.hold("a"):
            raise ValueError("the agent blew up")
    assert lim.in_flight.count == 0


def test_a_broken_limiter_does_not_break_the_service(monkeypatch):
    """A limiter is a guard, not a dependency. If it cannot decide, it lets the
    customer through rather than taking the service down with it."""
    lim = limits.Limits(limits.Visitors(1, 1), limits.InFlight(4))
    monkeypatch.setattr(lim.visitors, "record", lambda *a, **k: 1 / 0)
    with lim.hold("a"):
        pass


def test_refusals_are_counted_for_metrics():
    lim = limits.Limits(limits.Visitors(1, 50), limits.InFlight(1))
    with lim.hold("a"):
        pass
    with pytest.raises(limits.Refused):
        with lim.hold("a"):
            pass
    assert lim.snapshot()["refused_rate"] == 1
    assert lim.snapshot()["refused_busy"] == 0
