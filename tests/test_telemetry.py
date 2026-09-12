"""Tests for the telemetry layer.

The contract: with no connection string configured, everything here is a no-op
and the application behaves identically. Tests, CI and local development must
never need an Azure resource, and a monitoring outage must never take the
service down.
"""

import pytest

from services.orchestrator import telemetry


@pytest.fixture(autouse=True)
def telemetry_off(monkeypatch):
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.setattr(telemetry, "_ENABLED", False)
    monkeypatch.setattr(telemetry, "_tracer", None)
    yield


def test_setup_is_off_without_a_connection_string():
    assert telemetry.setup() is False
    assert telemetry.enabled() is False


def test_setup_never_raises_on_a_bad_connection_string(monkeypatch):
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "this is not valid")
    assert telemetry.setup() is False  # logged, not raised


def test_span_yields_none_when_off():
    with telemetry.span("x", agent="diagnostics") as s:
        assert s is None


def test_set_tolerates_a_none_span():
    telemetry.set(None, tokens=5, agent="diagnostics")  # must not raise


def test_exceptions_still_propagate():
    with pytest.raises(ValueError):
        with telemetry.span("x"):
            raise ValueError("expected")


def test_attribute_keys_are_namespaced():
    assert telemetry._key("agent") == "autoassist.agent"
    assert telemetry._key("autoassist.agent") == "autoassist.agent"


@pytest.mark.parametrize("value,expected", [
    ("text", "text"),
    (True, True),
    (42, 42),
    (3.5, 3.5),
    (["a", 1], ["a", "1"]),
    (("a", "b"), ["a", "b"]),
])
def test_values_are_coerced_to_types_otel_accepts(value, expected):
    assert telemetry._value(value) == expected


def test_unsupported_values_become_strings():
    assert isinstance(telemetry._value({"a": 1}), str)
    assert isinstance(telemetry._value(None), str)
