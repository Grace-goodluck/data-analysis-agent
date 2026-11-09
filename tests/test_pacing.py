import pytest

from fakes import FakeClock
from data_analysis_agent.pacing import Pacer


def test_first_call_never_waits():
    clock = FakeClock()
    Pacer(13.0, clock.clock, clock.sleep).wait_turn()
    assert clock.sleeps == []


def test_calls_spaced_start_to_start():
    clock = FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    pacer.wait_turn()
    clock.t += 4.0
    pacer.wait_turn()
    pacer.wait_turn()
    assert clock.sleeps == [pytest.approx(9.0), pytest.approx(13.0)]


def test_no_wait_when_enough_time_passed():
    clock = FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    pacer.wait_turn()
    clock.t += 20.0
    pacer.wait_turn()
    assert clock.sleeps == []


def test_default_spacing_is_13s():
    assert Pacer().min_interval == 13.0
