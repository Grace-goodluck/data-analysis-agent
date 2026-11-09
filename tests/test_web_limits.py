"""Decision 5: 3 analyses per visitor per hour, 20 per day in total, reset at Pacific midnight."""

from datetime import datetime, timezone

import pytest

from fakes import FakeNow
from data_analysis_agent.web.limits import Limiter, Limits, seconds_until_pacific_midnight

# 2026-11-08 is after the 2026-11-01 DST change: Pacific midnight is 08:00 UTC.
JUST_BEFORE_PACIFIC_MIDNIGHT = datetime(2026, 11, 9, 7, 59, tzinfo=timezone.utc)


def test_defaults_match_scope():
    assert Limits() == Limits(analyses_per_visitor_per_hour=3, analyses_per_day=20)


def test_three_analyses_per_visitor_per_rolling_hour():
    now = FakeNow()
    limiter = Limiter(now=now)
    for _ in range(3):
        assert limiter.try_start("203.0.113.7") is None
        now.advance(minutes=10)

    refusal = limiter.try_start("203.0.113.7")
    assert refusal is not None and "3 analyses for this hour" in refusal.message
    assert refusal.retry_after_s == 30 * 60  # first analysis was 30 min ago

    assert limiter.try_start("198.51.100.2") is None  # another visitor is unaffected

    now.advance(minutes=30)  # the first analysis is now an hour old
    assert limiter.try_start("203.0.113.7") is None


def test_refused_attempts_are_not_counted():
    limiter = Limiter(Limits(analyses_per_visitor_per_hour=1, analyses_per_day=2), now=FakeNow())
    assert limiter.try_start("a") is None
    for _ in range(5):
        assert limiter.try_start("a") is not None
    assert limiter.try_start("b") is None  # the daily total only counted 2


def test_twenty_analyses_per_day_in_total():
    limiter = Limiter(now=FakeNow())
    for i in range(20):
        assert limiter.try_start(f"visitor-{i}") is None
    refusal = limiter.try_start("visitor-new")
    assert refusal is not None and "limit of 20 analyses for today" in refusal.message
    assert "midnight Pacific" in refusal.message


def test_daily_total_resets_at_pacific_midnight_not_utc_midnight():
    now = FakeNow(datetime(2026, 11, 8, 23, 0, tzinfo=timezone.utc))  # 15:00 Pacific
    limiter = Limiter(Limits(analyses_per_visitor_per_hour=100, analyses_per_day=1), now=now)
    assert limiter.try_start("a") is None
    assert limiter.try_start("a") is not None

    now.advance(hours=2)  # 01:00 UTC on the 9th, still the 8th in Pacific time
    assert limiter.try_start("a") is not None

    now.t = JUST_BEFORE_PACIFIC_MIDNIGHT
    refusal = limiter.try_start("a")
    assert refusal is not None and refusal.retry_after_s == 60

    now.advance(minutes=2)  # 00:01 Pacific on the 9th
    assert limiter.try_start("a") is None


@pytest.mark.parametrize(
    "utc, seconds",
    [
        (datetime(2026, 11, 9, 7, 59, tzinfo=timezone.utc), 60),  # PST, UTC-8
        (datetime(2026, 7, 1, 6, 59, tzinfo=timezone.utc), 60),  # PDT, UTC-7
        (datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc), 23 * 3600),  # DST starts that day: a 23-hour day
    ],
)
def test_seconds_until_pacific_midnight(utc, seconds):
    assert seconds_until_pacific_midnight(utc) == seconds
