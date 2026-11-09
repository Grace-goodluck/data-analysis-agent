"""Demo limits (decision 5), in memory: they reset when the service restarts.

- 3 analyses per visitor per rolling hour (visitor = first X-Forwarded-For IP).
- 20 analyses per day in total — the real protection, and the cap on Gemini calls
  (continued 3: no separate summary cap).
"Daily" resets at midnight Pacific time, matching Gemini's quota.
"""

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class Limits:
    analyses_per_visitor_per_hour: int = 3
    analyses_per_day: int = 20


@dataclass(frozen=True)
class Refusal:
    message: str
    retry_after_s: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def seconds_until_pacific_midnight(now: datetime) -> int:
    local = now.astimezone(PACIFIC)
    midnight = datetime.combine(local.date() + timedelta(days=1), time(0), tzinfo=PACIFIC)
    # Subtract in UTC: aware datetimes sharing a tzinfo subtract by wall clock, ignoring DST.
    return max(1, math.ceil((midnight.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()))


class Limiter:
    def __init__(self, limits: Limits = Limits(), now: Callable[[], datetime] = utc_now) -> None:
        self.limits = limits
        self._now = now
        self._lock = threading.Lock()
        self._visits: dict[str, deque[datetime]] = {}
        self._day: date | None = None
        self._today = 0

    def _roll_day(self, now: datetime) -> None:  # caller holds the lock
        today = now.astimezone(PACIFIC).date()
        if today != self._day:
            self._day = today
            self._today = 0
            self._visits = {v: q for v, q in self._visits.items() if q and now - q[-1] < HOUR}

    def try_start(self, visitor: str) -> Refusal | None:
        """None: allowed and counted. Refusal: not counted, with a message for the visitor."""
        with self._lock:
            now = self._now()
            self._roll_day(now)
            visits = self._visits.setdefault(visitor, deque())
            while visits and now - visits[0] >= HOUR:
                visits.popleft()

            if len(visits) >= self.limits.analyses_per_visitor_per_hour:
                wait = max(1, math.ceil((visits[0] + HOUR - now).total_seconds()))
                return Refusal(
                    f"You have used your {self.limits.analyses_per_visitor_per_hour} analyses for this hour. "
                    f"Try again in about {math.ceil(wait / 60)} minutes.",
                    wait,
                )
            if self._today >= self.limits.analyses_per_day:
                return Refusal(
                    f"The demo has reached its limit of {self.limits.analyses_per_day} analyses for today. "
                    "It resets at midnight Pacific time.",
                    seconds_until_pacific_midnight(now),
                )
            visits.append(now)
            self._today += 1
            return None
