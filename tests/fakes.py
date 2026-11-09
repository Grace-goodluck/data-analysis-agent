"""Fakes and fixture helpers. No test touches the network or Gemini."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
KNOWN = "known_properties.csv"


def fixture_path(name: str) -> Path:
    return FIXTURES / name


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# A summary of known_properties.csv whose every number is in the computed statistics.
GOOD_SUMMARY = (
    "The file has 24 rows and 6 columns. The age column has 3 missing values (12.5%), "
    "and there is 1 duplicate row.\n\n"
    "Visits rose from 3 in 2025-01 to 9 in 2025-04. North is the most common clinic, with 6 rows. "
    "Hours and score are strongly associated."
)
GOOD_STEPS = ["Remove the 1 duplicate row.", "Find out why 3 ages are missing.", "Check the 4 charts."]

# The same, plus a number that was never computed.
INVENTED_SUMMARY = GOOD_SUMMARY + " The average wait was 9,876 minutes."


def reply(summary: str = GOOD_SUMMARY, steps: list[str] | None = None) -> str:
    return json.dumps({"summary": summary, "next_steps": GOOD_STEPS if steps is None else steps})


class FakeLLM:
    """Returns scripted replies in order and records every prompt.

    A scripted item that is an Exception instance is raised instead.
    """

    def __init__(self, replies: list[object], model: str = "fake-model") -> None:
        self.model = model
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._replies:
            raise AssertionError(f"FakeLLM got an unexpected call #{len(self.prompts)}:\n{prompt[:300]}")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """Stands in for time.monotonic/time.sleep: sleeping advances the clock."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeNow:
    """Stands in for utc_now in the web limiter; advance() moves time on."""

    def __init__(self, start: datetime = datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **delta: float) -> None:
        self.t += timedelta(**delta)


class ImmediateExecutor:
    """Runs a web job synchronously, so tests need no threads."""

    def submit(self, fn, /, *args):
        fn(*args)
