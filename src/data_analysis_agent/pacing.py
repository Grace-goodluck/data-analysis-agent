"""Start-to-start spacing between outgoing Gemini calls (same design as website_audit_agent).

One analysis makes one Gemini call, so in practice this only spaces retries.
Cache hits make no call and never wait.
"""

import time
from collections.abc import Callable

from data_analysis_agent.config import MIN_CALL_SPACING_S


class Pacer:
    """Start-to-start spacing between outgoing calls."""

    def __init__(
        self,
        min_interval: float = MIN_CALL_SPACING_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None

    def wait_turn(self) -> None:
        if self._last_start is not None:
            remaining = self._last_start + self.min_interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last_start = self._clock()
