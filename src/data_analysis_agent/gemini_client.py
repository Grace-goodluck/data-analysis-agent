"""Gemini as an LLM: cache first, then a paced fresh call.

Same design as website_audit_agent (SCOPE decision 3):
- DiskCache (command line): one JSON file per reply in results/cache/, keyed by
  sha256(model + full prompt). Reruns with the same prompt are free.
- MemoryCache (web demo, continued 2): the same, in memory only — nothing is written to disk,
  because the prompt holds column names and category labels from the upload.
- Pacer: every attempt, retries included, waits its turn. Cache hits never wait.
- Timeout (continued 5): each attempt gets 60 s. No reply in time raises TimeoutError, which
  counts as a dropped connection and follows the same retry rules. The SDK's own HTTP timeout
  is set to the same 60 s, so an abandoned call also ends at the network level.
- Retry: a 503, a dropped connection or a timeout is retried (command line: 30 s, then 60 s;
  web: once after 10 s); a 429 quota error is never retried.
- make_gemini_call: the only place that touches the google-genai SDK.
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import httpx

from data_analysis_agent.config import GEMINI_RETRY_WAITS_S, GEMINI_TIMEOUT_S
from data_analysis_agent.pacing import Pacer
from data_analysis_agent.ports import LLMError


def _status_code(exc: BaseException) -> int | None:
    """The SDK's APIError carries an int .code; otherwise fall back to its message,
    which starts with the status code ("503 UNAVAILABLE...")."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    head = str(exc).lstrip()[:3]
    return int(head) if head.isdigit() else None


def is_unavailable(exc: BaseException) -> bool:
    """A Gemini 503."""
    return _status_code(exc) == 503


def is_deadline_exceeded(exc: BaseException) -> bool:
    """A Gemini 504 DEADLINE_EXCEEDED: the server enforcing our 60 s timeout (the SDK sends it
    as the request deadline), so it counts as a timeout (continued 5)."""
    return _status_code(exc) == 504


# A dropped connection or timeout as it comes out of the SDK (which calls httpx),
# plus the built-in equivalents.
_DROPPED = (
    ConnectionError,
    TimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def is_dropped_connection(exc: BaseException) -> bool:
    return isinstance(exc, _DROPPED)


def is_retryable(exc: BaseException) -> bool:
    """503, dropped connection or timeout (ours, or the server's 504). Never a 429."""
    return is_unavailable(exc) or is_deadline_exceeded(exc) or is_dropped_connection(exc)


class Cache(Protocol):
    def get(self, model: str, prompt: str) -> str | None: ...

    def put(self, model: str, prompt: str, response: str) -> None: ...


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(model: str, prompt: str) -> str:
        blob = json.dumps({"model": model, "prompt": prompt}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, model: str, prompt: str) -> Path:
        return self.directory / f"{self.key(model, prompt)}.json"

    def get(self, model: str, prompt: str) -> str | None:
        path = self._path(model, prompt)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        # Guard against a (vanishingly unlikely) hash collision or a hand-edited file.
        if entry.get("model") != model or entry.get("prompt") != prompt:
            return None
        response = entry.get("response")
        return response if isinstance(response, str) else None

    def put(self, model: str, prompt: str, response: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(model, prompt)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"model": model, "prompt": prompt, "response": response}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)


class MemoryCache:
    """In-memory only, oldest entries forgotten first; cleared when the service restarts."""

    def __init__(self, max_entries: int = 200) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model: str, prompt: str) -> str | None:
        with self._lock:
            return self._entries.get((model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._entries[(model, prompt)] = response
            self._entries.move_to_end((model, prompt))
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _event_wait(event: threading.Event, seconds: float) -> None:
    event.wait(seconds)


def call_with_timeout(
    call: Callable[[str], str],
    prompt: str,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[threading.Event, float], None] = _event_wait,
) -> str:
    """Run call(prompt) on a worker thread; raise TimeoutError if it has not answered by the deadline.

    `wait(done, seconds)` blocks until the call finishes or `seconds` pass; tests inject a fake
    clock and wait so a call that never answers times out without real waiting. A timed-out call's
    thread is a daemon and is abandoned (the SDK's HTTP timeout ends it soon after).
    """
    done = threading.Event()
    outcome: dict[str, object] = {}

    def work() -> None:
        try:
            outcome["text"] = call(prompt)
        except BaseException as exc:  # handed back to the caller below
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=work, name="gemini-call", daemon=True).start()
    deadline = clock() + timeout_s
    while not done.is_set():
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(f"no reply from Gemini within {timeout_s:g} s")
        wait(done, remaining)
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["text"]  # type: ignore[return-value]


class GeminiLLM:
    def __init__(
        self,
        model: str,
        call: Callable[[str], str],
        cache: Cache,
        pacer: Pacer,
        retry_waits: Sequence[float] = GEMINI_RETRY_WAITS_S,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = GEMINI_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[threading.Event, float], None] = _event_wait,
    ) -> None:
        self.model = model
        self._call = call
        self._cache = cache
        self._pacer = pacer
        self._retry_waits = tuple(retry_waits)
        self._sleep = sleep
        self._timeout_s = timeout_s
        self._clock = clock
        self._wait = wait

    def generate(self, prompt: str) -> str:
        cached = self._cache.get(self.model, prompt)
        if cached is not None:
            return cached

        waits = iter(self._retry_waits)
        while True:
            self._pacer.wait_turn()  # every attempt, retries included, goes through the pacer
            try:
                text = call_with_timeout(self._call, prompt, self._timeout_s, self._clock, self._wait)
                break
            except Exception as exc:  # SDK raises many types; all mean "this call failed"
                wait = next(waits, None) if is_retryable(exc) else None
                if wait is None:  # not retryable, or retries used up
                    raise LLMError(f"Gemini call failed: {exc}") from exc
                self._sleep(wait)
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Gemini returned an empty reply")

        self._cache.put(self.model, prompt, text)
        return text


def make_gemini_call(model: str, api_key: str, timeout_s: float = GEMINI_TIMEOUT_S) -> Callable[[str], str]:
    from google import genai  # imported here so tests never need the SDK
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))  # ms

    def call(prompt: str) -> str:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text or ""

    return call
