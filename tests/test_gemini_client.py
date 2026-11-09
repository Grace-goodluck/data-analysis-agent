"""Ported from website_audit_agent: same cache, pacer and retry rules (decision 3; continued 1–2)."""

import threading

import httpx
import pytest

from fakes import FakeClock
from data_analysis_agent.config import GEMINI_TIMEOUT_S, WEB_GEMINI_RETRY_WAITS_S
from data_analysis_agent.gemini_client import (
    DiskCache,
    GeminiLLM,
    MemoryCache,
    call_with_timeout,
    is_retryable,
)
from data_analysis_agent.pacing import Pacer
from data_analysis_agent.ports import LLMError


class RecordingCall:
    def __init__(self, reply: object = "reply") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _llm(cache, call, clock=None, model="gemini-test", **kwargs):
    clock = clock or FakeClock()
    pacer = Pacer(13.0, clock.clock, clock.sleep)
    llm = GeminiLLM(model, call, cache, pacer, sleep=clock.sleep, clock=clock.clock, wait=clock.wait, **kwargs)
    return llm, clock


def test_cache_hit_skips_api_and_pacer(tmp_path):
    call = RecordingCall("hello")
    llm, clock = _llm(DiskCache(tmp_path), call)

    assert llm.generate("p") == "hello"
    assert llm.generate("p") == "hello"
    assert call.prompts == ["p"]  # second answer came from disk
    assert clock.sleeps == []


def test_disk_cache_survives_a_new_client(tmp_path):
    first, _ = _llm(DiskCache(tmp_path), RecordingCall("cached"))
    first.generate("p")

    second_call = RecordingCall("fresh")
    second, _ = _llm(DiskCache(tmp_path), second_call)
    assert second.generate("p") == "cached"
    assert second_call.prompts == []


def test_cache_keyed_by_model_and_full_prompt(tmp_path):
    for cache in (DiskCache(tmp_path), MemoryCache()):
        cache.put("model-a", "prompt", "A")
        assert cache.get("model-a", "prompt") == "A"
        assert cache.get("model-b", "prompt") is None
        assert cache.get("model-a", "prompt ") is None


def test_memory_cache_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    call = RecordingCall("hello")
    llm, _ = _llm(MemoryCache(), call)
    assert llm.generate("p") == "hello"
    assert llm.generate("p") == "hello"
    assert call.prompts == ["p"]
    assert list(tmp_path.iterdir()) == []


def test_memory_cache_forgets_oldest_first():
    cache = MemoryCache(max_entries=2)
    cache.put("m", "a", "A")
    cache.put("m", "b", "B")
    cache.put("m", "c", "C")
    assert len(cache) == 2
    assert cache.get("m", "a") is None
    assert cache.get("m", "c") == "C"


def test_fresh_calls_spaced_at_least_13s():
    llm, clock = _llm(MemoryCache(), RecordingCall())
    llm.generate("one")
    clock.t += 5.0
    llm.generate("two")
    llm.generate("three")
    assert clock.sleeps == [pytest.approx(8.0), pytest.approx(13.0)]


def test_sdk_failure_raises_llm_error_and_is_not_cached(tmp_path):
    llm, _ = _llm(DiskCache(tmp_path), RecordingCall(RuntimeError("network down")))
    with pytest.raises(LLMError, match="network down"):
        llm.generate("p")
    assert DiskCache(tmp_path).get("gemini-test", "p") is None


# --- retries ----------------------------------------------------------------------

class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError: an int .code and a 'CODE STATUS. {...}' message."""

    def __init__(self, code: int, status: str) -> None:
        super().__init__(f"{code} {status}. {{'error': {{'code': {code}}}}}")
        self.code = code


class ScriptedCall:
    """Raises or returns each scripted outcome in turn; the last one repeats."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_503_retried_twice_then_reported(tmp_path):
    call = ScriptedCall([FakeAPIError(503, "UNAVAILABLE")])
    llm, clock = _llm(DiskCache(tmp_path), call)
    with pytest.raises(LLMError, match=r"Gemini call failed: 503 UNAVAILABLE"):
        llm.generate("p")
    assert call.calls == 3  # first try + 2 retries
    assert clock.sleeps == [30.0, 60.0]
    assert DiskCache(tmp_path).get("gemini-test", "p") is None


def test_web_retries_once_after_10s():
    call = ScriptedCall([FakeAPIError(503, "UNAVAILABLE")])
    llm, clock = _llm(MemoryCache(), call, retry_waits=WEB_GEMINI_RETRY_WAITS_S)
    with pytest.raises(LLMError):
        llm.generate("p")
    assert call.calls == 2  # first try + 1 retry
    assert clock.sleeps == [pytest.approx(10.0), pytest.approx(3.0)]  # 10 s wait, then the pacer's remaining 3 s


def test_429_not_retried():
    call = ScriptedCall([FakeAPIError(429, "RESOURCE_EXHAUSTED")])
    llm, clock = _llm(MemoryCache(), call)
    with pytest.raises(LLMError, match="429 RESOURCE_EXHAUSTED"):
        llm.generate("p")
    assert call.calls == 1
    assert clock.sleeps == []


def test_success_on_retry_is_used_and_cached(tmp_path):
    call = ScriptedCall([FakeAPIError(503, "UNAVAILABLE"), "ok"])
    llm, clock = _llm(DiskCache(tmp_path), call)
    assert llm.generate("p") == "ok"
    assert call.calls == 2
    assert clock.sleeps == [30.0]
    assert DiskCache(tmp_path).get("gemini-test", "p") == "ok"


def test_503_detected_from_message_without_code():
    call = ScriptedCall([Exception("503 UNAVAILABLE. high demand"), "ok"])
    llm, _ = _llm(MemoryCache(), call)
    assert llm.generate("p") == "ok"
    assert call.calls == 2


@pytest.mark.parametrize(
    "exc",
    [
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadTimeout("timed out"),
        ConnectionResetError("reset by peer"),
    ],
    ids=["server-disconnected", "read-timeout", "connection-reset"],
)
def test_dropped_connection_or_timeout_is_retried(exc):
    call = ScriptedCall([exc, "ok"])
    llm, clock = _llm(MemoryCache(), call)
    assert llm.generate("p") == "ok"
    assert call.calls == 2
    assert clock.sleeps == [30.0]


def test_dropped_connection_retried_twice_then_reported():
    call = ScriptedCall([httpx.RemoteProtocolError("Server disconnected without sending a response.")])
    llm, clock = _llm(MemoryCache(), call)
    with pytest.raises(LLMError, match="Server disconnected"):
        llm.generate("p")
    assert call.calls == 3
    assert clock.sleeps == [30.0, 60.0]


# --- timeout (continued 5) ---------------------------------------------------------------

class NeverAnswers:
    """A Gemini call that hangs until the test releases it. The injected clock makes it time out
    without real waiting; releasing it at the end lets the abandoned worker threads finish."""

    def __init__(self, then: list[str] | None = None) -> None:
        self.release = threading.Event()
        self.calls = 0
        self.then = list(then or [])  # replies for the calls after the first (hanging) one

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if self.calls > 1 and self.then:
            return self.then.pop(0)
        self.release.wait()
        return "too late"


@pytest.fixture
def never_answers():
    calls = []

    def make(then=None):
        call = NeverAnswers(then)
        calls.append(call)
        return call

    yield make
    for call in calls:
        call.release.set()


def test_timeout_after_60s_is_retried_twice_on_the_command_line(tmp_path, never_answers):
    call = never_answers()
    llm, clock = _llm(DiskCache(tmp_path), call)
    start = clock.t
    with pytest.raises(LLMError, match="no reply from Gemini within 60 s"):
        llm.generate("p")
    assert call.calls == 3  # first try + 2 retries, as for a dropped connection
    assert clock.waits == [60.0, 60.0, 60.0]  # each attempt waited its full 60 s
    assert clock.sleeps == [30.0, 60.0]  # the command line's retry waits
    assert clock.t - start == pytest.approx(3 * 60 + 30 + 60)
    assert DiskCache(tmp_path).get("gemini-test", "p") is None


def test_timeout_on_the_web_is_retried_once(never_answers):
    call = never_answers()
    llm, clock = _llm(MemoryCache(), call, retry_waits=WEB_GEMINI_RETRY_WAITS_S)
    with pytest.raises(LLMError, match="no reply from Gemini within 60 s"):
        llm.generate("p")
    assert call.calls == 2  # first try + 1 retry
    assert clock.waits == [60.0, 60.0]
    assert clock.sleeps == [10.0]


def test_answer_after_a_timeout_is_used_and_cached(never_answers):
    call = never_answers(then=["ok"])
    cache = MemoryCache()
    llm, clock = _llm(cache, call)
    assert llm.generate("p") == "ok"
    assert call.calls == 2
    assert clock.waits == [60.0]
    assert clock.sleeps == [30.0]
    assert cache.get("gemini-test", "p") == "ok"


def test_timeout_is_a_dropped_connection():
    assert is_retryable(TimeoutError("no reply from Gemini within 60 s"))


def test_server_side_deadline_504_is_retried_like_a_timeout():
    """Seen in the live run of 2026-11-09: the SDK sends our 60 s as the request deadline and the
    server answers 504 DEADLINE_EXCEEDED before our own timeout fires."""
    call = ScriptedCall([FakeAPIError(504, "DEADLINE_EXCEEDED")])
    llm, clock = _llm(MemoryCache(), call)
    with pytest.raises(LLMError, match="504 DEADLINE_EXCEEDED"):
        llm.generate("p")
    assert call.calls == 3
    assert clock.sleeps == [30.0, 60.0]


def test_504_detected_from_message_without_code():
    assert is_retryable(Exception("504 DEADLINE_EXCEEDED. {'error': {'code': 504}}"))
    assert not is_retryable(Exception("429 RESOURCE_EXHAUSTED"))
    assert not is_retryable(Exception("400 INVALID_ARGUMENT"))


def test_default_timeout_is_60s():
    assert GEMINI_TIMEOUT_S == 60.0


def test_quick_answer_is_not_affected_by_the_timeout():
    call = RecordingCall("fast")
    assert call_with_timeout(call, "p", 60.0) == "fast"  # real clock: returns at once


def test_errors_inside_the_call_come_back_unchanged():
    with pytest.raises(RuntimeError, match="boom"):
        call_with_timeout(RecordingCall(RuntimeError("boom")), "p", 60.0)


def test_empty_reply_raises_llm_error():
    llm, _ = _llm(MemoryCache(), RecordingCall("   "))
    with pytest.raises(LLMError):
        llm.generate("p")
