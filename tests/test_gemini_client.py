"""Ported from website_audit_agent: same cache, pacer and retry rules (decision 3; continued 1–2)."""

import httpx
import pytest

from fakes import FakeClock
from data_analysis_agent.config import WEB_GEMINI_RETRY_WAITS_S
from data_analysis_agent.gemini_client import DiskCache, GeminiLLM, MemoryCache
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
    return GeminiLLM(model, call, cache, pacer, sleep=clock.sleep, **kwargs), clock


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


def test_empty_reply_raises_llm_error():
    llm, _ = _llm(MemoryCache(), RecordingCall("   "))
    with pytest.raises(LLMError):
        llm.generate("p")
