"""Gemini summary and next steps (decision 3). Never raises: a failure becomes Summary(available=False).

A reply is used only if every number in it matches a computed value (numbers.py);
otherwise the report says "summary unavailable" and still shows all the statistics.
"""

import json
import re

from data_analysis_agent.config import MAX_NEXT_STEPS, NO_KEY_REASON
from data_analysis_agent.models import Summary
from data_analysis_agent.numbers import allowed_from, unverified
from data_analysis_agent.ports import LLM, LLMError
from data_analysis_agent.prompts import build_prompt

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def parse_reply(reply: str) -> tuple[str, list[str]]:
    """(summary, next_steps). Raises ValueError when the reply is not the expected JSON."""
    data = json.loads(_strip_fence(reply))
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    text = data.get("summary")
    if not isinstance(text, str) or not text.strip():
        raise ValueError('reply has no "summary" text')
    steps = data.get("next_steps", [])
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError('"next_steps" is not a list of strings')
    return text.strip(), [s.strip() for s in steps if s.strip()][:MAX_NEXT_STEPS]


def summarise(prompt_stats: dict, llm: LLM | None) -> Summary:
    if llm is None:
        return Summary(available=False, code="no_key", reason=NO_KEY_REASON)
    try:
        reply = llm.generate(build_prompt(prompt_stats))
    except LLMError as exc:
        return Summary(available=False, code="llm_failed", reason=str(exc))
    try:
        text, steps = parse_reply(reply)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        return Summary(available=False, code="unreadable", reason=f"Gemini reply could not be read: {exc}")

    bad = unverified([text, *steps], allowed_from(prompt_stats))
    if bad:
        return Summary(
            available=False,
            code="rejected",
            reason="summary rejected: number(s) not in the computed statistics: " + ", ".join(bad),
        )
    return Summary(available=True, text=text, next_steps=tuple(steps))
