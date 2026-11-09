"""The number check (decision 3): every number in the summary must match a computed value.

Allowed values: every number in stats_for_prompt() — the exact dictionary Gemini saw —
including the report's own counts (charts, top-N sizes). For correlations and the trend's
% change, the size without its sign also counts ("a negative correlation of 0.79").

Extraction:
- column names, category values and other labels from the statistics are removed first,
  so "Q1", "col_2" or a column called "2024 sales" are not read as numbers;
- dates ("2025-01-05", "2025-01", "05/01/2025") count as their year;
- handles 1,203 / -0.79 / −0.79 (Unicode minus) / 12.5% / ranges such as 242–382 (two numbers);
- digits stuck to letters ("3rd", "Q4", "5MB") are ignored;
- numbers written as words ("three") are not checked: the prompt asks for digits.

A number passes if:
- some allowed value rounds to it at the precision written (0.79 and 0.8 both pass for 0.787;
  0.85 does not), or
- it is a whole number and some allowed value of 100 or more rounds to it at 2 or 3
  significant figures ("about 1,200" for 1,203 — but not "1,000"), or
- it is a year (1900–2100, written without decimals or %) within the date column's range.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

_ABS_KEYS = {"r", "change_first_to_last_period_pct"}
_HAS_LETTER = re.compile(r"[^\W\d_]")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?!\d)")
_SLASH_DATE = re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/(\d{4})(?!\d)")
_NUMBER = re.compile(r"(?<![\w.])([-−]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)(?!\w)")


@dataclass(frozen=True)
class Allowed:
    values: frozenset[float]
    years: tuple[int, int] | None
    names: tuple[str, ...]  # labels removed from the text before extraction


@dataclass(frozen=True)
class Found:
    raw: str
    value: float
    decimals: int
    percent: bool


def allowed_from(prompt_stats: dict) -> Allowed:
    values: set[float] = set()
    names: set[str] = set()

    def walk(node: object, key: str | None = None) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            values.add(float(node))
            if key in _ABS_KEYS:
                values.add(abs(float(node)))
        elif isinstance(node, str):
            if len(node) >= 2 and _HAS_LETTER.search(node):
                names.add(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, key)

    walk(prompt_stats)

    years = None
    trend = prompt_stats.get("trend")
    if isinstance(trend, dict) and trend.get("first_date") and trend.get("last_date"):
        years = (int(str(trend["first_date"])[:4]), int(str(trend["last_date"])[:4]))
    return Allowed(frozenset(values), years, tuple(sorted(names, key=len, reverse=True)))


def find_numbers(text: str, names: Iterable[str] = ()) -> list[Found]:
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", text)
    text = _ISO_DATE.sub(lambda m: f" {m.group(1)} ", text)
    text = _SLASH_DATE.sub(lambda m: f" {m.group(1)} ", text)

    found: list[Found] = []
    for m in _NUMBER.finditer(text):
        sign, whole, fraction, percent = m.groups()
        value = float(whole.replace(",", "") + (fraction or ""))
        if sign:
            value = -value
        found.append(Found(m.group(0), value, len(fraction) - 1 if fraction else 0, bool(percent)))
    return found


def _round_sig(value: float, sig: int) -> float:
    if value == 0:
        return 0.0
    return round(value, sig - 1 - math.floor(math.log10(abs(value))))


def is_allowed(number: Found, allowed: Allowed) -> bool:
    tolerance = 0.5 * 10 ** -number.decimals + 1e-9
    for v in allowed.values:
        if abs(number.value - v) <= tolerance:
            return True
        if number.decimals == 0 and abs(v) >= 100:
            if any(math.isclose(number.value, _round_sig(v, sig), abs_tol=1e-9) for sig in (2, 3)):
                return True
    if (
        allowed.years is not None
        and number.decimals == 0
        and not number.percent
        and 1900 <= number.value <= 2100
        and allowed.years[0] <= number.value <= allowed.years[1]
    ):
        return True
    return False


def unverified(texts: Iterable[str], allowed: Allowed) -> list[str]:
    """Numbers (as written) that match no computed value, in order, without repeats."""
    bad: list[str] = []
    for text in texts:
        for number in find_numbers(text, allowed.names):
            if not is_allowed(number, allowed) and number.raw not in bad:
                bad.append(number.raw)
    return bad
