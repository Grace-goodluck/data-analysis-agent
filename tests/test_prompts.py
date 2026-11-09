"""Decision 3: Gemini sees the computed statistics only, never the raw rows."""

import json

import pandas as pd

from fakes import KNOWN, fixture_path
from data_analysis_agent.profile import profile
from data_analysis_agent.prompts import build_prompt, stats_for_prompt


def prompt_stats():
    return stats_for_prompt(profile(pd.read_csv(fixture_path(KNOWN))), n_charts=4)


def test_prompt_contains_statistics_as_json():
    stats = prompt_stats()
    prompt = build_prompt(stats)
    payload = json.loads(prompt.split("Statistics (JSON):", 1)[1])
    assert payload == stats
    assert payload["file"] == {
        "rows": 24, "columns": 6, "missing_values_total": 3, "duplicate_rows": 1, "duplicate_rows_pct": 4.17,
        "column_types": {"numeric": 3, "date": 1, "category": 1, "identifier or free text": 1},
    }
    assert payload["trend"]["rows_per_period"] == [
        {"period": "2025-01", "rows": 3}, {"period": "2025-02", "rows": 5},
        {"period": "2025-03", "rows": 7}, {"period": "2025-04", "rows": 9},
    ]
    assert payload["trend"]["change_first_to_last_period_pct"] == 200.0
    assert payload["report"]["charts"] == 4


def test_prompt_never_contains_raw_rows():
    prompt = build_prompt(prompt_stats())
    for raw in ("P007", "P023", "2025-02-17", "P001,2025-01-05"):  # identifiers and individual dates
        assert raw not in prompt


def test_prompt_states_the_rules():
    prompt = build_prompt(prompt_stats())
    assert "Every number you write must appear in the statistics" in prompt
    assert '"next_steps"' in prompt


def test_rounding():
    stats = prompt_stats()
    hours = next(n for n in stats["numeric_summaries"] if n["column"] == "hours")
    assert hours["mean"] == 12.38  # 12.375 to 2 decimals (banker's rounding would give 12.38 too)
