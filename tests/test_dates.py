"""Dates are detected by trying to parse text columns (pre-decided)."""

import pandas as pd
import pytest

from data_analysis_agent.profile import build_trend, choose_period, parse_dates


def test_iso_dates_detected():
    parsed = parse_dates(pd.Series(["2025-01-01", "2025-02-15", None, "2025-03-31"]))
    assert parsed is not None
    assert parsed.notna().sum() == 3
    assert parsed.isna().iloc[2]  # missing stays missing, aligned to the original index


def test_day_month_year_detected():
    parsed = parse_dates(pd.Series(["13/01/2025", "14/02/2025", "28/03/2025"]))
    assert parsed is not None
    assert list(parsed.dt.month) == [1, 2, 3]


def test_written_dates_detected():
    assert parse_dates(pd.Series(["5 January 2025", "12 February 2025", "1 March 2025"])) is not None


@pytest.mark.parametrize(
    "values",
    [
        ["Emergency", "Outpatient", "Surgery"],  # plain text
        ["12", "7", "2025", "31"],  # numbers stored as text are not dates
        ["R00001", "R00002", "R00003"],  # identifiers
    ],
    ids=["text", "numbers-as-text", "identifiers"],
)
def test_non_dates_rejected(values):
    assert parse_dates(pd.Series(values)) is None


def test_mostly_dates_accepted_but_half_rejected():
    good = [f"2025-01-{d:02d}" for d in range(1, 20)]
    assert parse_dates(pd.Series(good + ["not a date"])) is not None  # 95% parse
    pending = [f"pending {i}" for i in range(1, 6)]
    assert parse_dates(pd.Series(good[:5] + pending)) is None  # 50%


@pytest.mark.parametrize(
    "span_days, period",
    [(0, "month"), (183, "month"), (184, "quarter"), (364, "quarter"), (3 * 366, "quarter"), (3 * 366 + 1, "year")],
)
def test_choose_period(span_days, period):
    assert choose_period(span_days) == period


def test_quarterly_trend_over_one_year():
    dates = pd.to_datetime(pd.Series(["2025-01-01", "2025-02-01", "2025-05-01", "2025-08-01", "2025-12-31"]))
    trend = build_trend("admission_date", dates, unparsed=0)
    assert trend.period == "quarter"
    assert [(p.label, p.count) for p in trend.counts] == [
        ("2025-Q1", 2), ("2025-Q2", 1), ("2025-Q3", 1), ("2025-Q4", 1),
    ]


def test_yearly_trend_over_five_years():
    dates = pd.to_datetime(pd.Series(["2020-06-01", "2021-06-01", "2021-07-01", "2025-06-01"]))
    trend = build_trend("d", dates, unparsed=0)
    assert trend.period == "year"
    assert [p.label for p in trend.counts] == ["2020", "2021", "2022", "2023", "2024", "2025"]
    assert [p.count for p in trend.counts] == [1, 2, 0, 0, 0, 1]
