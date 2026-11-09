"""The deterministic analysis on a fixture with known, deliberate properties (SCOPE: Evaluation).

Expected findings for tests/fixtures/known_properties.csv, written by hand before the first run:
- 24 rows (23 unique + 1 exact duplicate of the last row), 6 columns
- types: record_id identifier, visit_date date, clinic category, hours/score/age numeric
- missing values: age 3 (12.5%), all other columns 0
- duplicate rows: 1
- hours: count 24, mean 12.375, median 12.5, min 1, max 23
- strongest correlation: hours vs score, strongly positive (score = 2 × hours + 10, ±1), r > 0.99
- clinic: 12 distinct; North 6, South 5, East 3, West 2, then 8 values with 1 row each
  (10 shown, ties alphabetical: Alder…Fern; Gorse and Hazel are the 2 "other" rows)
- visit_date: 2025-01-05 to 2025-04-23, monthly counts rising 3, 5, 7, 9 (+200%)
"""

from datetime import date

import pandas as pd
import pytest

from fakes import KNOWN, fixture_path
from data_analysis_agent.profile import profile


@pytest.fixture(scope="module")
def stats():
    return profile(pd.read_csv(fixture_path(KNOWN)))


def test_shape_and_types(stats):
    assert (stats.rows, stats.columns) == (24, 6)
    kinds = {c.name: c.kind for c in stats.column_info}
    assert kinds == {
        "record_id": "identifier",
        "visit_date": "date",
        "clinic": "category",
        "hours": "numeric",
        "score": "numeric",
        "age": "numeric",
    }


def test_missing_values(stats):
    missing = {c.name: c.missing for c in stats.column_info}
    assert missing == {"record_id": 0, "visit_date": 0, "clinic": 0, "hours": 0, "score": 0, "age": 3}
    age = next(c for c in stats.column_info if c.name == "age")
    assert age.missing_pct == pytest.approx(12.5)
    assert stats.missing_total == 3


def test_duplicate_rows(stats):
    assert stats.duplicate_rows == 1
    assert stats.duplicate_pct == pytest.approx(100 / 24)


def test_numeric_summary(stats):
    hours = next(n for n in stats.numeric if n.column == "hours")
    assert hours.count == 24
    assert hours.mean == pytest.approx(12.375)
    assert hours.median == pytest.approx(12.5)
    assert (hours.min, hours.max) == (1.0, 23.0)
    age = next(n for n in stats.numeric if n.column == "age")
    assert age.count == 21  # missing values are not counted


def test_strongest_correlation(stats):
    top = stats.correlations[0]
    assert {top.a, top.b} == {"hours", "score"}
    assert top.r > 0.99
    assert top.strength == "strong positive"
    assert top.rows_used == 24
    assert len(stats.correlations) == 3  # 3 numeric columns -> 3 pairs (cap is 5)


def test_top_categories_capped_at_ten(stats):
    (clinic,) = stats.categories
    assert clinic.column == "clinic"
    assert clinic.distinct == 12
    assert [(c.value, c.count) for c in clinic.shown] == [
        ("North", 6), ("South", 5), ("East", 3), ("West", 2),
        ("Alder", 1), ("Birch", 1), ("Cedar", 1), ("Daisy", 1), ("Elm", 1), ("Fern", 1),
    ]
    assert clinic.other_count == 2
    assert clinic.shown[0].pct == pytest.approx(25.0)


def test_identifier_values_are_never_listed(stats):
    assert all(top.column != "record_id" for top in stats.categories)


def test_date_trend(stats):
    trend = stats.trend
    assert trend is not None
    assert trend.column == "visit_date"
    assert trend.period == "month"
    assert (trend.first_date, trend.last_date) == (date(2025, 1, 5), date(2025, 4, 23))
    assert [(p.label, p.count) for p in trend.counts] == [
        ("2025-01", 3), ("2025-02", 5), ("2025-03", 7), ("2025-04", 9),
    ]
    assert trend.change_pct == pytest.approx(200.0)
    assert trend.unparsed == 0


def test_numeric_row_number_is_an_identifier_not_a_measure():
    df = pd.DataFrame({"row": range(1, 31), "value": [i % 7 for i in range(30)]})
    kinds = {c.name: c.kind for c in profile(df).column_info}
    assert kinds == {"row": "identifier", "value": "numeric"}


def test_empty_column_and_no_correlations():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "blank": [None, None, None]})
    stats = profile(df)
    assert {c.name: c.kind for c in stats.column_info} == {"a": "numeric", "blank": "empty"}
    assert stats.correlations == ()
    assert stats.trend is None


def test_gaps_in_the_trend_count_as_zero():
    df = pd.DataFrame({"when": ["2025-01-10", "2025-01-20", "2025-03-05"]})
    trend = profile(df).trend
    assert [(p.label, p.count) for p in trend.counts] == [("2025-01", 2), ("2025-02", 0), ("2025-03", 1)]
