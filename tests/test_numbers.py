"""The number check: every number in the summary must match a computed value (decision 3)."""

import pytest

from data_analysis_agent.numbers import allowed_from, find_numbers, unverified

# Shaped like stats_for_prompt(), with the practice dataset's headline numbers.
STATS = {
    "file": {"rows": 1203, "columns": 9, "duplicate_rows": 3, "duplicate_rows_pct": 0.25},
    "columns": [{"name": "satisfaction_score", "missing": 60, "missing_pct": 4.99}],
    "correlations": [{"columns": ["wait_minutes", "satisfaction_score"], "r": -0.787}],
    "top_categories": [{"column": "department", "most_common": [{"value": "Emergency", "rows": 431}]}],
    "trend": {
        "date_column": "admission_date",
        "first_date": "2025-01-01",
        "last_date": "2025-12-31",
        "rows_per_period": [{"period": "2025-Q1", "rows": 242}, {"period": "2025-Q4", "rows": 382}],
        "change_first_to_last_period_pct": 57.85,
    },
    "report": {"charts": 4, "max_categories_shown_per_column": 10},
}
ALLOWED = allowed_from(STATS)


@pytest.mark.parametrize(
    "text",
    [
        "The file has 1,203 rows and 9 columns.",
        "There are about 1,200 rows.",  # 1,203 at 2 or 3 significant figures
        "Wait and satisfaction are negatively correlated (r = -0.79).",
        "Wait and satisfaction are negatively correlated (r = −0.79).",  # Unicode minus
        "a negative correlation of 0.79",  # size without the sign, for correlations
        "r is about -0.8",
        "satisfaction_score has 60 missing values (4.99%).",
        "satisfaction_score has 60 missing values (about 5%).",
        "There are 3 duplicate rows.",
        "Quarterly counts rose from 242 in Q1 to 382 in Q4.",  # Q1/Q4 are labels, not numbers
        "Counts rose from 242 to 382 (242–382).",  # a range is two numbers
        "Counts rose by 57.85% (about 58%) over 2025.",
        "From 2025-01-01 to 2025-12-31.",  # dates count as their year
        "Emergency is the largest department with 431 rows.",
        "The report has 4 charts and shows at most 10 values per column.",
        "Check the 1st and 2nd quarters.",  # digits stuck to letters are not numbers
        "Next, look at wait_minutes and satisfaction_score.",
    ],
)
def test_accepted(text):
    assert unverified([text], ALLOWED) == []


@pytest.mark.parametrize(
    "text, bad",
    [
        ("The average wait was 45 minutes.", ["45"]),
        ("There are about 1,000 rows.", ["1,000"]),  # not a sensible rounding of 1,203
        ("The correlation is -0.85.", ["-0.85"]),
        ("6% of satisfaction scores are missing.", ["6%"]),
        ("The report has 7 charts.", ["7"]),  # small counts must still match
        ("Admissions have risen since 2019.", ["2019"]),  # year outside the date range
        ("The correlation's size is 0.787, written +0.79 too.", []),  # "+" is not read as a minus
        ("Two problems: 45 and 46.", ["45", "46"]),
    ],
)
def test_rejected(text, bad):
    assert unverified([text], ALLOWED) == bad


def test_negative_number_needs_a_negative_value():
    allowed = allowed_from({"mean": 12.0})
    assert unverified(["The mean is -12."], allowed) == ["-12"]


def test_sign_dropping_is_only_for_correlations_and_change():
    allowed = allowed_from({"min": -5.0})
    assert unverified(["The minimum is 5."], allowed) == ["5"]


def test_years_without_a_date_column_must_match_a_value():
    allowed = allowed_from({"rows": 40})
    assert unverified(["Data from 2025."], allowed) == ["2025"]


def test_checks_next_steps_too_and_reports_each_number_once():
    assert unverified(["45 rows.", "Remove 45 rows and 46 more."], ALLOWED) == ["45", "46"]


def test_find_numbers_parsing():
    found = find_numbers("1,203 rows, r = −0.79, 12.5%, Q1, 3rd, 2025-03-04")
    assert [(f.value, f.decimals, f.percent) for f in found] == [
        (1203.0, 0, False),
        (-0.79, 2, False),
        (12.5, 1, True),
        (2025.0, 0, False),
    ]


def test_labels_with_digits_are_removed_before_extraction():
    allowed = allowed_from({"columns": [{"name": "sales 2019", "missing": 0}]})
    assert unverified(["sales 2019 has 0 missing values."], allowed) == []
