"""Decision 2: 3–4 charts, drawn in memory as PNG bytes."""

import pandas as pd

from fakes import KNOWN, fixture_path
from data_analysis_agent.charts import make_charts
from data_analysis_agent.profile import profile

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def charts_for(df):
    return make_charts(df, profile(df))


def test_fixture_gets_four_charts_in_order():
    charts = charts_for(pd.read_csv(fixture_path(KNOWN)))
    assert [c.title for c in charts] == [
        "Rows per month (visit_date)",
        "Missing values per column",
        "Distribution of hours",
        "Most common values in clinic",
    ]
    for chart in charts:
        assert chart.png.startswith(PNG_SIGNATURE)
        assert len(chart.png) > 1000
        assert chart.b64  # embeddable


def test_without_dates_the_strongest_pair_is_plotted_first():
    df = pd.DataFrame({"x": range(30), "y": [2 * i + (i % 3) for i in range(30)], "z": [i % 5 for i in range(30)]})
    # x is 0..29, always increasing and all different: an identifier. y and z are the measures.
    charts = charts_for(df)
    assert charts[0].title.startswith("y vs z") or charts[0].title.startswith("z vs y")
    assert 3 <= len(charts) <= 4


def test_never_more_than_four():
    df = pd.DataFrame({f"n{i}": [(j * (i + 1)) % 11 for j in range(40)] for i in range(6)})
    df["cat"] = ["a", "b", "c", "d"] * 10
    assert len(charts_for(df)) == 4


def test_awkward_column_names_do_not_break_drawing():
    df = pd.DataFrame({"price $ (USD)": [1.5, 2.5, 3.5, 4.0], "<b>label</b>": ["a", "b", "a", "c"]})
    charts = charts_for(df)
    assert all(c.png.startswith(PNG_SIGNATURE) for c in charts)
