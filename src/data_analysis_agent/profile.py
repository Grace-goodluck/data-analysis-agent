"""The deterministic analysis (decision 2): DataFrame -> Stats. No AI, no I/O.

Column types:
- numeric: a numeric column, unless it looks like a row number / ID (whole numbers,
  all different, and named "...id" or always increasing) -> identifier;
- date: a text column where at least 90% of non-empty values parse as dates
  (and the values are not just numbers stored as text);
- identifier: a text column with 20+ values that are more than 90% unique
  (IDs, names, free text) — counted, never listed;
- category: any other text column (also true/false columns);
- empty: no values at all.
"""

import re
import warnings

import pandas as pd

from data_analysis_agent.config import (
    DATE_MIN_SHARE,
    IDENTIFIER_MIN_ROWS,
    IDENTIFIER_UNIQUE_SHARE,
    MAX_CATEGORIES,
    MAX_CORRELATIONS,
)
from data_analysis_agent.models import (
    CategoryCount,
    ColumnInfo,
    Correlation,
    NumericSummary,
    PeriodCount,
    Stats,
    TopCategories,
    Trend,
)

_NUMERIC_TEXT = r"[-+]?\d+(?:[.,]\d+)?"
_ID_NAME = re.compile(r"(?:^|[_\s-])id$", re.IGNORECASE)
_FREQ = {"month": "M", "quarter": "Q", "year": "Y"}
MAX_LABEL_CHARS = 80


def _non_empty_text(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[values != ""]


def parse_dates(series: pd.Series) -> pd.Series | None:
    """Parsed dates aligned to the series' index (NaT where missing or unreadable),
    or None if this is not a date column."""
    values = _non_empty_text(series)
    if values.empty:
        return None
    if values.str.fullmatch(_NUMERIC_TEXT).mean() > 0.5:  # numbers stored as text, not dates
        return None
    if values.str.contains(r"\d", regex=True).mean() < DATE_MIN_SHARE:
        return None

    best: pd.Series | None = None
    best_share = 0.0
    for options in (
        {"format": "ISO8601"},
        {"format": "mixed", "dayfirst": False},
        {"format": "mixed", "dayfirst": True},
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(values, errors="coerce", **options)
        except (ValueError, TypeError, OverflowError):
            continue
        if not pd.api.types.is_datetime64_any_dtype(parsed):  # e.g. mixed time zones
            continue
        share = float(parsed.notna().mean())
        if share > best_share:
            best, best_share = parsed, share
        if share == 1.0:
            break

    if best is None or best_share < DATE_MIN_SHARE:
        return None
    if best.dt.tz is not None:
        best = best.dt.tz_convert(None)
    return best.reindex(series.index)


def _is_numeric_identifier(name: str, series: pd.Series) -> bool:
    values = series.dropna()
    if len(values) < IDENTIFIER_MIN_ROWS:
        return False
    if not pd.api.types.is_integer_dtype(values) and not bool((values % 1 == 0).all()):
        return False
    if values.nunique() != len(values):
        return False
    return bool(_ID_NAME.search(name.strip())) or bool(values.is_monotonic_increasing)


def _label(value: str) -> str:
    return value if len(value) <= MAX_LABEL_CHARS else value[: MAX_LABEL_CHARS - 1] + "…"


def _numeric_summary(name: str, series: pd.Series) -> NumericSummary:
    values = series.dropna().astype(float)
    return NumericSummary(
        column=name,
        count=int(values.count()),
        mean=float(values.mean()),
        median=float(values.median()),
        min=float(values.min()),
        max=float(values.max()),
        std=float(values.std()) if len(values) > 1 else None,
    )


def _top_categories(name: str, series: pd.Series, rows: int) -> TopCategories:
    counts = series.dropna().astype(str).value_counts()
    # Most common first; ties in alphabetical order, so the result never depends on row order.
    ordered = sorted(((str(v), int(c)) for v, c in counts.items()), key=lambda vc: (-vc[1], vc[0]))
    shown = ordered[:MAX_CATEGORIES]
    return TopCategories(
        column=name,
        distinct=len(ordered),
        shown=tuple(CategoryCount(_label(v), c, c / rows * 100) for v, c in shown),
        other_count=sum(c for _, c in ordered[MAX_CATEGORIES:]),
    )


def _correlations(df: pd.DataFrame, keys: list) -> tuple[Correlation, ...]:
    usable = [k for k in keys if df[k].nunique(dropna=True) > 1]
    if len(usable) < 2:
        return ()
    data = df[usable].astype(float)
    matrix = data.corr(method="pearson", min_periods=3)
    pairs: list[Correlation] = []
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            r = matrix.loc[a, b]
            if pd.notna(r):
                rows_used = int(data[[a, b]].dropna().shape[0])
                pairs.append(Correlation(str(a), str(b), float(r), rows_used))
    pairs.sort(key=lambda c: (-abs(c.r), c.a, c.b))
    return tuple(pairs[:MAX_CORRELATIONS])


def _period_label(p: pd.Period, period: str) -> str:
    if period == "month":
        return f"{p.year}-{p.month:02d}"
    if period == "quarter":
        return f"{p.year}-Q{p.quarter}"
    return f"{p.year}"


def choose_period(span_days: int) -> str:
    """Monthly up to ~6 months of data, quarterly up to ~3 years, then yearly."""
    if span_days <= 183:
        return "month"
    if span_days <= 3 * 366:
        return "quarter"
    return "year"


def build_trend(name: str, parsed: pd.Series, unparsed: int) -> Trend | None:
    valid = parsed.dropna()
    if valid.empty:
        return None
    first, last = valid.min(), valid.max()
    period = choose_period((last - first).days)
    freq = _FREQ[period]
    periods = valid.dt.to_period(freq)
    counts = periods.value_counts()
    full = pd.period_range(start=periods.min(), end=periods.max(), freq=freq)
    counts = counts.reindex(full, fill_value=0)  # periods with no rows count as 0
    return Trend(
        column=name,
        period=period,
        first_date=first.date(),
        last_date=last.date(),
        counts=tuple(PeriodCount(_period_label(p, period), int(c)) for p, c in counts.items()),
        unparsed=unparsed,
    )


def profile(df: pd.DataFrame) -> Stats:
    rows = len(df)
    infos: list[ColumnInfo] = []
    numeric_keys: list = []
    category_keys: list = []
    dates: list[tuple[str, pd.Series, int]] = []

    for key in df.columns:
        series = df[key]
        name = str(key)
        missing = int(series.isna().sum())
        non_null = rows - missing
        distinct = int(series.nunique(dropna=True))

        if non_null == 0:
            kind = "empty"
        elif pd.api.types.is_bool_dtype(series):
            kind = "category"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "identifier" if _is_numeric_identifier(name, series) else "numeric"
        else:
            parsed = parse_dates(series)
            if parsed is not None:
                kind = "date"
                unparsed = len(_non_empty_text(series)) - int(parsed.notna().sum())
                dates.append((name, parsed, unparsed))
            elif non_null >= IDENTIFIER_MIN_ROWS and distinct / non_null > IDENTIFIER_UNIQUE_SHARE:
                kind = "identifier"
            else:
                kind = "category"

        if kind == "numeric":
            numeric_keys.append(key)
        elif kind == "category":
            category_keys.append(key)
        infos.append(ColumnInfo(name, kind, missing, missing / rows * 100 if rows else 0.0, distinct))

    trend = None
    if dates:
        name, parsed, unparsed = max(dates, key=lambda d: int(d[1].notna().sum()))  # first on ties
        trend = build_trend(name, parsed, unparsed)

    return Stats(
        rows=rows,
        columns=len(df.columns),
        column_info=tuple(infos),
        duplicate_rows=int(df.duplicated().sum()),
        numeric=tuple(_numeric_summary(str(k), df[k]) for k in numeric_keys),
        categories=tuple(_top_categories(str(k), df[k], rows) for k in category_keys),
        correlations=_correlations(df, numeric_keys),
        trend=trend,
    )
