"""3–4 charts as PNG bytes, drawn in memory (decision 2); nothing is written to disk.

Uses matplotlib's Figure directly (no pyplot), so charts can be drawn on the
web demo's worker threads.

Order of preference:
1. rows over time (if a date column was found), else a scatter of the strongest correlated pair;
2. missing values per column (if any are missing);
3. distribution of the first numeric column;
4. most common values of the first category column;
then, to reach 3–4, the scatter (if not used yet), more distributions, more category charts.
"""

import io
import math
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402  (after choosing the backend)

import pandas as pd  # noqa: E402

from data_analysis_agent.config import MAX_CHARTS  # noqa: E402
from data_analysis_agent.models import Chart, Correlation, Stats, TopCategories, Trend  # noqa: E402

COLOR = "#1f5fae"
MAX_SCATTER_POINTS = 5000
MAX_MISSING_BARS = 15


def _t(text: str, limit: int = 40) -> str:
    """Column names and values are untrusted text: shorten, and stop "$" starting matplotlib maths."""
    text = str(text)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("$", r"\$")


def _figure(height: float = 3.6) -> tuple[Figure, object]:
    fig = Figure(figsize=(7.0, height), dpi=100, layout="constrained")
    return fig, fig.add_subplot()


def _png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


def trend_chart(trend: Trend) -> Chart:
    title = f"Rows per {trend.period} ({trend.column})"
    fig, ax = _figure()
    labels = [p.label for p in trend.counts]
    counts = [p.count for p in trend.counts]
    positions = list(range(len(counts)))
    ax.plot(positions, counts, marker="o", color=COLOR)
    step = max(1, math.ceil(len(labels) / 12))
    ticks = positions[::step]
    ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=45 if len(ticks) > 6 else 0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Rows")
    ax.set_title(_t(title, 80))
    ax.grid(axis="y", alpha=0.3)
    return Chart(title, _png(fig))


def missing_chart(stats: Stats) -> Chart:
    title = "Missing values per column"
    columns = sorted((c for c in stats.column_info if c.missing), key=lambda c: (-c.missing, c.name))
    columns = columns[:MAX_MISSING_BARS]
    fig, ax = _figure(height=max(2.4, 0.4 * len(columns) + 1.2))
    names = [_t(c.name) for c in reversed(columns)]
    ax.barh(range(len(names)), [c.missing for c in reversed(columns)], color=COLOR)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Missing values")
    ax.set_title(title)
    return Chart(title, _png(fig))


def histogram_chart(values: pd.Series, name: str) -> Chart:
    title = f"Distribution of {name}"
    data = values.dropna().astype(float)
    fig, ax = _figure()
    bins = min(30, max(5, int(math.sqrt(len(data))))) if len(data) else 5
    ax.hist(data, bins=bins, color=COLOR, edgecolor="white")
    ax.set_xlabel(_t(name))
    ax.set_ylabel("Rows")
    ax.set_title(_t(title, 80))
    return Chart(title, _png(fig))


def category_chart(top: TopCategories) -> Chart:
    title = f"Most common values in {top.column}"
    shown = list(reversed(top.shown))
    fig, ax = _figure(height=max(2.4, 0.35 * len(shown) + 1.2))
    ax.barh(range(len(shown)), [c.count for c in shown], color=COLOR)
    ax.set_yticks(range(len(shown)), [_t(c.value, 30) for c in shown])
    ax.set_xlabel("Rows")
    ax.set_title(_t(title, 80))
    return Chart(title, _png(fig))


def scatter_chart(x: pd.Series, y: pd.Series, corr: Correlation) -> Chart:
    title = f"{corr.a} vs {corr.b} (r = {corr.r:.2f})"
    data = pd.concat([x.astype(float), y.astype(float)], axis=1).dropna()
    data.columns = ["x", "y"]
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
    fig, ax = _figure()
    ax.scatter(data["x"], data["y"], s=10, alpha=0.4, color=COLOR)
    ax.set_xlabel(_t(corr.a))
    ax.set_ylabel(_t(corr.b))
    ax.set_title(_t(title, 80))
    return Chart(title, _png(fig))


def make_charts(df: pd.DataFrame, stats: Stats) -> tuple[Chart, ...]:
    columns = {str(c): c for c in df.columns}
    numeric = [n.column for n in stats.numeric]
    corr = stats.correlations[0] if stats.correlations else None

    def scatter() -> Chart:
        assert corr is not None
        return scatter_chart(df[columns[corr.a]], df[columns[corr.b]], corr)

    plans: list[Callable[[], Chart]] = []
    if stats.trend is not None:
        plans.append(lambda: trend_chart(stats.trend))
    elif corr is not None:
        plans.append(scatter)
    if stats.missing_total:
        plans.append(lambda: missing_chart(stats))
    if numeric:
        plans.append(lambda: histogram_chart(df[columns[numeric[0]]], numeric[0]))
    if stats.categories:
        plans.append(lambda: category_chart(stats.categories[0]))

    # Fillers, only drawn if needed to reach MAX_CHARTS.
    if stats.trend is not None and corr is not None:
        plans.append(scatter)
    for name in numeric[1:]:
        plans.append(lambda name=name: histogram_chart(df[columns[name]], name))
    for top in stats.categories[1:]:
        plans.append(lambda top=top: category_chart(top))

    return tuple(plan() for plan in plans[:MAX_CHARTS])
