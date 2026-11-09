"""The Gemini prompt (decision 3): computed statistics only, never the raw rows.

stats_for_prompt() is also the reference for the number check: every number in
the summary must match a number in this dictionary (numbers.py).
Identifier / free-text columns are counted but their values are never listed.
"""

import json

from data_analysis_agent.config import MAX_CATEGORIES, MAX_CORRELATIONS
from data_analysis_agent.models import Stats


def _num(x: float) -> float:
    """2 decimals; very small values keep 3 significant figures so they don't become 0."""
    x = float(x)
    if x == 0 or abs(x) >= 0.1:
        return round(x, 2)
    return float(f"{x:.3g}")


def stats_for_prompt(stats: Stats, n_charts: int) -> dict:
    trend = None
    if stats.trend is not None:
        t = stats.trend
        trend = {
            "date_column": t.column,
            "first_date": t.first_date.isoformat(),
            "last_date": t.last_date.isoformat(),
            "period": t.period,
            "rows_per_period": [{"period": p.label, "rows": p.count} for p in t.counts],
            "unparsed_dates": t.unparsed,
        }
        if t.change_pct is not None:
            trend["change_first_to_last_period_pct"] = _num(t.change_pct)

    return {
        "file": {
            "rows": stats.rows,
            "columns": stats.columns,
            "missing_values_total": stats.missing_total,
            "duplicate_rows": stats.duplicate_rows,
            "duplicate_rows_pct": _num(stats.duplicate_pct),
            "column_types": stats.kind_counts,
        },
        "columns": [
            {
                "name": c.name,
                "type": c.kind_label,
                "missing": c.missing,
                "missing_pct": _num(c.missing_pct),
                "distinct_values": c.distinct,
            }
            for c in stats.column_info
        ],
        "numeric_summaries": [
            {
                "column": n.column,
                "count": n.count,
                "mean": _num(n.mean),
                "median": _num(n.median),
                "min": _num(n.min),
                "max": _num(n.max),
                "std": None if n.std is None else _num(n.std),
            }
            for n in stats.numeric
        ],
        "top_categories": [
            {
                "column": top.column,
                "distinct_values": top.distinct,
                "most_common": [{"value": c.value, "rows": c.count, "pct": _num(c.pct)} for c in top.shown],
                "rows_with_other_values": top.other_count,
            }
            for top in stats.categories
        ],
        "correlations": [
            {"columns": [c.a, c.b], "r": _num(c.r), "strength": c.strength, "rows_used": c.rows_used}
            for c in stats.correlations
        ],
        "trend": trend,
        "report": {
            "charts": n_charts,
            "max_categories_shown_per_column": MAX_CATEGORIES,
            "max_correlation_pairs_shown": MAX_CORRELATIONS,
        },
    }


PROMPT = """You are a careful data analyst. Below are statistics computed from one CSV file. \
You do not see the rows themselves. Write a plain-English summary for a reader who is not a \
specialist, and suggest next steps.

Rules:
- Use only the statistics below. Do not guess what the data is about beyond the column names and values shown.
- Every number you write must appear in the statistics. You may round it (for example 0.787 as 0.79, \
or 1,203 as about 1,200). Do not calculate new numbers: no sums, differences, ratios or averages of your own.
- Write numbers as digits. Mention a year only if it lies within the date range given.
- Describe correlations as associations, not causes.
- Cover: what the data contains, data-quality issues (missing values, duplicate rows), key statistics, \
trends over time and relationships between columns, where the statistics show them.
- Reply with JSON only, no other text:
  {{"summary": "<2 to 4 short paragraphs separated by a blank line>", "next_steps": ["<step>", "..."]}}
  with 3 to 5 next steps.

Statistics (JSON):
{stats}
"""


def build_prompt(prompt_stats: dict) -> str:
    return PROMPT.format(stats=json.dumps(prompt_stats, ensure_ascii=False, indent=1))
