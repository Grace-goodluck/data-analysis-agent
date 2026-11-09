"""Plain data passed between the analysis, the summary and the report."""

import base64
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

ColumnKind = Literal["numeric", "date", "category", "identifier", "empty"]
KIND_LABELS: dict[str, str] = {
    "numeric": "numeric",
    "date": "date",
    "category": "category",
    "identifier": "identifier or free text",
    "empty": "empty",
}


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: ColumnKind
    missing: int
    missing_pct: float
    distinct: int

    @property
    def kind_label(self) -> str:
        return KIND_LABELS[self.kind]


@dataclass(frozen=True)
class NumericSummary:
    column: str
    count: int
    mean: float
    median: float
    min: float
    max: float
    std: float | None  # None with fewer than 2 values


@dataclass(frozen=True)
class CategoryCount:
    value: str
    count: int
    pct: float  # of all rows


@dataclass(frozen=True)
class TopCategories:
    column: str
    distinct: int
    shown: tuple[CategoryCount, ...]
    other_count: int  # rows holding values beyond the ones shown


@dataclass(frozen=True)
class Correlation:
    a: str
    b: str
    r: float  # Pearson
    rows_used: int

    @property
    def strength(self) -> str:
        size = abs(self.r)
        if size < 0.1:
            return "little or no"
        word = "strong" if size >= 0.7 else "moderate" if size >= 0.4 else "weak"
        return f"{word} {'positive' if self.r > 0 else 'negative'}"


@dataclass(frozen=True)
class PeriodCount:
    label: str  # "2025-01", "2025-Q1" or "2025"
    count: int


Period = Literal["month", "quarter", "year"]


@dataclass(frozen=True)
class Trend:
    column: str
    period: Period
    first_date: date
    last_date: date
    counts: tuple[PeriodCount, ...]
    unparsed: int  # non-empty values that could not be read as dates

    @property
    def change_pct(self) -> float | None:
        """Change from the first period's count to the last's, in %."""
        if len(self.counts) < 2 or self.counts[0].count == 0:
            return None
        first, last = self.counts[0].count, self.counts[-1].count
        return (last - first) / first * 100


@dataclass(frozen=True)
class Stats:
    rows: int
    columns: int
    column_info: tuple[ColumnInfo, ...]
    duplicate_rows: int
    numeric: tuple[NumericSummary, ...]
    categories: tuple[TopCategories, ...]
    correlations: tuple[Correlation, ...]
    trend: Trend | None

    @property
    def missing_total(self) -> int:
        return sum(c.missing for c in self.column_info)

    @property
    def duplicate_pct(self) -> float:
        return self.duplicate_rows / self.rows * 100 if self.rows else 0.0

    @property
    def kind_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in KIND_LABELS.values()}
        for c in self.column_info:
            counts[c.kind_label] += 1
        return {label: n for label, n in counts.items() if n}


@dataclass(frozen=True)
class Chart:
    title: str
    png: bytes

    @property
    def b64(self) -> str:
        return base64.b64encode(self.png).decode("ascii")


# "" when available; otherwise why not, so the web demo can show a public reason.
SummaryCode = Literal["", "no_key", "llm_failed", "unreadable", "rejected"]


@dataclass(frozen=True)
class Summary:
    available: bool
    text: str = ""
    next_steps: tuple[str, ...] = ()
    code: SummaryCode = ""
    reason: str = ""


@dataclass(frozen=True)
class Report:
    source_name: str
    analysed_at: datetime
    stats: Stats
    charts: tuple[Chart, ...]
    summary: Summary
    model: str | None

    @property
    def ok(self) -> bool:
        return self.summary.available
