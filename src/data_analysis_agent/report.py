"""Self-contained HTML report (continued 4): charts embedded as base64 PNGs, no other files.

ENV is shared with the web demo, so its results page shows exactly the same report body.
Autoescape is on for every template: column names and values come from the uploaded file.
"""

import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from data_analysis_agent.models import Report

TEMPLATES_DIR = Path(__file__).parent / "templates"


def fmt_num(x: object) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return f"{int(x):,}"
        if x != 0 and abs(x) < 0.1:
            return f"{x:.3g}"
        return f"{x:,.2f}"
    return str(x)


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


ENV = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True, trim_blocks=True, lstrip_blocks=True)
ENV.filters["num"] = fmt_num
ENV.filters["pct"] = fmt_pct
ENV.filters["paragraphs"] = paragraphs


def render_html(report: Report, download_url: str | None = None) -> str:
    return ENV.get_template("report.html").render(report=report, download_url=download_url)


def report_filename(report: Report) -> str:
    stem = Path(report.source_name).stem
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:50] or "data"
    stamp = (report.analysed_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{safe}-{stamp}.html"


def write_report(report: Report, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / report_filename(report)
    path.write_text(render_html(report), encoding="utf-8")
    return path
