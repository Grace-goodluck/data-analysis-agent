"""Continued 4: one self-contained HTML report, charts embedded, every CSV value escaped."""

import re
from datetime import datetime

from fakes import KNOWN, FakeLLM, fixture_bytes, reply
from data_analysis_agent.analyse import run_analysis
from data_analysis_agent.report import fmt_num, render_html, report_filename, write_report

WHEN = datetime(2026, 11, 8, 14, 30, 5)


def known_report(llm=None):
    return run_analysis(fixture_bytes(KNOWN), KNOWN, llm, now=WHEN)


def test_report_is_self_contained_with_embedded_charts():
    html = render_html(known_report(FakeLLM([reply()])))
    assert html.startswith("<!doctype html>")
    assert html.count('src="data:image/png;base64,') == 4
    assert not re.search(r'(src|href)="(?!data:)', html)  # no external files at all
    assert "24 rows × 6 columns" in html
    assert "analysed 2026-11-08 14:30" in html


def test_report_shows_the_findings():
    html = render_html(known_report())
    assert "Duplicate rows: 1 (4.2% of rows)." in html
    assert "age 3 (12.5%)" in html
    assert "<td>North</td><td class=\"n\">6</td>" in html
    assert "<td>hours and score</td>" in html
    assert "<td>2025-04</td><td class=\"n\">9</td>" in html
    assert "Other values" in html


def test_csv_content_is_escaped():
    data = b'<script>alert(1)</script>,value\n<img src=x onerror=alert(2)>,1\nb,2\nc,3\n'
    html = render_html(run_analysis(data, "evil.csv", None, now=WHEN))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x" not in html


def test_no_summary_says_so_and_keeps_statistics():
    html = render_html(known_report(None))
    assert "Summary unavailable" in html and "GEMINI_API_KEY is not set" in html
    assert "<h2>Numeric columns</h2>" in html


def test_report_filename_is_safe():
    report = known_report()
    assert report_filename(report) == "known_properties-20261108-143005.html"
    odd = run_analysis(b"a,b\n1,2\n", "../my data (v2).csv", None, now=WHEN)
    assert report_filename(odd) == "my-data-v2-20261108-143005.html"


def test_write_report(tmp_path):
    path = write_report(known_report(), tmp_path / "reports")
    assert path.parent == tmp_path / "reports"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_fmt_num():
    assert fmt_num(1203) == "1,203"
    assert fmt_num(12.375) == "12.38"
    assert fmt_num(23.0) == "23"
    assert fmt_num(0.0123) == "0.0123"
    assert fmt_num(None) == "–"
