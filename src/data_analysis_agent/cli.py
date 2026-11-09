"""Command line: data-analysis <file.csv>. Wires the real Gemini client, writes the HTML report.

Exit codes: 0 report with summary; 1 report written but summary unavailable;
2 bad arguments or a file that cannot be analysed.
"""

import argparse
import sys
from pathlib import Path

from data_analysis_agent.analyse import run_analysis
from data_analysis_agent.config import MAX_UPLOAD_BYTES, load_config
from data_analysis_agent.gemini_client import DiskCache, GeminiLLM, make_gemini_call
from data_analysis_agent.load import LoadError, size_label
from data_analysis_agent.pacing import Pacer
from data_analysis_agent.report import write_report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="data-analysis",
        description="Analyse one CSV file and write a self-contained HTML report to reports/.",
    )
    parser.add_argument("csv", type=Path, help="path to a .csv file (at most 5 MB and 50,000 rows)")
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        parser.error(f"file not found: {args.csv}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path: Path = args.csv
    if path.stat().st_size > MAX_UPLOAD_BYTES:  # don't read a huge file into memory just to reject it
        print(f"Cannot analyse {path.name}: the file is larger than {size_label(MAX_UPLOAD_BYTES)}.", file=sys.stderr)
        return 2

    config = load_config()
    llm = None
    if config.gemini_api_key:
        call = make_gemini_call(config.model, config.gemini_api_key)
        llm = GeminiLLM(config.model, call, DiskCache(config.cache_dir), Pacer())  # 30 s, 60 s retries

    try:
        report = run_analysis(path.read_bytes(), path.name, llm)
    except LoadError as exc:
        print(f"Cannot analyse {path.name}: {exc}", file=sys.stderr)
        return 2

    out = write_report(report, config.reports_dir)
    print(f"Report written: {out}")
    if not report.summary.available:
        print(f"Summary unavailable: {report.summary.reason}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
