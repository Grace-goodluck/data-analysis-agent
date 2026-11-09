# Data Analysis Agent

Analyses one CSV file and writes a plain-English report: what the data contains, data-quality issues,
key statistics, trends and relationships, with charts and suggested next steps. The analysis is
deterministic Python (pandas, matplotlib); one Gemini call writes the summary from the computed
statistics only — never the rows — and every number in it is checked against those statistics.
See [SCOPE.md](SCOPE.md) for the design decisions.

## Command line

```
uv sync
uv run data-analysis path/to/file.csv
```

The report is written to `reports/<name>-YYYYMMDD-HHMMSS.html`: one self-contained HTML file with the
charts embedded. Exit codes: 0 report with summary; 1 report written without a summary; 2 bad
arguments or a file that cannot be analysed (not .csv, empty, over 5 MB or 50,000 rows, unreadable).

Environment variables:
- `GEMINI_API_KEY` — without it, the statistics and charts are still reported and the summary is marked unavailable.
- `ANALYSIS_AGENT_MODEL` — defaults to `gemini-3.8-flash`.

Gemini replies are cached in `results/cache/`. A 503 or dropped connection is retried twice
(after 30 s, then 60 s); a 429 quota error is never retried.

## The number check

Every number in the summary and next steps must match a computed value, allowing sensible rounding
(0.787 may be written 0.79 or 0.8; 1,203 may be written "about 1,200"). Years must fall within the
date column's range; small counts such as "4 charts" must match the report's own counts. If any
number does not match, the summary is rejected and the report says "Summary unavailable".

## Web demo (Render)

```
uv sync --extra web
uv run data-analysis-web    # http://localhost:8000
```

Upload one CSV, get the same report in the page, with a download link for the HTML file.
Uploads are analysed in memory and never stored; the Gemini cache is in memory too. Limits:
3 analyses per visitor per hour and 20 per day in total (reset at midnight Pacific time).
On the web, a failed Gemini call is retried once, after 10 s.

Deploy on Render as a free **Web Service** with runtime **Docker** (it builds the `Dockerfile`; run
`uv lock` first so `uv.lock` exists):

- Environment: add `GEMINI_API_KEY` as a secret environment variable in the Render dashboard — a
  separate key from its own Google project. It is never baked into the image.
- The app listens on the port Render provides in `PORT`.
- Health check path: `/health` (answers GET and HEAD).

Free Render services sleep when idle; an UptimeRobot monitor pings `/health` to keep the demo awake.

## Tests

```
uv run pytest
```

Tests use small CSV fixtures and a fake Gemini; no test touches the network.
