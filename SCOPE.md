# Data Analysis Agent — SCOPE

## Goal
Analyse one CSV file and produce a plain-English report: what the data contains, data-quality issues, key statistics, trends and relationships, with charts and suggested next steps.

## Design decisions (2026-11-08)
1. Input: one CSV, at most 5 MB and 50,000 rows. Other files are rejected with a clear message.
2. Analysis is deterministic Python (pandas), no AI: row/column counts; column types; missing values per column; duplicate rows; numeric summaries (mean, median, min, max, std); top categories for text columns; correlations between numeric columns (strongest pairs); trend over time if a date column is detected. Plus 3–4 charts (matplotlib).
3. One Gemini call (gemini-3.8-flash, env var ANALYSIS_AGENT_MODEL) writes the summary and next steps from the computed statistics only — never the raw rows. Every number in the summary must match a computed value (sensible rounding allowed); otherwise the summary is rejected and the report says "summary unavailable". Same cache, shared pacer and retry rules as website_audit_agent (503 and dropped connections retried twice; 429 never). If Gemini fails, the report is still written from the statistics.
4. Interface: a web page (upload → report) and a command-line tool; logic separate from input/output. Hosted on Render (free plan, Docker), with UptimeRobot pinging /health (GET and HEAD).
5. Safety: uploads are analysed in memory and never stored. Visible notice: "Don't upload personal or confidential data." Limits: 3 analyses per visitor per hour, 20 per day in total. A separate Gemini key from its own Google project, stored as a Render secret.

## Evaluation
- Unit tests use small CSV fixtures and a fake Gemini: no test touches the network.
- A fixture CSV with known, deliberate properties (missing values, a duplicate row, a strong correlation, a date trend); expected findings written by hand before running.
- A summary containing a number not in the computed statistics must be rejected (test).
- One live run on a practice dataset, expected findings written before running.

## Out of scope for v1
Excel files, multiple files, joining datasets, machine-learning predictions, editing the data.
