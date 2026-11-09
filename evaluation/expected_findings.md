# Expected findings — practice_hospital_data.csv (synthetic, written 2026-11-08 before the first run)

- Rows: 1,203; columns: 9
- Missing values: satisfaction_score 60; patient_age 26; all other columns 0
- Duplicate rows: 3
- Strongest correlation: wait_minutes vs satisfaction_score, strongly negative (about -0.79)
- Date column: admission_date; quarterly counts rise through 2025: Q1 242, Q2 254, Q3 325, Q4 382
- Top department: Emergency (431); smallest: Surgery (130)
- Note: the dataset is synthetic — made-up records, no real patients.

## After the run (2026-11-09)

Command: `uv run data-analysis evaluation/practice_hospital_data.csv` with `ANALYSIS_AGENT_MODEL=gemini-3.8-flash`.
Report: `reports/practice_hospital_data-20261109-072236.html`. Exit code 0.

| Expected | Report | Result |
|---|---|---|
| Rows 1,203; columns 9 | 1,203 rows × 9 columns | Matched |
| Missing: satisfaction_score 60; patient_age 26; others 0 | satisfaction_score 60 (5.0%), patient_age 26 (2.2%), all other columns 0 (86 in total) | Matched |
| Duplicate rows: 3 | 3 (0.2% of rows) | Matched |
| Strongest correlation: wait_minutes vs satisfaction_score, about -0.79 | Top pair wait_minutes and satisfaction_score, r = -0.79, "strong negative" (1,143 rows used) | Matched |
| Date column admission_date; quarterly Q1 242, Q2 254, Q3 325, Q4 382 | admission_date detected, 2025-01-01 to 2025-12-31, rows per quarter 242 / 254 / 325 / 382 | Matched |
| Top department Emergency (431); smallest Surgery (130) | Emergency 431 (35.8%) first, Surgery 130 (10.8%) last of 5 | Matched |
| Dataset is synthetic | Not stated in the report: the analysis cannot know where data came from. Note for the reader, not a finding to detect | Not applicable |

Number check: **passed**. The Gemini summary and its 4 next steps were accepted; every number in them
(e.g. 1203, 86, 2.16%, 0.25%, 38.9, 57.85%, -0.79, -0.17, 0.12, years 2025) matches a computed value.
The summary repeats the findings above correctly and describes the correlations as associations.

Observations:
- The run took over 5 minutes, far more than the analysis plus a single Gemini call should need.
  Cause not yet investigated (first-run matplotlib font cache, or a slow Gemini response).
- record_id was correctly treated as an identifier (1,200 distinct of 1,203: the 3 duplicates repeat their ID),
  so no IDs were listed or sent to Gemini.
