"""Configuration from environment variables only — never loads a .env file.

Paths are relative to the current working directory: run the command-line
tool from the project root so the cache lands in results/cache/ and reports
in reports/.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-3.8-flash"  # decision 3; override with ANALYSIS_AGENT_MODEL

# Input limits (decision 1)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 50_000

# Analysis
MAX_CATEGORIES = 10  # most common values shown per text column
MAX_CORRELATIONS = 5  # strongest pairs of numeric columns
MAX_CHARTS = 4  # decision 2: 3–4 charts
MAX_NEXT_STEPS = 6
DATE_MIN_SHARE = 0.9  # a text column is a date column if at least 90% of its values parse as dates
IDENTIFIER_MIN_ROWS = 20  # below this, a mostly-unique text column is still treated as categories
IDENTIFIER_UNIQUE_SHARE = 0.9  # text columns more unique than this are identifiers or free text

# Gemini (decision 3; continued 1)
MIN_CALL_SPACING_S = 13.0
GEMINI_TIMEOUT_S = 60.0  # continued 5: no reply within 60 s counts as a dropped connection
GEMINI_RETRY_WAITS_S = (30.0, 60.0)  # command line: 503 / dropped connection retried twice; 429 never
WEB_GEMINI_RETRY_WAITS_S = (10.0,)  # web demo: one retry after 10 s

CACHE_DIR = Path("results/cache")  # command line only (continued 2)
REPORTS_DIR = Path("reports")

NO_KEY_REASON = "GEMINI_API_KEY is not set"


@dataclass(frozen=True)
class Config:
    gemini_api_key: str | None  # None: the statistics are still reported, the summary is unavailable
    model: str = DEFAULT_MODEL
    cache_dir: Path = CACHE_DIR
    reports_dir: Path = REPORTS_DIR


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    api_key = env.get("GEMINI_API_KEY", "").strip() or None
    model = env.get("ANALYSIS_AGENT_MODEL", "").strip() or DEFAULT_MODEL
    return Config(gemini_api_key=api_key, model=model)
