"""The whole analysis, with no input/output of its own (decision 4): bytes or DataFrame in, Report out.

The command line and the web demo both call this; only they read files, write reports or serve pages.
"""

from datetime import datetime

import pandas as pd

from data_analysis_agent.charts import make_charts
from data_analysis_agent.load import load_csv
from data_analysis_agent.models import Report
from data_analysis_agent.ports import LLM
from data_analysis_agent.profile import profile
from data_analysis_agent.prompts import stats_for_prompt
from data_analysis_agent.summary import summarise


def analyse_dataframe(
    df: pd.DataFrame,
    source_name: str,
    llm: LLM | None,
    now: datetime | None = None,
) -> Report:
    stats = profile(df)
    charts = make_charts(df, stats)
    summary = summarise(stats_for_prompt(stats, n_charts=len(charts)), llm)
    return Report(
        source_name=source_name,
        analysed_at=now or datetime.now(),
        stats=stats,
        charts=charts,
        summary=summary,
        model=llm.model if llm is not None else None,
    )


def run_analysis(data: bytes, filename: str, llm: LLM | None, now: datetime | None = None) -> Report:
    """Raises LoadError if the file cannot be analysed."""
    return analyse_dataframe(load_csv(data, filename), filename, llm, now)
