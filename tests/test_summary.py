"""Decision 3: one Gemini call writes the summary; a number not in the statistics rejects it."""

import json

import pytest

from fakes import GOOD_STEPS, GOOD_SUMMARY, INVENTED_SUMMARY, KNOWN, FakeLLM, fixture_bytes, reply
from data_analysis_agent.analyse import run_analysis
from data_analysis_agent.ports import LLMError
from data_analysis_agent.report import render_html
from data_analysis_agent.summary import parse_reply, summarise


def analyse(llm):
    return run_analysis(fixture_bytes(KNOWN), KNOWN, llm)


# --- the required rejection test (SCOPE: Evaluation) ---------------------------------------

def test_summary_with_a_number_not_in_the_statistics_is_rejected():
    llm = FakeLLM([reply(INVENTED_SUMMARY)])
    report = analyse(llm)

    assert len(llm.prompts) == 1
    assert report.summary.available is False
    assert report.summary.code == "rejected"
    assert "9,876" in report.summary.reason

    html = render_html(report)
    assert "Summary unavailable" in html
    assert "The average wait was" not in html  # no part of the rejected summary is shown
    assert "Remove the 1 duplicate row." not in html  # nor its next steps
    for section in ("Data quality", "Columns", "Numeric columns", "Most common values",
                    "Relationships between numeric columns", "Trend over time", "Charts"):
        assert f"<h2>{section}</h2>" in html  # the statistics are still reported in full


# --- accepted and other failures ------------------------------------------------------------

def test_summary_whose_numbers_all_match_is_shown():
    report = analyse(FakeLLM([reply()]))
    assert report.summary.available, report.summary.reason
    assert report.summary.text == GOOD_SUMMARY
    assert report.summary.next_steps == tuple(GOOD_STEPS)
    html = render_html(report)
    assert "North is the most common clinic, with 6 rows." in html
    assert "<li>Check the 4 charts.</li>" in html
    assert "fake-model" in html


def test_invented_number_in_next_steps_also_rejects():
    report = analyse(FakeLLM([reply(steps=["Collect 500 more rows."])]))
    assert report.summary.code == "rejected"
    assert "500" in report.summary.reason


def test_gemini_failure_still_reports_statistics():
    report = analyse(FakeLLM([LLMError("Gemini call failed: 503 UNAVAILABLE")]))
    assert report.summary.code == "llm_failed"
    assert report.stats.rows == 24 and len(report.charts) == 4
    assert "Summary unavailable" in render_html(report)


def test_unreadable_reply():
    report = analyse(FakeLLM(["Here is your summary: the data looks fine."]))
    assert report.summary.code == "unreadable"


def test_no_key_means_no_call():
    report = analyse(None)
    assert report.summary.code == "no_key"
    assert report.model is None


def test_fenced_json_reply_is_accepted():
    fenced = "```json\n" + reply() + "\n```"
    assert analyse(FakeLLM([fenced])).summary.available


@pytest.mark.parametrize(
    "text, message",
    [
        ("[]", "not a JSON object"),
        (json.dumps({"summary": ""}), 'no "summary"'),
        (json.dumps({"summary": "ok", "next_steps": "do things"}), "not a list of strings"),
    ],
)
def test_parse_reply_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_reply(text)


def test_parse_reply_caps_and_trims_steps():
    text, steps = parse_reply(json.dumps({"summary": " s ", "next_steps": [" a ", "", *"bcdefgh"]}))
    assert text == "s"
    assert steps == ["a", "b", "c", "d", "e", "f"]


def test_summarise_uses_the_numbers_gemini_saw():
    stats = {"file": {"rows": 10}}
    assert summarise(stats, FakeLLM([json.dumps({"summary": "There are 10 rows."})])).available
    assert not summarise(stats, FakeLLM([json.dumps({"summary": "There are 11 rows."})])).available
