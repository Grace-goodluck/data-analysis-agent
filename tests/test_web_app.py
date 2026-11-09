"""The web layer, in-process: FakeLLM, fake clock, jobs run immediately. No network."""

import logging

import pytest
from fastapi.testclient import TestClient

from fakes import INVENTED_SUMMARY, KNOWN, FakeLLM, FakeNow, ImmediateExecutor, fixture_bytes, reply
from data_analysis_agent.ports import LLMError
from data_analysis_agent.web import app as web_app
from data_analysis_agent.web.app import WebSettings, create_app
from data_analysis_agent.web.limits import Limiter, Limits

NOTICE = "Don't upload personal or confidential data."
TINY_SECRET_CSV = b"name,value\nZEBRA-SECRET-42,1\nB,2\nC,3\n"
TINY_REPLY = reply("The file has 3 rows and 2 columns.", steps=["Add more rows."])


def make_client(llm="default", limits=Limits(), settings=None):
    llm = FakeLLM([reply()] * 25) if llm == "default" else llm
    settings = settings or WebSettings(limits=limits)
    app = create_app(
        llm=llm,
        limiter=Limiter(settings.limits, now=FakeNow()),
        settings=settings,
        executor=ImmediateExecutor(),
    )
    return TestClient(app), llm


def upload(client, data=None, filename=KNOWN, ip="203.0.113.7"):
    data = fixture_bytes(KNOWN) if data is None else data
    return client.post(
        "/analyse",
        files={"file": (filename, data, "text/csv")},
        headers={"X-Forwarded-For": f"{ip}, 10.0.0.1"},
        follow_redirects=False,
    )


def analyse(client, **kwargs):
    response = upload(client, **kwargs)
    assert response.status_code == 303, response.text
    return client.get(response.headers["location"])


# --- form and notice --------------------------------------------------------------------

def test_form_shows_notice_and_file_input():
    client, _ = make_client()
    page = client.get("/")
    assert page.status_code == 200
    assert NOTICE in page.text
    assert 'enctype="multipart/form-data"' in page.text
    assert 'name="file" type="file"' in page.text
    assert "never stored" in page.text


# --- happy path ---------------------------------------------------------------------------

def test_upload_shows_the_report_with_summary():
    client, llm = make_client()
    page = analyse(client)
    assert page.status_code == 200
    assert NOTICE in page.text
    assert "North is the most common clinic, with 6 rows." in page.text
    assert "<h2>Data quality</h2>" in page.text and "<h2>Trend over time</h2>" in page.text
    assert page.text.count('src="data:image/png;base64,') == 4
    assert len(llm.prompts) == 1
    assert "P007" not in llm.prompts[0]  # Gemini never sees the rows


def test_download_is_the_same_self_contained_html():
    client, _ = make_client()
    location = upload(client).headers["location"]
    download = client.get(f"{location}/report.html")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/html")
    assert 'attachment; filename="known_properties-' in download.headers["content-disposition"]
    assert download.text.startswith("<!doctype html>")
    assert download.text.count('src="data:image/png;base64,') == 4


def test_csv_content_is_escaped_on_the_page():
    client, _ = make_client(llm=FakeLLM([TINY_REPLY]))
    page = analyse(client, data=b"<script>alert(1)</script>,value\nx,1\ny,2\nz,3\n", filename="evil.csv")
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page.text


# --- bad files: clear messages, no quota used ---------------------------------------------

@pytest.mark.parametrize(
    "data, filename, message",
    [
        (b"a,b\n1,2\n", "data.txt", "Only .csv files can be analysed."),
        (b"a,b\n", "data.csv", "header row but no data rows"),
        (b"", "data.csv", "The file is empty."),
    ],
)
def test_bad_files_rejected_with_a_clear_message(data, filename, message):
    client, llm = make_client()
    response = upload(client, data=data, filename=filename)
    assert response.status_code == 400
    assert message in response.text
    assert llm.prompts == []


def test_missing_file_field():
    client, _ = make_client()
    response = client.post("/analyse", data={"other": "x"}, follow_redirects=False)
    assert response.status_code == 400 and "Please choose a CSV file." in response.text


def test_rejected_files_do_not_use_up_the_visitor_quota():
    client, _ = make_client()
    for _ in range(5):
        assert upload(client, data=b"", filename="x.csv").status_code == 400
    for _ in range(3):
        assert upload(client).status_code == 303


def test_oversized_upload_refused_before_reading():
    client, llm = make_client(settings=WebSettings(max_upload_bytes=1000))
    response = upload(client, data=b"a\n" + b"1\n" * 40_000)  # ~80 KB: over the limit plus form overhead
    assert response.status_code == 413
    assert "larger than 1000 bytes" in response.text
    assert llm.prompts == []


def test_upload_just_over_the_limit_rejected_by_the_loader():
    client, _ = make_client(settings=WebSettings(max_upload_bytes=2048))
    response = upload(client, data=b"a,b\n" + b"1,2\n" * 1000)  # ~4 KB
    assert response.status_code == 400
    assert "larger than 2 KB" in response.text


# --- summary unavailable: generic public reasons ------------------------------------------

def test_rejected_summary_shows_statistics_and_a_public_reason():
    client, _ = make_client(llm=FakeLLM([reply(INVENTED_SUMMARY)]))
    page = analyse(client)
    assert "Summary unavailable" in page.text
    assert "contained a number that is not in the computed statistics" in page.text
    assert "9,876" not in page.text and "average wait" not in page.text
    assert "<h2>Numeric columns</h2>" in page.text


def test_gemini_failure_shows_a_generic_reason():
    client, _ = make_client(llm=FakeLLM([LLMError("Gemini call failed: 503 UNAVAILABLE internal-detail")]))
    page = analyse(client)
    assert "the AI service is busy or failed" in page.text
    assert "503" not in page.text and "internal-detail" not in page.text


def test_no_key_says_summary_not_configured():
    client, _ = make_client(llm=None)
    page = analyse(client)
    assert "the AI summary is not configured on this demo" in page.text
    assert "<h2>Charts</h2>" in page.text


# --- limits through the web ----------------------------------------------------------------

def test_fourth_analysis_in_an_hour_is_refused_with_retry_after():
    client, llm = make_client()
    for _ in range(3):
        assert upload(client).status_code == 303
    refused = upload(client)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0
    assert "3 analyses for this hour" in refused.text
    assert len(llm.prompts) == 3  # nothing analysed
    assert upload(client, ip="198.51.100.2").status_code == 303  # another visitor


def test_daily_total_cap_refuses_new_visitors():
    client, _ = make_client(limits=Limits(analyses_per_day=2))
    assert upload(client, ip="192.0.2.1").status_code == 303
    assert upload(client, ip="192.0.2.2").status_code == 303
    refused = upload(client, ip="192.0.2.3")
    assert refused.status_code == 429 and "limit of 2 analyses for today" in refused.text


# --- results, logs, health, port, secrets ---------------------------------------------------

def test_unknown_job_is_404():
    client, _ = make_client()
    assert client.get("/analyse/not-a-real-id").status_code == 404
    assert client.get("/analyse/not-a-real-id/report.html").status_code == 404


def test_logs_never_contain_file_name_columns_or_values(caplog):
    client, _ = make_client(llm=FakeLLM([TINY_REPLY]))
    with caplog.at_level(logging.INFO, logger="data_analysis_agent.web"):
        analyse(client, data=TINY_SECRET_CSV, filename="patients_private.csv")
        upload(client, data=b"ZEBRA-SECRET-42\n", filename="patients_private.csv")  # header only: rejected
    text = caplog.text
    assert "rows=3 cols=2" in text and "summary" in text and "result=bad_file" in text
    for secret in ("ZEBRA-SECRET-42", "patients_private", "name", "value"):
        assert secret not in text


@pytest.mark.parametrize("method", ["GET", "HEAD"])  # UptimeRobot's HTTP monitors send HEAD
def test_health_is_quick_and_runs_no_analysis_or_gemini(method):
    llm = FakeLLM([])  # any Gemini call fails the test
    client, _ = make_client(llm=llm, limits=Limits(analyses_per_visitor_per_hour=1))
    for _ in range(3):
        response = client.request(method, "/health")
        assert response.status_code == 200
        if method == "GET":
            assert response.text == "ok"
    assert llm.prompts == []


def test_port_from_env():
    assert web_app.port_from_env({"PORT": "10000"}) == 10000
    assert web_app.port_from_env({"PORT": " 8080 "}) == 8080
    assert web_app.port_from_env({}) == 8000
    assert web_app.port_from_env({"PORT": ""}) == 8000
    with pytest.raises(ValueError):
        web_app.port_from_env({"PORT": "not-a-port"})


def test_main_runs_uvicorn_on_port_from_env(monkeypatch):
    import sys
    import types

    calls = []
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: calls.append((app, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("PORT", "10000")

    web_app.main()
    assert calls == [(
        "data_analysis_agent.web.app:build_default_app",
        {"factory": True, "host": "0.0.0.0", "port": 10000, "access_log": False},
    )]


def test_default_app_reads_key_never_shows_it_and_writes_nothing_to_disk(monkeypatch, tmp_path):
    secret = "test-secret-key-do-not-show"
    seen = {}

    def fake_make_call(model, api_key):
        seen["key"] = api_key
        return lambda prompt: reply()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr(web_app, "make_gemini_call", fake_make_call)

    client = TestClient(web_app.build_default_app(executor=ImmediateExecutor()))
    assert seen["key"] == secret
    page = analyse(client)
    assert "North is the most common clinic" in page.text
    for path in ("/", "/health", "/analyse/nope"):
        assert secret not in client.get(path).text
    assert secret not in page.text
    assert list(tmp_path.iterdir()) == []  # no cache, report or upload written (continued 2)
