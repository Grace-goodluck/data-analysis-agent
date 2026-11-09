"""Command line: exit codes, report file, disk cache (continued 2)."""

import pytest

from fakes import KNOWN, fixture_path, reply
from data_analysis_agent import cli


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # reports/ and results/cache/ are relative to the working directory
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANALYSIS_AGENT_MODEL", raising=False)
    return tmp_path


def test_without_key_writes_report_and_exits_1(workdir, capsys):
    assert cli.main([str(fixture_path(KNOWN))]) == 1
    reports = list((workdir / "reports").glob("known_properties-*.html"))
    assert len(reports) == 1
    out, err = capsys.readouterr()
    assert "Report written:" in out
    assert "Summary unavailable: GEMINI_API_KEY is not set" in err


def test_with_key_uses_gemini_and_disk_cache(workdir, monkeypatch):
    calls = []

    def fake_make_call(model, api_key):
        assert (model, api_key) == ("gemini-3.8-flash", "test-key")

        def call(prompt):
            calls.append(prompt)
            return reply()

        return call

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "make_gemini_call", fake_make_call)

    assert cli.main([str(fixture_path(KNOWN))]) == 0
    assert cli.main([str(fixture_path(KNOWN))]) == 0
    assert len(calls) == 1  # the second run came from results/cache/
    assert len(list((workdir / "results" / "cache").glob("*.json"))) == 1


def test_bad_file_exits_2(workdir, capsys):
    bad = workdir / "empty.csv"
    bad.write_bytes(b"a,b\n")
    assert cli.main([str(bad)]) == 2
    assert "header row but no data rows" in capsys.readouterr().err
    assert not (workdir / "reports").exists()


def test_oversized_file_exits_2_without_reading_it(workdir, capsys):
    big = workdir / "big.csv"
    big.write_bytes(b"a\n" + b"1\n" * (3 * 1024 * 1024))  # ~6 MB
    assert cli.main([str(big)]) == 2
    assert "larger than 5 MB" in capsys.readouterr().err


def test_missing_file_is_an_argument_error(workdir):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(workdir / "nope.csv")])
    assert exc.value.code == 2
