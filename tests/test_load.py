"""Decision 1: one CSV, at most 5 MB and 50,000 rows; anything else is rejected with a clear message."""

import pytest

from fakes import KNOWN, fixture_bytes
from data_analysis_agent.load import LoadError, load_csv, size_label


def test_fixture_loads():
    df = load_csv(fixture_bytes(KNOWN), KNOWN)
    assert df.shape == (24, 6)


@pytest.mark.parametrize(
    "data, filename, message",
    [
        (b"a,b\n1,2\n", "data.txt", "Only .csv files"),
        (b"a,b\n1,2\n", "data.xlsx", "Only .csv files"),
        (b"", "data.csv", "empty"),
        (b"  \n\n", "data.csv", "empty"),
        (b"a,b\n", "data.csv", "header row but no data rows"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image.csv", "does not look like a CSV"),
        (b'a,b\n"unclosed,2\n', "data.csv", "could not be read as CSV"),
    ],
    ids=["txt", "xlsx", "empty", "blank", "header-only", "binary", "bad-quote"],
)
def test_rejections_have_clear_messages(data, filename, message):
    with pytest.raises(LoadError, match=message):
        load_csv(data, filename)


def test_extension_is_case_insensitive():
    assert load_csv(b"a,b\n1,2\n", "DATA.CSV").shape == (1, 2)


def test_too_large():
    data = b"a,b\n" + b"1,2\n" * 1000
    with pytest.raises(LoadError, match="larger than 2 KB"):
        load_csv(data, "big.csv", max_bytes=2048)


def test_too_many_rows():
    data = b"a\n" + b"1\n" * 50_001
    with pytest.raises(LoadError, match="more than 50,000 rows"):
        load_csv(data, "rows.csv")
    assert len(load_csv(b"a\n" + b"1\n" * 50_000, "rows.csv")) == 50_000


def test_semicolon_separated_file():
    df = load_csv(b"name;value\nA;1\nB;2\n", "semi.csv")
    assert list(df.columns) == ["name", "value"]


def test_byte_order_mark_and_latin1():
    assert list(load_csv("﻿city,n\nZürich,1\n".encode("utf-8"), "bom.csv").columns) == ["city", "n"]
    df = load_csv("city,n\nZürich,1\n".encode("latin-1"), "latin.csv")
    assert df.loc[0, "city"] == "Zürich"


def test_size_label():
    assert size_label(5 * 1024 * 1024) == "5 MB"
    assert size_label(2048) == "2 KB"
    assert size_label(500) == "500 bytes"
