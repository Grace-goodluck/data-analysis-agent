"""CSV bytes -> DataFrame, or LoadError with a message fit to show the person (decision 1).

Everything happens in memory; nothing is written to disk.
"""

import io

import pandas as pd

from data_analysis_agent.config import MAX_ROWS, MAX_UPLOAD_BYTES


class LoadError(Exception):
    """The file cannot be analysed. The message is safe to show to the person who sent it."""


def size_label(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):g} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:g} KB"
    return f"{n_bytes} bytes"


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")  # never fails


def _parse(text: str, max_rows: int, sep: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, nrows=max_rows + 1, low_memory=False, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise LoadError("The file is empty.") from None
    except (pd.errors.ParserError, ValueError):
        raise LoadError(
            "The file could not be read as CSV. Check that every row has the same number of fields."
        ) from None


def load_csv(
    data: bytes,
    filename: str,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_rows: int = MAX_ROWS,
) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise LoadError("Only .csv files can be analysed.")
    if not data.strip():
        raise LoadError("The file is empty.")
    if len(data) > max_bytes:
        raise LoadError(f"The file is larger than {size_label(max_bytes)}.")
    if b"\x00" in data[:8192]:
        raise LoadError("The file does not look like a CSV text file.")

    text = _decode(data)
    df = _parse(text, max_rows, ",")
    first_line = text.lstrip("﻿").split("\n", 1)[0]
    if df.shape[1] == 1 and ";" in first_line:  # common in countries that use a decimal comma
        df = _parse(text, max_rows, ";")

    if df.shape[1] == 0:
        raise LoadError("The file has no columns.")
    if len(df) == 0:
        raise LoadError("The file has a header row but no data rows.")
    if len(df) > max_rows:
        raise LoadError(f"The file has more than {max_rows:,} rows.")
    return df
