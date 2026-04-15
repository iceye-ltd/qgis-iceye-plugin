# coding=utf-8
"""Tests for parse_iso8601_datetime (Python 3.10 fromisoformat quirks)."""

from datetime import datetime, timezone

import pytest

from ICEYE_toolbox.core.metadata import parse_iso8601_datetime


@pytest.mark.parametrize(
    (
        "raw",
        "expected_year",
        "expected_month",
        "expected_day",
        "expected_hour",
        "expected_micro",
    ),
    [
        ("2025-11-09T14:15:25.87+00:00", 2025, 11, 9, 14, 870000),
        ("2025-11-09T14:15:25.870000+00:00", 2025, 11, 9, 14, 870000),
        ("2025-11-09T14:15:25+00:00", 2025, 11, 9, 14, 0),
        ("2025-11-09T14:15:25.1234567+00:00", 2025, 11, 9, 14, 123456),
    ],
)
def test_parse_iso8601_datetime_fractions(
    raw: str,
    expected_year: int,
    expected_month: int,
    expected_day: int,
    expected_hour: int,
    expected_micro: int,
) -> None:
    """Two-digit and long fractional seconds normalize for fromisoformat."""
    dt = parse_iso8601_datetime(raw)
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    assert dt.year == expected_year
    assert dt.month == expected_month
    assert dt.day == expected_day
    assert dt.hour == expected_hour
    assert dt.microsecond == expected_micro


def test_parse_iso8601_datetime_z_suffix() -> None:
    """Z suffix maps to UTC."""
    dt = parse_iso8601_datetime("2025-01-02T03:04:05Z")
    assert dt == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_parse_iso8601_datetime_invalid() -> None:
    """Empty, None, and garbage inputs return None."""
    assert parse_iso8601_datetime("") is None
    assert parse_iso8601_datetime(None) is None
    assert parse_iso8601_datetime("not-a-date") is None
