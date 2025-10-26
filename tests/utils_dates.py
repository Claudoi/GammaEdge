# tests/utils_dates.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl


def _try_polars_datetime_range_new(
    start: datetime,
    end: datetime,
    interval: str,
    tz: str | None,
    closed: str,
) -> list[datetime] | None:
    """
    Try modern Polars API: pl.datetime_range(start=..., end=..., interval=..., closed=...).
    Returns a Python list or None if signature mismatch in this Polars version.
    """
    try:
        out = pl.datetime_range(  # type: ignore[attr-defined]
            start=start,
            end=end,
            interval=interval,
            time_zone=tz,  # modern Polars
            closed=closed,  # 'both'|'left'|'right'|'none'
        )
        # Some versions return Expr unless eager; normalize to Python list.
        if hasattr(out, "to_list"):
            return out.to_list()  # type: ignore[no-any-return]
        return pl.Series(out).to_list()
    except (TypeError, AttributeError):
        return None


def _try_polars_datetime_range_old(
    start: datetime,
    end: datetime,
    interval: str,
    tz: str | None,
    closed: str,
) -> list[datetime] | None:
    """
    Try older Polars API variants:
    - datetime_range(low=..., high=..., interval=..., closed=..., eager=True)
    - datetime_range(start=..., end=..., interval=..., eager=True) without time_zone
    """
    # Variant 1: low/high + eager
    try:
        out = pl.datetime_range(  # type: ignore[attr-defined]
            low=start,
            high=end,
            interval=interval,
            closed=closed,
            eager=True,
        )
        return out.to_list()
    except (TypeError, AttributeError):
        pass

    # Variant 2: start/end + eager (no time_zone)
    try:
        out = pl.datetime_range(  # type: ignore[attr-defined]
            start=start,
            end=end,
            interval=interval,
            closed=closed,
            eager=True,
        )
        return out.to_list()
    except (TypeError, AttributeError):
        return None


def _fallback_python_datetime_list(
    start: datetime,
    end: datetime,
    interval_days: int,
) -> list[datetime]:
    """
    Pure-Python fallback. Inclusive range with step = interval_days.
    """
    step = timedelta(days=interval_days)
    cur = start
    out: list[datetime] = []
    while cur <= end:
        out.append(cur)
        cur = cur + step
    return out


def make_dates(
    T: int,
    *,
    start: datetime | None = None,
    tz: str | None = None,
    closed: str = "both",
    interval: str = "1d",
) -> list[datetime]:
    """
    Version-agnostic date range for tests.
    """
    if T <= 0:
        return []

    # Build start and end
    if start is None:
        if tz:
            start = (
                datetime(2024, 1, 1, tzinfo=timezone.utc)
                if tz.upper() == "UTC"
                else datetime(2024, 1, 1)
            )
        else:
            start = datetime(2024, 1, 1)

    # For Polars attempts we pass explicit end = start + (T-1) days.
    end = start + timedelta(days=T - 1)

    # Try modern Polars API first
    dates = _try_polars_datetime_range_new(start, end, interval, tz, closed)
    if dates is not None and len(dates) == T:
        return dates

    # Then try old Polars API variants
    dates = _try_polars_datetime_range_old(start, end, interval, tz, closed)
    if dates is not None and len(dates) == T:
        return dates

    # Fallback to pure-Python generation
    try:
        if interval.lower().endswith("d"):
            days = int(interval[:-1])
            days = max(days, 1)
        else:
            days = 1
    except Exception:
        days = 1

    return _fallback_python_datetime_list(start, end, days)
