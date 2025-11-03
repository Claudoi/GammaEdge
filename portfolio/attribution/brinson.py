# portfolio/attribution/brinson.py

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# We reuse the robust normalizer you just fixed.
from portfolio.backtest.brinson_utils import coerce_brinson_timeseries_to_long


@dataclass(frozen=True)
class BrinsonResult:
    """Container for Brinson time-series output."""

    df: pl.DataFrame  # columns: date, group_id, alloc, select, interact, total


def _validate_long_schema(df: pl.DataFrame) -> None:
    needed = {"date", "group_id", "alloc", "select", "interact", "total"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def compute_brinson_timeseries(frame: pl.DataFrame) -> BrinsonResult:
    """
    Normalize any supported input to Brinson long format and return it.

    This is a thin, predictable wrapper around `coerce_brinson_timeseries_to_long` so
    higher layers can depend on a stable function name in the attribution module.
    """
    # 1) Normalize to the canonical long format you just stabilized.
    long_df = coerce_brinson_timeseries_to_long(frame)

    # 2) Validate schema for downstream consumers.
    _validate_long_schema(long_df)

    # 3) Return as a structured result (keeps API future-proof).
    return BrinsonResult(df=long_df)
