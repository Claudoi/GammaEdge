# portfolio/attribution/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

Method = Literal["generic", "brinson", "euler"]


@dataclass
class AttributionResult:
    """
    Lightweight container for attribution outputs.

    For now it only stores:
    - method: label of the method ('generic', 'brinson', 'euler', ...)
    - contributions: Polars DataFrame with per-date/per-asset contributions
    - meta: optional dict with metadata (column names, etc.)
    """

    method: Method
    contributions: pl.DataFrame
    meta: dict[str, Any] | None = None


def _ensure_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalize to LazyFrame so we can chain joins and selects consistently.
    """
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def compute_portfolio_contributions(
    *,
    weights: pl.DataFrame | pl.LazyFrame,
    returns: pl.DataFrame | pl.LazyFrame,
    date_col: str = "date",
    method: Method = "generic",
) -> AttributionResult:
    """
    Compute simple portfolio contributions: contrib = weight * return.

    Parameters
    ----------
    weights:
        Polars DataFrame/LazyFrame with `date_col` and one column
        per asset/bucket with portfolio weights.
    returns:
        Polars DataFrame/LazyFrame with `date_col` and one column
        per asset/bucket with returns, on the same time grid as `weights`.
    date_col:
        Name of the date column used to align both DataFrames.
    method:
        Label of the attribution method. Currently informational only,
        but keeps the API ready for plugging in 'brinson' / 'euler', etc.

    Returns
    -------
    AttributionResult
        `contributions` has the schema:
        [date_col] + same asset columns as `weights`/`returns`,
        with values equal to weight * return.
    """
    w_lazy = _ensure_lazy(weights)
    r_lazy = _ensure_lazy(returns)

    # Inner join on date; return columns get a '_ret' suffix
    joined = w_lazy.join(r_lazy, on=date_col, how="inner", suffix="_ret")

    # Asset columns: all except date and the '_ret' suffixed ones
    asset_cols = [c for c in joined.schema if c != date_col and not c.endswith("_ret")]

    # Build expressions: date + contributions per asset
    exprs: list[pl.Expr] = [pl.col(date_col)]
    for c in asset_cols:
        exprs.append((pl.col(c) * pl.col(f"{c}_ret")).alias(c))

    contributions = joined.select(exprs).collect()

    return AttributionResult(
        method=method,
        contributions=contributions,
        meta={"date_col": date_col, "asset_cols": asset_cols},
    )
