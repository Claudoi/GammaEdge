# portfolio/attribution/brinson.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from portfolio.attribution.brinson_utils import coerce_brinson_timeseries_to_long
from portfolio.attribution.engine import compute_portfolio_contributions

BRINSON_METRICS = ["alloc", "select", "interact", "total"]


@dataclass
class BrinsonAttribution:
    """
    Simple container for Brinson results.

    - timeseries: long-format DataFrame (date, group_id, metrics).
    - by_group: aggregated by group_id.
    - total: single-row DataFrame with period totals.
    """

    timeseries: pl.DataFrame
    by_group: pl.DataFrame
    total: pl.DataFrame


def _build_agg_exprs(how: Literal["sum", "mean"]) -> list[pl.Expr]:
    """
    Build aggregation expressions for Brinson metrics.

    `how` is restricted to "sum" or "mean" at the type level, so we do not
    need a runtime error branch.
    """
    if how == "sum":
        return [pl.col(m).sum().alias(m) for m in BRINSON_METRICS]
    # At this point, by type, it can only be "mean".
    return [pl.col(m).mean().alias(m) for m in BRINSON_METRICS]


def compute_brinson_attribution(
    df: pl.DataFrame,
    how: Literal["sum", "mean"] = "sum",
) -> BrinsonAttribution:
    """
    Normalize any "Brinson-style" timeseries DataFrame and return:

    - timeseries: standard long format via `coerce_brinson_timeseries_to_long`.
    - by_group: aggregated by group_id (sum or mean).
    - total: metrics aggregated over the whole period.
    """
    ts = coerce_brinson_timeseries_to_long(df)
    agg_exprs = _build_agg_exprs(how)

    by_group = ts.group_by("group_id").agg(agg_exprs).sort("group_id")
    total = ts.select([pl.col(m).sum().alias(m) for m in BRINSON_METRICS])

    return BrinsonAttribution(timeseries=ts, by_group=by_group, total=total)


def _call_portfolio_contributions_from_long(
    df: pl.DataFrame,
    weights_col: str,
    returns_col: str,
    date_col: str,
) -> pl.DataFrame:
    """
    Given a *long* df (date, asset, w, r), build the wide-format DataFrames
    expected by `compute_portfolio_contributions` and return a long df with
    contributions per (date, asset).
    """
    if "asset" not in df.columns:
        msg = "Expected a long dataframe with an 'asset' column."
        raise ValueError(msg)

    # 1) Pivot to wide format for weights and returns
    weights_wide = (
        df.select([date_col, "asset", weights_col])
        .pivot(values=weights_col, index=date_col, on="asset")
        .sort(date_col)
    )

    returns_wide = (
        df.select([date_col, "asset", returns_col])
        .pivot(values=returns_col, index=date_col, on="asset")
        .sort(date_col)
    )

    # 2) Call the engine in wide format
    contrib_wide = compute_portfolio_contributions(
        weights=weights_wide,
        returns=returns_wide,
        date_col=date_col,
        method="brinson",
    ).contributions

    # 3) Back to long: (date, asset, contribution)
    contrib_long = contrib_wide.melt(
        id_vars=date_col,
        variable_name="asset",
        value_name="contribution",
    )

    return contrib_long


def run_brinson_engine(
    df: pl.DataFrame,
    weights_col: str = "w",
    returns_col: str = "r",
    date_col: str = "date",
) -> pl.DataFrame:
    """
    Integrate a long-format Brinson-style dataframe with the attribution engine.

    Returns the original dataframe enriched with contributions per
    asset/date as computed by `compute_portfolio_contributions`.
    """
    contrib_long = _call_portfolio_contributions_from_long(
        df=df,
        weights_col=weights_col,
        returns_col=returns_col,
        date_col=date_col,
    )

    out = df.join(
        contrib_long,
        on=[date_col, "asset"],
        how="left",
    )
    return out
