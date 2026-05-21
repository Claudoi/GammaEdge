"""Shared helpers used by multiple Streamlit pages.

Consolidates utilities that previously lived in individual pages (and were
copy-pasted). Adding new pages? Prefer importing helpers from here instead
of re-implementing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def to_pandas(df: Any):
    """Safely convert Polars or other table-like objects to pandas.

    On failure, logs a warning and returns the original object so callers
    that pass it to ``st.dataframe`` still degrade gracefully.

    Canonical version: matches ``06_Reporting._to_pandas`` (warns + ``exc_info``).
    The previous ``04_Backtest._to_pandas`` only debug-logged and is unified here.
    """
    try:
        return df.to_pandas()
    except Exception as exc:
        logger.warning("to_pandas conversion failed: %s", exc, exc_info=True)
        return df


def ensure_turnover_with_drift(bt: dict, df_wide: pl.DataFrame) -> tuple[list, np.ndarray]:
    """Ensure we have a turnover time series.

    1) Try to use what the engine returns (DataFrame or array).
    2) If missing, reconstruct turnover between rebalance dates using drifted
       weights.

    Canonical version: matches ``04_Backtest._ensure_turnover_with_drift``
    (includes a debug log on the unpacking failure branch). The version
    previously duplicated in ``07_Scenarios`` was functionally identical.
    """
    to_obj = bt.get("turnover")
    if to_obj is not None:
        try:
            # Case 1: Polars DataFrame with columns [date, turnover]
            if isinstance(to_obj, pl.DataFrame) and "turnover" in to_obj.columns:
                dates = (
                    to_obj.get_column("date").to_list()
                    if "date" in to_obj.columns
                    else list(range(to_obj.height))
                )
                vals = to_obj.get_column("turnover").to_numpy()
                return dates, np.asarray(vals, float)

            # Case 2: pandas DataFrame with columns [date, turnover]
            if isinstance(to_obj, pd.DataFrame) and "turnover" in to_obj.columns:
                dates = (
                    to_obj["date"].tolist()
                    if "date" in to_obj.columns
                    else list(range(len(to_obj)))
                )
                return dates, np.asarray(to_obj["turnover"].values, float)

            # Case 3: plain array-like
            arr = np.asarray(to_obj, float).ravel()
            if arr.size > 0:
                rb_dates = bt.get("rebalance_dates")
                dates = (
                    list(rb_dates)[-arr.size :]
                    if rb_dates is not None
                    else list(bt["dates"][-arr.size :])
                )
                return dates, arr
        except Exception:
            logger.debug(
                "turnover unpacking failed, falling back to reconstruction",
                exc_info=True,
            )

    # Fallback: reconstruct from pre/post-drift weights
    rb_dates = list(bt.get("rebalance_dates", []))
    W_reb = np.asarray(bt.get("weights", []), float)
    tick = list(bt.get("tickers", []))

    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0] or df_wide is None:
        return [], np.array([], float)

    have = set(df_wide.columns)
    miss = [t for t in tick if t not in have]
    if miss:
        df_wide = df_wide.with_columns(**{c: pl.lit(0.0, dtype=pl.Float64) for c in miss})
    df_wide = df_wide.select(["date", *tick]).sort("date")
    idx_map = {d: i for i, d in enumerate(df_wide.get_column("date").to_list())}

    def _norm(w: np.ndarray) -> np.ndarray:
        s = float(np.sum(w))
        return (w / s) if s > 1e-12 else w

    turns: list[float] = []
    out_dates: list[Any] = []
    w_prev = _norm(W_reb[0])

    for k in range(len(rb_dates) - 1):
        d0, d1 = rb_dates[k], rb_dates[k + 1]
        i0, i1 = idx_map.get(d0), idx_map.get(d1)

        if i0 is None or i1 is None or i1 <= i0:
            w_pre = w_prev
        else:
            R_seg = df_wide.slice(i0, i1 - i0).select(tick).to_numpy()
            G = np.prod(1.0 + np.nan_to_num(R_seg, nan=0.0), axis=0)
            w_pre = _norm(w_prev * G)

        w_new = _norm(W_reb[k + 1])
        to_k = 0.5 * float(np.sum(np.abs(w_new - w_pre)))
        turns.append(to_k)
        out_dates.append(d1)
        w_prev = w_new

    return out_dates, np.asarray(turns, float)


def safe_metrics_to_dict(metrics_df_obj: Any) -> dict[str, float] | None:
    """Coerce a 2-column metrics dataframe (Polars or pandas) into ``{name: value}``.

    Returns ``None`` if the input cannot be coerced. Used by the Attribution
    page (and any other page that wants to forward KPIs into the reporter).
    """
    try:
        pdf = metrics_df_obj.to_pandas() if hasattr(metrics_df_obj, "to_pandas") else metrics_df_obj
        return {str(k): float(v) for k, v in zip(pdf.iloc[:, 0], pdf.iloc[:, 1], strict=False)}
    except Exception:
        logger.debug("Could not coerce metrics_df to dict", exc_info=True)
        return None


__all__ = [
    "to_pandas",
    "ensure_turnover_with_drift",
    "safe_metrics_to_dict",
]
