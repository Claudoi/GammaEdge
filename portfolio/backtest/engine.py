# portfolio/backtest/engine.py  (añadir)
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import polars as pl

def _rebalance_dates_from_freq(dates: pl.Series, freq: str = "1mo") -> pl.Series:
    """
    Devuelve puntos de rebalance (último día de cada ventana).
    """
    df = pl.DataFrame({"date": dates})
    out = (
        df.lazy()
          .group_by_dynamic("date", every=freq, closed="right", label="right")
          .agg(pl.col("date").last().alias("rb_date"))
          .select("rb_date")
          .collect()["rb_date"]
    )
    return out

def backtest_rebalanced(
    df_ret_wide: pl.DataFrame,                  # ['date', tickers...], ordenada
    *,
    lookback: int = 252,
    rebalance_freq: str = "1mo",
    cost_bps: float = 0.0,
    allocator: Callable[[pl.DataFrame], np.ndarray],  # recibe ventana (lookback) y devuelve w (N,)
    bench_weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Backtest sencillo con rebalance periódico y costes lineales en turnover.
    - allocator recibe returns wide de la ventana (últimos 'lookback' puntos) y devuelve pesos para el próximo periodo.
    - bench_weights (opcional) para calcular tracking error con respecto a benchmark estático.
    """
    assert "date" in df_ret_wide.columns
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    N = len(tickers)
    df = df_ret_wide.sort("date")
    dates = df["date"]

    rb = _rebalance_dates_from_freq(dates, rebalance_freq)
    rb_set = set(rb.to_list())

    W = []
    TO = []
    equity = []
    te_series = []

    w_prev = np.full(N, 1.0 / N)
    eq = 1.0

    for i in range(lookback, df.height):
        d = dates[i]
        # rebalance si toca
        if d in rb_set:
            win = df.slice(i - lookback, lookback)
            w_new = allocator(win)  # (N,)
            w_new = np.asarray(w_new, dtype=float)
            w_new = w_new / max(w_new.sum(), 1e-12)

            # costes (turnover * cost_bps)
            to = float(np.sum(np.abs(w_new - w_prev)))
            cost = to * (cost_bps / 10000.0)
            TO.append(to)
            W.append(w_new.copy())
            w_prev = w_new.copy()
        # aplicar retorno del día i (si ya tenemos w_prev)
        r = df.row(i, named=True)
        rets = np.array([r[t] for t in tickers], dtype=float)
        eq *= (1.0 + float(np.nansum(w_prev * rets)) - cost if "cost" in locals() else (1.0 + float(np.nansum(w_prev * rets))))
        cost = 0.0
        equity.append(eq)

        # TE diario vs benchmark estático (si se pasa)
        if bench_weights is not None:
            v = w_prev - bench_weights
            S = np.outer(rets, rets)  # proxy Σ_t; para TE diario aproximado usa var real si la tienes
            te_daily = float(np.sqrt(max(v @ S @ v, 0.0)))
            te_series.append(te_daily)

    out = {
        "dates": dates[lookback:].to_list(),
        "equity": np.array(equity),
        "weights": np.array(W) if W else np.zeros((0, N)),
        "turnover": np.array(TO),
        "te_daily_proxy": np.array(te_series) if te_series else None,
        "tickers": tickers,
    }
    return out
