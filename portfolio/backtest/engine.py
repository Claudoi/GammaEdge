# portfolio/backtest/engine.py
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
import polars as pl

from portfolio.core.compat import dataclass_compat as dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Types / dataclasses
# ──────────────────────────────────────────────────────────────────────────────

RebalanceFreq = Literal["D", "W", "M", "Q"]  # daily, weekly, monthly, quarterly (pandas resampling)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start: str | None = None
    end: str | None = None
    rebalance: RebalanceFreq | str = "M"  # accepts "M"/"W"… or dynamic like "1mo"/"1w"/"3mo"
    fees_bps: float = 0.0  # round-trip fees in bps
    slippage_bps: float = 0.0  # slippage in bps (added to fees)
    initial_capital: float = 1.0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity: pl.DataFrame  # ["date","equity","ret"]
    weights: pl.DataFrame  # long: ["date","ticker","weight"]
    trades: pl.DataFrame  # long: ["date","ticker","d_weight","cost_bps"]
    stats: dict  # aggregated metrics (CAGR, Sharpe, MaxDD, …)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_wide_prices(df_prices_long: pl.DataFrame) -> pl.DataFrame:
    """
    Convert long ["date","ticker","price"] to wide ["date", tickers...]
    """
    return (
        df_prices_long.select(
            [
                pl.col("date").alias("date"),
                pl.col("ticker").cast(pl.Utf8).alias("ticker"),
                pl.col("price").cast(pl.Float64).alias("price"),
            ]
        )
        .pivot(index="date", columns="ticker", values="price")
        .sort("date")
    )


def _dates_to_rebalance_pandas(dates: pd.DatetimeIndex, freq: RebalanceFreq) -> pd.DatetimeIndex:
    """
    Rebalance by last observation of each resample period (pandas “D/W/M/Q”).
    """
    s = pd.Series(index=dates, data=1.0)
    ix = s.resample(freq).last().index
    if len(dates) and (dates[0] not in ix):
        ix = ix.insert(0, dates[0])
    return ix.intersection(dates)


def _rebalance_dates_from_freq_polars(dates: pl.Series, freq: str = "1mo") -> pl.Series:
    """
    Dynamic-window rebalance using Polars (“1w”, “1mo”, “3mo”, …).
    Returns last date of each dynamic group.
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


def _quick_stats_from_equity(equity_df: pl.DataFrame) -> dict:
    """
    Quick metrics without external deps (CAGR, Sharpe, MaxDD).
    Assumes ~252 trading days for annualization.
    """
    if equity_df.is_empty() or equity_df.height <= 1:
        return {}
    ann = 252.0
    r = equity_df["ret"].to_numpy()
    mu_a = float(np.nanmean(r)) * ann
    vol_a = float(np.nanstd(r, ddof=1)) * np.sqrt(ann)
    sharpe = mu_a / vol_a if vol_a > 1e-12 else np.nan

    curve = equity_df["equity"].to_numpy()
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    maxdd = float(np.nanmin(dd))

    # crude CAGR using number of steps ~ height
    cagr = float((curve[-1] / max(curve[0], 1e-12)) ** (ann / max(1, equity_df.height)) - 1.0)

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd}


# ──────────────────────────────────────────────────────────────────────────────
# 1) Classic engine (dict output) — compatible with current UI
# ──────────────────────────────────────────────────────────────────────────────


def backtest_rebalanced(
    df_ret_wide: pl.DataFrame,  # ['date', tickers...], sorted
    *,
    lookback: int = 252,
    rebalance_freq: str = "1mo",  # “1w”, “1mo”, “3mo”, …
    cost_bps: float = 0.0,
    allocator: Callable[
        [pl.DataFrame], np.ndarray
    ],  # receives trailing window (lookback) → weights (N,)
    bench_weights: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Simple periodic-rebalance backtest with linear turnover costs.
    - allocator receives the last 'lookback' rows of returns (wide) and must return next-period weights.
    - bench_weights (optional) enables a daily TE proxy vs static benchmark.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("df_ret_wide must contain 'date' column")

    tickers = [c for c in df_ret_wide.columns if c != "date"]
    N = len(tickers)
    df = df_ret_wide.sort("date")
    dates = df["date"]

    # Rebalance dates (Polars dynamic)
    rb = _rebalance_dates_from_freq_polars(dates, rebalance_freq)
    rb_set = set(rb.to_list())

    W: list[np.ndarray] = []  # weights at each rebalance (rows align with RB_DATES)
    TO: list[float] = []  # turnover at each rebalance
    RB_DATES: list[object] = []  # dates of each rebalance (same length as TO and W)
    equity: list[float] = []  # daily equity curve (from lookback onward)
    te_series: list[float] = []  # daily TE proxy

    w_prev = np.full(N, 1.0 / max(N, 1))
    eq = 1.0

    # Main loop
    for i in range(lookback, df.height):
        d = dates[i]
        cost = 0.0  # rebalance cost applied on this step

        # Rebalance when scheduled
        if d in rb_set:
            win = df.slice(i - lookback, lookback)
            w_new = allocator(win)  # (N,)
            w_new = np.asarray(w_new, dtype=float)

            s = float(np.sum(w_new))
            w_new = w_new / s if s > 1e-12 else np.full(N, 1.0 / max(N, 1))

            # turnover & cost (bps on absolute weight change)
            to = float(np.nansum(np.abs(w_new - w_prev)))
            cost = to * (cost_bps / 10000.0)

            TO.append(to)
            W.append(w_new.copy())
            RB_DATES.append(d)  # <— store the actual rebalance date
            w_prev = w_new.copy()

        # apply portfolio return for day i
        r = df.row(i, named=True)
        rets = np.array([r[t] for t in tickers], dtype=float)
        port_ret = float(np.nansum(w_prev * rets))
        eq *= 1.0 + port_ret - cost
        equity.append(eq)

        # daily TE proxy vs static benchmark (if provided)
        if bench_weights is not None:
            v = w_prev - bench_weights
            S = np.outer(rets, rets)  # crude Σ_t; replace with a better estimator if available
            te_daily = float(np.sqrt(max(v @ S @ v, 0.0)))
            te_series.append(te_daily)

    out = {
        "dates": dates[lookback:].to_list(),  # daily grid from lookback
        "equity": np.array(equity, dtype=float),  # daily equity
        "weights": np.array(W, dtype=float) if W else np.zeros((0, N), float),  # (n_rebalances, N)
        "rebalance_dates": RB_DATES,  # <— NEW: list of rebalance dates
        "turnover": pl.DataFrame(
            {  # <— CHANGED: DF with ['date','turnover']
                "date": RB_DATES,
                "turnover": np.array(TO, dtype=float),
            }
        ),
        "te_daily_proxy": np.array(te_series, dtype=float) if te_series else None,
        "tickers": tickers,
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Engines that return BacktestResult (equity/weights/trades/metrics)
# ──────────────────────────────────────────────────────────────────────────────


def backtest_equal_weight_buy_hold(
    df_prices_long: pl.DataFrame,
    *,
    cfg: BacktestConfig = BacktestConfig(),
    benchmark: str | None = None,  # optional benchmark ticker (not used in baseline)
) -> BacktestResult:
    """
    Baseline: Buy & Hold equal-weight. Rebalances only at start (or per cfg.rebalance if you wish).
    No leverage.
    """
    prices = _ensure_wide_prices(df_prices_long)

    if cfg.start:
        prices = prices.filter(pl.col("date") >= pl.lit(cfg.start))
    if cfg.end:
        prices = prices.filter(pl.col("date") <= pl.lit(cfg.end))

    if prices.height < 2:
        empty = pl.DataFrame({"date": [], "equity": [], "ret": []})
        return BacktestResult(equity=empty, weights=empty, trades=empty, stats={})

    prices = prices.sort("date")
    dates = pd.DatetimeIndex(prices["date"].to_pandas())
    tickers = [c for c in prices.columns if c != "date"]

    # Rebalance dates via pandas or dynamic polars-style strings
    if isinstance(cfg.rebalance, str) and cfg.rebalance.lower().endswith(("w", "mo", "q", "d")):
        rb = _rebalance_dates_from_freq_polars(pl.Series(dates), str(cfg.rebalance))
        rb_mask = pl.Series(
            values=dates.isin(pd.DatetimeIndex(rb.to_list())), name="is_rb"
        ).to_numpy()
    else:
        rb_ix = _dates_to_rebalance_pandas(
            dates, cfg.rebalance if isinstance(cfg.rebalance, str) else "M"
        )
        rb_mask = dates.isin(rb_ix)

    fee_cost = (cfg.fees_bps + cfg.slippage_bps) / 1e4  # proportional cost

    # Price array
    P = prices.select(tickers).to_numpy()  # (T, N)

    equity = float(cfg.initial_capital)
    prev_w = np.zeros(len(tickers), dtype=float)

    equity_curve = [equity]
    rets = [0.0]
    weights_rows: list[dict] = []
    trades_rows: list[dict] = []

    for t in range(1, P.shape[0]):
        pt = P[t, :]
        p0 = P[t - 1, :]

        valid = np.isfinite(pt) & np.isfinite(p0)
        asset_ret = np.ones_like(pt)
        asset_ret[valid] = pt[valid] / p0[valid]
        port_ret = float(np.nansum(prev_w * (asset_ret - 1.0)))
        equity *= 1.0 + port_ret
        equity_curve.append(equity)
        rets.append(port_ret)

        # Rebalance if scheduled
        if rb_mask[t]:
            valid_now = np.isfinite(pt)
            n_valid = int(np.sum(valid_now))
            if n_valid > 0:
                w_target = np.zeros_like(prev_w)
                w_target[valid_now] = 1.0 / n_valid
            else:
                w_target = prev_w

            d_w = w_target - prev_w
            turnover = float(np.nansum(np.abs(d_w)))
            if turnover > 0:
                equity *= 1.0 - turnover * fee_cost

            for i, tk in enumerate(tickers):
                dw = float(d_w[i])
                if np.isfinite(dw) and dw != 0.0:
                    trades_rows.append(
                        {
                            "date": dates[t],
                            "ticker": tk,
                            "d_weight": dw,
                            "cost_bps": (cfg.fees_bps + cfg.slippage_bps),
                        }
                    )
            prev_w = w_target

        # Store weights at day t
        for i, tk in enumerate(tickers):
            w_i = float(prev_w[i])
            if np.isfinite(w_i) and w_i != 0.0:
                weights_rows.append({"date": dates[t], "ticker": tk, "weight": w_i})

    equity_df = pl.from_pandas(pd.DataFrame({"date": dates, "equity": equity_curve, "ret": rets}))
    weights_df = (
        pl.from_pandas(pd.DataFrame(weights_rows))
        if weights_rows
        else pl.DataFrame({"date": [], "ticker": [], "weight": []})
    )
    trades_df = (
        pl.from_pandas(pd.DataFrame(trades_rows))
        if trades_rows
        else pl.DataFrame({"date": [], "ticker": [], "d_weight": [], "cost_bps": []})
    )

    stats = _quick_stats_from_equity(equity_df)
    return BacktestResult(equity=equity_df, weights=weights_df, trades=trades_df, stats=stats)


def backtest_from_returns_alloc(
    df_ret_wide: pl.DataFrame,
    *,
    lookback: int,
    rebalance_freq: str,
    cost_bps: float,
    allocator: Callable[[pl.DataFrame], np.ndarray],
    bench_weights: np.ndarray | None = None,
) -> BacktestResult:
    """
    Classic engine variant that takes wide returns but returns a BacktestResult
    (equity + long weights/trades + stats).
    """
    out = backtest_rebalanced(
        df_ret_wide=df_ret_wide,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        cost_bps=cost_bps,
        allocator=allocator,
        bench_weights=bench_weights,
    )

    dates = pd.DatetimeIndex(out["dates"])
    equity = np.asarray(out["equity"], float)
    tickers = list(out["tickers"])
    W = np.asarray(out["weights"], float)  # (n_rebalances, N) or (0, N) if empty

    # Daily equity/ret from curve
    if equity.size > 0:
        ret = np.concatenate([[0.0], (equity[1:] / equity[:-1]) - 1.0])
    else:
        ret = np.array([], float)
    equity_df = pl.from_pandas(pd.DataFrame({"date": dates, "equity": equity, "ret": ret}))

    # Long weights (rebalance dates)
    weights_rows: list[dict] = []
    trades_rows: list[dict] = []
    if W.size > 0:
        for k in range(W.shape[0]):
            d = dates[min(k, len(dates) - 1)]
            w_k = W[k, :]
            for i, tk in enumerate(tickers):
                wi = float(w_k[i])
                if np.isfinite(wi) and wi != 0.0:
                    weights_rows.append({"date": d, "ticker": tk, "weight": wi})
            if k > 0:
                d_w = W[k, :] - W[k - 1, :]
                for i, tk in enumerate(tickers):
                    dw = float(d_w[i])
                    if np.isfinite(dw) and dw != 0.0:
                        trades_rows.append(
                            {"date": d, "ticker": tk, "d_weight": dw, "cost_bps": cost_bps}
                        )

    weights_df = (
        pl.from_pandas(pd.DataFrame(weights_rows))
        if weights_rows
        else pl.DataFrame({"date": [], "ticker": [], "weight": []})
    )
    trades_df = (
        pl.from_pandas(pd.DataFrame(trades_rows))
        if trades_rows
        else pl.DataFrame({"date": [], "ticker": [], "d_weight": [], "cost_bps": []})
    )

    stats = _quick_stats_from_equity(equity_df)
    return BacktestResult(equity=equity_df, weights=weights_df, trades=trades_df, stats=stats)
