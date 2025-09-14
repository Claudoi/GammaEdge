# portfolio/backtest/engine.py
from __future__ import annotations

from typing import Callable, Literal
import numpy as np
import pandas as pd
import polars as pl

from portfolio.core.compat import dataclass_compat as dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Tipos / dataclasses (estilo “pro”)
# ──────────────────────────────────────────────────────────────────────────────

RebalanceFreq = Literal["D", "W", "M", "Q"]  # diario, semanal, mensual, trimestral (para precios)

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start: str | None = None
    end: str | None = None
    rebalance: RebalanceFreq | str = "M"  # acepta "M"/"W"… o dinámico tipo "1mo"/"1w"/"3mo"
    fees_bps: float = 0.0        # comisiones (round-trip) en bps
    slippage_bps: float = 0.0    # deslizamiento en bps (se suma a fees)
    initial_capital: float = 1.0

@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity: pl.DataFrame            # ["date","equity","ret"]
    weights: pl.DataFrame           # long: ["date","ticker","weight"]
    trades: pl.DataFrame            # long: ["date","ticker","d_weight","cost_bps"]
    stats: dict                     # métricas agregadas (CAGR, Sharpe, MaxDD, etc.)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers comunes
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_wide_prices(df_prices_long: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte long ["date","ticker","price"] a wide ["date", tickers...]
    """
    return (
        df_prices_long
        .select([
            pl.col("date").alias("date"),
            pl.col("ticker").cast(pl.Utf8).alias("ticker"),
            pl.col("price").cast(pl.Float64).alias("price"),
        ])
        .pivot(index="date", columns="ticker", values="price")
        .sort("date")
    )

def _dates_to_rebalance_pandas(dates: pd.DatetimeIndex, freq: RebalanceFreq) -> pd.DatetimeIndex:
    """
    Rebalanceo por última observación del periodo para frecuencias pandas (“D/W/M/Q”).
    """
    s = pd.Series(index=dates, data=1.0)
    ix = s.resample(freq).last().index
    if len(dates) and (dates[0] not in ix):
        ix = ix.insert(0, dates[0])
    return ix.intersection(dates)

def _rebalance_dates_from_freq_polars(dates: pl.Series, freq: str = "1mo") -> pl.Series:
    """
    Rebalanceo usando ventana dinámica de Polars (“1w”, “1mo”, “3mo”, …).
    Devuelve la última fecha de cada grupo.
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
    Métricas rápidas sin depender de otros módulos (CAGR, Sharpe, MaxDD).
    Supone frecuencia diaria aprox 252 para anualizar.
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

    # aprox simple de CAGR con nº de pasos ~ height
    cagr = float((curve[-1] / max(curve[0], 1e-12)) ** (ann / max(1, equity_df.height)) - 1.0)

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd}


# ──────────────────────────────────────────────────────────────────────────────
# 1) Motor que YA te funcionaba (compat con la UI actual)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_rebalanced(
    df_ret_wide: pl.DataFrame,                  # ['date', tickers...], ordenada
    *,
    lookback: int = 252,
    rebalance_freq: str = "1mo",               # “1w”, “1mo”, “3mo”, …
    cost_bps: float = 0.0,
    allocator: Callable[[pl.DataFrame], np.ndarray],  # recibe ventana (lookback) y devuelve w (N,)
    bench_weights: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Backtest sencillo con rebalance periódico y costes lineales en turnover.
    - allocator recibe returns wide de la ventana (últimos 'lookback' puntos) y devuelve pesos para el próximo periodo.
    - bench_weights (opcional) para calcular un proxy de TE diario.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("df_ret_wide must contain 'date' column")

    tickers = [c for c in df_ret_wide.columns if c != "date"]
    N = len(tickers)
    df = df_ret_wide.sort("date")
    dates = df["date"]

    # Fechas de rebalanceo (Polars dinámico)
    rb = _rebalance_dates_from_freq_polars(dates, rebalance_freq)
    rb_set = set(rb.to_list())

    W = []
    TO = []
    equity = []
    te_series = []

    w_prev = np.full(N, 1.0 / max(N, 1))
    eq = 1.0

    # Bucle principal
    for i in range(lookback, df.height):
        d = dates[i]
        cost = 0.0  # costo del rebalanceo para este paso

        # Rebalanceo si toca
        if d in rb_set:
            win = df.slice(i - lookback, lookback)
            w_new = allocator(win)  # (N,)
            w_new = np.asarray(w_new, dtype=float)
            s = float(np.sum(w_new))
            w_new = (w_new / s) if s > 1e-12 else np.full(N, 1.0 / max(N, 1))

            # costes (turnover * cost_bps)
            to = float(np.nansum(np.abs(w_new - w_prev)))
            cost = to * (cost_bps / 10000.0)
            TO.append(to)
            W.append(w_new.copy())
            w_prev = w_new.copy()

        # aplicar retorno del día i
        r = df.row(i, named=True)
        rets = np.array([r[t] for t in tickers], dtype=float)
        port_ret = float(np.nansum(w_prev * rets))
        eq *= (1.0 + port_ret - cost)
        equity.append(eq)

        # TE diario proxy vs benchmark estático (si se pasa)
        if bench_weights is not None:
            v = w_prev - bench_weights
            S = np.outer(rets, rets)  # proxy Σ_t; usar var real si la tienes
            te_daily = float(np.sqrt(max(v @ S @ v, 0.0)))
            te_series.append(te_daily)

    out = {
        "dates": dates[lookback:].to_list(),
        "equity": np.array(equity),
        "weights": np.array(W) if W else np.zeros((0, N)),
        "turnover": np.array(TO, float),
        "te_daily_proxy": np.array(te_series, float) if te_series else None,
        "tickers": tickers,
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Motores “pro” que devuelven BacktestResult (equity/pesos/trades/metrics)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_equal_weight_buy_hold(
    df_prices_long: pl.DataFrame,
    *,
    cfg: BacktestConfig = BacktestConfig(),
    benchmark: str | None = None,   # ticker para curva de referencia (opcional)
) -> BacktestResult:
    """
    Baseline: Buy & Hold equal-weight. Rebalancea sólo al inicio (o según cfg.rebalance si quieres).
    Sin apalancamiento.
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

    # Fechas de rebalanceo estilo pandas (“D/W/M/Q”) o dinámico (“1mo/1w/3mo”)
    if isinstance(cfg.rebalance, str) and cfg.rebalance.lower().endswith(("w", "mo", "q", "d")):
        rb = _rebalance_dates_from_freq_polars(pl.Series(dates), str(cfg.rebalance))
        rb_mask = pl.Series(values=dates.isin(pd.DatetimeIndex(rb.to_list())), name="is_rb").to_numpy()
    else:
        rb_ix = _dates_to_rebalance_pandas(dates, cfg.rebalance if isinstance(cfg.rebalance, str) else "M")
        rb_mask = dates.isin(rb_ix)

    fee_cost = (cfg.fees_bps + cfg.slippage_bps) / 1e4  # proporción

    # Arrays de precios
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
        equity *= (1.0 + port_ret)
        equity_curve.append(equity)
        rets.append(port_ret)

        # Rebalanceo si toca
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
                equity *= (1.0 - turnover * fee_cost)

            for i, tk in enumerate(tickers):
                dw = float(d_w[i])
                if np.isfinite(dw) and dw != 0.0:
                    trades_rows.append(
                        {"date": dates[t], "ticker": tk, "d_weight": dw, "cost_bps": (cfg.fees_bps + cfg.slippage_bps)}
                    )
            prev_w = w_target

        # Guarda pesos del día t
        for i, tk in enumerate(tickers):
            w_i = float(prev_w[i])
            if np.isfinite(w_i) and w_i != 0.0:
                weights_rows.append({"date": dates[t], "ticker": tk, "weight": w_i})

    equity_df = pl.from_pandas(pd.DataFrame({"date": dates, "equity": equity_curve, "ret": rets}))
    weights_df = pl.from_pandas(pd.DataFrame(weights_rows)) if weights_rows else pl.DataFrame(
        {"date": [], "ticker": [], "weight": []}
    )
    trades_df = pl.from_pandas(pd.DataFrame(trades_rows)) if trades_rows else pl.DataFrame(
        {"date": [], "ticker": [], "d_weight": [], "cost_bps": []}
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
    Variante del motor clásico basada en retornos wide, pero devolviendo BacktestResult
    (equity + pesos/trades en formato largo + métricas).
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
    W = np.asarray(out["weights"], float)  # (n_rebalances, N) o (0, N) si vacío

    # Equity/ret diarios desde la curva
    if equity.size > 0:
        ret = np.concatenate([[0.0], (equity[1:] / equity[:-1]) - 1.0])
    else:
        ret = np.array([], float)
    equity_df = pl.from_pandas(pd.DataFrame({"date": dates, "equity": equity, "ret": ret}))

    # Pesos en formato largo (sólo en rebalances)
    weights_rows: list[dict] = []
    trades_rows: list[dict] = []
    if W.size > 0:
        # aproximamos trades como delta entre rebalances consecutivos
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
                        trades_rows.append({"date": d, "ticker": tk, "d_weight": dw, "cost_bps": cost_bps})

    weights_df = pl.from_pandas(pd.DataFrame(weights_rows)) if weights_rows else pl.DataFrame(
        {"date": [], "ticker": [], "weight": []}
    )
    trades_df = pl.from_pandas(pd.DataFrame(trades_rows)) if trades_rows else pl.DataFrame(
        {"date": [], "ticker": [], "d_weight": [], "cost_bps": []}
    )

    stats = _quick_stats_from_equity(equity_df)
    return BacktestResult(equity=equity_df, weights=weights_df, trades=trades_df, stats=stats)