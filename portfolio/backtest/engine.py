# portfolio/backtest/engine.py
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

from portfolio.core.compat import dataclass_compat as dataclass  # noqa: E402  # after logger setup

# ──────────────────────────────────────────────────────────────────────────────
# Types / dataclasses
# ──────────────────────────────────────────────────────────────────────────────

RebalanceFreq = Literal["D", "W", "M", "Q"]  # daily, weekly, monthly, quarterly (pandas resampling)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start: str | None = None
    end: str | None = None
    rebalance: RebalanceFreq | str = "M"  # "M"/"W"… o dinámicos "1mo"/"1w"/"3mo"
    fees_bps: float = 0.0  # round-trip fees en bps
    slippage_bps: float = 0.0  # slippage en bps (se suma a fees)
    initial_capital: float = 1.0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity: pl.DataFrame  # ["date","equity","ret"]
    weights: pl.DataFrame  # long: ["date","ticker","weight"]
    trades: pl.DataFrame  # long: ["date","ticker","d_weight","cost_bps"]
    stats: dict  # métricas agregadas (CAGR, Sharpe, MaxDD, …)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _pivot_prices_compat(df_prices_long: pl.DataFrame) -> pl.DataFrame:
    """
    Compat layer para Polars pivot:
      - Versiones nuevas: pivot(index=..., columns=..., values=...)
      - Versiones antiguas: pivot(index=..., values=..., on=...)
    Devuelve wide con columnas ['date', <tickers...>] ordenado por 'date'.
    """
    dfc = df_prices_long.select(
        [
            pl.col("date").alias("date"),
            pl.col("ticker").cast(pl.Utf8).alias("ticker"),
            pl.col("price").cast(pl.Float64).alias("price"),
        ]
    )
    pivot_fn = getattr(dfc, "pivot", None)
    if pivot_fn is None:
        raise AttributeError("DataFrame has no method 'pivot'")

    try:
        wide = dfc.pivot(index="date", columns="ticker", values="price")  # type: ignore[call-arg]
    except TypeError:
        # Fallback API antigua
        wide = dfc.pivot(index="date", values="price", on="ticker")
    return wide.sort("date")


def _ensure_wide_prices(df_prices_long: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte long ["date","ticker","price"] → wide ["date", tickers...]
    con compatibilidad entre versiones de Polars.
    """
    return _pivot_prices_compat(df_prices_long)


def _dates_to_rebalance_pandas(dates: pd.DatetimeIndex, freq: RebalanceFreq) -> pd.DatetimeIndex:
    """
    Rebalanceo por última observación de cada período (pandas “D/W/M/Q”).
    """
    s = pd.Series(index=dates, data=1.0)
    ix = s.resample(freq).last().index  # DatetimeIndex
    if len(dates) and (dates[0] not in ix):
        ix = ix.insert(0, dates[0])
    # mypy a veces infiere Index; forzamos DatetimeIndex explícito
    return cast(pd.DatetimeIndex, ix.intersection(dates))


def _rebalance_dates_from_freq_polars(dates: pl.Series, freq: str = "1mo") -> pl.Series:
    """
    Rebalanceo dinámico usando Polars (“1w”, “1mo”, “3mo”, …).
    Devuelve última fecha de cada ventana dinámica.
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
    Métricas rápidas sin dependencias extra (CAGR, Sharpe, MaxDD).
    Asume ~252 días de trading para anualizar.
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

    # CAGR aproximado usando número de pasos ~ height
    cagr = float((curve[-1] / max(curve[0], 1e-12)) ** (ann / max(1, equity_df.height)) - 1.0)

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd}


# ──────────────────────────────────────────────────────────────────────────────
# 1) Vectorized engine (Fast) — Ideal para simulaciones masivas
# ──────────────────────────────────────────────────────────────────────────────


def backtest_vectorized(
    df_ret_wide: pl.DataFrame,
    *,
    lookback: int = 252,
    rebalance_freq: str = "1mo",
    cost_bps: float = 0.0,
    allocator: Callable[[pl.DataFrame], np.ndarray],
    bench_weights: np.ndarray | None = None,
    # New Impact params (accepted for signature compatibility, currently ignored or linear-only)
    impact_model: str = "linear",
    impact_c: float = 1.0,
    df_volume: pl.DataFrame | None = None,
) -> dict[str, object]:
    """
    Motor vectorizado de alta velocidad.
    - Calcula rebalanceos solo en fechas clave.
    - Interpola drift de pesos (buy-and-hold) entre rebalanceos usando expresiones de Polars.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("df_ret_wide must contain 'date' column")

    # Ordenar y asegurar fechas
    df = df_ret_wide.sort("date")

    # Polars distinguishes null (missing) from NaN (float not-a-number).
    # We must check both to catch numpy np.nan values stored as float columns.
    tickers_all = [c for c in df.columns if c != "date"]
    float_cols = {t for t in tickers_all if df[t].dtype in (pl.Float32, pl.Float64)}
    total_nulls = sum(df.select(tickers_all).null_count().row(0))
    total_nans = sum(df.select([pl.col(t).is_nan().sum().alias(t) for t in float_cols]).row(0))
    if total_nulls > 0 or total_nans > 0:
        not_null_mask = pl.all_horizontal(
            *[
                pl.col(t).is_not_null()
                & (pl.col(t).is_nan().not_() if t in float_cols else pl.lit(True))
                for t in tickers_all
            ]
        )
        valid_rows = df.filter(not_null_mask)
        if valid_rows.is_empty():
            return {"error": "No common valid date across all tickers"}
        first_valid = valid_rows["date"].min()
        logger.warning(
            "Data has NaN gaps across tickers. Truncating backtest to first "
            "common valid date: %s (dropped %d rows)",
            first_valid,
            df.filter(pl.col("date") < first_valid).height,
        )
        df = valid_rows

    dates = df["date"]

    # 1. Identificar fechas de rebalanceo
    rebal_dates = _rebalance_dates_from_freq_polars(
        pl.Series(list(dates)), str(rebalance_freq).lower()
    )
    # Filtrar para empezar despues del lookback
    if dates.len() <= lookback:
        return {"error": "Not enough history"}

    start_date = dates[lookback]
    rebal_dates = rebal_dates.filter(rebal_dates >= start_date)

    # Asegurar que la ultima fecha este incluida si no coincide
    last_dt = dates[-1]
    if rebal_dates[-1] != last_dt:
        rebal_dates = rebal_dates.extend(pl.Series([last_dt]))

    # 2. Loop de Allocator (inevitable, pero solo en K fechas)
    #    Calculamos Weights Target (W_tgt) en cada fecha de rebalanceo.
    rebal_indices = []

    # Necesitamos mapear fechas a indices para hacer slicing rapido
    # Crear map date->idx
    date_to_idx = {d: i for i, d in enumerate(dates)}

    W_targets = []
    rebal_dates_list = rebal_dates.to_list()
    valid_rebal_dates = []

    tickers = [c for c in df.columns if c != "date"]
    N = len(tickers)

    for rd in rebal_dates_list:
        idx = date_to_idx.get(rd)
        if idx is None or idx < lookback:
            continue

        # Ventana trailing
        # Polars slice es rapido
        # df[i-lookback : i]
        chunk = df.slice(idx - lookback, lookback)

        # Call allocator
        w = allocator(chunk)
        w = np.asarray(w, dtype=float)
        # Normalize
        s = np.sum(w)
        if s > 1e-12:
            w /= s
        else:
            w = np.full(N, 1.0 / N)

        W_targets.append(w)
        valid_rebal_dates.append(rd)
        rebal_indices.append(idx)

    if not W_targets:
        return {}

    # Note: W_targets remains a list of arrays for downstream indexing.
    # The (K, N) matrix is materialized later only where needed.

    # 3. Drift de pesos Vectorizado

    curr_eq = 1.0

    # Alineamos df de retornos total
    df_u = df.filter(pl.col("date") >= valid_rebal_dates[0])

    # Matrix T x N
    rets_vals = df_u.select(tickers).to_numpy()
    dates_vals = df_u["date"].to_list()

    # Mapa fechas -> fila en rets_vals
    d2row = {d: i for i, d in enumerate(dates_vals)}

    # Listas para reconstruir series diarias
    full_equity = []
    full_to = []  # turnover

    # Primer peso
    curr_w = np.zeros(N)

    for k in range(len(valid_rebal_dates)):
        # Fecha de rebalanceo actual
        r_date = valid_rebal_dates[k]

        # Fecha del SIGUIENTE rebalanceo
        next_r_date = valid_rebal_dates[k + 1] if k < len(valid_rebal_dates) - 1 else dates_vals[-1]

        # Indices en matriz local numpy
        idx_start = d2row[r_date]
        idx_end = d2row[next_r_date]  # inclusive (el loop excluye el ultimo usualmente, cuidado)

        # Nuevo peso target
        w_tgt = W_targets[k]

        # 1. Calcular Turnover y Coste del Rebalanceo
        #    Delta = |w_tgt - w_drifted_prev|
        #    Si k=0, w_drifted_prev = 0

        delta = np.abs(w_tgt - curr_w)
        turnover = np.sum(delta)
        cost = turnover * (cost_bps / 10000.0)

        # Restar coste al equity actual (impacto de una vez)
        curr_eq *= 1.0 - cost

        full_to.append((r_date, turnover))

        # 2. Simular Buy & Hold dentro del periodo

        # Slice returns
        # Si es el ultimo punto, incluirlo
        end_slice = idx_end if k < len(valid_rebal_dates) - 1 else idx_end + 1

        block_rets = rets_vals[idx_start:end_slice, :]  # Forma (days, N)

        # Vectorized Wealth Index (Cumulative Return) del bloque
        # r_cum = (1+r).cumprod()
        # wealth_t = w_tgt * r_cum
        # port_val_t = sum(wealth_t)

        geom_rets = 1.0 + block_rets
        cum_geom = np.cumprod(geom_rets, axis=0)  # (days, N)

        # Valor de cada activo en el portfolio (norm to 1 at start)
        # holdings_val(t, i) = w_tgt[i] * cum_geom(t, i)
        holdings_val = cum_geom * w_tgt[None, :]  # broadcast

        # Equity curve relativa del bloque (multiplicador)
        block_eq_rel = np.sum(holdings_val, axis=1)

        # Valid points to append
        valid_points = block_eq_rel
        if k < len(valid_rebal_dates) - 1:
            valid_points = block_eq_rel[:-1]

            w_drifted = holdings_val[-1] / np.sum(holdings_val[-1])
            curr_w = w_drifted
        else:
            # Ultimo bloque
            pass

        # Append equity
        full_equity.extend(curr_eq * valid_points)
        curr_eq = full_equity[-1]

    # Trim or pad?
    res_len = len(full_equity)
    req_len = len(dates_vals)

    final_eq = np.array(full_equity)
    final_dates = dates_vals

    if res_len != req_len:
        # Ajuste fino defensive
        L = min(res_len, req_len)
        final_eq = final_eq[:L]
        final_dates = final_dates[:L]

    # Weights DF (sparse snapshots)
    w_rows = []
    for i, rd in enumerate(valid_rebal_dates):
        wt = W_targets[i]
        for j, tk in enumerate(tickers):
            if abs(wt[j]) > 1e-6:
                w_rows.append({"date": rd, "ticker": tk, "weight": wt[j]})

    # Turnover DF
    to_df = pl.DataFrame(full_to, schema=["date", "turnover"], orient="row")

    out: dict[str, object] = {
        "dates": final_dates,
        "equity": final_eq,
        "weights": np.array(W_targets),  # Snapshot matrix
        "rebalance_dates": valid_rebal_dates,
        "turnover": to_df,
        "tickers": tickers,
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Classic engine (dict output) — compatible con la UI actual
# ──────────────────────────────────────────────────────────────────────────────


def backtest_rebalanced(
    df_ret_wide: pl.DataFrame,  # ['date', tickers...], ordenado
    *,
    lookback: int = 252,
    rebalance_freq: str = "1mo",  # “1w”, “1mo”, “3mo”, …
    cost_bps: float = 0.0,
    allocator: Callable[[pl.DataFrame], np.ndarray],  # ventana trailing (lookback) → w (N,)
    bench_weights: np.ndarray | None = None,
    # Market Impact params
    impact_model: str = "linear",  # "linear" (bps) or "sqrt" (volume)
    df_volume: pl.DataFrame | None = None,  # Required for sqrt
    impact_c: float = 1.0,  # Coeff for sqrt model (usually 0.5 - 1.0)
) -> dict[str, object]:
    """
    Backtest simple con rebalanceo periódico y coste lineal por turnover.
    - `allocator` recibe las últimas `lookback` filas de retornos (wide) y devuelve pesos del siguiente periodo.
    - `bench_weights` (opcional) habilita un proxy diario de TE vs benchmark estático.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("df_ret_wide must contain 'date' column")

    tickers = [c for c in df_ret_wide.columns if c != "date"]
    N = len(tickers)
    df = df_ret_wide.sort("date")
    dates = df["date"]

    rb = _rebalance_dates_from_freq_polars(pl.Series(list(dates)), str(rebalance_freq).lower())
    rb_set = set(rb.to_list())

    W: list[np.ndarray] = []  # pesos en cada rebalance
    TO: list[float] = []  # turnover por rebalance
    RB_DATES: list[object] = []  # fechas de rebalance
    equity: list[float] = []  # equity diario (desde lookback)
    costs_list: list[float] = []  # daily cost fraction
    te_series: list[float] = []  # TE diario (proxy)

    w_prev = np.full(N, 1.0 / max(N, 1), dtype=float)
    eq = 1.0

    # Loop principal
    for i in range(lookback, df.height):
        d = dates[i]
        cost = 0.0  # coste aplicado en este paso si hay rebalance

        # Rebalance cuando toca
        if d in rb_set:
            win = df.slice(i - lookback, lookback)
            w_new = allocator(win)  # (N,)
            w_new = np.asarray(w_new, dtype=float)

            s = float(np.nansum(w_new))
            w_new = w_new / s if s > 1e-12 else np.full(N, 1.0 / max(N, 1), dtype=float)

            # turnover & coste (bps sobre |Δw| o sqrt law)
            to = float(np.nansum(np.abs(w_new - w_prev)))

            cost = 0.0
            if impact_model == "linear":
                cost = to * (cost_bps / 10000.0)
            elif impact_model == "sqrt":
                # Square Root Law: Cost ~ c * sigma * sqrt(Trade / Vol)
                # TradeSize($) ~ |w_new - w_prev| * CurrentEquity
                # Need: df_volume row for today (d).
                if df_volume is not None:
                    trade_pct = np.abs(w_new - w_prev)
                    trade_amt = trade_pct * eq

                    # Extract vol
                    # Vol lookup is slow, but we are in a loop anyway.
                    # Optimization: Pre-lookup or join before loop. Given rebal freq is low, fetching is ok.
                    try:
                        # Filter df_volume for date d
                        vol_row = (
                            df_volume.filter(pl.col("date") == d)
                            .select(tickers)
                            .to_numpy()
                            .flatten()
                        )
                        vol_row = np.maximum(
                            np.nan_to_num(vol_row, nan=1e6), 1.0
                        )  # ensure non-zero

                        sigma_d = 0.02  # simple daily vol assumption (or pass it in?)

                        # Market Impact ($)
                        # Cost($) = c * sigma * sum [ Trade_j^(1.5) / Vol_j^(0.5) ]
                        impact_usd = (
                            impact_c * sigma_d * np.nansum((trade_amt**1.5) / np.sqrt(vol_row))
                        )

                        # Cost as fractional return drag: Cost($) / Equity($)
                        cost = impact_usd / eq
                    except Exception:
                        cost = to * 0.0010  # fallback 10bps
                else:
                    cost = to * 0.0010

            TO.append(to)
            W.append(w_new.copy())
            RB_DATES.append(d)
            w_prev = w_new.copy()

        # aplicar retorno de cartera del día i
        row = df.row(i, named=True)
        rets = np.array([row[t] for t in tickers], dtype=float)
        port_ret = float(np.nansum(w_prev * rets))

        # Store daily cost for visualization
        # eq *= 1.0 + port_ret - cost
        # Gross return factor: 1 + port_ret
        # Net return factor: 1 + port_ret - cost

        eq *= 1.0 + port_ret - cost
        equity.append(eq)

        costs_list.append(cost)  # Needs initialization outside loop

        # TE proxy diario vs benchmark estático
        if bench_weights is not None:
            v = w_prev - bench_weights
            S = np.outer(rets, rets)  # Σ_t muy crudo; sustituible por estimador mejor
            te_daily = float(np.sqrt(max(v @ S @ v, 0.0)))
            te_series.append(te_daily)

    out: dict[str, object] = {
        "dates": dates[lookback:].to_list(),  # rejilla diaria desde lookback
        "equity": np.array(equity, dtype=float),
        "costs": np.array(costs_list, dtype=float),  # New
        "weights": np.array(W, dtype=float) if W else np.zeros((0, N), float),
        "rebalance_dates": RB_DATES,
        "turnover": pl.DataFrame({"date": RB_DATES, "turnover": np.array(TO, dtype=float)}),
        "te_daily_proxy": np.array(te_series, dtype=float) if te_series else None,
        "tickers": tickers,
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Motores que devuelven BacktestResult (equity/weights/trades/metrics)
# ──────────────────────────────────────────────────────────────────────────────


def backtest_equal_weight_buy_hold(
    df_prices_long: pl.DataFrame,
    *,
    cfg: BacktestConfig = BacktestConfig(),
    benchmark: str | None = None,  # opcional (no usado en baseline)
) -> BacktestResult:
    """
    Baseline: Buy & Hold equal-weight. Rebalance solo al inicio (o según cfg.rebalance si quisieras).
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

    # Fechas de rebalanceo: pandas (“D/W/M/Q”) o dinámicas estilo Polars (“1w”/“1mo”/…)
    if isinstance(cfg.rebalance, str) and cfg.rebalance.lower().endswith(("w", "mo", "q", "d")):
        rb = _rebalance_dates_from_freq_polars(pl.Series(list(dates)), str(cfg.rebalance))
        rb_list = pd.DatetimeIndex(rb.to_list())
        rb_mask = dates.isin(rb_list)
    else:
        rb_ix = _dates_to_rebalance_pandas(
            dates, cast(RebalanceFreq, (cfg.rebalance if isinstance(cfg.rebalance, str) else "M"))
        )
        rb_mask = dates.isin(rb_ix)

    fee_cost = (cfg.fees_bps + cfg.slippage_bps) / 1e4  # coste proporcional

    # Matriz de precios
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

        # Rebalance cuando toca
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

        # Guardar pesos del día t
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
    Variante del motor “classic” que recibe retornos wide y devuelve BacktestResult
    (equity + weights/trades en largo + stats).
    """
    out = backtest_rebalanced(
        df_ret_wide=df_ret_wide,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        cost_bps=cost_bps,
        allocator=allocator,
        bench_weights=bench_weights,
    )

    dates = pd.DatetimeIndex(cast(list[object], out["dates"]))
    equity = np.asarray(cast(np.ndarray, out["equity"]), float)
    tickers = cast(list[str], out["tickers"])
    W = np.asarray(cast(np.ndarray, out["weights"]), float)

    # Equity/ret diarios a partir de la curva
    if equity.size > 0:
        ret = np.concatenate([[0.0], (equity[1:] / equity[:-1]) - 1.0])
    else:
        ret = np.array([], float)
    equity_df = pl.from_pandas(pd.DataFrame({"date": dates, "equity": equity, "ret": ret}))

    # Pesos/trades en formato largo (en fechas de rebalanceo)
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
