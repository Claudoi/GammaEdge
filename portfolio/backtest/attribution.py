# Performance attribution
# portfolio/backtest/attribution.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import polars as pl

from portfolio.core.compat import dataclass_compat as dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Constantes y tipos
# ──────────────────────────────────────────────────────────────────────────────

EPS: float = 1e-12  # para evitar divisiones por ~0 y mejorar estabilidad


@dataclass(frozen=True, slots=True)
class DailyAlignment:
    dates: list[Any]
    tickers: list[str]
    returns: np.ndarray  # (T, N) daily asset returns
    weights: np.ndarray  # (T, N) daily portfolio weights (post-rebalance, stepwise)


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """Contenedor estándar para resultados Brinson–Fachler (vectorizado)."""

    date: np.ndarray  # (T,)
    alloc: np.ndarray  # (T,)
    select: np.ndarray  # (T,)
    interact: np.ndarray  # (T,)
    total: np.ndarray  # (T,)
    # si alguna vez devolvemos por grupo, podemos añadir aquí estructuras extra


# ──────────────────────────────────────────────────────────────────────────────
# Alignment helpers
# ──────────────────────────────────────────────────────────────────────────────


def align_returns_and_weights(
    df_ret_wide: pl.DataFrame,  # ['date', T1, T2, ...] sorted
    daily_weights: np.ndarray,  # (T, N) daily weights
) -> DailyAlignment:
    """
    Alinea retornos diarios (wide) con weights diarios ya expandidos.
    Valida shapes y devuelve numpy arrays.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("df_ret_wide must contain 'date' column")

    df = df_ret_wide.sort("date")
    tickers = [c for c in df.columns if c != "date"]
    T = df.height
    N = len(tickers)

    if daily_weights.shape != (T, N):
        raise ValueError(f"incompatible dims: weights {daily_weights.shape} vs returns {(T, N)}")

    R = np.column_stack([df.get_column(t).to_numpy() for t in tickers])
    # NaN-safe: tratamos NaN/inf como 0 para no contaminar contribuciones
    R = np.nan_to_num(R, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return DailyAlignment(
        dates=df.get_column("date").to_list(),
        tickers=tickers,
        returns=R,
        weights=daily_weights.astype(float, copy=False),
    )


def expand_rebalance_weights(
    dates: Iterable,  # full daily dates
    rb_dates: Iterable,  # rebalance dates
    W_reb: np.ndarray,  # (n_reb, N) weights at each rebalance
) -> np.ndarray:
    """
    Expande weights de rebalanceo a frecuencia diaria (stepwise/forward-fill).
    Asume tipos comparables entre 'dates' y 'rb_dates'.
    """
    dates = list(dates)
    rb_dates = list(rb_dates)
    n_reb, N = W_reb.shape
    if len(rb_dates) != n_reb:
        raise ValueError("len(rb_dates) must match W_reb.shape[0]")

    rb_ix = {d: i for i, d in enumerate(rb_dates)}
    out = np.zeros((len(dates), N), dtype=float)

    last_w: np.ndarray | None = None
    for i, d in enumerate(dates):
        if d in rb_ix:
            last_w = W_reb[rb_ix[d]].astype(float, copy=False)
            s = last_w.sum()
            last_w = last_w / (s if abs(s) > EPS else 1.0)
        if last_w is None:
            # antes del primer rebalanceo: equal weight
            last_w = np.full(N, 1.0 / max(N, 1), dtype=float)
        out[i, :] = last_w
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Contributions by asset and by group
# ──────────────────────────────────────────────────────────────────────────────


def contributions_by_asset(aln: DailyAlignment) -> pl.DataFrame:
    """
    Contribución diaria por activo: c_{t,i} = w_{t,i} * r_{t,i}.
    Devuelve DF long: ['date','ticker','contrib','ret','weight'].
    """
    w = aln.weights
    r = aln.returns
    c = w * r  # (T, N)

    T, N = c.shape
    rows = []
    for j, name in enumerate(aln.tickers):
        rows.append(
            pl.DataFrame(
                {
                    "date": aln.dates,
                    "ticker": [name] * T,
                    "contrib": c[:, j],
                    "ret": r[:, j],
                    "weight": w[:, j],
                }
            )
        )
    return pl.concat(rows)


def contributions_by_group(
    aln: DailyAlignment,
    group_map: dict[str, str],
    *,
    other_label: str = "OTHER",
) -> pl.DataFrame:
    """
    Atribución agrupada (suma contribuciones por grupo).
    Tickers no mapeados -> 'other_label'.
    Devuelve DF long: ['date','group','contrib','weight'].
    """
    df_asset = contributions_by_asset(aln)
    df_asset = df_asset.with_columns(
        pl.col("ticker").replace(group_map, default=other_label).alias("group")
    )
    df_grp = (
        df_asset.group_by(["date", "group"])
        .agg(
            [
                pl.col("contrib").sum().alias("contrib"),
                pl.col("weight").sum().alias("weight"),
            ]
        )
        .sort(["date", "group"])
    )
    return df_grp


# ──────────────────────────────────────────────────────────────────────────────
# Brinson–Fachler (básico por periodo) + acumulado (loop)
# ──────────────────────────────────────────────────────────────────────────────


def brinson_fachler_period(
    w_p: np.ndarray,
    r_p: np.ndarray,  # kept for backward compat; we use r := r_p
    w_b: np.ndarray,
    r_b: np.ndarray,  # kept for backward compat; we use r := r_b
    groups: list[int],  # asset→group indices [0..G-1]
) -> tuple[float, float, float]:
    """
    Brinson–Fachler (single period):
    Allocation:  Σ_g (w_p_g - w_b_g) * (r_b_g - r_b)
    Selection:   Σ_g w_b_g * (r_p_g - r_b_g)    [r_p_g, r_b_g are group returns]
    Interaction: Σ_g (w_p_g - w_b_g) * (r_p_g - r_b_g)
    Notes:
      - r_p and r_b are the same asset-level return vector in practice; we use one r.
      - Group returns are computed with within-group normalized weights.
    """
    # Use a single return vector (asset-level) for both portfolio and benchmark math
    r = np.asarray(r_p, dtype=float)
    w_p = np.asarray(w_p, dtype=float)
    w_b = np.asarray(w_b, dtype=float)

    groups_arr = np.asarray(groups, dtype=int)
    G = int(groups_arr.max()) + 1 if groups_arr.size else 0

    # Total benchmark return r_b_total = sum_i w_b_i * r_i
    rb_total = float(np.sum(w_b * r))

    A = 0.0
    S = 0.0
    interaction = 0.0

    for g in range(G):
        idx = groups_arr == g
        wp_g = float(np.sum(w_p[idx]))
        wb_g = float(np.sum(w_b[idx]))

        # Within-group normalized weights (guarded by EPS)
        # Benchmark group return r_b_g
        rb_g_num = float(np.sum(w_b[idx] * r[idx]))
        rb_g_den = wb_g if wb_g > EPS else 1.0
        rb_g = rb_g_num / rb_g_den

        # Portfolio group return r_p_g
        rp_g_num = float(np.sum(w_p[idx] * r[idx]))
        rp_g_den = wp_g if wp_g > EPS else 1.0
        rp_g = rp_g_num / rp_g_den

        diff_w = wp_g - wb_g
        diff_r = rp_g - rb_g

        # Brinson–Fachler:
        A += diff_w * (rb_g - rb_total)  # <-- change vs BHB
        S += wb_g * diff_r
        interaction += diff_w * diff_r

    # Tiny clamp to remove floating-point dust
    if abs(A) < 1e-15:
        A = 0.0
    if abs(S) < 1e-15:
        S = 0.0
    if abs(interaction) < 1e-15:
        interaction = 0.0

    return A, S, interaction


def brinson_fachler_cumulative(
    aln: DailyAlignment,
    bench_weights_daily: np.ndarray,  # (T, N)
    groups_idx: list[int],  # asset→group indices [0..G-1]
    *,
    return_aggregate_row: bool = False,
) -> pl.DataFrame:
    """
    Arithmetic linking of Brinson–Fachler through time.
    Output: ['date','alloc','select','interact','total'] per period.
    If return_aggregate_row=True, appends an 'AGG' summary row with arithmetic sums.
    """
    R = aln.returns
    Wp = aln.weights
    Wb = bench_weights_daily
    if Wb.shape != Wp.shape:
        raise ValueError("bench_weights_daily must have the same shape as portfolio weights")

    rows: list[tuple] = []
    for t in range(R.shape[0]):
        a, s, inter = brinson_fachler_period(Wp[t], R[t], Wb[t], R[t], groups_idx)

        # clamp per-period BEFORE appending
        if abs(a) < 1e-15:
            a = 0.0
        if abs(s) < 1e-15:
            s = 0.0
        if abs(inter) < 1e-15:
            inter = 0.0
        rows.append((aln.dates[t], a, s, inter, a + s + inter))

    df = pl.DataFrame(
        rows,
        schema=["date", "alloc", "select", "interact", "total"],
        orient="row",
    ).sort("date")

    if return_aggregate_row:
        agg = df.select(
            [
                pl.lit("AGG").alias("date"),
                pl.col("alloc").sum().alias("alloc"),
                pl.col("select").sum().alias("select"),
                pl.col("interact").sum().alias("interact"),
                pl.col("total").sum().alias("total"),
            ]
        )
        df = pl.concat([df, agg])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Brinson–Fachler vectorizado (rápido y estable)
# ──────────────────────────────────────────────────────────────────────────────


def brinson_fachler_vectorized(
    aln: DailyAlignment,
    bench_weights_daily: np.ndarray,
    groups_idx: list[int] | np.ndarray,
    *,
    cumulative: bool = True,
) -> AttributionResult:
    R = np.asarray(aln.returns, dtype=float)
    Wp = np.asarray(aln.weights, dtype=float)
    Wb = np.asarray(bench_weights_daily, dtype=float)

    if Wb.shape != Wp.shape:
        raise ValueError("bench_weights_daily must have same shape as portfolio weights")

    # Sanitize returns
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    T, N = Wp.shape
    gidx = np.asarray(groups_idx, dtype=int)
    if gidx.shape[0] != N:
        raise ValueError("groups_idx must have length N")

    G = int(gidx.max()) + 1 if gidx.size else 0
    H = np.zeros((N, G), dtype=np.float64)
    if G > 0:
        H[np.arange(N), gidx] = 1.0

    Wp_g = Wp @ H
    Wb_g = Wb @ H

    Rp_g = np.divide((Wp * R) @ H, np.clip(Wp_g, EPS, None), where=Wp_g > EPS)
    Rb_g = np.divide((Wb * R) @ H, np.clip(Wb_g, EPS, None), where=Wb_g > EPS)

    Rb_t = np.sum(Wb * R, axis=1)  # (T,)

    diff_w = Wp_g - Wb_g
    diff_r = Rp_g - Rb_g

    alloc = np.sum(diff_w * (Rb_g - Rb_t[:, None]), axis=1)
    select = np.sum(Wb_g * diff_r, axis=1)
    inter = np.sum(diff_w * diff_r, axis=1)
    total = alloc + select + inter

    if cumulative:
        alloc = np.cumsum(alloc)
        select = np.cumsum(select)
        inter = np.cumsum(inter)
        total = np.cumsum(total)

    # clamps
    alloc = np.where(np.abs(alloc) < 1e-15, 0.0, alloc)
    select = np.where(np.abs(select) < 1e-15, 0.0, select)
    inter = np.where(np.abs(inter) < 1e-15, 0.0, inter)
    total = np.where(np.abs(total) < 1e-15, 0.0, total)

    # force +0.0 (cosmetic; avoids -0.0 in plots)
    alloc = alloc + 0.0
    select = select + 0.0
    inter = inter + 0.0
    total = total + 0.0

    return AttributionResult(
        date=np.array(aln.dates),
        alloc=alloc,
        select=select,
        interact=inter,
        total=total,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summaries & high-level API (usado por la UI)
# ──────────────────────────────────────────────────────────────────────────────


def _daily_alignment_from_bt(bt: dict, df_ret_wide: pl.DataFrame) -> DailyAlignment:
    """
    Construye DailyAlignment desde un dict de backtest:
    - expande weights de rebalance a la malla diaria
    - alinea retornos wide a misma malla y orden de tickers
    """
    dates = bt["dates"]
    tickers = bt["tickers"]
    W_reb = np.asarray(bt["weights"], dtype=float)  # (n_reb, N)
    rb_dates = bt.get("rebalance_dates", [])
    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0]:
        raise ValueError("Invalid 'weights'/'rebalance_dates' in bt.")

    df = df_ret_wide.filter(pl.col("date").is_in(dates)).sort("date").select(["date", *tickers])

    W_daily = expand_rebalance_weights(dates, rb_dates, W_reb)  # (T, N)
    return align_returns_and_weights(df, W_daily)


def top_contributors(
    bt: dict,
    df_ret_wide: pl.DataFrame,
    *,
    top_n: int = 10,
    sign: Literal["pos", "neg", "both"] = "both",
) -> pl.DataFrame:
    """
    Top contribuyentes (acumulado) desde bt + retornos.
    Salida: ['ticker','contrib_total'].
    """
    aln = _daily_alignment_from_bt(bt, df_ret_wide)
    df_asset = contributions_by_asset(aln)

    agg = (
        df_asset.group_by("ticker")
        .agg(pl.col("contrib").sum().alias("contrib_total"))
        .sort("contrib_total", descending=True)
    )
    if sign == "pos":
        return agg.filter(pl.col("contrib_total") > 0).head(top_n)
    if sign == "neg":
        return (
            agg.filter(pl.col("contrib_total") < 0)
            .sort("contrib_total", descending=False)
            .head(top_n)
        )

    pos = agg.filter(pl.col("contrib_total") > 0).head(top_n)
    neg = (
        agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False).head(top_n)
    )
    return pl.concat([pos, neg])


def group_contrib(
    bt: dict,
    df_ret_wide: pl.DataFrame,
    *,
    groups_map: dict[str, str],
    other_label: str = "OTHER",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Contribuciones agrupadas desde bt + retornos.
    Devuelve:
      - df_group_total: ['group','contrib_total','weight_avg'] (acumulado)
      - df_group_daily: ['date','group','contrib','weight']   (serie diaria)
    """
    aln = _daily_alignment_from_bt(bt, df_ret_wide)
    df_daily = contributions_by_group(aln, groups_map, other_label=other_label)

    df_total = (
        df_daily.group_by("group")
        .agg(
            [
                pl.col("contrib").sum().alias("contrib_total"),
                pl.col("weight").mean().alias("weight_avg"),
            ]
        )
        .sort("contrib_total", descending=True)
    )
    return df_total, df_daily


# ──────────────────────────────────────────────────────────────────────────────
# Advanced helpers (seguros; no rompen API existente)
# ──────────────────────────────────────────────────────────────────────────────


def align_with_ipo_mask(aln: DailyAlignment) -> DailyAlignment:
    """
    IPO-safe alignment:
    - Peso 0 para cada activo antes de su primera observación válida.
    - Renormalización por fila; si la fila queda en ~0, EW sobre disponibles.
    """
    R = np.asarray(aln.returns, dtype=float)
    W = np.asarray(aln.weights, dtype=float).copy()
    T, N = R.shape

    valid = np.isfinite(R) & (np.abs(R) > 1e-15)

    t0 = np.full(N, T, dtype=int)
    for j in range(N):
        nz = np.nonzero(valid[:, j])[0]
        if nz.size:
            t0[j] = int(nz[0])

    for j in range(N):
        if t0[j] > 0:
            W[: t0[j], j] = 0.0

    row_sum = W.sum(axis=1, keepdims=True)
    available = np.arange(T)[:, None] >= t0[None, :]
    n_avail = available.sum(axis=1, keepdims=True).astype(float)

    zero_rows = (np.abs(row_sum) < EPS).ravel()
    if np.any(zero_rows):
        W[zero_rows, :] = 0.0
        W[zero_rows, :] = np.where(available[zero_rows, :], 1.0, 0.0)
        W[zero_rows, :] /= np.clip(n_avail[zero_rows, :], 1.0, None)

    nz_rows = ~zero_rows
    if np.any(nz_rows):
        W[nz_rows, :] /= np.clip(row_sum[nz_rows, :], EPS, None)

    return DailyAlignment(
        dates=list(aln.dates),
        tickers=list(aln.tickers),
        returns=R,
        weights=W,
    )


def build_groups_idx(
    tickers: list[str],
    meta_df: pl.DataFrame | None,
    *,
    col: str = "sector",
    other_label: str = "OTHER",
) -> tuple[list[int], list[str], dict[str, str]]:
    group_map_out: dict[str, str]

    if meta_df is None or "ticker" not in meta_df.columns or col not in meta_df.columns:
        group_map_out = {tk: tk for tk in tickers}
        labels = list(tickers)
        idx = list(range(len(tickers)))
        return idx, labels, group_map_out

    lut = dict(zip(meta_df["ticker"].to_list(), meta_df[col].to_list()))
    label_order: list[str] = []
    group_map_out = {}
    for tk in tickers:
        g = lut.get(tk, other_label)
        group_map_out[tk] = g
        if g not in label_order:
            label_order.append(g)
    label_to_id = {g: i for i, g in enumerate(label_order)}
    groups_idx = [label_to_id[group_map_out[tk]] for tk in tickers]
    return groups_idx, label_order, group_map_out


def coerce_benchmark_weights(
    Wb: np.ndarray | None,
    T: int,
    N: int,
    *,
    scheme: Literal["EW", "cash"] = "EW",
) -> np.ndarray:
    """
    Asegura shape (T, N) para weights del benchmark.
    Si Wb es None:
      - "EW": equal-weight diario
      - "cash": todos ceros
    Cualquier fila ~0 se renormaliza a EW por seguridad.
    """
    if Wb is None or (isinstance(Wb, np.ndarray) and Wb.size == 0):
        if scheme == "cash":
            Wb = np.zeros((T, N), dtype=float)
        else:
            Wb = np.full((T, N), 1.0 / max(N, 1), dtype=float)
    else:
        Wb = np.asarray(Wb, dtype=float)
        if Wb.ndim == 1:
            if Wb.shape[0] != N:
                raise ValueError("1D Wb must have length N")
            Wb = np.tile(Wb, (T, 1))
        if Wb.shape != (T, N):
            raise ValueError(f"Wb must be shape (T,N), got {Wb.shape}")

    row_sum = Wb.sum(axis=1, keepdims=True)
    zero_rows = (np.abs(row_sum) < EPS).ravel()
    if np.any(zero_rows):
        Wb[zero_rows, :] = 1.0
        Wb[zero_rows, :] /= float(N) if N > 0 else 1.0
    else:
        Wb /= np.clip(row_sum, EPS, None)
    return Wb


def top_contributors_from_asset(
    df_asset_contrib: pl.DataFrame,
    *,
    top_n: int = 10,
    sign: Literal["pos", "neg", "both"] = "both",
) -> pl.DataFrame:
    """
    Versión DF pura para top contributors.
    Entrada requiere ['ticker','contrib'].
    """
    if "ticker" not in df_asset_contrib.columns or "contrib" not in df_asset_contrib.columns:
        raise ValueError("df_asset_contrib must contain columns 'ticker' and 'contrib'.")

    agg = (
        df_asset_contrib.group_by("ticker")
        .agg(pl.col("contrib").sum().alias("contrib_total"))
        .sort("contrib_total", descending=True)
    )
    if sign == "pos":
        return agg.filter(pl.col("contrib_total") > 0).head(top_n)
    if sign == "neg":
        return (
            agg.filter(pl.col("contrib_total") < 0)
            .sort("contrib_total", descending=False)
            .head(top_n)
        )
    pos = agg.filter(pl.col("contrib_total") > 0).head(top_n)
    neg = (
        agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False).head(top_n)
    )
    return pl.concat([pos, neg])


def contributions_share_by_group(df_group_daily: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte contribución diaria de grupo a 'share of total' por fecha:
      share_{t,g} = contrib_{t,g} / sum_g contrib_{t,g}
    Si la suma diaria ~0, devuelve 0 para evitar blow-ups.
    """
    req = {"date", "group", "contrib"}
    if not req.issubset(set(df_group_daily.columns)):
        raise ValueError("df_group_daily must contain 'date','group','contrib'")

    total = df_group_daily.group_by("date").agg(pl.col("contrib").sum().alias("tot"))
    out = (
        df_group_daily.join(total, on="date", how="left")
        .with_columns(
            (
                pl.col("contrib")
                / pl.when(pl.col("tot").abs() > EPS).then(pl.col("tot")).otherwise(1.0)
            ).alias("share")
        )
        .select(["date", "group", "share"])
        .sort(["date", "group"])
    )
    return out


def brinson_fachler_timeseries(
    aln: DailyAlignment,
    bench_weights_daily: np.ndarray,
    groups_idx: list[int],
    *,
    cumulative: bool = True,
    by_group: bool = False,  # reservado: podríamos extender a salida por grupo
) -> pl.DataFrame:
    """
    Brinson–Fachler en el tiempo.
    Si cumulative=True → cumulativos; else → por periodo.
    Salida: ['date','alloc','select','interact','total']
    """
    res = brinson_fachler_vectorized(
        aln=aln,
        bench_weights_daily=bench_weights_daily,
        groups_idx=groups_idx,
        cumulative=cumulative,
    )
    df = pl.DataFrame(
        {
            "date": list(res.date),
            "alloc": res.alloc,
            "select": res.select,
            "interact": res.interact,
            "total": res.total,
        }
    ).sort("date")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Euler Risk Attribution (volatility contributions)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EulerRCResult:
    """Container for Euler risk attribution at a single time step."""

    sigma: float  # portfolio volatility (per-step, sqrt of variance)
    mrc: np.ndarray  # marginal risk contributions, shape (N,)
    rc: np.ndarray  # total risk contributions, shape (N,)  with sum(rc) == sigma (≈)
    tickers: list[str]


def _euler_single(weights: np.ndarray, cov: np.ndarray) -> EulerRCResult:
    """
    Compute Euler risk contributions for a single time step.
    sigma = sqrt(w^T Σ w)
    MRC = (Σ w) / sigma
    RC_i = w_i * MRC_i
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("cov must be a square matrix")
    if S.shape[0] != w.size:
        raise ValueError("cov dim must match number of assets")

    # Ensure symmetry and numerical stability
    S = 0.5 * (S + S.T)
    # Portfolio variance and sigma
    var = float(w @ (S @ w))
    var = max(var, 0.0)
    sigma = float(np.sqrt(var))

    if sigma < EPS:
        # Degenerate: zero risk; define zeros safely
        mrc = np.zeros_like(w)
        rc = np.zeros_like(w)
        return EulerRCResult(sigma=sigma, mrc=mrc, rc=rc, tickers=[])

    grad = S @ w  # gradient of variance wrt w is 2 Σ w, but for sigma we want Σ w / sigma
    mrc = grad / sigma  # Σ w / sigma
    rc = w * mrc
    return EulerRCResult(sigma=sigma, mrc=mrc, rc=rc, tickers=[])


def euler_rc_by_asset(
    aln: DailyAlignment,
    *,
    cov: np.ndarray | list[np.ndarray] | None = None,
    cov_builder: Any | None = None,
    window: int = 60,
    ddof: int = 1,
) -> pl.DataFrame:
    """
    Euler RC per day by asset.

    Inputs:
      - aln: DailyAlignment with returns and daily portfolio weights.
      - cov:
          • None → build rolling covariance from aln.returns with 'window' and 'ddof'.
          • np.ndarray (N,N) → same covariance for all dates.
          • list[np.ndarray] length T → one covariance per date.
      - cov_builder (callable optional): cov_builder(R_slice: np.ndarray) -> np.ndarray
          • If provided, overrides default covariance estimator.
      - window, ddof: parameters for rolling sample covariance if cov is None.

    Output (long DF):
      ['date','ticker','rc','mrc','sigma']  with one row per asset-date.
      The daily identity holds: group_by('date').sum('rc') ≈ sigma (first() by date).
    """
    R = np.asarray(aln.returns, dtype=float)  # (T, N)
    W = np.asarray(aln.weights, dtype=float)  # (T, N)
    T, N = R.shape
    if W.shape != (T, N):
        raise ValueError("aln.weights must match returns shape")

    # Build covariances timeline
    if cov is None:
        # Rolling covariance on returns (rows=time)
        covs: list[np.ndarray] = []
        for t in range(T):
            slice_R = R[: t + 1, :] if t + 1 < max(window, 2) else R[t - window + 1 : t + 1, :]

            if cov_builder is not None:
                C = np.asarray(cov_builder(slice_R), dtype=float)
            else:
                # Sample covariance (columns=assets)
                X = slice_R - np.nanmean(slice_R, axis=0, keepdims=True)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                # shape (window, N) → covariance (N, N)
                denom = max(X.shape[0] - ddof, 1)
                C = (X.T @ X) / float(denom)
            # Stabilize: symmetrize + tiny ridge
            C = 0.5 * (C + C.T)
            if N > 0:
                C.flat[:: N + 1] += 1e-12  # tiny ridge
            covs.append(C)
    elif isinstance(cov, list):
        if len(cov) != T:
            raise ValueError("cov list length must equal T")
        covs = [np.asarray(C, dtype=float) for C in cov]
    else:
        C = np.asarray(cov, dtype=float)
        covs = [C for _ in range(T)]

    rows = []
    for t in range(T):
        res = _euler_single(W[t], covs[t])
        sigma = res.sigma
        rc = res.rc
        # Guard against tiny negatives due to FP errors
        rc = np.where(np.abs(rc) < 1e-15, 0.0, rc)

        for j, tk in enumerate(aln.tickers):
            rows.append((aln.dates[t], tk, float(rc[j]), float(res.mrc[j]), float(sigma)))

    return pl.DataFrame(
        rows,
        schema=["date", "ticker", "rc", "mrc", "sigma"],
        orient="row",
    ).sort(["date", "ticker"])


def euler_rc_by_group(
    aln: DailyAlignment,
    *,
    groups_map: dict[str, str],
    cov: np.ndarray | list[np.ndarray] | None = None,
    cov_builder: Any | None = None,
    window: int = 60,
    ddof: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Euler RC per day aggregated by group.

    Returns:
      - df_group_daily: ['date','group','rc','sigma']  (daily series)
      - df_group_total: ['group','rc_total','share']   (time-aggregated)
    """
    df_rc_asset = euler_rc_by_asset(
        aln,
        cov=cov,
        cov_builder=cov_builder,
        window=window,
        ddof=ddof,
    ).with_columns(pl.col("ticker").replace(groups_map, default="OTHER").alias("group"))

    df_group_daily = (
        df_rc_asset.group_by(["date", "group"])
        .agg(
            [
                pl.col("rc").sum().alias("rc"),
                pl.col("sigma").first().alias("sigma"),
            ]
        )
        .sort(["date", "group"])
    )

    df_group_total = (
        df_group_daily.group_by("group")
        .agg(pl.col("rc").sum().alias("rc_total"))
        .with_columns((pl.col("rc_total") / pl.col("rc_total").sum().alias("tot")).alias("share"))
        .sort("rc_total", descending=True)
    )
    return df_group_daily, df_group_total
