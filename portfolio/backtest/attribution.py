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
    r_p: np.ndarray,  # portfolio
    w_b: np.ndarray,
    r_b: np.ndarray,  # benchmark
    groups: list[int],  # asset→group indices [0..G-1]
) -> tuple[float, float, float]:
    """
    Brinson–Fachler (un periodo):
    - Allocation:  Σ_g (w_p_g - w_b_g) * r_b_g
    - Selection:   Σ_g w_b_g * (r_p_g - r_b_g)
    - Interaction: Σ_g (w_p_g - w_b_g) * (r_p_g - r_b_g)
    """
    groups_arr = np.asarray(groups, dtype=int)
    G = int(groups_arr.max()) + 1 if len(groups_arr) else 0
    A = S = interaction = 0.0
    for g in range(G):
        idx = groups_arr == g
        wp_g = float(np.sum(w_p[idx]))
        wb_g = float(np.sum(w_b[idx]))
        rb_g = float(np.sum(w_b[idx] * r_b[idx])) / (wb_g if wb_g > EPS else 1.0)
        rp_g = float(np.sum(w_p[idx] * r_p[idx])) / (wp_g if wp_g > EPS else 1.0)

        diff_w = wp_g - wb_g
        diff_r = rp_g - rb_g
        A += diff_w * rb_g
        S += wb_g * diff_r
        interaction += diff_w * diff_r
    return A, S, interaction


def brinson_fachler_cumulative(
    aln: DailyAlignment,
    bench_weights_daily: np.ndarray,  # (T, N)
    groups_idx: list[int],  # asset→group indices [0..G-1]
) -> pl.DataFrame:
    """
    Suma aritmética de Brinson–Fachler en el tiempo.
    Salida: ['date','alloc','select','interact','total'].
    """
    R = aln.returns
    Wp = aln.weights
    Wb = bench_weights_daily
    if Wb.shape != Wp.shape:
        raise ValueError("bench_weights_daily must have the same shape as portfolio weights")

    rows = []
    for t in range(R.shape[0]):
        a, s, inter = brinson_fachler_period(Wp[t], R[t], Wb[t], R[t], groups_idx)
        rows.append((aln.dates[t], a, s, inter, a + s + inter))
    return pl.DataFrame(rows, schema=["date", "alloc", "select", "interact", "total"]).sort("date")


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
    """
    Descomposición Brinson–Fachler completamente vectorizada.
    Complejidad O(T·G), con G = nº de grupos.
    """
    R = np.asarray(aln.returns, dtype=float)
    Wp = np.asarray(aln.weights, dtype=float)
    Wb = np.asarray(bench_weights_daily, dtype=float)

    if Wb.shape != Wp.shape:
        raise ValueError("bench_weights_daily must have same shape as portfolio weights")

    T, N = Wp.shape
    gidx = np.asarray(groups_idx, dtype=int)
    if gidx.shape[0] != N:
        raise ValueError("groups_idx must have length N")

    G = int(gidx.max()) + 1 if gidx.size else 0
    H = np.zeros((N, G), dtype=np.float64)
    if G > 0:
        H[np.arange(N), gidx] = 1.0

    # weights y retornos por grupo
    Wp_g = Wp @ H  # (T, G)
    Wb_g = Wb @ H  # (T, G)
    Rp_g = np.divide((Wp * R) @ H, np.clip(Wp_g, EPS, None), where=Wp_g > EPS)
    Rb_g = np.divide((Wb * R) @ H, np.clip(Wb_g, EPS, None), where=Wb_g > EPS)

    diff_w = Wp_g - Wb_g
    diff_r = Rp_g - Rb_g

    alloc = np.sum(diff_w * Rb_g, axis=1)  # (T,)
    select = np.sum(Wb_g * diff_r, axis=1)  # (T,)
    inter = np.sum(diff_w * diff_r, axis=1)  # (T,)
    total = alloc + select + inter

    if cumulative:
        alloc = np.cumsum(alloc)
        select = np.cumsum(select)
        inter = np.cumsum(inter)
        total = np.cumsum(total)

    return AttributionResult(
        date=np.asarray(aln.dates),
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
