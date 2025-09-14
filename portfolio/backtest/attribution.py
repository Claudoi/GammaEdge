# Performance attribution
# portfolio/backtest/attribution.py
from __future__ import annotations

from portfolio.core.compat import dataclass_compat as dataclass
from typing import Callable, Iterable

import numpy as np
import polars as pl


# ──────────────────────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DailyAlignment:
    dates: list
    tickers: list[str]
    returns: np.ndarray  # shape (T, N)  daily asset returns
    weights: np.ndarray  # shape (T, N)  daily portfolio weights (post-rebalance, stepwise)


# ──────────────────────────────────────────────────────────────────────────────
# Alineación y expansión de pesos
# ──────────────────────────────────────────────────────────────────────────────

def align_returns_and_weights(
    df_ret_wide: pl.DataFrame,                  # ['date', T1, T2, ...] ordenado
    daily_weights: np.ndarray,                  # (T, N) ya en frecuencia diaria
) -> DailyAlignment:
    """
    Alinea retornos y pesos ya diarios. Valida shapes y devuelve np arrays.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("df_ret_wide debe contener columna 'date'")

    df = df_ret_wide.sort("date")
    tickers = [c for c in df.columns if c != "date"]
    T = df.height
    N = len(tickers)

    if daily_weights.shape != (T, N):
        raise ValueError(f"dims incompatibles: weights {daily_weights.shape} vs returns {(T, N)}")

    R = np.column_stack([df.get_column(t).to_numpy() for t in tickers])
    # NaN-safe → tratamos NaN como 0 en retorno para no contaminar contribuciones
    R = np.nan_to_num(R, copy=False, nan=0.0)

    return DailyAlignment(
        dates=df.get_column("date").to_list(),
        tickers=tickers,
        returns=R,
        weights=daily_weights.astype(float, copy=False),
    )


def expand_rebalance_weights(
    dates: Iterable,                 # fechas diarias completas
    rb_dates: Iterable,              # fechas en las que hay rebalanceo
    W_reb: np.ndarray,               # (n_reb, N) pesos en cada rebalance
) -> np.ndarray:
    """
    Expande pesos de rebalance (stepwise hold) a diario (forward-fill hasta próximo rebalance).
    - `dates` y `rb_dates` deben ser comparables (mismo tipo y timezone).
    """
    dates = list(dates)
    rb_dates = list(rb_dates)
    n_reb, N = W_reb.shape
    if len(rb_dates) != n_reb:
        raise ValueError("len(rb_dates) debe coincidir con W_reb.shape[0]")

    # mapa fecha_reb → índice de rebalance
    rb_ix = {d: i for i, d in enumerate(rb_dates)}
    out = np.zeros((len(dates), N), dtype=float)

    last_w: np.ndarray | None = None
    for i, d in enumerate(dates):
        if d in rb_ix:
            last_w = W_reb[rb_ix[d]].astype(float, copy=False)
            s = last_w.sum()
            last_w = last_w / (s if abs(s) > 1e-12 else 1.0)
        if last_w is None:
            # Antes del primer rebalance: igual ponderación
            last_w = np.full(N, 1.0 / N, dtype=float)
        out[i, :] = last_w
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Atribución por activo y por grupo
# ──────────────────────────────────────────────────────────────────────────────

def contributions_by_asset(aln: DailyAlignment) -> pl.DataFrame:
    """
    Contribución diaria por activo: c_ti = w_ti * r_ti
    Devuelve un DataFrame Polars largo: ['date','ticker','contrib','ret','weight'].
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
    Atribución agrupada (suma de contribuciones por grupo).
    group_map: ticker -> grupo (sector/país/lo_que_sea). Si falta alguno, va a OTHER.
    Devuelve DF largo: ['date','group','contrib','weight'] (weight = suma de pesos del grupo).
    """
    df_asset = contributions_by_asset(aln)
    df_asset = df_asset.with_columns(
        pl.col("ticker").map_elements(lambda t: group_map.get(t, other_label)).alias("group")
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
# Brinson-Fachler (sencillo)
# ──────────────────────────────────────────────────────────────────────────────

def brinson_fachler_period(
    w_p: np.ndarray, r_p: np.ndarray,     # cartera
    w_b: np.ndarray, r_b: np.ndarray,     # benchmark
    groups: list[int],                     # asignación por activo a grupos [0..G-1]
) -> tuple[float, float, float]:
    """
    Brinson-Fachler en una ventana (un periodo):
    - Asignación: sum_g (w_p_g - w_b_g) * r_b_g
    - Selección:  sum_g w_b_g * (r_p_g - r_b_g)
    - Interacción: sum_g (w_p_g - w_b_g) * (r_p_g - r_b_g)
    Retorna (A, S, I).
    """
    groups = np.asarray(groups, dtype=int)
    G = int(groups.max()) + 1 if len(groups) else 0
    A = S = I = 0.0
    for g in range(G):
        idx = (groups == g)
        wp_g = float(np.sum(w_p[idx]))
        wb_g = float(np.sum(w_b[idx]))
        # medias ponderadas de retorno en el grupo
        rb_g = float(np.sum(w_b[idx] * r_b[idx]) / (wb_g if wb_g > 1e-16 else 1.0))
        rp_g = float(np.sum(w_p[idx] * r_p[idx]) / (wp_g if wp_g > 1e-16 else 1.0))

        A += (wp_g - wb_g) * rb_g
        S += wb_g * (rp_g - rb_g)
        I += (wp_g - wb_g) * (rp_g - rb_g)
    return A, S, I


def brinson_fachler_cumulative(
    aln: DailyAlignment,
    bench_weights_daily: np.ndarray,   # (T, N) benchmark estático o time-varying
    groups_idx: list[int],             # mapping de tickers a índices de grupo [0..G-1]
) -> pl.DataFrame:
    """
    Agrega Brinson-Fachler por día (periodo) y devuelve acumulado (suma aritmética).
    Columns: ['date','alloc','select','interact','total']
    """
    R = aln.returns
    Wp = aln.weights
    Wb = bench_weights_daily
    if Wb.shape != Wp.shape:
        raise ValueError("bench_weights_daily debe tener misma shape que portfolio weights")

    rows = []
    for t in range(R.shape[0]):
        a, s, inter = brinson_fachler_period(Wp[t], R[t], Wb[t], R[t], groups_idx)
        rows.append((aln.dates[t], a, s, inter, a + s + inter))
    return pl.DataFrame(rows, schema=["date", "alloc", "select", "interact", "total"]).sort("date")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de resumen
# ──────────────────────────────────────────────────────────────────────────────

def top_contributors(
    df_asset_contrib: pl.DataFrame,
    *,
    k: int = 10,
    sign: str = "both",  # "pos" | "neg" | "both"
) -> pl.DataFrame:
    """
    Toma DF largo de contributions_by_asset y devuelve top-k contribuyentes acumulados.
    """
    agg = (
        df_asset_contrib.group_by("ticker")
        .agg(pl.col("contrib").sum().alias("contrib_total"))
        .sort("contrib_total", descending=True)
    )
    if sign == "pos":
        agg = agg.filter(pl.col("contrib_total") > 0)
    elif sign == "neg":
        agg = agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False)

    if sign == "both":
        pos = agg.filter(pl.col("contrib_total") > 0).head(k)
        neg = (
            agg.filter(pl.col("contrib_total") < 0)
            .sort("contrib_total", descending=False)
            .head(k)
        )
        return pl.concat([pos, neg])
    return agg.head(k)