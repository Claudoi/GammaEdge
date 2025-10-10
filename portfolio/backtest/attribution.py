# Performance attribution
# portfolio/backtest/attribution.py
from __future__ import annotations

from portfolio.core.compat import dataclass_compat as dataclass
from typing import Iterable, Tuple

import numpy as np
import polars as pl


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DailyAlignment:
    dates: list
    tickers: list[str]
    returns: np.ndarray  # shape (T, N)  daily asset returns
    weights: np.ndarray  # shape (T, N)  daily portfolio weights (post-rebalance, stepwise)


# ──────────────────────────────────────────────────────────────────────────────
# Alignment helpers
# ──────────────────────────────────────────────────────────────────────────────

def align_returns_and_weights(
    df_ret_wide: pl.DataFrame,                  # ['date', T1, T2, ...] sorted
    daily_weights: np.ndarray,                  # (T, N) daily weights
) -> DailyAlignment:
    """
    Align wide daily returns with already-expanded daily weights.
    Validates shapes and returns numpy arrays.
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
    # NaN-safe: treat NaN returns as 0 to avoid contaminating contributions
    R = np.nan_to_num(R, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return DailyAlignment(
        dates=df.get_column("date").to_list(),
        tickers=tickers,
        returns=R,
        weights=daily_weights.astype(float, copy=False),
    )


def expand_rebalance_weights(
    dates: Iterable,                 # full daily dates
    rb_dates: Iterable,              # rebalance dates
    W_reb: np.ndarray,               # (n_reb, N) weights at each rebalance
) -> np.ndarray:
    """
    Expand stepwise rebalance weights to daily frequency (forward-fill until next rebalance).
    Assumes 'dates' and 'rb_dates' are directly comparable types.
    """
    dates = list(dates)
    rb_dates = list(rb_dates)
    n_reb, N = W_reb.shape
    if len(rb_dates) != n_reb:
        raise ValueError("len(rb_dates) must match W_reb.shape[0]")

    # map rebalance date → rebalance index
    rb_ix = {d: i for i, d in enumerate(rb_dates)}
    out = np.zeros((len(dates), N), dtype=float)

    last_w: np.ndarray | None = None
    for i, d in enumerate(dates):
        if d in rb_ix:
            last_w = W_reb[rb_ix[d]].astype(float, copy=False)
            s = last_w.sum()
            last_w = last_w / (s if abs(s) > 1e-12 else 1.0)
        if last_w is None:
            # Before first rebalance: equal-weight
            last_w = np.full(N, 1.0 / N, dtype=float)
        out[i, :] = last_w
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Contributions by asset and by group
# ──────────────────────────────────────────────────────────────────────────────

def contributions_by_asset(aln: DailyAlignment) -> pl.DataFrame:
    """
    Daily contribution per asset: c_{t,i} = w_{t,i} * r_{t,i}.
    Returns a long Polars DataFrame: ['date','ticker','contrib','ret','weight'].
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
    Grouped attribution (sum of contributions by group).
    If a ticker is not present in 'group_map', it falls back to 'other_label'.
    Returns a long DF: ['date','group','contrib','weight'] (weight = group weight sum).
    """
    df_asset = contributions_by_asset(aln)
    # Fast mapping using replace with default (Polars ≥0.19)
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
# Brinson–Fachler (basic)
# ──────────────────────────────────────────────────────────────────────────────

def brinson_fachler_period(
    w_p: np.ndarray, r_p: np.ndarray,     # portfolio
    w_b: np.ndarray, r_b: np.ndarray,     # benchmark
    groups: list[int],                    # asset→group indices [0..G-1]
) -> Tuple[float, float, float]:
    """
    Single-period Brinson–Fachler:
    - Allocation:  Σ_g (w_p_g - w_b_g) * r_b_g
    - Selection:   Σ_g w_b_g * (r_p_g - r_b_g)
    - Interaction: Σ_g (w_p_g - w_b_g) * (r_p_g - r_b_g)
    Returns (A, S, I).
    """
    groups = np.asarray(groups, dtype=int)
    G = int(groups.max()) + 1 if len(groups) else 0
    A = S = I = 0.0
    for g in range(G):
        idx = (groups == g)
        wp_g = float(np.sum(w_p[idx]))
        wb_g = float(np.sum(w_b[idx]))
        # weighted average returns inside the group (guard against zero weight)
        rb_g = float(np.sum(w_b[idx] * r_b[idx])) / wb_g if wb_g > 1e-16 else 0.0
        rp_g = float(np.sum(w_p[idx] * r_p[idx])) / wp_g if wp_g > 1e-16 else 0.0

        diff_w = (wp_g - wb_g)
        diff_r = (rp_g - rb_g)
        A += diff_w * rb_g
        S += wb_g * diff_r
        I += diff_w * diff_r
    return A, S, I


def brinson_fachler_cumulative(
    aln: DailyAlignment,
    bench_weights_daily: np.ndarray,   # (T, N)
    groups_idx: list[int],             # asset→group indices [0..G-1]
) -> pl.DataFrame:
    """
    Sum single-period Brinson–Fachler components over time (arithmetic sum).
    Output columns: ['date','alloc','select','interact','total'].
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
# Summaries & high-level API (used by the UI)
# ──────────────────────────────────────────────────────────────────────────────

def _daily_alignment_from_bt(bt: dict, df_ret_wide: pl.DataFrame) -> DailyAlignment:
    """
    Build DailyAlignment from the backtest dict:
    - expands rebalance weights to the daily grid
    - aligns wide returns to the same grid and ticker order
    """
    dates = bt["dates"]
    tickers = bt["tickers"]
    W_reb = np.asarray(bt["weights"], dtype=float)            # (n_reb, N)
    rb_dates = bt.get("rebalance_dates", [])
    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0]:
        raise ValueError("Invalid 'weights'/'rebalance_dates' in bt.")

    # Ensure returns are filtered and ordered to match 'dates' and 'tickers'
    df = (
        df_ret_wide
        .filter(pl.col("date").is_in(dates))
        .sort("date")
        .select(["date", *tickers])
    )

    W_daily = expand_rebalance_weights(dates, rb_dates, W_reb)  # (T, N)
    return align_returns_and_weights(df, W_daily)


def top_contributors(
    bt: dict,
    df_ret_wide: pl.DataFrame,
    *,
    top_n: int = 10,
    sign: str = "both",  # "pos" | "neg" | "both"
) -> pl.DataFrame:
    """
    High-level: compute top contributors (cumulative) directly from bt + returns.
    Returns Polars DF with ['ticker','contrib_total'].
    """
    aln = _daily_alignment_from_bt(bt, df_ret_wide)
    df_asset = contributions_by_asset(aln)

    agg = (
        df_asset.group_by("ticker")
        .agg(pl.col("contrib").sum().alias("contrib_total"))
        .sort("contrib_total", descending=True)
    )
    if sign == "pos":
        agg = agg.filter(pl.col("contrib_total") > 0)
        return agg.head(top_n)
    if sign == "neg":
        agg = agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False)
        return agg.head(top_n)

    # both: top positive and top negative
    pos = agg.filter(pl.col("contrib_total") > 0).head(top_n)
    neg = (
        agg.filter(pl.col("contrib_total") < 0)
        .sort("contrib_total", descending=False)
        .head(top_n)
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
    High-level: grouped contributions from bt + returns.
    Returns:
      - df_group_total: ['group','contrib_total','weight_avg'] (cumulative)
      - df_group_daily: ['date','group','contrib','weight']   (daily series)
    """
    aln = _daily_alignment_from_bt(bt, df_ret_wide)
    df_daily = contributions_by_group(aln, groups_map, other_label=other_label)

    df_total = (
        df_daily.group_by("group")
        .agg([
            pl.col("contrib").sum().alias("contrib_total"),
            pl.col("weight").mean().alias("weight_avg"),
        ])
        .sort("contrib_total", descending=True)
    )
    return df_total, df_daily





# ──────────────────────────────────────────────────────────────────────────────
# Advanced helpers (safe additions; do not replace existing functions)
# ──────────────────────────────────────────────────────────────────────────────

def align_with_ipo_mask(aln: DailyAlignment) -> DailyAlignment:
    """
    IPO-safe alignment:
    - Zero weight for each asset before its first valid return observation.
    - Row-wise renormalization (if a row has no available assets yet, fallback to EW
      across assets that have already 'IPO'ed by that date).
    - Returns a new DailyAlignment with the same dates/tickers and adjusted weights.
    """
    R = np.asarray(aln.returns, dtype=float)
    W = np.asarray(aln.weights, dtype=float).copy()
    T, N = R.shape

    # 'Valid' = finite and non-tiny absolute return (ignore tiny numerical noise)
    valid = np.isfinite(R) & (np.abs(R) > 1e-15)

    t0 = np.full(N, T, dtype=int)  # default 'after the end' (never valid)
    for j in range(N):
        nz = np.nonzero(valid[:, j])[0]
        if nz.size:
            t0[j] = int(nz[0])

    # Zero weight before inception for each asset
    for j in range(N):
        if t0[j] > 0:
            W[:t0[j], j] = 0.0

    # Row-wise renormalization; EW across available assets when a row sums ~0
    row_sum = W.sum(axis=1, keepdims=True)
    available = (np.arange(T)[:, None] >= t0[None, :])
    n_avail = available.sum(axis=1, keepdims=True).astype(float)

    zero_rows = (np.abs(row_sum) < 1e-15).ravel()
    if np.any(zero_rows):
        W[zero_rows, :] = 0.0
        W[zero_rows, :] = np.where(available[zero_rows, :], 1.0, 0.0)
        W[zero_rows, :] /= np.clip(n_avail[zero_rows, :], 1.0, None)

    nz_rows = ~zero_rows
    if np.any(nz_rows):
        W[nz_rows, :] /= np.clip(row_sum[nz_rows, :], 1e-15, None)

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
    """
    Build an asset→group index for Brinson and grouped reporting.

    Returns:
      - groups_idx: list[int] of length N, with group index for each ticker (0..G-1)
      - group_labels: list[str] of group names in index order
      - groups_map: dict[ticker -> group_name] (useful elsewhere in the app)

    If meta_df is None or col missing, falls back to identity grouping (each ticker = its own group).
    """
    if meta_df is None or "ticker" not in meta_df.columns or col not in meta_df.columns:
        # Identity grouping: one group per ticker
        groups_map = {tk: tk for tk in tickers}
        labels = list(tickers)
        idx = list(range(len(tickers)))
        return idx, labels, groups_map

    lut = dict(zip(meta_df["ticker"].to_list(), meta_df[col].to_list()))
    # Normalize groups preserving order of first appearance
    label_order: list[str] = []
    groups_map: dict[str, str] = {}
    for tk in tickers:
        g = lut.get(tk, other_label)
        groups_map[tk] = g
        if g not in label_order:
            label_order.append(g)
    label_to_id = {g: i for i, g in enumerate(label_order)}
    groups_idx = [label_to_id[groups_map[tk]] for tk in tickers]
    return groups_idx, label_order, groups_map


def coerce_benchmark_weights(
    Wb: np.ndarray | None,
    T: int,
    N: int,
    *,
    scheme: str = "EW",
) -> np.ndarray:
    """
    Ensure benchmark weights have shape (T, N). If Wb is None, build one by scheme:
      - "EW": equal-weight each day
      - "cash": all zeros (no benchmark exposure)
    Any row that sums ~0 is renormalized to EW (safety).
    """
    if Wb is None or Wb.size == 0:
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
    zero_rows = (np.abs(row_sum) < 1e-15).ravel()
    if np.any(zero_rows):
        Wb[zero_rows, :] = 1.0
        Wb[zero_rows, :] /= float(N) if N > 0 else 1.0
    else:
        Wb /= np.clip(row_sum, 1e-15, None)
    return Wb


def top_contributors_from_asset(
    df_asset_contrib: pl.DataFrame,
    *,
    top_n: int = 10,
    sign: str = "both",  # "pos", "neg", "both"
) -> pl.DataFrame:
    """
    Pure DF version of top contributors.
    Input DF must have ['ticker','contrib'] (optionally ['date','ret','weight'] present).
    Returns Polars DF with ['ticker','contrib_total'] for the requested sign.
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
    # both
    pos = agg.filter(pl.col("contrib_total") > 0).head(top_n)
    neg = (
        agg.filter(pl.col("contrib_total") < 0)
           .sort("contrib_total", descending=False)
           .head(top_n)
    )
    return pl.concat([pos, neg])


def contributions_share_by_group(df_group_daily: pl.DataFrame) -> pl.DataFrame:
    """
    Convert daily group contribution to 'share of total' per date:
      share_{t,g} = contrib_{t,g} / sum_g contrib_{t,g}
    If daily sum is ~0, returns 0 to avoid blow-ups.
    Output columns: ['date','group','share'].
    """
    req = {"date", "group", "contrib"}
    if not req.issubset(set(df_group_daily.columns)):
        raise ValueError("df_group_daily must contain 'date','group','contrib'")
    total = df_group_daily.group_by("date").agg(pl.col("contrib").sum().alias("tot"))
    out = (
        df_group_daily.join(total, on="date", how="left")
        .with_columns((pl.col("contrib") / pl.when(pl.abs(pl.col("tot")) > 1e-15).then(pl.col("tot")).otherwise(1.0)).alias("share"))
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
    by_group: bool = False,
) -> pl.DataFrame:
    """
    Brinson–Fachler over time, optionally:
      - cumulative: cumulative sum (True) or per-period (False)
      - by_group: return components per group (True) or just totals (False)

    If by_group=True, returns columns:
      ['date','group','alloc','select','interact','total']
    Else returns:
      ['date','alloc','select','interact','total'] (same as brinson_fachler_cumulative)
    """
    R = np.asarray(aln.returns, dtype=float)
    Wp = np.asarray(aln.weights, dtype=float)
    Wb = np.asarray(bench_weights_daily, dtype=float)
    if Wb.shape != Wp.shape:
        raise ValueError("bench_weights_daily must have same shape as portfolio weights")

    groups_idx = np.asarray(groups_idx, dtype=int)
    if groups_idx.shape[0] != Wp.shape[1]:
        raise ValueError("groups_idx must have length N")

    # Group membership matrix (N x G): one-hot
    G = int(groups_idx.max()) + 1 if groups_idx.size else 0
    N = Wp.shape[1]
    H = np.zeros((N, G), dtype=float)
    if G > 0:
        H[np.arange(N), groups_idx] = 1.0

    dates = list(aln.dates)
    rows = []

    for t in range(Wp.shape[0]):
        wp = Wp[t]        # (N,)
        wb = Wb[t]        # (N,)
        rp = R[t]         # (N,)
        rb = R[t]         # (N,)

        # group weights
        wp_g = H.T @ wp    # (G,)
        wb_g = H.T @ wb    # (G,)

        # group returns (guard zeros)
        rb_g = np.divide(H.T @ (wb * rb), np.clip(wb_g, 1e-16, None), where=wb_g > 1e-16)
        rp_g = np.divide(H.T @ (wp * rp), np.clip(wp_g, 1e-16, None), where=wp_g > 1e-16)

        diff_w = wp_g - wb_g
        diff_r = rp_g - rb_g

        alloc = diff_w * rb_g
        select = wb_g * diff_r
        inter = diff_w * diff_r
        total = alloc + select + inter

        if by_group:
            for g in range(G):
                rows.append((dates[t], int(g), alloc[g], select[g], inter[g], total[g]))
        else:
            rows.append((dates[t], float(alloc.sum()), float(select.sum()), float(inter.sum()), float(total.sum())))

    if by_group:
        out = pl.DataFrame(rows, schema=["date", "group_id", "alloc", "select", "interact", "total"]).sort("date")
        # Attach readable labels via group_id if needed outside; keep numeric id here to avoid expensive joins
        if cumulative:
            out = out.with_columns(
                pl.col("alloc").cum_sum().over("group_id"),
                pl.col("select").cum_sum().over("group_id"),
                pl.col("interact").cum_sum().over("group_id"),
                pl.col("total").cum_sum().over("group_id"),
            )
        return out
    else:
        out = pl.DataFrame(rows, schema=["date", "alloc", "select", "interact", "total"]).sort("date")
        if cumulative:
            out = out.with_columns(
                pl.col("alloc").cum_sum(),
                pl.col("select").cum_sum(),
                pl.col("interact").cum_sum(),
                pl.col("total").cum_sum(),
            )
        return out