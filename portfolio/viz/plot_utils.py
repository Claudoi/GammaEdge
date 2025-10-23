# portfolio/viz/plot_utils.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from portfolio.core.compat import dataclass_compat as dataclass

DataFrameLike = Union[pd.DataFrame, "pl.DataFrame"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers numéricos
# ──────────────────────────────────────────────────────────────────────────────

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True, slots=True)
class HeatmapOrder:
    clustered: bool = True
    method: Literal["single", "complete", "average", "ward"] = "average"
    optimal: bool = True  # optimal leaf ordering (reduce disonancias)


def _to_numpy_matrix(x: np.ndarray | pl.DataFrame) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pl.DataFrame):
        return x.to_numpy()
    raise TypeError("Expected np.ndarray or polars.DataFrame")


def _safe_corr_from_cov(Sigma: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square matrix")
    d = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    d[d < eps] = eps
    R = (Sigma / d[:, None]) / d[None, :]
    R = np.clip(R, -1.0, 1.0)
    # simetriza por estabilidad numérica
    return 0.5 * (R + R.T)


def _hierarchical_order(Corr: np.ndarray, order_cfg: HeatmapOrder) -> np.ndarray:
    # distancia "correlation distance": sqrt(0.5*(1-ρ))
    # bounded en [0,1]
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - Corr)))
    dvec = squareform(dist, checks=False)
    Z = linkage(dvec, method=order_cfg.method)
    if order_cfg.optimal:
        Z = optimal_leaf_ordering(Z, dvec)
    order = leaves_list(Z)
    return order


def _apply_order(mat: np.ndarray, order: np.ndarray) -> np.ndarray:
    return mat[np.ix_(order, order)]


def _to_pandas(df: DataFrameLike) -> pd.DataFrame:
    """Convert Polars DataFrame to Pandas if needed."""
    try:
        import polars as pl  # type: ignore

        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    except Exception:
        pass
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError("Unsupported DataFrame type. Expected pandas or polars.")


# ──────────────────────────────────────────────────────────────────────────────
# 1) Correlation Heatmap (clustered)
# ──────────────────────────────────────────────────────────────────────────────


def corr_heatmap(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    order: HeatmapOrder = HeatmapOrder(),
    zlim: tuple[float, float] = (-1.0, 1.0),
    title: str = "Correlation Heatmap (clustered)",
) -> go.Figure:
    """
    Heatmap de correlación con ordenamiento jerárquico.
    - Acepta Σ o ρ. Si is_cov=True, convierte a ρ primero.
    - Ordena por clustering (average por defecto) y leaf ordering óptimo (opcional).
    """
    M = _to_numpy_matrix(Sigma_or_Corr)
    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)

    n = Corr.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]

    if order.clustered and n >= 3:
        ord_idx = _hierarchical_order(Corr, order)
        Corr_ord = _apply_order(Corr, ord_idx)
        labels_ord = [labels[i] for i in ord_idx]
    else:
        Corr_ord, labels_ord = Corr, list(labels)

    fig = go.Figure(
        data=go.Heatmap(
            z=Corr_ord,
            x=labels_ord,
            y=labels_ord,
            zmin=zlim[0],
            zmax=zlim[1],
            colorbar=dict(title="ρ"),
            hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2) Correlation Dendrogram
# ──────────────────────────────────────────────────────────────────────────────

# Optional SciPy import with safe fallback
try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
except Exception:
    linkage = None
    dendrogram = None
    squareform = None


def corr_dendrogram(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    method: Literal["single", "complete", "average", "ward"] = "average",
    title: str = "Correlation Dendrogram",
) -> go.Figure:
    """
    Build a hierarchical clustering dendrogram from a correlation or covariance matrix.

    Parameters
    ----------
    Sigma_or_Corr : np.ndarray | pl.DataFrame
        Covariance or correlation matrix (NxN).
    labels : list[str] | None
        Asset labels for leaf nodes.
    is_cov : bool
        Whether the input is covariance (True) or already a correlation matrix.
    method : str
        Linkage method for hierarchical clustering (e.g., 'ward', 'average', etc.).
    title : str
        Plot title.
    """
    # ---- Sanitize input matrix ----
    M = (
        Sigma_or_Corr.to_numpy()
        if hasattr(Sigma_or_Corr, "to_numpy")
        else np.asarray(Sigma_or_Corr, dtype=float)
    )
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("corr_dendrogram: input must be a square matrix (NxN).")
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Convert covariance to correlation if necessary ----
    if is_cov:
        std = np.sqrt(np.clip(np.diag(M), 0.0, np.inf))
        std[std == 0.0] = 1.0
        Corr = M / np.outer(std, std)
    else:
        Corr = np.copy(M)

    # Clamp to [-1, 1] to avoid numerical issues
    Corr = np.clip(Corr, -1.0, 1.0)
    n = Corr.shape[0]

    # ---- Prepare labels ----
    if labels is None:
        labels = [f"A{i}" for i in range(n)]
    else:
        labels = list(labels)
        if len(labels) != n:
            # Adjust label list length for safety
            labels = (labels + [f"A{i}" for i in range(len(labels), n)])[:n]

    # ---- Compute pairwise distance from correlation ----
    # Distance = sqrt(0.5 * (1 - Corr)) ensures values in [0, 1]
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - Corr)))
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Fallback if SciPy is unavailable ----
    if linkage is None or dendrogram is None or squareform is None:
        fig = go.Figure()
        fig.update_layout(
            title=title + " (scipy missing)",
            xaxis_title="Assets",
            yaxis_title="Distance",
            template="plotly_white",
            height=420,
            margin=dict(l=60, r=20, t=60, b=80),
        )
        return fig

    # ---- Condensed vector expected by linkage ----
    dvec = squareform(dist, checks=False)

    # ---- Compute linkage and dendrogram (without plotting) ----
    Z = linkage(dvec, method=method)
    dn = dendrogram(Z, labels=labels, no_plot=True)

    icoord = np.asarray(dn["icoord"], dtype=float)
    dcoord = np.asarray(dn["dcoord"], dtype=float)
    xlbls = dn.get("ivl", labels)

    # ---- Build Plotly figure ----
    lines = []
    for xs, ys in zip(icoord, dcoord):  # Compatible with Python 3.9 (no strict argument)
        lines.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=lines)
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(5, 10 * n, 10)),  # Standard tick positions for dendrogram leaves
            ticktext=xlbls,
            tickangle=45,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(title="Distance", showgrid=True, zeroline=False),
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=20, t=60, b=120),
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3) Covariance Spectrum (eigenvalues)
# ──────────────────────────────────────────────────────────────────────────────


def covariance_spectrum(
    Sigma: np.ndarray | pl.DataFrame,
    *,
    title: str = "Covariance Spectrum (eigenvalues)",
) -> go.Figure:
    """
    Muestra los autovalores de Σ (ordenados), útil para diagnosticar
    condicionamiento y decidir shrinkage/regularización.
    """
    S = _to_numpy_matrix(Sigma)
    S = 0.5 * (S + S.T)
    vals = np.linalg.eigvalsh(S)
    vals_sorted = np.sort(vals)[::-1]
    cond = (vals_sorted[0] / max(vals_sorted[-1], 1e-16)) if vals_sorted.size else np.nan

    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.arange(1, len(vals_sorted) + 1),
                y=vals_sorted,
                mode="lines+markers",
                hovertemplate="idx=%{x}<br>λ=%{y:.4e}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"{title}  —  κ≈{cond:.2e}",
        xaxis_title="Index",
        yaxis_title="Eigenvalue (λ)",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Small internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _placeholder_figure(title: str, subtitle: str = "No data available") -> go.Figure:
    """Return a minimal placeholder figure instead of raising exceptions."""
    fig = go.Figure()
    fig.add_annotation(
        text=subtitle,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        align="center",
        font=dict(size=14),
    )
    fig.update_layout(title=title, margin=dict(l=60, r=20, t=60, b=60), showlegend=False)
    return fig


def _to_1d_float(x) -> np.ndarray:
    """Coerce input to a 1-D float array; on failure return an empty array."""
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return np.array([], dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# 4) Efficient Frontier
# ──────────────────────────────────────────────────────────────────────────────


def efficient_frontier(
    *args,
    mu: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
    rf: float = 0.0,
    risks_closed: Iterable[float] | None = None,
    rets_closed: Iterable[float] | None = None,
    risks_box: Iterable[float] | None = None,
    rets_box: Iterable[float] | None = None,
    msr_point: tuple[float, float] | None = None,
    minvar_point: tuple[float, float] | None = None,
    custom_points: dict[str, tuple[float, float]] | None = None,
    title: str = "Efficient Frontier",
    **kwargs,
) -> go.Figure:
    """
    Robust frontier figure that accepts either:
      • risks_closed/rets_closed and optional risks_box/rets_box, or
      • efficient_frontier(risks, rets) via *args compatibility.
    The function filters NaNs, sorts by risk, and never raises if inputs are partial.
    """

    # Backward compatibility: efficient_frontier(risks, rets)
    if len(args) == 2 and risks_closed is None and rets_closed is None:
        risks_closed = args[0]
        rets_closed = args[1]

    def _clean_and_sort(x_raw, y_raw):
        x = _to_1d_float(x_raw)
        y = _to_1d_float(y_raw)
        if x.size == 0 or y.size == 0:
            return np.array([]), np.array([])
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return x, y
        idx = np.argsort(x)
        return x[idx], y[idx]

    fig = go.Figure()

    # Clean/ordered series
    x_c, y_c = _clean_and_sort(risks_closed, rets_closed)
    x_b, y_b = _clean_and_sort(risks_box, rets_box)

    # Optional shaded gap between the two frontiers over common σ domain
    def _add_constraint_gap_fill(fig: go.Figure, x1, y1, x2, y2):
        try:
            if x1.size and x2.size:
                xmin = max(x1.min(), x2.min())
                xmax = min(x1.max(), x2.max())
                if np.isfinite([xmin, xmax]).all() and xmax > xmin:
                    xs = np.linspace(xmin, xmax, 200)
                    y1i = np.interp(xs, x1, y1)
                    y2i = np.interp(xs, x2, y2)
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([xs, xs[::-1]]),
                            y=np.concatenate([y1i, y2i[::-1]]),
                            fill="toself",
                            mode="lines",
                            line=dict(width=0),
                            fillcolor="rgba(99,110,250,0.15)",
                            name="Constraint gap",
                            hoverinfo="skip",
                            showlegend=True,
                        )
                    )
        except Exception:
            # Never fail on cosmetic fill
            pass

    if x_c.size and x_b.size:
        _add_constraint_gap_fill(fig, x_c, y_c, x_b, y_b)

    # Closed-form curve
    if x_c.size:
        fig.add_trace(
            go.Scatter(
                x=x_c,
                y=y_c,
                mode="lines",
                name="Closed-form (no box)",
                line=dict(width=2),
                hovertemplate="σ=%{x:.4f}<br>μ=%{y:.4f}<extra>Closed-form</extra>",
            )
        )

    # Box-projected curve
    if x_b.size:
        fig.add_trace(
            go.Scatter(
                x=x_b,
                y=y_b,
                mode="lines",
                name="Box-projected",
                line=dict(width=2, dash="dash"),
                hovertemplate="σ=%{x:.4f}<br>μ=%{y:.4f}<extra>Box-projected</extra>",
            )
        )

    # Key points (MinVar & MSR)
    if isinstance(minvar_point, (tuple, list)) and len(minvar_point) == 2:
        sx, ry = float(minvar_point[0]), float(minvar_point[1])
        if np.isfinite(sx) and np.isfinite(ry):
            fig.add_trace(
                go.Scatter(
                    x=[sx],
                    y=[ry],
                    mode="markers",
                    name="MinVar",
                    marker=dict(size=10, symbol="diamond"),
                    hovertemplate="MinVar<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>",
                )
            )

    title_suffix = ""
    if isinstance(msr_point, (tuple, list)) and len(msr_point) == 2:
        sx, ry = float(msr_point[0]), float(msr_point[1])
        if np.isfinite(sx) and np.isfinite(ry):
            fig.add_trace(
                go.Scatter(
                    x=[sx],
                    y=[ry],
                    mode="markers",
                    name="Max Sharpe",
                    marker=dict(size=11, symbol="star"),
                    hovertemplate="Max Sharpe<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>",
                )
            )
            # CAL (if rf is provided)
            if np.isfinite(rf) and sx > 1e-12:
                sharpe = (ry - rf) / sx
                title_suffix = f" · Sharpe*={sharpe:.2f}"
                fig.add_trace(
                    go.Scatter(
                        x=[0.0, sx],
                        y=[rf, ry],
                        mode="lines",
                        name="CAL",
                        line=dict(width=1, dash="dot"),
                        hovertemplate="CAL<extra></extra>",
                    )
                )

    # Custom points
    if isinstance(custom_points, dict) and custom_points:
        for label, pt in custom_points.items():
            if isinstance(pt, (tuple, list)) and len(pt) == 2:
                sx, ry = float(pt[0]), float(pt[1])
                if np.isfinite(sx) and np.isfinite(ry):
                    fig.add_trace(
                        go.Scatter(
                            x=[sx],
                            y=[ry],
                            mode="markers+text",
                            name=str(label),
                            text=[str(label)],
                            textposition="top center",
                            marker=dict(size=8),
                            hovertemplate=f"{label}<br>σ=%{{x:.4f}}<br>μ=%{{y:.4f}}<extra></extra>",
                        )
                    )

    # Auto ranges (do not crash if everything is empty)
    candidates_x = []
    candidates_y = []
    if x_c.size:
        candidates_x.append(x_c.max())
    if x_b.size:
        candidates_x.append(x_b.max())
    if isinstance(msr_point, (tuple, list)) and len(msr_point) == 2:
        candidates_x.append(float(msr_point[0]))
        candidates_y.append(float(msr_point[1]))
    if isinstance(minvar_point, (tuple, list)) and len(minvar_point) == 2:
        candidates_y.append(float(minvar_point[1]))
    if y_c.size:
        candidates_y.append(y_c.max())
    if y_b.size:
        candidates_y.append(y_b.max())

    x_max = np.nanmax(candidates_x) if candidates_x else np.nan
    y_max = np.nanmax(candidates_y) if candidates_y else np.nan

    if not (x_c.size or x_b.size or candidates_x or candidates_y):
        return _placeholder_figure(title, subtitle="Frontier data not available")

    fig.update_layout(
        title=title + title_suffix,
        xaxis=dict(
            title="Risk (σ)",
            rangemode="tozero",
            range=[0, x_max * 1.05] if np.isfinite(x_max) and x_max > 0 else None,
            showgrid=True,
            zeroline=True,
        ),
        yaxis=dict(
            title="Return (μ)",
            range=[None, y_max * 1.06] if np.isfinite(y_max) and y_max > 0 else None,
            showgrid=True,
            zeroline=True,
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5) Weights Barplot
# ──────────────────────────────────────────────────────────────────────────────


def weights_bar(
    weights,
    labels: Sequence[str],
    *,
    sort: bool = True,
    topn: int | None = None,
    horizontal: bool = True,
    title: str = "Portfolio Weights",
) -> go.Figure:
    """
    Bar plot of portfolio weights. Defensive against NaNs and inconsistent inputs.
    """
    w = _to_1d_float(weights)
    if w.size == 0 or labels is None or len(labels) == 0:
        return _placeholder_figure(title)

    # Align lengths defensively
    N = min(w.size, len(labels))
    if N == 0:
        return _placeholder_figure(title)
    w = w[:N]
    lab = list(labels)[:N]

    # Replace non-finite values with zeros for plotting
    w = np.where(np.isfinite(w), w, 0.0)

    idx = np.arange(N)
    if sort:
        idx = np.argsort(w)
    if topn is not None:
        topn = int(max(1, min(topn, N)))
        idx = idx[-topn:]

    w_plot = w[idx]
    l_plot = [lab[i] for i in idx]

    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=w_plot, y=l_plot, orientation="h", hovertemplate="%{y}: %{x:.2%}<extra></extra>"
            )
        )
        fig.update_layout(
            xaxis_tickformat=".0%",
            title=title,
            margin=dict(l=90, r=20, t=60, b=40),
        )
    else:
        fig = go.Figure(go.Bar(x=l_plot, y=w_plot, hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
        fig.update_layout(
            yaxis_tickformat=".0%",
            title=title,
            margin=dict(l=40, r=20, t=60, b=80),
        )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6) Weights Heatmap (scenarios)
# ──────────────────────────────────────────────────────────────────────────────


def weights_heatmap(
    W: np.ndarray,
    asset_labels: Sequence[str],
    scenario_labels: Sequence[str] | None = None,
    *,
    title: str = "Weights by Scenario",
) -> go.Figure:
    """
    Heatmap for scenario weights. Handles empty/degenerate arrays gracefully.
    """
    if W is None:
        return _placeholder_figure(title)
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.size == 0:
        return _placeholder_figure(title, subtitle="No weights matrix to display")

    S, N = W.shape
    # Defensive label alignment
    assets = list(asset_labels)[:N] if asset_labels is not None else [f"A{i}" for i in range(N)]
    if len(assets) != N:
        assets = [f"A{i}" for i in range(N)]
    scenarios = (
        list(scenario_labels)[:S] if scenario_labels is not None else [f"S{i}" for i in range(S)]
    )
    if len(scenarios) != S:
        scenarios = [f"S{i}" for i in range(S)]

    # Replace non-finites for plotting; keep the color range informative
    W_plot = np.where(np.isfinite(W), W, 0.0)
    zmin = np.nanmin(W_plot) if np.isfinite(W_plot).any() else 0.0
    zmax = np.nanmax(W_plot) if np.isfinite(W_plot).any() else 0.0
    if not np.isfinite(zmin):
        zmin = 0.0
    if not np.isfinite(zmax):
        zmax = 0.0
    if zmax == zmin:
        # Avoid a flat colorbar; widen slightly
        zmax = zmin + (abs(zmin) + 1e-6)

    fig = go.Figure(
        data=go.Heatmap(
            z=W_plot,
            x=assets,
            y=scenarios,
            colorbar=dict(title="weight"),
            zmin=zmin,
            zmax=zmax,
            hovertemplate="scenario=%{y}<br>asset=%{x}<br>w=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, automargin=True),
        margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


ArrayLike = np.ndarray  # simple alias here; adjust if you have a typed alias elsewhere


# ──────────────────────────────────────────────────────────────────────────────
# 1) Equity Curve + Drawdown
# ──────────────────────────────────────────────────────────────────────────────
def equity_and_drawdown(dates: Sequence, equity, *, title: str = "Equity & Drawdown") -> go.Figure:
    """
    Equity curve (level) and drawdown (as a ratio, formatted as %) on a secondary y-axis.
    """
    eq = np.asarray(equity, dtype=float)
    if len(dates) != len(eq):
        raise ValueError("`dates` and `equity` must have the same length.")

    # Sanitize values
    eq = np.where(np.isfinite(eq), eq, np.nan)

    # Drawdown as v/peak - 1 (negative while underwater)
    dd = np.full_like(eq, np.nan, dtype=float)
    peak = -np.inf
    for i, v in enumerate(eq):
        if np.isfinite(v):
            if v > peak:
                peak = v
            if np.isfinite(peak) and peak > 0:
                dd[i] = (v / peak) - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=eq,
            mode="lines",
            name="Equity",
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dd,
            mode="lines",
            name="Drawdown",
            yaxis="y2",
            fill="tozeroy",
            hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Equity"),
        yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right", tickformat=".0%"),
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2) Loss Distribution with VaR/ES markers
# ──────────────────────────────────────────────────────────────────────────────
def loss_distribution(
    losses: ArrayLike,
    *,
    alphas: Sequence[float] = (0.95, 0.99),
    bins: int = 60,
    title: str = "Loss Distribution with VaR / ES",
) -> go.Figure:
    """
    Plot a histogram of losses and annotate VaR/ES for given alphas.
    `losses` should be positive for losses (if you have PnL, pass `-PnL`).
    """
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    x.sort()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=x, nbinsx=bins, name="Losses", opacity=0.75, histnorm="probability")
    )

    for a in alphas:
        q = np.quantile(x, a)
        tail = x[x >= q]
        es = float(tail.mean()) if tail.size else np.nan
        fig.add_vline(
            x=q,
            line_dash="dash",
            annotation_text=f"VaR {int(a*100)}%: {q:.2f}",
            annotation_position="top",
        )
        if np.isfinite(es):
            fig.add_vline(
                x=es,
                line_dash="dot",
                annotation_text=f"ES {int(a*100)}%: {es:.2f}",
                annotation_position="top",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Loss",
        yaxis_title="Probability",
        margin=dict(l=60, r=20, t=60, b=60),
        bargap=0.02,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3) Scree Plot (explained variance)
# ──────────────────────────────────────────────────────────────────────────────
def scree_plot(
    eigvals: ArrayLike,
    *,
    title: str = "Scree Plot (Explained Variance)",
) -> go.Figure:
    """
    Bar chart of explained variance per eigenvalue with a cumulative line.
    Input eigenvalues do not need to be normalized.
    """
    lam = np.asarray(eigvals, dtype=float)
    lam = lam[np.isfinite(lam)]
    lam = np.clip(lam, 0.0, None)
    if lam.size == 0:
        lam = np.array([0.0])

    lam_sorted = np.sort(lam)[::-1]
    total = float(lam_sorted.sum()) if lam_sorted.size else 1.0
    exp = lam_sorted / max(total, 1e-16)
    cum = np.cumsum(exp)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(1, len(exp) + 1), y=exp, name="Explained"))
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(cum) + 1),
            y=cum,
            mode="lines+markers",
            name="Cumulative",
            yaxis="y2",
            hovertemplate="k=%{x}<br>cum=%{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Component (k)",
        yaxis=dict(title="Explained (share)", tickformat=".0%"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right", tickformat=".0%"),
        barmode="group",
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4) Correlation Network Graph (simple circular layout)
# ──────────────────────────────────────────────────────────────────────────────
def network_corr_graph(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    threshold: float = 0.3,  # draw edges with |ρ| >= threshold
    title: str = "Correlation Network Graph",
) -> go.Figure:
    """
    Correlation graph (without networkx): circular node layout, edges scaled by |ρ|.
    """
    # Expect helpers _to_numpy_matrix and _safe_corr_from_cov to exist in your module
    M = _to_numpy_matrix(Sigma_or_Corr)
    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)
    n = Corr.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)

    pos_x, pos_y, pos_w = [], [], []
    neg_x, neg_y, neg_w = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            rho = float(Corr[i, j])
            if np.isnan(rho) or abs(rho) < threshold:
                continue
            segx = [xs[i], xs[j], None]
            segy = [ys[i], ys[j], None]
            width = 1.0 + 3.0 * abs(rho)
            if rho >= 0:
                pos_x += segx
                pos_y += segy
                pos_w.append(width)
            else:
                neg_x += segx
                neg_y += segy
                neg_w.append(width)

    fig = go.Figure()
    if pos_x:
        fig.add_trace(
            go.Scatter(
                x=pos_x,
                y=pos_y,
                mode="lines",
                name="ρ ≥ 0",
                line=dict(width=np.mean(pos_w) if pos_w else 1.5, color="rgba(0,120,255,0.5)"),
                hoverinfo="none",
            )
        )
    if neg_x:
        fig.add_trace(
            go.Scatter(
                x=neg_x,
                y=neg_y,
                mode="lines",
                name="ρ < 0",
                line=dict(width=np.mean(neg_w) if neg_w else 1.5, color="rgba(255,80,80,0.5)"),
                hoverinfo="none",
            )
        )
    deg = (np.abs(Corr) >= threshold).sum(axis=0) - 1
    node_size = 10 + 3 * np.clip(deg, 0, None)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=list(labels),
            textposition="top center",
            marker=dict(size=node_size, line=dict(width=1, color="#333")),
            hovertemplate="%{text}<extra></extra>",
            name="assets",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5) Risk Contributions bar
# ──────────────────────────────────────────────────────────────────────────────
def risk_contributions_bar(
    rc: ArrayLike,
    labels: Sequence[str],
    *,
    sort: bool = True,
    topn: int | None = None,
    title: str = "Risk Contributions",
) -> go.Figure:
    """
    Horizontal bars for risk contributions (absolute or %).
    """
    v = np.asarray(rc, dtype=float)
    if v.ndim != 1 or len(v) != len(labels):
        raise ValueError("`rc` must be 1D and aligned with `labels`.")
    idx = np.arange(len(v))
    if sort:
        idx = np.argsort(v)
    if topn is not None:
        idx = idx[-topn:]
    v_plot = v[idx]
    l_plot = [labels[i] for i in idx]
    fig = go.Figure(
        go.Bar(x=v_plot, y=l_plot, orientation="h", hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    )
    fig.update_layout(
        title=title, margin=dict(l=80, r=20, t=60, b=40), xaxis_title="Contribution", yaxis_title=""
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6) Rolling lines (multi-series)
# ──────────────────────────────────────────────────────────────────────────────
def rolling_lines(
    dates: Sequence,
    series_dict: dict[str, ArrayLike],
    *,
    title: str = "Rolling Metrics",
) -> go.Figure:
    """
    One line per series in `series_dict = {"label": aligned_array, ...}`.
    """
    fig = go.Figure()
    for name, arr in series_dict.items():
        y = np.asarray(arr, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y,
                mode="lines",
                name=name,
                hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7) Correlation Heatmap (WebGL)
# ──────────────────────────────────────────────────────────────────────────────
def corr_heatmap_gl(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    order: HeatmapOrder = None,  # keep your real type if available
    zlim: tuple[float, float] = (-1.0, 1.0),
    title: str = "Correlation Heatmap (WebGL)",
) -> go.Figure:
    """
    Same as a standard correlation heatmap but using Heatmapgl (faster for N > ~200).
    Expects helper functions _to_numpy_matrix / _safe_corr_from_cov / _hierarchical_order / _apply_order.
    """
    if order is None:
        order = HeatmapOrder()  # relies on your existing class

    M = _to_numpy_matrix(Sigma_or_Corr)
    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)
    n = Corr.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]

    if getattr(order, "clustered", False) and n >= 3:
        ord_idx = _hierarchical_order(Corr, order)
        Corr_ord = _apply_order(Corr, ord_idx)
        labels_ord = [labels[i] for i in ord_idx]
    else:
        Corr_ord, labels_ord = Corr, list(labels)

    fig = go.Figure(
        data=go.Heatmapgl(
            z=Corr_ord,
            x=labels_ord,
            y=labels_ord,
            zmin=zlim[0],
            zmax=zlim[1],
            colorbar=dict(title="ρ"),
            hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8) Weights path vs γ
# ──────────────────────────────────────────────────────────────────────────────
def weights_path_gammas(
    Ws: np.ndarray,
    gammas: Sequence[float],
    labels: Sequence[str],
    *,
    topn: int = 20,
    title: str = "Weights path vs γ (top names)",
) -> go.Figure:
    """
    Each row of `Ws` corresponds to a γ. Plot trajectories of weights against log10(γ)
    for the top `topn` names by peak weight across the sweep.
    """
    if Ws.ndim != 2:
        raise ValueError("`Ws` must be 2D (n_gamma x N).")
    nG, N = Ws.shape
    g = np.asarray(gammas, dtype=float)
    x = np.log10(g)
    peak = np.max(Ws, axis=0)
    idx = np.argsort(peak)[::-1][: min(topn, N)]
    fig = go.Figure()
    for i in idx:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=Ws[:, i],
                mode="lines",
                name=labels[i],
                hovertemplate="log10 γ=%{x:.2f}<br>w=%{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, xaxis_title="log10(γ)", yaxis_title="Weight", yaxis_tickformat=".0%"
    )
    return fig


def turnover_vs_gamma(
    Ws: np.ndarray, w_ref: np.ndarray, gammas: Sequence[float], *, title: str = "Turnover vs γ"
) -> go.Figure:
    """
    L1/L2 turnover vs γ using `Ws` (n_gamma x N) and a reference portfolio `w_ref`.
    """
    L1 = np.sum(np.abs(Ws - w_ref[None, :]), axis=1)
    L2 = np.sqrt(np.sum((Ws - w_ref[None, :]) ** 2, axis=1))
    x = np.log10(np.asarray(gammas, dtype=float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=L1, mode="lines+markers", name="L1"))
    fig.add_trace(go.Scatter(x=x, y=L2, mode="lines+markers", name="L2"))
    fig.update_layout(title=title, xaxis_title="log10(γ)", yaxis_title="Turnover")
    return fig


def te_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_bench: np.ndarray,
    Ws: np.ndarray,
    *,
    title: str = "Tracking-Error Frontier",
    annualize: bool = False,
    periods_per_year: int = 252,
) -> go.Figure:
    """
    TE frontier: plot per-period (or annualized) expected return vs tracking error for a set of portfolios.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    w_bench = np.asarray(w_bench, dtype=float).reshape(-1)
    Ws = np.asarray(Ws, dtype=float)

    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("`Sigma` must be (n, n).")
    if w_bench.shape[0] != n or Ws.shape[1] != n:
        raise ValueError("Weights dimensions must match `mu/Sigma`.")

    dW = Ws - w_bench[None, :]
    mu_p = Ws @ mu
    te = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dW, Sigma, dW), 0.0))

    if annualize:
        mu_p = (1 + mu_p) ** periods_per_year - 1.0
        te = te * np.sqrt(periods_per_year)

    fig = go.Figure(
        go.Scatter(
            x=te,
            y=mu_p,
            mode="markers+lines",
            name="Portfolios",
            hovertemplate="TE: %{x:.2%}<br>μ: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=("Tracking Error (annualized)" if annualize else "Tracking Error (per period)"),
        yaxis_title=(
            "Expected Return (annualized)" if annualize else "Expected Return (per period)"
        ),
    )
    fig.update_xaxes(tickformat=".2%")
    fig.update_yaxes(tickformat=".2%")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9) Backtest & Attribution plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_equity(dates: list, equity: np.ndarray, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure(go.Scatter(x=dates, y=equity, mode="lines", name="Equity"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="NAV", template="plotly_white")
    return fig


def plot_drawdown(dates: list, equity: np.ndarray, title: str = "Drawdown") -> go.Figure:
    cummax = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(cummax, 1e-12)) - 1.0
    fig = go.Figure(go.Scatter(x=dates, y=dd, mode="lines", name="Drawdown"))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown", template="plotly_white"
    )
    return fig


def plot_weights_heatmap(
    dates: list, tickers: list[str], W: np.ndarray, title="Weights"
) -> go.Figure:
    avg_w = np.asarray(W, dtype=float).mean(axis=0)
    order = np.argsort(-avg_w)
    tickers_ord = [tickers[i] for i in order]
    W_ord = np.asarray(W, dtype=float)[:, order]
    fig = go.Figure(
        data=go.Heatmap(
            z=W_ord.T,
            x=dates,
            y=tickers_ord,
            coloraxis="coloraxis",
            hovertemplate="Ticker: %{y}<br>Date: %{x}<br>Weight: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, coloraxis_colorscale="Blues", template="plotly_white")
    return fig


def plot_turnover(dates_or_df, turnover=None, title: str = "Turnover") -> go.Figure:
    """
    Accepts either:
      - (dates, turnover_array)
      - A DataFrame with columns ['date', 'turnover'] (Polars or Pandas)
    Robust to Polars vs Pandas.
    """
    if turnover is None:
        df = dates_or_df
        # Polars
        if isinstance(df, pl.DataFrame) and {"date", "turnover"}.issubset(set(df.columns)):
            x = df.get_column("date").to_list()
            y = df.get_column("turnover").to_numpy()
        # Pandas
        elif hasattr(df, "columns") and {"date", "turnover"}.issubset(set(df.columns)):
            x = df["date"].tolist()
            y = np.asarray(df["turnover"].values, dtype=float)
        else:
            raise ValueError("If you pass a DataFrame it must have columns ['date','turnover'].")
    else:
        x = list(dates_or_df)
        y = np.asarray(turnover, dtype=float)
        if len(x) != len(y):
            raise ValueError("`dates` and `turnover` must have the same length.")

    y = np.where(np.isfinite(y), y, np.nan)
    y = np.clip(y, 0.0, None)
    fig = go.Figure(
        go.Scatter(
            x=x, y=y, mode="lines", name="Turnover", hovertemplate="%{x}<br>%{y:.1%}<extra></extra>"
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Turnover (%)",
        yaxis=dict(tickformat=".0%"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig


def plot_tracking_error(
    dates: list, te: np.ndarray, title="Tracking Error (daily proxy)"
) -> go.Figure:
    te = np.asarray(te, dtype=float)
    if len(dates) != len(te):
        raise ValueError("`dates` and `te` must have the same length.")
    te = np.where(np.isfinite(te), te, np.nan)
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=te,
            mode="lines",
            name="TE",
            hovertemplate="%{x}<br>TE: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="TE (%)",
        yaxis=dict(tickformat=".2%"),
        template="plotly_white",
    )
    return fig


def plot_top_contributors(df_top: pl.DataFrame, title="Top Contributors") -> go.Figure:
    pdf = df_top.select(["ticker", "contrib_total"]).to_pandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
    pdf = pdf.sort_values("contrib_total", ascending=False)
    fig = go.Figure(go.Bar(x=pdf["ticker"], y=pdf["contrib_total"]))
    fig.update_layout(
        title=title, xaxis_title="Ticker", yaxis_title="Total Contribution", template="plotly_white"
    )
    return fig


def plot_group_contrib(df_group_total: pl.DataFrame, title="Group Contributions") -> go.Figure:
    pdf = df_group_total.select(["group", "contrib_total"]).to_pandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
    pdf = pdf.sort_values("contrib_total", ascending=False)
    fig = go.Figure(go.Bar(x=pdf["group"], y=pdf["contrib_total"]))
    fig.update_layout(
        title=title, xaxis_title="Group", yaxis_title="Total Contribution", template="plotly_white"
    )
    return fig


def plot_group_contrib_area(df: pl.DataFrame, title="Group Contributions Over Time") -> go.Figure:
    req = {"date", "group", "contrib"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Missing columns: {req}")
    pdf = (
        df.to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "group", "contrib"])
        .sort_values("date")
    )
    fig = px.area(
        pdf, x="date", y="contrib", color="group", title=title, labels={"contrib": "Contribution"}
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Contribution",
        yaxis_tickformat=".2%",
        legend_title="Group",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def plot_brinson_cumulative(
    df_brinson: pl.DataFrame, title="Brinson-Fachler Attribution"
) -> go.Figure:
    cols = {"date", "alloc", "select", "interact", "total"}
    if not cols.issubset(set(df_brinson.columns)):
        raise ValueError(f"Missing columns in df_brinson. Expected: {cols}")
    pdf = df_brinson.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure()
    for col in ["alloc", "select", "interact", "total"]:
        fig.add_trace(
            go.Scatter(
                x=pdf["date"],
                y=pdf[col],
                mode="lines",
                name=col.capitalize(),
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Attribution (%)",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
    )
    return fig


# =========================
# Advanced attribution plots (non-breaking additions)
# =========================


def plot_asset_contrib_heatmap_adv(
    df_asset_daily: pl.DataFrame,
    *,
    topk_by_abs_total: int = 30,
    title: str = "Asset Daily Contribution Heatmap",
) -> go.Figure:

    req = {"date", "ticker", "contrib"}
    if not req.issubset(set(df_asset_daily.columns)):
        raise ValueError("df_asset_daily must contain 'date','ticker','contrib'")

    totals = (
        df_asset_daily.group_by("ticker")
        .agg(pl.col("contrib").sum().abs().alias("abs_total"))
        .sort("abs_total", descending=True)
        .head(topk_by_abs_total)
    )
    keep = set(totals["ticker"].to_list())
    df_top = df_asset_daily.filter(pl.col("ticker").is_in(keep))

    wide = (
        df_top.select(["date", "ticker", "contrib"])
        .pivot(values="contrib", index="date", columns="ticker")
        .sort("date")
    )
    pdf = wide.to_pandas().set_index("date")
    fig = px.imshow(
        pdf.T,
        aspect="auto",
        origin="lower",
        title=title,
        labels=dict(x="Date", y="Ticker", color="Contribution"),
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_cumulative_contrib_curves_adv(
    df_asset_daily: pl.DataFrame,
    *,
    topk_by_abs_total: int = 10,
    title: str = "Cumulative Contribution (Top-k assets)",
) -> go.Figure:

    req = {"date", "ticker", "contrib"}
    if not req.issubset(set(df_asset_daily.columns)):
        raise ValueError("df_asset_daily must contain 'date','ticker','contrib'")

    totals = (
        df_asset_daily.group_by("ticker")
        .agg(pl.col("contrib").sum().abs().alias("abs_total"))
        .sort("abs_total", descending=True)
        .head(topk_by_abs_total)
    )
    keep = set(totals["ticker"].to_list())
    df_top = (
        df_asset_daily.filter(pl.col("ticker").is_in(keep))
        .sort(["ticker", "date"])
        .with_columns(pl.col("contrib").cum_sum().over("ticker").alias("cum_contrib"))
    )
    pdf = df_top.select(["date", "ticker", "cum_contrib"]).to_pandas()
    fig = px.line(
        pdf,
        x="date",
        y="cum_contrib",
        color="ticker",
        title=title,
        labels={"cum_contrib": "Cumulative"},
    )
    fig.update_layout(template="plotly_white")
    fig.update_yaxes(tickformat=".2%")
    return fig


def plot_brinson_components_bar_adv(
    df_brinson: pl.DataFrame,
    *,
    title: str = "Brinson Components (final snapshot)",
    as_percent: bool = True,
) -> go.Figure:

    needed = {"date", "alloc", "select", "interact", "total"}
    if not needed.issubset(set(df_brinson.columns)):
        raise ValueError("df_brinson must contain 'date','alloc','select','interact','total'")
    last = (
        df_brinson.sort("date").tail(1).select(["alloc", "select", "interact", "total"]).to_pandas()
    )
    last = last.replace([np.inf, -np.inf], np.nan).dropna()
    vals = last.iloc[0].values if len(last) else [0, 0, 0, 0]
    fig = go.Figure(go.Bar(x=["Allocation", "Selection", "Interaction", "Total"], y=vals))
    fig.update_layout(
        title=title, template="plotly_white", xaxis_title="", yaxis_title="Attribution"
    )
    if as_percent:
        fig.update_yaxes(tickformat=".2%")
    return fig


def plot_brinson_components_area_adv(
    df_brinson: pl.DataFrame,
    *,
    title: str = "Brinson Components Over Time",
    as_percent: bool = True,
) -> go.Figure:

    need = {"date", "alloc", "select", "interact"}
    if not need.issubset(set(df_brinson.columns)):
        raise ValueError("df_brinson must contain 'date','alloc','select','interact'")
    pdf = (
        df_brinson.select(["date", "alloc", "select", "interact"])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("date")
    )
    pdf_m = pdf.melt(id_vars="date", var_name="component", value_name="value")
    fig = px.area(
        pdf_m, x="date", y="value", color="component", title=title, labels={"value": "Attribution"}
    )
    fig.update_layout(template="plotly_white", legend_title="Component")
    if as_percent:
        fig.update_yaxes(tickformat=".0%")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Extra attribution plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_contrib_heatmap_daily(
    df_asset_daily: pl.DataFrame,
    *,
    title: str = "Daily Contribution Heatmap",
    tickers_order: list[str] | None = None,
) -> go.Figure:
    """
    Heatmap of daily contributions by ticker.
    Expects columns ['date','ticker','contrib'] in df_asset_daily.
    Useful to spot regime shifts and concentration of contribution.
    """
    req = {"date", "ticker", "contrib"}
    if not req.issubset(set(df_asset_daily.columns)):
        raise ValueError("df_asset_daily must include 'date','ticker','contrib'.")

    pdf = (
        df_asset_daily.select(["date", "ticker", "contrib"])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "ticker", "contrib"])
    )
    # Optional user order (e.g., sort by total contribution elsewhere)
    if tickers_order:
        cat = pd.Categorical(pdf["ticker"], categories=tickers_order, ordered=True)
        pdf = pdf.assign(ticker=cat).sort_values(["ticker", "date"])
    pivot = pdf.pivot_table(
        index="ticker", columns="date", values="contrib", aggfunc="sum", fill_value=0.0
    )
    # Ensure increasing date order on columns
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            colorbar=dict(title="Contribution"),
            hovertemplate="Date=%{x}<br>Ticker=%{y}<br>Contrib=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Ticker",
        template="plotly_white",
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def plot_top_contributors_waterfall(
    df_cum: pl.DataFrame,
    *,
    k: int = 10,
    orientation: str = "v",
    title: str = "Cumulative Contribution (Waterfall)",
) -> go.Figure:

    req = {"ticker", "contrib_total"}
    if not req.issubset(set(df_cum.columns)):
        raise ValueError("df_cum must include 'ticker','contrib_total'.")

    pdf = (
        df_cum.select(["ticker", "contrib_total"])
        .sort("contrib_total", descending=True)
        .head(k)
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    labels = pdf["ticker"].tolist()
    vals = pdf["contrib_total"].values.astype(float)

    # Positive/negative steps; no running total bar at the end (keeps it compact)
    measure = ["relative"] * len(labels)

    if orientation not in ("v", "h"):
        orientation = "v"

    fig = go.Figure(
        go.Waterfall(
            orientation="v" if orientation == "v" else "h",
            measure=measure,
            x=labels if orientation == "v" else None,
            y=vals if orientation == "v" else None,
            text=[f"{v:.4f}" for v in vals],
            textposition="auto",
            connector={"line": {"width": 1}},
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=("Ticker" if orientation == "v" else "Contribution"),
        yaxis_title=("Contribution" if orientation == "v" else "Ticker"),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def plot_group_share_area_from_share(
    df_share: pl.DataFrame,
    *,
    title: str = "Group Share of Total Contribution",
) -> go.Figure:

    req = {"date", "group", "share"}
    if not req.issubset(set(df_share.columns)):
        raise ValueError("df_share must include 'date','group','share'.")

    pdf = (
        df_share.select(["date", "group", "share"])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "group", "share"])
        .sort_values("date")
    )
    fig = px.area(
        pdf,
        x="date",
        y="share",
        color="group",
        title=title,
        labels={"share": "Share of total"},
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Share",
        yaxis_tickformat=".0%",
        legend_title="Group",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def plot_brinson_by_group_area(
    df_brinson_g: pl.DataFrame,
    group_labels: list[str],
    *,
    component: str = "total",  # "alloc" | "select" | "interact" | "total"
    title: str | None = None,
) -> go.Figure:

    req = {"date", "group_id", "alloc", "select", "interact", "total"}
    if not req.issubset(set(df_brinson_g.columns)):
        raise ValueError(
            "df_brinson_g must come from brinson_fachler_timeseries(..., by_group=True)."
        )

    if component not in {"alloc", "select", "interact", "total"}:
        raise ValueError("component must be one of {'alloc','select','interact','total'}.")

    pdf = (
        df_brinson_g.select(["date", "group_id", component])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    # Attach readable labels
    gid = pdf["group_id"].astype(int).values
    labels = [group_labels[i] if 0 <= i < len(group_labels) else f"G{i}" for i in gid]
    pdf = pdf.assign(group=labels).drop(columns=["group_id"]).sort_values("date")

    fig = px.area(
        pdf,
        x="date",
        y=component,
        color="group",
        title=title or f"Brinson by Group – {component.capitalize()} (Cumulative)",
        labels={component: component.capitalize()},
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=f"{component.capitalize()}",
        yaxis_tickformat=".2%",
        legend_title="Group",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Brinson – extras (safe add-ons)
# ──────────────────────────────────────────────────────────────────────────────
def plot_brinson_cumulative_components(
    df_brinson: pl.DataFrame,
    title: str = "Cumulative Brinson–Fachler Attribution",
) -> go.Figure:
    """
    Lines of cumulative allocation/select/interaction/total over time.
    Expects df_brinson with: ['date','alloc','select','interact','total'].
    """
    need = {"date", "alloc", "select", "interact", "total"}
    if not need.issubset(set(df_brinson.columns)):
        raise ValueError(f"Missing columns in df_brinson: {need}")

    df_cum = df_brinson.with_columns(
        [
            pl.col("alloc").cum_sum().alias("alloc_cum"),
            pl.col("select").cum_sum().alias("select_cum"),
            pl.col("interact").cum_sum().alias("interact_cum"),
            pl.col("total").cum_sum().alias("total_cum"),
        ]
    )
    pdf = df_cum.to_pandas().replace([np.inf, -np.inf], np.nan).dropna(subset=["date"])

    fig = go.Figure()
    for col in ["alloc_cum", "select_cum", "interact_cum", "total_cum"]:
        fig.add_trace(
            go.Scatter(
                x=pdf["date"],
                y=pdf[col],
                mode="lines",
                name=col.replace("_cum", "").capitalize(),
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Attribution (%)",
        yaxis_tickformat=".1%",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_brinson_final_bar(
    df_brinson: pl.DataFrame,
    title: str = "Final Attribution Breakdown",
) -> go.Figure:
    """
    Single bar chart of the last cumulative values for each component.
    Expects df_brinson with: ['date','alloc','select','interact','total'].
    """
    need = {"alloc", "select", "interact", "total"}
    if not need.issubset(set(df_brinson.columns)):
        raise ValueError(f"Missing columns in df_brinson: {need}")

    # take last row; if empty, make zeros
    if df_brinson.height == 0:
        comps = ["alloc", "select", "interact", "total"]
        vals = [0.0, 0.0, 0.0, 0.0]
    else:
        last = df_brinson.tail(1).select(list(need)).to_pandas().iloc[0]
        comps = ["alloc", "select", "interact", "total"]
        vals = [float(last[c]) for c in comps]

    fig = go.Figure(
        go.Bar(
            x=[c.capitalize() for c in comps],
            y=vals,
            text=[f"{v:.2%}" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Attribution (%)",
        yaxis_tickformat=".1%",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Brinson – extras (safe add-ons)
# ──────────────────────────────────────────────────────────────────────────────


def plot_scenario_equity_panel(results: list, title: str = "Scenario Equity Panel") -> go.Figure:

    fig = go.Figure()
    for r in results:
        dates = r.bt.get("dates", [])
        eq = r.bt.get("equity", [])
        fig.add_trace(go.Scatter(x=dates, y=eq, mode="lines", name=r.name))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Equity", template="plotly_white"
    )
    return fig


def plot_scenario_metrics_bars(
    df_metrics: pl.DataFrame, title: str = "Scenario Metrics"
) -> go.Figure:

    pdf = df_metrics.to_pandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.bar(
        pdf.melt(id_vars="scenario"),
        x="scenario",
        y="value",
        color="variable",
        barmode="group",
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-20)
    return fig


def plot_weights_delta_heatmap(
    tickers: list[str],
    W_base: np.ndarray,
    W_scn: np.ndarray,
    title: str = "Δ Weights (Scenario - Baseline)",
) -> go.Figure:

    base = (W_base[-1] if W_base.ndim == 2 else W_base).astype(float)
    scn = (W_scn[-1] if W_scn.ndim == 2 else W_scn).astype(float)
    d = scn - base
    fig = go.Figure(
        data=go.Heatmap(
            z=d.reshape(1, -1),
            x=tickers,
            y=["Δw"],
            colorbar=dict(title="Δw"),
        )
    )
    fig.update_layout(title=title, template="plotly_white", xaxis_nticks=min(40, len(tickers)))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────


def plot_tornado_sensitivity(
    df_sens: DataFrameLike,
    metric_label: str = "CAGR",
    down_label: str = "Down",
    up_label: str = "Up",
    sort_by: str = "min_delta",
) -> go.Figure:

    pdf = _to_pandas(df_sens).copy()

    # Normalize naming
    if "asset" not in pdf.columns and "name" in pdf.columns:
        pdf.rename(columns={"name": "asset"}, inplace=True)

    required = {"asset", "base", "down", "up"}
    missing = required - set(pdf.columns)
    if missing:
        raise ValueError(f"plot_tornado_sensitivity: missing columns {missing}")

    # Type coercion and NaN handling
    for c in ["base", "down", "up"]:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0.0)

    # Compute deltas
    pdf["down_delta"] = pdf["down"] - pdf["base"]
    pdf["up_delta"] = pdf["up"] - pdf["base"]

    # Sort by most adverse delta
    pdf["min_delta"] = np.minimum(pdf["down_delta"].values, pdf["up_delta"].values)
    pdf.sort_values(by=sort_by, inplace=True)
    pdf.reset_index(drop=True, inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=pdf["asset"], x=pdf["down_delta"], orientation="h", name=down_label))
    fig.add_trace(go.Bar(y=pdf["asset"], x=pdf["up_delta"], orientation="h", name=up_label))
    fig.update_layout(
        title=f"Tornado Sensitivity — Metric: {metric_label}",
        barmode="overlay",
        xaxis_title=f"Δ {metric_label} vs Base",
        yaxis_title="Asset",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_equity_compare(
    dates: Sequence,
    equity_a: Sequence[float],
    equity_b: Sequence[float],
    name_a: str = "Baseline",
    name_b: str = "Scenario",
    title: str | None = None,
) -> go.Figure:

    title = title or f"Equity Comparison — {name_a} vs {name_b}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(dates), y=list(equity_a), mode="lines", name=name_a))
    fig.add_trace(go.Scatter(x=list(dates), y=list(equity_b), mode="lines", name=name_b))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_drawdown_compare(
    dates: Sequence,
    equity_a: Sequence[float],
    equity_b: Sequence[float],
    name_a: str = "Baseline",
    name_b: str = "Scenario",
    title: str | None = None,
) -> go.Figure:
    """Compare drawdowns (%) between two equity curves."""

    def _dd_curve(eq: Sequence[float]) -> np.ndarray:
        eq = np.asarray(eq, dtype=float)
        if eq.size == 0:
            return np.array([], dtype=float)
        cummax = np.maximum.accumulate(eq)
        dd = (eq / np.maximum(cummax, 1e-12)) - 1.0
        return dd

    dd_a = _dd_curve(equity_a) * 100.0
    dd_b = _dd_curve(equity_b) * 100.0
    title = title or f"Drawdown Comparison — {name_a} vs {name_b}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(dates), y=list(dd_a), mode="lines", name=name_a))
    fig.add_trace(go.Scatter(x=list(dates), y=list(dd_b), mode="lines", name=name_b))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_metric_delta_bars(
    df_metrics: DataFrameLike,
    baseline_value: float,
    metric_col: str = "CAGR",
    scenario_name_col: str = "Scenario",
    title: str | None = None,
) -> go.Figure:

    pdf = _to_pandas(df_metrics).copy()
    for c in [metric_col]:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    if scenario_name_col not in pdf.columns or metric_col not in pdf.columns:
        raise ValueError("plot_metric_delta_bars: required columns not found.")

    pdf["delta"] = pdf[metric_col] - float(baseline_value)
    pdf.sort_values(by="delta", inplace=True)
    title = title or f"Δ{metric_col} vs Baseline"

    fig = go.Figure()
    fig.add_trace(go.Bar(y=pdf[scenario_name_col], x=pdf["delta"], orientation="h"))
    fig.update_layout(
        title=title,
        xaxis_title=f"Δ{metric_col}",
        yaxis_title="Scenario",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_weights_compare_heatmap(
    dates: Sequence,
    tickers: Sequence[str],
    weights_a: np.ndarray | Sequence[Sequence[float]],
    weights_b: np.ndarray | Sequence[Sequence[float]],
    name_a: str = "Baseline",
    name_b: str = "Scenario",
    mode: str = "delta",  # {"delta", "absolute"}
    title: str | None = None,
    zmax_abs: float | None = None,
) -> go.Figure:

    # Coerce to numpy arrays
    A = np.asarray(weights_a, dtype=float)
    B = np.asarray(weights_b, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("weights_a and weights_b must be 2D arrays (T, N).")
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: weights_a {A.shape} vs weights_b {B.shape}.")
    T, N = A.shape
    if len(dates) != T:
        raise ValueError(f"'dates' length {len(dates)} must match T={T}.")
    if len(tickers) != N:
        raise ValueError(f"'tickers' length {len(tickers)} must match N={N}.")

    # Clean NaNs/Infs
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    if mode not in {"delta", "absolute"}:
        raise ValueError("mode must be either 'delta' or 'absolute'.")

    if mode == "delta":
        Z = B - A
        # Symmetric color range around zero
        max_abs = np.max(np.abs(Z)) if Z.size else 1.0
        if zmax_abs is not None and zmax_abs > 0:
            vmax = float(zmax_abs)
        else:
            vmax = float(max_abs) if max_abs > 0 else 1e-6
        vmin = -vmax
        colorscale = "RdBu"  # diverging
        zmid = 0.0
        default_title = f"Weights Heatmap — {name_b} vs {name_a} (Δ)"
    else:
        Z = B
        vmax = float(np.max(Z)) if Z.size else 1.0
        vmin = 0.0
        colorscale = "Blues"  # sequential
        zmid = None
        default_title = f"Weights Heatmap — {name_b} (absolute weights)"

    title = title or default_title

    # Convert dates to strings for cleaner axis labels (Plotly heatmaps like strings)
    x_labels = list(tickers)
    y_labels = [str(d) for d in dates]

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            zmid=zmid,
            colorbar=dict(title=("Δw" if mode == "delta" else "w")),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Assets",
        yaxis_title="Date",
        template="plotly_white",
        margin=dict(t=60, r=10, b=40, l=60),
    )
    return fig
