"""
Plot utilities for GammaEdge: robust, typed, and lint-clean.
"""

from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Union, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from numpy.typing import NDArray

# SciPy soft dependency (optional)
try:
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform

    _SCIPY_OK = True
except Exception:  # pragma: no cover
    sch = SimpleNamespace(linkage=None, optimal_leaf_ordering=None, leaves_list=None)
    squareform = None
    _SCIPY_OK = False

pd.set_option("future.no_silent_downcasting", True)

# ============================================================================
# Tipos
# ============================================================================
DataFrameLike = Union[pd.DataFrame, pl.DataFrame]
ArrayLike = Union[NDArray[np.float64], Sequence[float]]

# SciPy soft dependency (optional)
try:
    import scipy.cluster.hierarchy as sch  # noqa: F401
    from scipy.spatial.distance import squareform  # noqa: F401
except Exception:  # pragma: no cover
    sch = None
    squareform = None

# ============================================================================
# Plotly rendering helpers (centralized config, no deprecations)
# ============================================================================

DEFAULT_PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "filename": "gammaedge_plot"},
}


def apply_fig_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def show_plot(
    fig: go.Figure,
    *,
    config: dict[str, Any] | None = None,
    st_obj: Any | None = None,
    key: str | None = None,
) -> None:
    fig = apply_fig_defaults(fig)
    cfg = dict(DEFAULT_PLOTLY_CONFIG)
    if config:
        cfg.update(config)

    if st_obj is None:
        import streamlit as st

        target = st
    else:
        target = st_obj

    target.plotly_chart(fig, config=cfg, key=key)


def fig_to_html(
    fig: go.Figure,
    *,
    include_plotlyjs: str = "cdn",
    config: dict[str, Any] | None = None,
    full_html: bool = False,
) -> str:
    fig = apply_fig_defaults(fig)
    cfg = dict(DEFAULT_PLOTLY_CONFIG)
    if config:
        cfg.update(config)
    html = cast(
        str,
        fig.to_html(
            full_html=full_html,
            include_plotlyjs=include_plotlyjs,
            config=cfg,
        ),
    )
    return html


# ============================================================================
# Helpers numéricos / utilidades
# ============================================================================


@dataclass(frozen=True, slots=True)
class HeatmapOrder:
    clustered: bool = True
    method: Literal["single", "complete", "average", "ward"] = "average"
    optimal: bool = True  # optimal leaf ordering


def _to_numpy_matrix(x: np.ndarray | pl.DataFrame) -> NDArray[np.float64]:
    """Convert np.ndarray or Polars DataFrame to a 2D np.ndarray[float64]."""
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, pl.DataFrame):
        arr = x.to_numpy()
        return np.asarray(arr, dtype=np.float64)
    raise TypeError("Expected np.ndarray or pl.DataFrame.")


def _safe_corr_from_cov(Sigma: np.ndarray, eps: float = 1e-16) -> NDArray[np.float64]:
    """Convierte Σ a ρ de forma segura y simetriza."""
    S = np.asarray(Sigma, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Sigma must be a square matrix")
    d = np.sqrt(np.clip(np.diag(S), 0.0, None))
    d[d < eps] = eps
    R = (S / d[:, None]) / d[None, :]
    R = np.clip(R, -1.0, 1.0)
    return np.asarray(0.5 * (R + R.T), dtype=float)


def _hierarchical_order(Corr: np.ndarray, order_cfg: HeatmapOrder) -> NDArray[np.int64]:
    C = np.asarray(Corr, dtype=float)
    n = C.shape[0]
    if not _SCIPY_OK or n < 3 or sch is None or squareform is None:
        return np.arange(n, dtype=np.int64)  # <- int64 explícito

    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - C)))
    dvec = squareform(dist, checks=False)
    Z = sch.linkage(dvec, method=order_cfg.method)
    if order_cfg.optimal:
        Z = sch.optimal_leaf_ordering(Z, dvec)
    ord_idx = np.asarray(sch.leaves_list(Z), dtype=np.int64)
    return ord_idx


def _apply_order(mat: np.ndarray, order: np.ndarray) -> NDArray[np.float64]:
    M = np.asarray(mat, dtype=np.float64)
    idx = np.asarray(order, dtype=np.int64)
    return np.asarray(M[np.ix_(idx, idx)], dtype=np.float64)


def _to_pandas(df: DataFrameLike) -> pd.DataFrame:
    """Convierte Polars → Pandas si es necesario."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError("Unsupported DataFrame type. Expected pandas or polars.")


def _pivot_compat(df: pl.DataFrame, *, values: str, index: str, columns: str) -> pl.DataFrame:
    pivot_fn = getattr(df, "pivot", None)
    if pivot_fn is None:
        raise AttributeError("DataFrame has no method 'pivot'")

    p: Any = pivot_fn  # <- evitar chequeo de signatura
    try:
        out = p(index=index, columns=columns, values=values)
    except TypeError:
        out = p(index=index, values=values, on=columns)
    return pl.DataFrame(out)


def _placeholder_figure(title: str, subtitle: str = "No data available") -> go.Figure:
    """Figura placeholder defensiva."""
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


def _to_1d_float(x: Any) -> NDArray[np.float64]:
    """Convierte a vector 1D float, seguro para mypy y ruff."""
    if x is None:
        return np.asarray([], dtype=float)
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.ravel()
    return np.asarray(arr, dtype=float)


# ============================================================================
# 1) Correlation Heatmap (clustered)
# ============================================================================


def corr_heatmap(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    order: HeatmapOrder = HeatmapOrder(),
    zlim: tuple[float, float] = (-1.0, 1.0),
    title: str = "Correlation Heatmap (clustered)",
) -> go.Figure:
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 2) Correlation Dendrogram
# ============================================================================


def corr_dendrogram(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    method: Literal["single", "complete", "average", "ward"] = "average",
    title: str = "Correlation Dendrogram",
) -> go.Figure:
    M = _to_numpy_matrix(Sigma_or_Corr).astype(float, copy=False)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("corr_dendrogram: input must be a square matrix (NxN).")
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)
    Corr = np.clip(Corr, -1.0, 1.0)
    n = Corr.shape[0]

    if labels is None:
        labels = [f"A{i}" for i in range(n)]
    else:
        labels = list(labels)
        if len(labels) != n:
            labels = (labels + [f"A{i}" for i in range(len(labels), n)])[:n]

    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - Corr)))

    # ... (tras construir 'dist' y antes de crear 'fig') ...
    if not _SCIPY_OK or sch is None or squareform is None:
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

    dvec = squareform(dist, checks=False)
    Z = sch.linkage(dvec, method=method)
    dn = sch.dendrogram(Z, labels=labels, no_plot=True)

    icoord = cast(np.ndarray, np.asarray(dn["icoord"], dtype=float))
    dcoord = cast(np.ndarray, np.asarray(dn["dcoord"], dtype=float))
    xlbls: list[str] = dn.get("ivl", labels)

    lines: list[go.Scatter] = []
    for xs, ys in zip(icoord, dcoord, strict=False):
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
            tickvals=list(range(5, 10 * n, 10)),
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


# ============================================================================
# 3) Covariance Spectrum (eigenvalues)
# ============================================================================
def _eigvalsh_safe(S: np.ndarray) -> NDArray[np.float64]:
    """Safe eigenvalue computation for nearly singular matrices."""
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    with np.errstate(all="ignore"):
        try:
            return np.linalg.eigvalsh(S)
        except np.linalg.LinAlgError:
            return np.linalg.eigvalsh(S + 1e-12 * np.eye(S.shape[0]))


def covariance_spectrum(
    Sigma: np.ndarray | pl.DataFrame,
    *,
    title: str = "Covariance Spectrum (eigenvalues)",
) -> go.Figure:
    S = _to_numpy_matrix(Sigma).astype(float, copy=False)
    S = 0.5 * (S + S.T)
    vals = _eigvalsh_safe(S)
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 4) Efficient Frontier
# ============================================================================


def efficient_frontier(
    *args: Any,
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
    **kwargs: Any,
) -> go.Figure:
    """
    Frontier robust:
      • Acepta (riesgo, retorno) ya preparados (closed-form y/o box-projected).
      • Compatibilidad: efficient_frontier(risks, rets) vía *args.
      • Añade puntos clave (MSR/MinVar), CAL, y puntos personalizados.
      • Defensivo con NaNs / arrays vacíos. Totalmente tipado para mypy.
    """
    # Compat: efficient_frontier(risks, rets)
    if len(args) == 2 and risks_closed is None and rets_closed is None:
        risks_closed = cast(Iterable[float], args[0])
        rets_closed = cast(Iterable[float], args[1])

    def _clean_and_sort(
        x_raw: Iterable[float] | None,
        y_raw: Iterable[float] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = _to_1d_float(x_raw)
        y = _to_1d_float(y_raw)
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return x, y
        idx = np.argsort(x)
        return x[idx], y[idx]

    fig = go.Figure()

    # Series limpias/ordenadas
    x_c, y_c = _clean_and_sort(risks_closed, rets_closed)
    x_b, y_b = _clean_and_sort(risks_box, rets_box)

    # Relleno “gap” entre frontiers si ambas están presentes
    def _add_constraint_gap_fill(
        fig_: go.Figure,
        x1: NDArray[np.float64],
        y1: NDArray[np.float64],
        x2: NDArray[np.float64],
        y2: NDArray[np.float64],
    ) -> None:
        try:
            if x1.size and x2.size:
                xmin = float(max(x1.min(), x2.min()))
                xmax = float(min(x1.max(), x2.max()))
                if np.isfinite(np.asarray([xmin, xmax], dtype=float)).all() and xmax > xmin:
                    xs = np.linspace(xmin, xmax, 200)
                    y1i = np.interp(xs, x1, y1)
                    y2i = np.interp(xs, x2, y2)
                    fig_.add_trace(
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
            # No romper por un fill estético
            pass

    if x_c.size and x_b.size:
        _add_constraint_gap_fill(fig, x_c, y_c, x_b, y_b)

    # Curva closed-form
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

    # Curva box-projected
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

    # Puntos clave
    title_suffix = ""
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
            # CAL si hay rf y σ>0
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

    # Puntos personalizados
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

    # Auto-rangos
    candidates_x: list[float] = []
    candidates_y: list[float] = []
    if x_c.size:
        candidates_x.append(float(x_c.max()))
    if x_b.size:
        candidates_x.append(float(x_b.max()))
    if isinstance(msr_point, (tuple, list)) and len(msr_point) == 2:
        candidates_x.append(float(msr_point[0]))
        candidates_y.append(float(msr_point[1]))
    if isinstance(minvar_point, (tuple, list)) and len(minvar_point) == 2:
        candidates_y.append(float(minvar_point[1]))
    if y_c.size:
        candidates_y.append(float(y_c.max()))
    if y_b.size:
        candidates_y.append(float(y_b.max()))

    x_max = np.nanmax(candidates_x) if candidates_x else np.nan
    y_max = np.nanmax(candidates_y) if candidates_y else np.nan

    # Si no hay nada que pintar
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 5) Weights Barplot
# ============================================================================


def weights_bar(
    weights: ArrayLike,
    labels: Sequence[str],
    *,
    sort: bool = True,
    topn: int | None = None,
    horizontal: bool = True,
    title: str = "Portfolio Weights",
) -> go.Figure:
    w = _to_1d_float(weights)
    if w.size == 0 or labels is None or len(labels) == 0:
        return _placeholder_figure(title)

    N = min(w.size, len(labels))
    if N == 0:
        return _placeholder_figure(title)
    w, labels = w[:N], list(labels)[:N]
    w = np.nan_to_num(w)

    idx = np.lexsort((np.arange(N), w)) if sort else np.arange(N)
    if topn:
        topn = int(max(1, min(topn, N)))
        idx = idx[-topn:]

    w_sel = w[idx]
    labs_sel = [labels[i] for i in idx]

    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=w_sel,
                y=labs_sel,
                orientation="h",
                hovertemplate="%{y}: %{x:.2%}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_tickformat=".0%",
            title=title,
            margin=dict(l=90, r=20, t=60, b=40),
            template="plotly_white",
        )
    else:
        fig = go.Figure(go.Bar(x=labs_sel, y=w_sel, hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
        fig.update_layout(
            yaxis_tickformat=".0%",
            title=title,
            margin=dict(l=40, r=20, t=60, b=80),
            template="plotly_white",
        )
    return fig


# ============================================================================
# 6) Weights Heatmap (scenarios)
# ============================================================================


def weights_heatmap(
    W: np.ndarray,
    asset_labels: Sequence[str],
    scenario_labels: Sequence[str] | None = None,
    *,
    title: str = "Weights by Scenario",
) -> go.Figure:
    """2D heatmap of weights across scenarios."""
    if W is None or np.ndim(W) != 2:
        return _placeholder_figure(title)

    S, N = W.shape
    assets = list(asset_labels)[:N] if asset_labels is not None else [f"A{i}" for i in range(N)]
    if len(assets) != N:
        assets = [f"A{i}" for i in range(N)]
    scenarios = (
        list(scenario_labels)[:S] if scenario_labels is not None else [f"S{i}" for i in range(S)]
    )
    if len(scenarios) != S:
        scenarios = [f"S{i}" for i in range(S)]

    W_plot = np.where(np.isfinite(W), W, 0.0)
    zmin = float(np.nanmin(W_plot)) if np.isfinite(W_plot).any() else 0.0
    zmax = float(np.nanmax(W_plot)) if np.isfinite(W_plot).any() else 0.0
    if not np.isfinite(zmin):
        zmin = 0.0
    if not np.isfinite(zmax):
        zmax = 0.0
    if zmax == zmin:
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 7) Equity & Drawdown
# ============================================================================


def equity_and_drawdown(
    dates: Sequence[Any],
    equity: np.ndarray | Sequence[float],
    *,
    title: str = "Equity & Drawdown",
) -> go.Figure:
    eq = np.asarray(equity, dtype=float)
    if len(dates) != len(eq):
        raise ValueError("`dates` and `equity` must have the same length.")
    eq = np.where(np.isfinite(eq), eq, np.nan)

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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 8) Loss Distribution with VaR/ES markers
# ============================================================================


def loss_distribution(
    losses: ArrayLike,
    *,
    alphas: Sequence[float] = (0.95, 0.99),
    bins: int = 60,
    title: str = "Loss Distribution with VaR / ES",
) -> go.Figure:
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    x.sort()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=x, nbinsx=bins, name="Losses", opacity=0.75, histnorm="probability")
    )

    for a in alphas:
        q = float(np.quantile(x, a)) if x.size else np.nan
        tail = x[x >= q] if x.size else np.array([], dtype=float)
        es = float(tail.mean()) if tail.size else np.nan
        if np.isfinite(q):
            fig.add_vline(
                x=q,
                line_dash="dash",
                annotation_text=f"VaR {int(a * 100)}%: {q:.2f}",
                annotation_position="top",
            )
        if np.isfinite(es):
            fig.add_vline(
                x=es,
                line_dash="dot",
                annotation_text=f"ES {int(a * 100)}%: {es:.2f}",
                annotation_position="top",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Loss",
        yaxis_title="Probability",
        margin=dict(l=60, r=20, t=60, b=60),
        bargap=0.02,
        template="plotly_white",
    )
    return fig


# ============================================================================
# 9) Scree Plot (explained variance)
# ============================================================================


def scree_plot(
    eigvals: ArrayLike,
    *,
    title: str = "Scree Plot (Explained Variance)",
) -> go.Figure:
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 10) Correlation Network Graph (simple circular layout)
# ============================================================================


def network_corr_graph(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    threshold: float = 0.3,  # draw edges with |ρ| >= threshold
    title: str = "Correlation Network Graph",
) -> go.Figure:
    M = _to_numpy_matrix(Sigma_or_Corr)
    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)
    n = Corr.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)

    edges_pos: list[tuple[float, float, float, float]] = []
    edges_neg: list[tuple[float, float, float, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            rho = float(Corr[i, j])
            if np.isnan(rho) or abs(rho) < threshold:
                continue
            x1, y1, x2, y2 = xs[i], ys[i], xs[j], ys[j]
            if rho >= 0:
                edges_pos.append((x1, y1, x2, y2))
            else:
                edges_neg.append((x1, y1, x2, y2))

    def _edges_to_segments(
        edges: list[tuple[float, float, float, float]],
    ) -> tuple[list[float], list[float]]:
        xs_e: list[float] = []
        ys_e: list[float] = []
        for x1, y1, x2, y2 in edges:
            xs_e += [x1, x2, None]  # type: ignore[list-item]
            ys_e += [y1, y2, None]  # type: ignore[list-item]
        return xs_e, ys_e

    xs_pos, ys_pos = _edges_to_segments(edges_pos)
    xs_neg, ys_neg = _edges_to_segments(edges_neg)

    def _avg_width(edges: list[tuple[float, float, float, float]]) -> float:
        if not edges:
            return 1.5
        # approximate width scaling by count
        return 1.0 + 3.0 * min(1.0, len(edges) / max(1, n * (n - 1) / 2))

    fig = go.Figure()
    if xs_pos:
        fig.add_trace(
            go.Scatter(
                x=xs_pos,
                y=ys_pos,
                mode="lines",
                name="ρ ≥ 0",
                line=dict(width=_avg_width(edges_pos), color="rgba(0,120,255,0.5)"),
                hoverinfo="none",
            )
        )
    if xs_neg:
        fig.add_trace(
            go.Scatter(
                x=xs_neg,
                y=ys_neg,
                mode="lines",
                name="ρ < 0",
                line=dict(width=_avg_width(edges_neg), color="rgba(255,80,80,0.5)"),
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 11) Risk Contributions bar
# ============================================================================


def risk_contributions_bar(
    rc: ArrayLike,
    labels: Sequence[str],
    *,
    sort: bool = True,
    topn: int | None = None,
    title: str = "Risk Contributions",
) -> go.Figure:
    v = _to_1d_float(rc)
    if v.ndim != 1 or len(v) != len(labels):
        raise ValueError("`rc` must be 1D and aligned with `labels`.")
    idx = np.arange(len(v))
    if sort:
        idx = np.argsort(v)
    if topn:
        idx = idx[-topn:]
    v_plot = v[idx]
    labs_sel = [labels[i] for i in idx]
    fig = go.Figure(
        go.Bar(x=v_plot, y=labs_sel, orientation="h", hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    )
    fig.update_layout(
        title=title,
        margin=dict(l=80, r=20, t=60, b=40),
        xaxis_title="Contribution",
        yaxis_title="",
        template="plotly_white",
    )
    return fig


# ============================================================================
# 12) Rolling lines (multi-series)
# ============================================================================


def rolling_lines(
    dates: Sequence[Any],
    series_dict: dict[str, ArrayLike],
    *,
    title: str = "Rolling Metrics",
) -> go.Figure:
    fig = go.Figure()
    for name, arr in series_dict.items():
        y = _to_1d_float(arr)
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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 13) Correlation Heatmap (WebGL)
# ============================================================================


def corr_heatmap_gl(
    Sigma_or_Corr: np.ndarray | pl.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    is_cov: bool = True,
    order: HeatmapOrder | None = None,
    zlim: tuple[float, float] = (-1.0, 1.0),
    title: str = "Correlation Heatmap (WebGL)",
) -> go.Figure:
    if order is None:
        order = HeatmapOrder()

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
        template="plotly_white",
    )
    return fig


# ============================================================================
# 14) Weights path vs γ / Turnover vs γ / TE frontier
# ============================================================================


def weights_path_gammas(
    Ws: np.ndarray,
    gammas: Sequence[float],
    labels: Sequence[str],
    *,
    topn: int = 20,
    title: str = "Weights path vs γ (top names)",
) -> go.Figure:
    if Ws.ndim != 2:
        raise ValueError("`Ws` must be 2D (n_gamma x N).")
    nG, N = Ws.shape
    g = np.asarray(gammas, dtype=float)
    if g.size != nG:
        raise ValueError("`gammas` length must match `Ws.shape[0]`.")
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
        title=title,
        xaxis_title="log10(γ)",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        template="plotly_white",
    )
    return fig


def turnover_vs_gamma(
    Ws: np.ndarray, w_ref: np.ndarray, gammas: Sequence[float], *, title: str = "Turnover vs γ"
) -> go.Figure:
    if Ws.ndim != 2:
        raise ValueError("`Ws` must be 2D.")
    if w_ref.ndim != 1 or w_ref.shape[0] != Ws.shape[1]:
        raise ValueError("`w_ref` must be shape (N,) and match `Ws` columns.")
    L1 = np.sum(np.abs(Ws - w_ref[None, :]), axis=1)
    L2 = np.sqrt(np.sum((Ws - w_ref[None, :]) ** 2, axis=1))
    g = np.asarray(gammas, dtype=float)
    if g.size != Ws.shape[0]:
        raise ValueError("`gammas` length must match `Ws.shape[0]`.")
    x = np.log10(g)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=L1, mode="lines+markers", name="L1"))
    fig.add_trace(go.Scatter(x=x, y=L2, mode="lines+markers", name="L2"))
    fig.update_layout(
        title=title, xaxis_title="log10(γ)", yaxis_title="Turnover", template="plotly_white"
    )
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
        template="plotly_white",
    )
    fig.update_xaxes(tickformat=".2%")
    fig.update_yaxes(tickformat=".2%")
    return fig


# ============================================================================
# 15) Backtest & Attribution plots
# ============================================================================


def plot_equity(dates: list[Any], equity: np.ndarray, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure(go.Scatter(x=dates, y=equity, mode="lines", name="Equity"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="NAV", template="plotly_white")
    return fig


def plot_drawdown(dates: list[Any], equity: np.ndarray, title: str = "Drawdown") -> go.Figure:
    cummax = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(cummax, 1e-12)) - 1.0
    fig = go.Figure(go.Scatter(x=dates, y=dd, mode="lines", name="Drawdown"))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown", template="plotly_white"
    )
    return fig


def plot_weights_heatmap(
    dates: list[Any], tickers: list[str], W: np.ndarray, title: str = "Weights"
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


def plot_turnover(
    dates_or_df: Any, turnover: ArrayLike | None = None, title: str = "Turnover"
) -> go.Figure:
    if turnover is None:
        df = dates_or_df
        if isinstance(df, pl.DataFrame) and {"date", "turnover"}.issubset(set(df.columns)):
            x = df.get_column("date").to_list()
            y = df.get_column("turnover").to_numpy()
        elif hasattr(df, "columns") and {"date", "turnover"}.issubset(set(df.columns)):
            x = df["date"].tolist()
            y = np.asarray(df["turnover"].values, dtype=float)
        else:
            raise ValueError("If you pass a DataFrame it must have columns ['date','turnover'].")
    else:
        x = list(dates_or_df)
        y = _to_1d_float(turnover)
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
    dates: list[Any], te: np.ndarray, title: str = "Tracking Error (daily proxy)"
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


def plot_top_contributors(df_top: pl.DataFrame, title: str = "Top Contributors") -> go.Figure:
    pdf = df_top.select(["ticker", "contrib_total"]).to_pandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
    pdf = pdf.sort_values("contrib_total", ascending=False)
    fig = go.Figure(go.Bar(x=pdf["ticker"], y=pdf["contrib_total"]))
    fig.update_layout(
        title=title, xaxis_title="Ticker", yaxis_title="Total Contribution", template="plotly_white"
    )
    return fig


def plot_group_contrib_area(
    df: pl.DataFrame, title: str = "Group Contributions Over Time"
) -> go.Figure:
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
    df_brinson: pl.DataFrame, title: str = "Brinson-Fachler Attribution"
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
# Advanced attribution plots
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

    wide = _pivot_compat(
        df_top.select(["date", "ticker", "contrib"]),
        values="contrib",
        index="date",
        columns="ticker",
    ).sort("date")

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


# ============================================================================
# 16) Extra attribution plots
# ============================================================================


def plot_contrib_heatmap_daily(
    df_asset_daily: pl.DataFrame,
    *,
    title: str = "Daily Contribution Heatmap",
    tickers_order: list[str] | None = None,
) -> go.Figure:
    """
    Heatmap de contribución por ticker (filas) vs fecha (columnas).

    Espera en df_asset_daily: ["date", "ticker", "contrib"].
    - Respeta `tickers_order` si se pasa.
    - Ordena columnas por fecha natural (convierte a datetime y quita tz).
    - Paleta divergente centrada en 0 y escala simétrica (mejor lectura ±).
    """
    req = {"date", "ticker", "contrib"}
    if not req.issubset(set(df_asset_daily.columns)):
        raise ValueError("df_asset_daily must include 'date','ticker','contrib'.")

    # ---- 1) A pandas + limpieza homogénea ----
    pdf = (
        df_asset_daily.select(["date", "ticker", "contrib"])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "ticker", "contrib"])
    )

    # Normaliza fechas (orden natural incluso si venían como string/tz)
    pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")
    pdf["date"] = pdf["date"].dt.tz_localize(None)
    pdf = pdf.dropna(subset=["date"])

    # ---- 2) Orden de tickers ----
    if tickers_order:
        # Respeta orden explícito
        pdf["ticker"] = pd.Categorical(pdf["ticker"], categories=tickers_order, ordered=True)
    else:
        # Si no se pasa orden: por contribución absoluta total descendente
        tot = (
            pdf.groupby("ticker", as_index=False)["contrib"]
            .sum(numeric_only=True)
            .assign(abs_total=lambda d: d["contrib"].abs())
            .sort_values("abs_total", ascending=False)["ticker"]
            .tolist()
        )
        pdf["ticker"] = pd.Categorical(pdf["ticker"], categories=tot, ordered=True)

    pdf = pdf.sort_values(["ticker", "date"])

    # ---- 3) Pivot (suma si hay duplicados ticker-fecha) ----
    pivot = pdf.pivot_table(
        index="ticker", columns="date", values="contrib", aggfunc="sum", fill_value=0.0
    )

    # Asegura orden cronológico de columnas
    pivot = pivot.sort_index(axis=1)

    if pivot.size == 0:
        return _placeholder_figure(title, subtitle="No data after cleaning")

    # ---- 4) Escala simétrica y paleta divergente centrada en 0 ----
    z = pivot.values.astype(float, copy=False)
    if np.all(~np.isfinite(z)):
        return _placeholder_figure(title, subtitle="No finite values to plot")

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(np.abs(z)))
    if vmax <= 0:
        vmax = 1e-6  # evita escala degenerada

    # ---- 5) Figura ----
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns,  # fechas (DatetimeIndex)
            y=pivot.index.astype(str),  # tickers en el orden elegido
            zmin=-vmax,
            zmax=vmax,
            zmid=0.0,
            colorscale="RdBu",
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
    group_labels: list[str] | None = None,
    *,
    component: str = "total",  # "alloc" | "select" | "interact" | "total"
    title: str | None = None,
) -> go.Figure:
    # columnas mínimas
    metrics = {"alloc", "select", "interact", "total"}
    req_base = {"date"} | metrics
    cols = set(df_brinson_g.columns)
    if component not in {"alloc", "select", "interact", "total"}:
        raise ValueError("component must be one of {'alloc','select','interact','total'}.")

    # 0) caso minimalista: no hay 'group_id' ni 'group' ⇒ inventamos grupo 0
    if req_base.issubset(cols) and "group_id" not in cols and "group" not in cols:
        df_brinson_g = df_brinson_g.with_columns(pl.lit(0).alias("group_id"))

    # 1) normalizar a group_id
    if "group_id" not in df_brinson_g.columns and "group" in df_brinson_g.columns:
        uniq = [g for g in df_brinson_g.get_column("group").unique().to_list() if g is not None]
        mapping = {g: i for i, g in enumerate(sorted(uniq))}
        df_brinson_g = df_brinson_g.with_columns(
            pl.col("group").map_elements(lambda g: mapping.get(g, -1)).alias("group_id")
        )

    # 2) validar contrato final
    need = {"date", "group_id", component}
    if not need.issubset(set(df_brinson_g.columns)):
        raise ValueError(
            f"df_brinson_g must include {sorted(need)}; got {sorted(df_brinson_g.columns)}"
        )

    # 3) a pandas, saneando infinitos/NaN
    pdf = (
        df_brinson_g.select(["date", "group_id", component])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    pdf["group_id"] = pdf["group_id"].astype(int)

    # 4) labels
    gids = sorted(pdf["group_id"].unique().tolist())
    if group_labels is None:
        group_labels = [f"G{i}" for i in range(max(gids) + 1)]
    # map
    labels = [group_labels[i] if 0 <= i < len(group_labels) else f"G{i}" for i in pdf["group_id"]]
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


def plot_brinson_cumulative_components(
    df_brinson: pl.DataFrame,
    title: str = "Cumulative Brinson–Fachler Attribution",
) -> go.Figure:
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
    need = {"alloc", "select", "interact", "total"}
    if not need.issubset(set(df_brinson.columns)):
        raise ValueError(f"Missing columns in df_brinson: {need}")

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


# ============================================================================
# 17) Group contrib (barra simple) — alias estable para la app
# ============================================================================


def plot_group_contrib(
    df_group_total: pl.DataFrame,
    title: str = "Group Contributions",
) -> go.Figure:
    """
    Barras de contribución por grupo (snapshot total).
    Mantiene la firma esperada por la app.
    """
    pdf = df_group_total.select(["group", "contrib_total"]).to_pandas()
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
    pdf = pdf.sort_values("contrib_total", ascending=False)
    fig = go.Figure(go.Bar(x=pdf["group"], y=pdf["contrib_total"]))
    fig.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title="Total Contribution",
        template="plotly_white",
    )
    return fig


# ============================================================================
# 18) Panel de equity por escenarios
# ============================================================================


def plot_scenario_equity_panel(
    results: list[Any], title: str = "Scenario Equity Panel"
) -> go.Figure:
    """
    Espera una lista de objetos con:
      - r.name (str)
      - r.bt["dates"], r.bt["equity"] (secuencia alineada)
    """
    fig = go.Figure()
    for r in results:
        dates = r.bt.get("dates", []) if hasattr(r, "bt") else []
        eq = r.bt.get("equity", []) if hasattr(r, "bt") else []
        name = getattr(r, "name", "scenario")
        fig.add_trace(go.Scatter(x=dates, y=eq, mode="lines", name=str(name)))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# 19) Barras de métricas por escenario
# ============================================================================


def plot_scenario_metrics_bars(
    df_metrics: pl.DataFrame, title: str = "Scenario Metrics"
) -> go.Figure:
    """
    df_metrics: polars con al menos columna 'scenario' y varias métricas numéricas.
    Hace un melt y agrupa en barras apiladas/agrupadas.
    """
    pdf = df_metrics.to_pandas().replace([np.inf, -np.inf], np.nan).dropna(subset=["scenario"])
    # Filtra columnas numéricas excepto 'scenario'
    numeric_cols = [
        c for c in pdf.columns if c != "scenario" and pd.api.types.is_numeric_dtype(pdf[c])
    ]
    if not numeric_cols:
        return _placeholder_figure(title, subtitle="No numeric metrics to plot")
    melted = pdf.melt(
        id_vars="scenario", value_vars=numeric_cols, var_name="metric", value_name="value"
    )
    fig = px.bar(
        melted,
        x="scenario",
        y="value",
        color="metric",
        barmode="group",
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_tickangle=-20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# 20) Heatmap Δ-weights (escenario vs baseline)
# ============================================================================


def plot_weights_delta_heatmap(
    tickers: list[str],
    W_base: np.ndarray,
    W_scn: np.ndarray,
    title: str = "Δ Weights (Scenario - Baseline)",
) -> go.Figure:
    """
    Acepta series o últimos vectores; hace reshape(1,-1) y plotea un heatmap horizontal.
    """
    base = (W_base[-1] if W_base.ndim == 2 else W_base).astype(float)
    scn = (W_scn[-1] if W_scn.ndim == 2 else W_scn).astype(float)
    if base.shape != scn.shape:
        raise ValueError("W_base and W_scn must be same shape (N,)")
    if len(tickers) != base.shape[0]:
        raise ValueError("tickers length must match N.")
    d = np.nan_to_num(scn - base, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(np.abs(d))) if d.size else 1.0
    vmax = vmax if vmax > 0 else 1e-6

    fig = go.Figure(
        data=go.Heatmap(
            z=d.reshape(1, -1),
            x=tickers,
            y=["Δw"],
            zmin=-vmax,
            zmax=vmax,
            zmid=0.0,
            colorscale="RdBu",
            colorbar=dict(title="Δw"),
            hovertemplate="Ticker: %{x}<br>Δw: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_nticks=min(40, len(tickers)),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ============================================================================
# 21) Tornado de sensibilidad (down/up vs base)
# ============================================================================


def plot_tornado_sensitivity(
    df_sens: DataFrameLike,
    metric_label: str = "CAGR",
    down_label: str = "Down",
    up_label: str = "Up",
    sort_by: str = "min_delta",
    title: str | None = None,
) -> go.Figure:
    """
    df_sens con columnas: ['asset'| 'name', 'base', 'down', 'up'].
    Grafica barras horizontales para Δ down y Δ up ordenadas por el lado más adverso.
    """
    pdf = _to_pandas(df_sens).copy()

    if "asset" not in pdf.columns and "name" in pdf.columns:
        pdf.rename(columns={"name": "asset"}, inplace=True)

    required = {"asset", "base", "down", "up"}
    missing = required - set(pdf.columns)
    if missing:
        raise ValueError(f"plot_tornado_sensitivity: missing columns {missing}")

    for c in ["base", "down", "up"]:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0.0)

    down_delta = (pdf["down"] - pdf["base"]).to_numpy(dtype=float)
    up_delta = (pdf["up"] - pdf["base"]).to_numpy(dtype=float)
    min_delta = np.minimum(down_delta, up_delta)

    pdf["down_delta"] = down_delta
    pdf["up_delta"] = up_delta
    pdf["min_delta"] = min_delta
    pdf.sort_values(by=sort_by, inplace=True)
    pdf.reset_index(drop=True, inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=pdf["asset"], x=pdf["down_delta"], orientation="h", name=down_label))
    fig.add_trace(go.Bar(y=pdf["asset"], x=pdf["up_delta"], orientation="h", name=up_label))
    fig.update_layout(
        title=title or f"Tornado Sensitivity — Metric: {metric_label}",
        barmode="overlay",
        xaxis_title=f"Δ {metric_label} vs Base",
        yaxis_title="Asset",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=30, t=60, b=40),
    )
    return fig


# ============================================================================
# 22) Comparación de equity entre dos curvas
# ============================================================================


def plot_equity_compare(
    dates: Sequence[Any],
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


# ============================================================================
# 23) Comparación de drawdowns (%)
# ============================================================================


def plot_drawdown_compare(
    dates: Sequence[Any],
    equity_a: Sequence[float],
    equity_b: Sequence[float],
    name_a: str = "Baseline",
    name_b: str = "Scenario",
    title: str | None = None,
) -> go.Figure:
    def _dd_curve(eq: Sequence[float]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(eq, dtype=np.float64)
        if arr.size == 0:
            return np.empty(0, dtype=np.float64)
        # mypy no sabe tipar accumulate correctamente -> cast al final
        cummax = np.maximum.accumulate(arr)
        dd = (arr / np.maximum(cummax, 1e-12)) - 1.0
        return cast(NDArray[np.float64], dd)

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


# ============================================================================
# 24) Barras de Δ-métrica vs baseline
# ============================================================================


def plot_metric_delta_bars(
    df_metrics: DataFrameLike,
    baseline_value: float,
    metric_col: str = "CAGR",
    scenario_name_col: str = "Scenario",
    title: str | None = None,
) -> go.Figure:
    pdf = _to_pandas(df_metrics).copy()
    if scenario_name_col not in pdf.columns or metric_col not in pdf.columns:
        raise ValueError("plot_metric_delta_bars: required columns not found.")
    pdf[metric_col] = pd.to_numeric(pdf[metric_col], errors="coerce")
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
        margin=dict(l=80, r=30, t=60, b=40),
    )
    return fig


# ============================================================================
# 25) Heatmap de comparación de weights (δ o absoluto) en el tiempo
# ============================================================================


def plot_weights_compare_heatmap(
    dates: Sequence[Any],
    tickers: Sequence[str],
    weights_a: np.ndarray | Sequence[Sequence[float]],
    weights_b: np.ndarray | Sequence[Sequence[float]],
    name_a: str = "Baseline",
    name_b: str = "Scenario",
    mode: str = "delta",  # {"delta", "absolute"}
    title: str | None = None,
    zmax_abs: float | None = None,
) -> go.Figure:
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

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    if mode not in {"delta", "absolute"}:
        raise ValueError("mode must be either 'delta' or 'absolute'.")

    if mode == "delta":
        Z = B - A
        max_abs = float(np.max(np.abs(Z))) if Z.size else 1.0
        vmax = (
            float(zmax_abs)
            if (zmax_abs is not None and zmax_abs > 0)
            else (max_abs if max_abs > 0 else 1e-6)
        )
        vmin = -vmax
        colorscale = "RdBu"
        zmid = 0.0
        default_title = f"Weights Heatmap — {name_b} vs {name_a} (Δ)"
    else:
        Z = B
        vmax = float(np.max(Z)) if Z.size else 1.0
        vmin = 0.0
        colorscale = "Blues"
        zmid = None
        default_title = f"Weights Heatmap — {name_b} (absolute weights)"

    title = title or default_title

    # Eje Y como strings para legibilidad en heatmap
    y_labels = [str(d) for d in dates]
    x_labels = list(tickers)

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
            hovertemplate="Date=%{y}<br>Ticker=%{x}<br>Value=%{z:.2%}<extra></extra>",
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


# ============================================================================
# 26) Plots de Brinson avanzados
# ============================================================================


def plot_brinson_group_bar(
    df: pl.DataFrame | pd.DataFrame,
    *,
    title: str = "Brinson attribution by group",
    metric: str | None = None,
) -> go.Figure:
    pdf = _to_pandas(df)

    if "group" in pdf.columns:
        group_col = "group"
    elif "group_id" in pdf.columns:
        group_col = "group_id"
    else:
        raise ValueError("Expected a 'group' or 'group_id' column in Brinson group dataframe.")

    x = pdf[group_col].astype(str)

    fig = go.Figure()

    if metric is not None:
        if metric not in pdf.columns:
            raise ValueError(f"Metric '{metric}' not found in Brinson group dataframe.")
        y = pdf[metric]
        fig.add_bar(
            x=x,
            y=y,
            name=metric,
            hovertemplate=f"Group: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>",
        )
    else:
        metrics = [c for c in ("alloc", "select", "interact", "total") if c in pdf.columns]
        if not metrics:
            raise ValueError(
                "No Brinson metric columns found (expected one of: "
                "'alloc', 'select', 'interact', 'total')."
            )
        for m in metrics:
            fig.add_bar(
                x=x,
                y=pdf[m],
                name=m.capitalize(),
                hovertemplate=f"Group: %{{x}}<br>{m}: %{{y:.4f}}<extra></extra>",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title="Contribution",
        barmode="group",
        legend_title_text="Component",
    )
    return fig


def plot_brinson_timeseries(
    df: DataFrameLike,
    title: str = "Brinson timeseries",
    *,
    metric: str | None = None,
) -> go.Figure:
    pdf = _to_pandas(df).copy()

    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date")

    metrics = [m for m in ["alloc", "select", "interact", "total"] if m in pdf.columns]
    if metric is not None and metric in metrics:
        metrics = [metric]

    fig = go.Figure()
    for m in metrics:
        fig.add_trace(
            go.Scatter(
                x=pdf["date"],
                y=pdf[m],
                mode="lines",
                name=m.capitalize(),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Attribution",
        template="plotly_dark",
    )
    return fig


def plot_euler_contributions(
    df: pl.DataFrame | pd.DataFrame | pd.Series,
    *,
    title: str = "Euler risk contributions",
    top_n: int | None = None,
    as_percent: bool = False,
) -> go.Figure:
    if isinstance(df, pd.Series):
        pdf = df.to_frame(name="risk_contribution")
        pdf["asset"] = pdf.index.astype(str)
    elif isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        pdf = df.copy()
    else:
        raise TypeError(
            "Unsupported type for Euler contributions: expected pandas.Series, "
            "pandas.DataFrame or polars.DataFrame."
        )

    if "risk_contribution" in pdf.columns:
        contrib_col = "risk_contribution"
    elif "contribution" in pdf.columns:
        contrib_col = "contribution"
    else:
        raise ValueError("Expected a 'risk_contribution' or 'contribution' column for Euler plot.")

    if "asset" in pdf.columns:
        asset_col = "asset"
    elif "ticker" in pdf.columns:
        asset_col = "ticker"
    else:
        pdf["asset"] = pdf.index.astype(str)
        asset_col = "asset"

    pdf = pdf[[asset_col, contrib_col]].copy()
    pdf = pdf.dropna(subset=[contrib_col])

    if as_percent:
        total = float(pdf[contrib_col].sum())
        if abs(total) > 1e-12:
            pdf[contrib_col] = pdf[contrib_col] / total * 100.0
        y_label = "Risk contribution (%)"
    else:
        y_label = "Risk contribution"

    pdf["abs_contrib"] = pdf[contrib_col].abs()
    pdf = pdf.sort_values("abs_contrib", ascending=False)

    if top_n is not None and top_n > 0:
        pdf = pdf.head(top_n)

    fig = go.Figure(
        go.Bar(
            x=pdf[asset_col],
            y=pdf[contrib_col],
            hovertemplate="Asset: %{x}<br>Contribution: %{y:.6f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title=y_label,
    )
    return fig


def plot_factor_rc_bar(
    factor_rc: pd.Series | pd.DataFrame,
    *,
    title: str = "Factor risk contributions",
    as_percent: bool = False,
    sigma_p: float | None = None,
    top_n: int | None = None,
) -> go.Figure:
    if isinstance(factor_rc, pd.DataFrame):
        s = pd.Series(dtype=float) if factor_rc.shape[1] == 0 else factor_rc.iloc[:, 0]
    else:
        s = factor_rc

    s = s.astype(float)

    if as_percent:
        if sigma_p is not None and sigma_p > 1e-12:
            s = s / float(sigma_p) * 100.0
        else:
            total = float(s.sum())
            if abs(total) > 1e-12:
                s = s / total * 100.0
        y_label = "Risk contribution (%)"
    else:
        y_label = "Risk contribution"

    if top_n is not None and top_n > 0 and top_n < len(s):
        order = s.abs().sort_values(ascending=False).index[:top_n]
        s = s.loc[order]

    fig = go.Figure(
        go.Bar(
            x=s.index.astype(str),
            y=s.values,
            hovertemplate="Factor: %{x}<br>Contribution: %{y:.6f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Factor",
        yaxis_title=y_label,
    )
    return fig


def plot_factor_rc_heatmap(
    asset_factor_rc: pd.DataFrame,
    *,
    title: str = "Asset × Factor RC",
    as_percent: bool = False,
    sigma_p: float | None = None,
) -> go.Figure:
    df = asset_factor_rc.astype(float)

    if as_percent:
        if sigma_p is not None and sigma_p > 1e-12:
            df = df / float(sigma_p) * 100.0
        else:
            total = float(df.values.sum())
            if abs(total) > 1e-12:
                df = df / total * 100.0
        color_label = "RC (%)"
    else:
        color_label = "RC"

    fig = px.imshow(
        df.values,
        x=df.columns.astype(str),
        y=df.index.astype(str),
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="upper",
        labels=dict(color=color_label),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Factor",
        yaxis_title="Asset",
        margin=dict(t=60, r=10, b=40, l=60),
    )
    return fig
