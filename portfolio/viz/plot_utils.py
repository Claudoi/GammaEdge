# portfolio/viz/plot_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union, Literal, Dict

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import squareform


# ──────────────────────────────────────────────────────────────────────────────
# Helpers numéricos
# ──────────────────────────────────────────────────────────────────────────────

ArrayLike = Union[np.ndarray, Sequence[float]]

@dataclass(frozen=True, slots=True)
class HeatmapOrder:
    clustered: bool = True
    method: Literal["single", "complete", "average", "ward"] = "average"
    optimal: bool = True  # optimal leaf ordering (reduce disonancias)


def _to_numpy_matrix(x: Union[np.ndarray, pl.DataFrame]) -> np.ndarray:
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


# ──────────────────────────────────────────────────────────────────────────────
# 1) Correlation Heatmap (clustered)
# ──────────────────────────────────────────────────────────────────────────────

def corr_heatmap(
    Sigma_or_Corr: Union[np.ndarray, pl.DataFrame],
    labels: Optional[Sequence[str]] = None,
    *,
    is_cov: bool = True,
    order: HeatmapOrder = HeatmapOrder(),
    zlim: Tuple[float, float] = (-1.0, 1.0),
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

def corr_dendrogram(
    Sigma_or_Corr: Union[np.ndarray, pl.DataFrame],
    labels: Optional[Sequence[str]] = None,
    *,
    is_cov: bool = True,
    method: Literal["single", "complete", "average", "ward"] = "average",
    title: str = "Correlation Dendrogram",
) -> go.Figure:
    """
    Dendrograma jerárquico basado en la distancia de correlación.
    Útil para visualizar clústeres y justificar HRP/estrategias de agrupación.
    """
    M = _to_numpy_matrix(Sigma_or_Corr)
    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)
    n = Corr.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - Corr)))
    Z = linkage(squareform(dist, checks=False), method=method)
    # Usamos el dendrogram de scipy para extraer coordenadas y lo pintamos con Plotly
    dendro = dendrogram(Z, labels=list(labels), no_plot=True)
    icoord = np.array(dendro["icoord"])
    dcoord = np.array(dendro["dcoord"])
    xlbls = dendro["ivl"]

    data = []
    for xs, ys in zip(icoord, dcoord):
        data.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1)))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=list(range(5, 10 * n, 10)), ticktext=xlbls, tickangle=45),
        yaxis=dict(title="distance"),
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=120),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3) Covariance Spectrum (eigenvalues)
# ──────────────────────────────────────────────────────────────────────────────

def covariance_spectrum(
    Sigma: Union[np.ndarray, pl.DataFrame],
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
# 4) Efficient Frontier
# ──────────────────────────────────────────────────────────────────────────────

def efficient_frontier(
    risks: ArrayLike,
    rets: ArrayLike,
    *,
    msr_point: Optional[Tuple[float, float]] = None,
    minvar_point: Optional[Tuple[float, float]] = None,
    custom_points: Optional[Dict[str, Tuple[float, float]]] = None,
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Dibuja frontera eficiente (riesgo vs retorno). Añade marcadores clave:
    - MSR (max Sharpe)
    - MinVar
    - Cualquier otro (dict nombre → (σ, μ))
    """
    x = np.asarray(risks)
    y = np.asarray(rets)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Frontier"))

    if minvar_point is not None:
        fig.add_trace(go.Scatter(x=[minvar_point[0]], y=[minvar_point[1]], mode="markers",
                                 name="MinVar", marker=dict(size=10, symbol="diamond")))

    if msr_point is not None:
        fig.add_trace(go.Scatter(x=[msr_point[0]], y=[msr_point[1]], mode="markers",
                                 name="Max Sharpe", marker=dict(size=10, symbol="star")))

    if custom_points:
        for k, (sx, mu) in custom_points.items():
            fig.add_trace(go.Scatter(x=[sx], y=[mu], mode="markers", name=k, marker=dict(size=8)))

    fig.update_layout(
        title=title,
        xaxis_title="Risk (σ)",
        yaxis_title="Return (μ)",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5) Weights Barplot
# ──────────────────────────────────────────────────────────────────────────────

def weights_bar(
    weights: ArrayLike,
    labels: Sequence[str],
    *,
    sort: bool = True,
    topn: Optional[int] = None,
    horizontal: bool = True,
    title: str = "Portfolio Weights",
) -> go.Figure:
    """
    Barras de pesos. Útil para inspección de carteras óptimas o actuales.
    """
    w = np.asarray(weights).astype(float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if len(w) != len(labels):
        raise ValueError("weights and labels must align")

    idx = np.arange(len(w))
    if sort:
        idx = np.argsort(w)
    if topn is not None:
        idx = idx[-topn:]

    w_plot = w[idx]
    l_plot = [labels[i] for i in idx]

    if horizontal:
        fig = go.Figure(go.Bar(x=w_plot, y=l_plot, orientation="h", hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
        fig.update_layout(xaxis_tickformat=".0%", title=title, margin=dict(l=80, r=20, t=60, b=40))
    else:
        fig = go.Figure(go.Bar(x=l_plot, y=w_plot, hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
        fig.update_layout(yaxis_tickformat=".0%", title=title, margin=dict(l=40, r=20, t=60, b=80))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6) Weights Heatmap (escenarios)
# ──────────────────────────────────────────────────────────────────────────────

def weights_heatmap(
    W: np.ndarray,                      # shape (S, N) escenarios x activos
    asset_labels: Sequence[str],
    scenario_labels: Optional[Sequence[str]] = None,
    *,
    title: str = "Weights by Scenario",
) -> go.Figure:
    if W.ndim != 2:
        raise ValueError("W must be 2D (scenarios x assets)")
    S, N = W.shape
    if len(asset_labels) != N:
        raise ValueError("asset_labels length mismatch")
    if scenario_labels is None:
        scenario_labels = [f"S{i}" for i in range(S)]

    fig = go.Figure(
        data=go.Heatmap(
            z=W,
            x=asset_labels,
            y=scenario_labels,
            colorbar=dict(title="weight"),
            zmin=np.min(W), zmax=np.max(W),
            hovertemplate="scenario=%{y}<br>asset=%{x}<br>w=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis=dict(tickangle=45, automargin=True), margin=dict(l=60, r=20, t=60, b=80))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7) Equity Curve + Drawdown
# ──────────────────────────────────────────────────────────────────────────────

def equity_and_drawdown(
    dates: Sequence, equity: ArrayLike, *, title: str = "Equity & Drawdown"
) -> go.Figure:
    """
    Muestra curva de equity y drawdown (%) en eje secundario.
    """
    eq = np.asarray(equity).astype(float)
    eq = np.where(np.isfinite(eq), eq, np.nan)
    dd = np.zeros_like(eq)
    peak = -np.inf
    for i, v in enumerate(eq):
        peak = v if v > peak else peak
        dd[i] = 0.0 if peak <= 0 or not np.isfinite(peak) or not np.isfinite(v) else (v / peak - 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=eq, mode="lines", name="Equity", hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=dates, y=dd, mode="lines", name="Drawdown", yaxis="y2",
                             hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Equity"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%"),
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8) Loss Distribution with VaR/ES markers
# ──────────────────────────────────────────────────────────────────────────────

def loss_distribution(
    losses: ArrayLike,
    *,
    alphas: Sequence[float] = (0.95, 0.99),
    bins: int = 60,
    title: str = "Loss Distribution with VaR / ES",
) -> go.Figure:
    """
    losses: array de pérdidas (+ es pérdida; si trabajas con PnL usa -PnL).
    Dibuja histograma y marca VaR/ES en niveles alpha.
    """
    x = np.asarray(losses).astype(float)
    x = x[np.isfinite(x)]
    x.sort()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=bins, name="Losses", opacity=0.75, histnorm="probability"))

    for a in alphas:
        q = np.quantile(x, a)
        tail = x[x >= q]
        es = float(tail.mean()) if tail.size else np.nan
        fig.add_vline(x=q, line_dash="dash", annotation_text=f"VaR {int(a*100)}%: {q:.2f}", annotation_position="top")
        if np.isfinite(es):
            fig.add_vline(x=es, line_dash="dot", annotation_text=f"ES {int(a*100)}%: {es:.2f}", annotation_position="top")

    fig.update_layout(
        title=title,
        xaxis_title="Loss",
        yaxis_title="Probability",
        margin=dict(l=60, r=20, t=60, b=60),
        bargap=0.02,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Scree Plot (explained variance)
# ──────────────────────────────────────────────────────────────────────────────
def scree_plot(
    eigvals: ArrayLike,
    *,
    title: str = "Scree Plot (Explained Variance)",
) -> go.Figure:
    """
    Barras de varianza explicada por autovalor + línea de acumulada.
    Acepta eigenvalues no negativos (no hace falta que estén normalizados).
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
    fig.add_trace(go.Scatter(x=np.arange(1, len(cum) + 1), y=cum, mode="lines+markers",
                             name="Cumulative", yaxis="y2",
                             hovertemplate="k=%{x}<br>cum=%{y:.1%}<extra></extra>"))

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
# Correlation Network Graph
# ──────────────────────────────────────────────────────────────────────────────
def network_corr_graph(
    Sigma_or_Corr: Union[np.ndarray, pl.DataFrame],
    labels: Optional[Sequence[str]] = None,
    *,
    is_cov: bool = True,
    threshold: float = 0.3,        # dibuja aristas con |ρ|>=threshold
    title: str = "Correlation Network Graph",
) -> go.Figure:
    """
    Grafo de correlaciones con layout circular (sin networkx).
    - Nodos: activos
    - Aristas: |ρ_ij|>=threshold (ancho/alpha según |ρ|)
    - Color de arista: tono por signo (positivo/negativo)
    """
    M = _to_numpy_matrix(Sigma_or_Corr)
    Corr = _safe_corr_from_cov(M) if is_cov else np.copy(M)
    n = Corr.shape[0]
    if labels is None:
        labels = [f"A{i}" for i in range(n)]

    # Layout circular
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)

    edge_x, edge_y, edge_width, edge_color = [], [], [], []
    for i in range(n):
        for j in range(i+1, n):
            rho = float(Corr[i, j])
            if np.isnan(rho) or abs(rho) < threshold: 
                continue
            edge_x += [xs[i], xs[j], None]
            edge_y += [ys[i], ys[j], None]
            edge_width.append(1.0 + 3.0*abs(rho))
            edge_color.append("rgba(0,0,0,0)")  # dummy; se usa en traces separados

    # Para colorear por signo, creamos dos trazas: positivas y negativas
    pos_x, pos_y, pos_w = [], [], []
    neg_x, neg_y, neg_w = [], [], []
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            rho = float(Corr[i, j])
            if np.isnan(rho) or abs(rho) < threshold: 
                continue
            segx = [xs[i], xs[j], None]
            segy = [ys[i], ys[j], None]
            if rho >= 0:
                pos_x += segx; pos_y += segy; pos_w.append(1.0 + 3.0*abs(rho))
            else:
                neg_x += segx; neg_y += segy; neg_w.append(1.0 + 3.0*abs(rho))
            k += 1

    fig = go.Figure()

    if pos_x:
        fig.add_trace(go.Scatter(x=pos_x, y=pos_y, mode="lines", name="ρ ≥ 0",
                                 line=dict(width=np.mean(pos_w) if pos_w else 1.5, color="rgba(0,120,255,0.5)"),
                                 hoverinfo="none"))
    if neg_x:
        fig.add_trace(go.Scatter(x=neg_x, y=neg_y, mode="lines", name="ρ < 0",
                                 line=dict(width=np.mean(neg_w) if neg_w else 1.5, color="rgba(255,80,80,0.5)"),
                                 hoverinfo="none"))

    # Nodos (tamaño ~ grado)
    deg = (np.abs(Corr) >= threshold).sum(axis=0) - 1
    node_size = 10 + 3*np.clip(deg, 0, None)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=list(labels), textposition="top center",
        marker=dict(size=node_size, line=dict(width=1, color="#333")),
        hovertemplate="%{text}<extra></extra>", name="assets"
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20),
        height=600
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Risk Contributions bar
# ──────────────────────────────────────────────────────────────────────────────
def risk_contributions_bar(
    rc: ArrayLike,
    labels: Sequence[str],
    *,
    sort: bool = True,
    topn: Optional[int] = None,
    title: str = "Risk Contributions",
) -> go.Figure:
    """
    Barras horizontales de contribución al riesgo (por ejemplo, marginal*weight o RC%).
    `rc` puede estar en unidades absolutas o en porcentaje; aquí sólo graficamos.
    """
    v = np.asarray(rc, dtype=float)
    if v.ndim != 1 or len(v) != len(labels):
        raise ValueError("rc must be 1D and aligned with labels")
    idx = np.arange(len(v))
    if sort:
        idx = np.argsort(v)
    if topn is not None:
        idx = idx[-topn:]
    v_plot = v[idx]
    l_plot = [labels[i] for i in idx]

    fig = go.Figure(go.Bar(x=v_plot, y=l_plot, orientation="h",
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(
        title=title, margin=dict(l=80, r=20, t=60, b=40),
        xaxis_title="Contribution", yaxis_title=""
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Rolling lines (multi-series)
# ──────────────────────────────────────────────────────────────────────────────
def rolling_lines(
    dates: Sequence,
    series_dict: Dict[str, ArrayLike],
    *,
    title: str = "Rolling Metrics",
) -> go.Figure:
    """
    Línea por cada serie (p.ej. vol 26/52, corr rolling, etc.).
    `series_dict` = {"label": array_like_alineada_con_dates, ...}
    """
    fig = go.Figure()
    for name, arr in series_dict.items():
        y = np.asarray(arr, dtype=float)
        fig.add_trace(go.Scatter(x=dates, y=y, mode="lines", name=name,
                                 hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap WebGL (para N grandes)
# ──────────────────────────────────────────────────────────────────────────────
def corr_heatmap_gl(
    Sigma_or_Corr: Union[np.ndarray, pl.DataFrame],
    labels: Optional[Sequence[str]] = None,
    *,
    is_cov: bool = True,
    order: HeatmapOrder = HeatmapOrder(),
    zlim: Tuple[float, float] = (-1.0, 1.0),
    title: str = "Correlation Heatmap (WebGL)",
) -> go.Figure:
    """
    Igual que corr_heatmap pero usando Heatmapgl (más rápido para N>200).
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
        data=go.Heatmapgl(
            z=Corr_ord, x=labels_ord, y=labels_ord,
            zmin=zlim[0], zmax=zlim[1],
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



def weights_path_gammas(
    Ws: np.ndarray, gammas: Sequence[float], labels: Sequence[str], *, topn: int = 20,
    title: str = "Weights path vs γ (top names)"
) -> go.Figure:
    """
    Cada fila de Ws corresponde a un γ. Dibuja trayectorias de pesos por γ (log10).
    Muestra las top 'topn' por max peso a lo largo del barrido.
    """
    if Ws.ndim != 2:
        raise ValueError("Ws must be 2D (n_gamma x N)")
    nG, N = Ws.shape
    g = np.asarray(gammas)
    x = np.log10(g)
    peak = np.max(Ws, axis=0)
    idx = np.argsort(peak)[::-1][:min(topn, N)]
    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=x, y=Ws[:, i], mode="lines", name=labels[i],
                                 hovertemplate="log10 γ=%{x:.2f}<br>w=%{y:.2%}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="log10(γ)", yaxis_title="Weight", yaxis_tickformat=".0%")
    return fig

def turnover_vs_gamma(
    Ws: np.ndarray, w_ref: np.ndarray, gammas: Sequence[float], *,
    title: str = "Turnover vs γ"
) -> go.Figure:
    """
    Curva L1/L2 turnover vs γ tomando Ws (n_gamma x N).
    """
    L1 = np.sum(np.abs(Ws - w_ref[None, :]), axis=1)
    L2 = np.sqrt(np.sum((Ws - w_ref[None, :]) ** 2, axis=1))
    x = np.log10(np.asarray(gammas))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=L1, mode="lines+markers", name="L1"))
    fig.add_trace(go.Scatter(x=x, y=L2, mode="lines+markers", name="L2"))
    fig.update_layout(title=title, xaxis_title="log10(γ)", yaxis_title="Turnover")
    return fig

def te_frontier(
    mu: np.ndarray, Sigma: np.ndarray, w_bench: np.ndarray, Ws: np.ndarray, *,
    title: str = "Tracking-Error Frontier"
) -> go.Figure:
    """
    Dibuja μ_p vs TE_p para una familia de carteras (Ws).
    TE(w) = sqrt( (w-w_b)' Σ (w-w_b) )
    """
    dW = Ws - w_bench[None, :]
    mu_p = Ws @ mu
    te = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dW, Sigma, dW), 0.0))
    fig = go.Figure(go.Scatter(x=te, y=mu_p, mode="markers+lines"))
    fig.update_layout(title=title, xaxis_title="Tracking Error", yaxis_title="Expected Return")
    return fig