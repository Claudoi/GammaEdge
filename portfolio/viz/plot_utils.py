# Plotting utilities
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def corr_heatmap(Sigma: pd.DataFrame) -> go.Figure:
    # convierte a correlación
    d = np.sqrt(np.diag(Sigma.values))
    with np.errstate(invalid="ignore", divide="ignore"):
        Corr = Sigma.values / np.outer(d, d)
    Corr = np.nan_to_num(Corr, nan=0.0, posinf=0.0, neginf=0.0)

    # clustering para ordenar el heatmap (single-linkage sobre distancia 1-ρ)
    dist = np.sqrt(0.5 * (1 - Corr))
    Z = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(Z)
    labels = Sigma.columns.to_numpy()[order]
    C_ord = Corr[np.ix_(order, order)]

    fig = go.Figure(
        data=go.Heatmap(
            z=C_ord, x=labels, y=labels, zmin=-1, zmax=1, colorbar=dict(title="ρ")
        )
    )
    fig.update_layout(title="Correlation Heatmap (clustered)", xaxis_nticks=len(labels))
    return fig
