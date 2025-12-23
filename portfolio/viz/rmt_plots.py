import numpy as np
import plotly.graph_objects as go

from portfolio.optim.robust import marcenko_pastur_limits


def marcenko_pastur_pdf(
    var_eps: float, q: float, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Theoretical Marcenko-Pastur probability density function.
    q = T / N
    """
    lambda_min = var_eps * (1 - np.sqrt(1.0 / q)) ** 2
    lambda_max = var_eps * (1 + np.sqrt(1.0 / q)) ** 2

    ls = np.linspace(lambda_min, lambda_max, n_points)
    pdf = (q / (2 * np.pi * var_eps * ls)) * np.sqrt(
        np.maximum(0, (lambda_max - ls) * (ls - lambda_min))
    )
    return ls, pdf


def plot_eigenvalue_spectrum(
    Sigma: np.ndarray, T: int, N: int, title: str = "Eigenvalue Spectrum (RMT)"
) -> go.Figure:
    """
    Plots the histogram of empirical eigenvalues vs Theoretical Marcenko-Pastur PDF.
    """
    # 1. Empircal Eigenvalues of Correlation Matrix
    # S = Cov -> Corr
    S = np.asarray(Sigma, dtype=float)
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    vals = np.linalg.eigvalsh(C)

    # 2. Theoretical PDF
    q = float(T) / float(N)
    lambda_min, lambda_max = marcenko_pastur_limits(T, N, var_eps=1.0)

    x_pdf, y_pdf = marcenko_pastur_pdf(var_eps=1.0, q=q, n_points=100)

    fig = go.Figure()

    # Histogram of Empirical Evals
    fig.add_trace(
        go.Histogram(
            x=vals,
            histnorm="probability density",
            name="Empirical Eigenvalues",
            opacity=0.7,
            marker_color="#636EFA",
        )
    )

    # Theoretical PDF line
    fig.add_trace(
        go.Scatter(
            x=x_pdf,
            y=y_pdf,
            mode="lines",
            name="Marcenko-Pastur (Noise)",
            line=dict(color="#EF553B", width=3),
        )
    )

    # Cutoff line
    fig.add_vline(
        x=lambda_max,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text="Noise Cutoff",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Eigenvalue (λ)",
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(x=0.8, y=0.9),
        bargap=0.1,
    )
    return fig
