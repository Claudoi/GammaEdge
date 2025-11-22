# tests/quick/test_plot_turnover.py

import plotly.graph_objects as go

from portfolio.viz.plot_utils import plot_turnover


def test_plot_turnover_basic() -> None:
    dates = [1, 2, 3]
    vals = [0.1, 0.2, 0.15]

    fig = plot_turnover(dates, vals, title="Test TO")

    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 3
    assert len(fig.data[0].y) == 3
