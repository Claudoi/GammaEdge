# tests/unit/test_viz_attribution_plots.py

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from portfolio.viz.plot_utils import (
    plot_brinson_group_bar,
    plot_brinson_timeseries,
    plot_euler_contributions,
)

# -----------------------------
# Pequeños helpers de datos
# -----------------------------


def _sample_brinson_by_group_pd() -> pd.DataFrame:
    # DataFrame "por grupo" con todas las columnas clásicas
    return pd.DataFrame(
        {
            "group": ["A", "B", "C"],
            "alloc": [0.10, 0.05, -0.02],
            "select": [0.30, -0.10, 0.15],
            "interact": [0.00, 0.02, -0.01],
            "total": [0.40, -0.03, 0.12],
        }
    )


def _sample_brinson_by_group_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group_id": [0, 1],
            "alloc": [0.3, 0.7],
            "select": [1.1, 1.5],
            "interact": [0.0, 0.3],
            "total": [1.4, 1.9],
        }
    )


def _sample_brinson_timeseries() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "alloc": [0.1, 0.2, 0.05, 0.0],
            "select": [0.5, 0.6, 0.4, 0.2],
            "interact": [0.0, 0.0, 0.1, 0.0],
            "total": [0.6, 0.8, 0.55, 0.2],
        }
    )


def _sample_euler_contrib_df() -> pd.DataFrame:
    # Nota: la función ordena por |contribución| desc.
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "risk_contribution": [0.01, -0.03, 0.02],
        }
    )


# -----------------------------
# Tests Brinson group bar
# -----------------------------


def test_plot_brinson_group_bar_basic_pandas():
    df = _sample_brinson_by_group_pd()
    fig = plot_brinson_group_bar(df, title="Brinson Group")

    assert isinstance(fig, go.Figure)
    # Debe haber una barra por cada métrica presente
    # (alloc, select, interact, total) => 4 traces
    assert len(fig.data) == 4
    names = {tr.name for tr in fig.data}
    assert names == {"Alloc", "Select", "Interact", "Total"}

    # Eje x coincide con los grupos stringificados del DataFrame
    xvals = list(fig.data[0].x)
    assert xvals == list(df["group"].astype(str))


def test_plot_brinson_group_bar_polars_group_id():
    df = _sample_brinson_by_group_pl()
    fig = plot_brinson_group_bar(df, title="Brinson Group")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # x debe provenir de group_id convertido a str
    xvals = list(fig.data[0].x)
    assert xvals == [str(g) for g in df.get_column("group_id").to_list()]


# -----------------------------
# Tests Brinson timeseries
# -----------------------------


def test_plot_brinson_timeseries_all_components():
    df = _sample_brinson_timeseries()
    fig = plot_brinson_timeseries(df, title="Brinson TS")

    assert isinstance(fig, go.Figure)
    # 4 componentes: alloc, select, interact, total
    assert len(fig.data) == 4

    # Fechas ordenadas ascendente
    xs = list(fig.data[0].x)
    assert xs == sorted(xs)


def test_plot_brinson_timeseries_one_metric():
    df = _sample_brinson_timeseries()
    fig = plot_brinson_timeseries(df, title="Brinson TS (total)", metric="total")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "Total"


# -----------------------------
# Tests Euler contributions
# -----------------------------


def test_plot_euler_contributions_basic_df():
    df = _sample_euler_contrib_df()
    fig = plot_euler_contributions(df, title="Euler RC")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert isinstance(bar, go.Bar)

    # Orden esperado: por |contribución| desc => B (0.03), C (0.02), A (0.01)
    assert list(bar.x) == ["B", "C", "A"]
    assert list(bar.y) == [-0.03, 0.02, 0.01]


def test_plot_euler_contributions_topn():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "risk_contribution": [0.01, -0.03, 0.02, 0.10],
        }
    )
    fig = plot_euler_contributions(df, title="Euler RC top2", top_n=2)

    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    # Top2 por |contribución| => D (0.10) y B (0.03)
    assert list(bar.x) == ["D", "B"]
    assert list(bar.y) == [0.10, -0.03]


def test_plot_euler_contributions_series_input():
    s = pd.Series([0.02, -0.01, 0.03], index=["X", "Y", "Z"], name="risk_contribution")
    fig = plot_euler_contributions(s, title="Euler RC series")

    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    # Orden por |contribución| desc => Z (0.03), X (0.02), Y (0.01)
    assert list(bar.x) == ["Z", "X", "Y"]
    assert list(bar.y) == [0.03, 0.02, -0.01]
