# tests/unit/test_reporting_render_html_no_figs.py
from __future__ import annotations

from datetime import datetime as dt

import numpy as np
import polars as pl

from portfolio.backtest.reporting import (
    build_backtest_report,
    build_context,
    render_html,
)


def test_render_html_with_tables_and_no_figures():
    # Datos mínimos coherentes
    dates = [dt(2024, 1, d) for d in (1, 2, 3)]
    df_ret = pl.DataFrame({"date": dates, "A": [0.0, 0.01, 0.0], "B": [0.0, -0.005, 0.0]})
    equity = np.array([1.0, 1.01, 1.005], dtype=float)
    Wd = np.tile([0.5, 0.5], (3, 1))

    # Construye report para obtener tablas agregadas
    report = build_backtest_report(
        df_ret_wide=df_ret, daily_weights=Wd, equity=equity, group_map=None
    )

    ctx = build_context(
        portfolio_name="Smoke",
        period=("2024-01-01", "2024-01-03"),
        df_asset_cum=report.tables.get("contrib_asset_total"),
        df_group_total=report.tables.get("contrib_group_total"),
        df_brinson=None,
        extra_metrics={"Sharpe": 1.0},
    )

    # Sin figuras -> debe devolver HTML válido con cabeceras/metricas/tablas
    html = render_html(ctx, figures=[], page_title="T", h1="H")
    assert isinstance(html, (bytes, bytearray))
    body = html.decode("utf-8", errors="ignore")
    assert "Smoke" in body
    assert "Sharpe" in body
    assert "Backtest" in body or "Contributors" in body
