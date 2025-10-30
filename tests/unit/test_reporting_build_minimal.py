# tests/unit/test_reporting_build_minimal.py
from __future__ import annotations

from datetime import datetime as dt

import numpy as np
import polars as pl

from portfolio.backtest.reporting import build_backtest_report


def test_build_report_minimal_smoke():
    T = 5
    dates = [dt(2024, 1, d) for d in range(1, T + 1)]
    df_ret = pl.DataFrame(
        {
            "date": dates,
            "A": [0.0, 0.01, -0.005, 0.002, 0.0],
            "B": [0.0, 0.00, 0.003, 0.001, 0.0],
            "C": [0.0, -0.02, 0.004, 0.000, 0.0],
        }
    )
    equity = np.array([1.0, 1.01, 1.005, 1.007, 1.007], dtype=float)
    Wd = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (T, 1))

    report = build_backtest_report(
        df_ret_wide=df_ret,
        daily_weights=Wd,
        equity=equity,
        group_map={"A": "G1", "B": "G1", "C": "G2"},
    )

    # sanity: figuras y tablas clave existen
    assert "equity" in report.figures and "drawdown" in report.figures
    assert "weights" in report.figures and "top_contrib" in report.figures
    assert "contrib_asset_total" in report.tables
    assert "contrib_group_total" in report.tables
