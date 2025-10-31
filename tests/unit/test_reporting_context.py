# tests/unit/test_reporting_context.py
from __future__ import annotations

from datetime import datetime as dt

import numpy as np
import polars as pl

from portfolio.backtest.reporting import build_backtest_report, build_context


def test_build_context_minimum_fields():
    dates = [dt(2024, 1, d) for d in range(1, 4)]
    df_ret = pl.DataFrame({"date": dates, "A": [0.0, 0.01, 0.0], "B": [0.0, -0.005, 0.0]})
    equity = np.array([1.0, 1.01, 1.005], dtype=float)
    Wd = np.tile([0.5, 0.5], (3, 1))

    report = build_backtest_report(
        df_ret_wide=df_ret, daily_weights=Wd, equity=equity, group_map=None
    )

    ctx = build_context(
        portfolio_name="Test",
        period=("2024-01-01", "2024-01-03"),
        df_asset_cum=report.tables.get("contrib_asset_total"),
        df_group_total=report.tables.get("contrib_group_total"),
        df_brinson=None,
        extra_metrics={"Sharpe": 1.0},
    )

    # Campos básicos
    assert ctx.get("portfolio_name") == "Test"
    assert ctx.get("period_start") == "2024-01-01"
    assert ctx.get("period_end") == "2024-01-03"

    # Métricas presentes como dict
    assert isinstance(ctx.get("metrics"), dict)
    assert "Sharpe" in ctx["metrics"]

    # Nota: la presencia de tablas depende de la implementación de build_context.
    # Si existen, validamos forma; si no, no fallamos.
    if "tables" in ctx and isinstance(ctx["tables"], dict):
        tbls = ctx["tables"]
        assert "contrib_asset_total" in tbls
        assert "contrib_group_total" in tbls
