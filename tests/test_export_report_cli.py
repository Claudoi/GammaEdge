from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from portfolio.backtest.reporting import build_backtest_report


def test_build_report_minimal(tmp_path):
    T, N = 30, 4
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(T)]
    eq = (1.0 + 0.001 * np.arange(T)).tolist()
    tickers = [f"T{i}" for i in range(N)]

    df_ret = pl.DataFrame({"date": dates, **{t: [0.001] * T for t in tickers}})
    W = np.tile(np.full(N, 1.0 / N), (T, 1))

    rpt = build_backtest_report(
        df_ret_wide=df_ret,
        daily_weights=W,
        equity=np.array(eq, dtype=float),
        group_map={t: ("A" if i % 2 == 0 else "B") for i, t in enumerate(tickers)},
        title="Test Report",
    )
    assert "equity" in rpt.figures and "drawdown" in rpt.figures
    assert "weights" in rpt.figures and "top_contrib" in rpt.figures
    assert "contrib_asset_total" in rpt.tables
    assert "contrib_group_total" in rpt.tables
