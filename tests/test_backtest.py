# Test for backtest engine
from datetime import date

import numpy as np
import polars as pl

from portfolio.backtest.engine import backtest_vectorized


def _make_allocator_ew():
    def alloc(df: pl.DataFrame) -> np.ndarray:
        n = len([c for c in df.columns if c != "date"])
        return np.full(n, 1.0 / n)
    return alloc


def test_backtest_nan_block_raises_warning(caplog):
    """CRITICAL-3: backtest con columna completamente NaN debe emitir warning
    y truncar al primer date donde todos los tickers tienen datos."""
    import logging
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2021, 6, 30), interval="1d", eager=True
    )
    n = len(dates)
    rng = np.random.default_rng(42)
    aapl = rng.normal(0.001, 0.02, n)
    # TSLA data starts 6 months in — first 126 rows are NaN
    tsla = np.full(n, np.nan)
    tsla[126:] = rng.normal(0.0008, 0.025, n - 126)

    df = pl.DataFrame({"date": dates, "AAPL": aapl, "TSLA": tsla})

    with caplog.at_level(logging.WARNING, logger="portfolio.backtest.engine"):
        result = backtest_vectorized(
            df,
            lookback=60,
            rebalance_freq="1mo",
            allocator=_make_allocator_ew(),
        )

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    equity = result["equity"]
    eq_vals = np.asarray(equity, dtype=float)
    assert not np.isnan(eq_vals).any(), "Equity curve contains NaN — NaN block not handled"
    assert any("gap" in r.message.lower() or "nan" in r.message.lower()
               for r in caplog.records), "Expected warning about data gaps"


def test_backtest_all_valid_unchanged():
    """Backtest con datos completos no debe cambiar comportamiento."""
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2021, 6, 30), interval="1d", eager=True
    )
    n = len(dates)
    rng = np.random.default_rng(0)
    df = pl.DataFrame({
        "date": dates,
        "A": rng.normal(0.001, 0.02, n),
        "B": rng.normal(0.0005, 0.015, n),
    })
    result = backtest_vectorized(
        df, lookback=60, rebalance_freq="1mo", allocator=_make_allocator_ew()
    )
    assert "error" not in result
    equity = result["equity"]
    eq_vals = np.asarray(equity, dtype=float)
    assert not np.isnan(eq_vals).any()
