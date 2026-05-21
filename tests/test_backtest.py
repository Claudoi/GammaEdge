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

    dates = pl.date_range(start=date(2020, 1, 2), end=date(2021, 6, 30), interval="1d", eager=True)
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
    assert any(
        "gap" in r.message.lower() or "nan" in r.message.lower() for r in caplog.records
    ), "Expected warning about data gaps"


def test_backtest_all_valid_unchanged():
    """Backtest con datos completos no debe cambiar comportamiento."""
    dates = pl.date_range(start=date(2020, 1, 2), end=date(2021, 6, 30), interval="1d", eager=True)
    n = len(dates)
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "date": dates,
            "A": rng.normal(0.001, 0.02, n),
            "B": rng.normal(0.0005, 0.015, n),
        }
    )
    result = backtest_vectorized(
        df, lookback=60, rebalance_freq="1mo", allocator=_make_allocator_ew()
    )
    assert "error" not in result
    equity = result["equity"]
    eq_vals = np.asarray(equity, dtype=float)
    assert not np.isnan(eq_vals).any()


def test_backtest_lookback_zero_handles_gracefully():
    """EDGE-2: lookback=0 should either return an error dict or handle gracefully."""
    dates = pl.date_range(start=date(2020, 1, 2), end=date(2021, 1, 2), interval="1d", eager=True)
    n = len(dates)
    rng = np.random.default_rng(5)
    df = pl.DataFrame({"date": dates, "A": rng.normal(0, 0.01, n)})

    # lookback=0 → no history → must not crash silently
    result = backtest_vectorized(
        df, lookback=0, rebalance_freq="1mo", allocator=_make_allocator_ew()
    )
    # Either returns an error or returns a valid result (no NaN)
    assert result is not None
    if "error" not in result:
        equity = result["equity"]
        eq_vals = (
            equity["equity"].to_numpy()
            if hasattr(equity, "columns")
            else np.asarray(equity, dtype=float)
        )
        assert not np.isnan(eq_vals).any(), "equity should not contain NaN even with lookback=0"


def test_backtest_equity_starts_at_one():
    """Equity curve must start at 1.0 (normalized)."""
    dates = pl.date_range(start=date(2020, 1, 2), end=date(2022, 1, 2), interval="1d", eager=True)
    n = len(dates)
    rng = np.random.default_rng(99)
    df = pl.DataFrame(
        {
            "date": dates,
            "A": rng.normal(0.001, 0.01, n),
            "B": rng.normal(0.0005, 0.008, n),
        }
    )

    result = backtest_vectorized(
        df, lookback=120, rebalance_freq="1mo", allocator=_make_allocator_ew()
    )
    assert "error" not in result
    equity = result["equity"]
    eq_vals = (
        equity["equity"].to_numpy()
        if hasattr(equity, "columns")
        else np.asarray(equity, dtype=float)
    )
    # First equity value should be very close to 1.0 (within one period's return)
    assert abs(eq_vals[0] - 1.0) < 0.05, f"First equity value too far from 1.0: {eq_vals[0]}"
    assert eq_vals[0] > 0, f"First equity value must be positive: {eq_vals[0]}"
