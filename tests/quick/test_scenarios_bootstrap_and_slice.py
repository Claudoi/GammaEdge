import numpy as np
import polars as pl

from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest.scenarios import (
    ScenarioConfig,
    ShockSpec,
    block_bootstrap_indices,
    historical_slice_returns,
    run_scenarios,
)


def test_block_bootstrap_indices_basic():
    T = 30
    block = 5
    idx = block_bootstrap_indices(T, block, seed=123)
    assert isinstance(idx, np.ndarray)
    assert idx.dtype == int
    assert len(idx) == T
    assert np.all(idx >= 0) and np.all(idx < T)


def test_historical_slice_inclusive_and_datetime():
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "A": np.linspace(0, 0.009, 10),
            "B": np.linspace(0, -0.009, 10),
        }
    )
    out = historical_slice_returns(df, "2024-01-03", "2024-01-05")
    assert out.height == 3
    # Datetime y límites inclusivos
    assert out.schema["date"] in (
        pl.Datetime,
        pl.Datetime("us"),
        pl.Datetime("ms"),
        pl.Datetime("ns"),
    )
    assert out["date"][0].date().isoformat() == "2024-01-03"
    assert out["date"][-1].date().isoformat() == "2024-01-05"


def _make_synthetic_returns_for_scenarios(
    n_days: int = 252, tickers: list[str] | None = None
) -> pl.DataFrame:
    """Small synthetic return panel for scenario tests."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(123)
    dates = np.arange(n_days, dtype="int64")
    df = pl.DataFrame({"date": dates})
    for tk in tickers:
        r = rng.normal(loc=0.0005, scale=0.01, size=n_days)
        df = df.with_columns(pl.Series(tk, r))
    # Cast date to Datetime to mimic real pipeline
    return df.with_columns(pl.col("date").cast(pl.Datetime))


def _equal_weight_allocator(df_ret_wide: pl.DataFrame):
    tickers = [c for c in df_ret_wide.columns if c != "date"]

    def alloc(win: pl.DataFrame) -> np.ndarray:
        n = len(tickers)
        if n == 0:
            return np.zeros(0, dtype=float)
        w = np.ones(n, dtype=float) / float(n)
        return w

    return alloc, tickers


def _cagr_from_equity(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return float("nan")
    r = equity[1:] / equity[:-1] - 1.0
    if r.size == 0:
        return float("nan")
    gross = float(np.prod(1.0 + r))
    return gross ** (252.0 / max(r.size, 1)) - 1.0


def test_run_scenarios_mean_shift_increases_cagr():
    """Positive mean_shift should not reduce CAGR vs baseline (within small tolerance)."""
    df = _make_synthetic_returns_for_scenarios()
    alloc, tickers = _equal_weight_allocator(df)
    n_assets = len(tickers)

    # Baseline backtest
    bt_base = backtest_rebalanced(
        df_ret_wide=df,
        lookback=60,
        rebalance_freq="1mo",
        cost_bps=0.0,
        allocator=alloc,
        bench_weights=np.full(n_assets, 1.0 / max(n_assets, 1)),
    )
    cagr_base = _cagr_from_equity(np.asarray(bt_base["equity"], dtype=float))

    # Scenario with positive mean_shift
    cfg = ScenarioConfig(
        name="MeanShiftUp",
        B=0,
        block=10,
        seed=42,
        shock=ShockSpec(mean_shift=0.001, cov_scale=1.0, crash=None),
    )

    results = run_scenarios(
        [cfg],
        df_ret_wide=df,
        allocator_factory=lambda: alloc,
        lookback=60,
        rebalance_freq="1mo",
        cost_bps=0.0,
        bench_weights=np.full(n_assets, 1.0 / max(n_assets, 1)),
    )

    assert len(results) == 1
    bt_scen = results[0].get("bt", {})
    cagr_scen = _cagr_from_equity(np.asarray(bt_scen.get("equity", []), dtype=float))

    # Allow small numerical noise but CAGR with positive drift should not be worse
    assert np.isfinite(cagr_base)
    assert np.isfinite(cagr_scen)
    assert cagr_scen >= cagr_base - 1e-4


def _maxdd_from_equity(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return float("nan")
    r = equity[1:] / equity[:-1] - 1.0
    if r.size == 0:
        return float("nan")
    eq_path = np.cumprod(1.0 + r)
    dd = 1.0 - eq_path / np.maximum.accumulate(eq_path)
    return float(np.max(dd))


def test_run_scenarios_crash_increases_max_drawdown():
    """A one-day negative crash should increase MaxDD vs baseline."""
    df = _make_synthetic_returns_for_scenarios()
    alloc, tickers = _equal_weight_allocator(df)
    n_assets = len(tickers)

    # Baseline
    bt_base = backtest_rebalanced(
        df_ret_wide=df,
        lookback=60,
        rebalance_freq="1mo",
        cost_bps=0.0,
        allocator=alloc,
        bench_weights=np.full(n_assets, 1.0 / max(n_assets, 1)),
    )
    m_base = bt_metrics.compute_backtest_metrics(bt_base)
    # be robust: if metrics DF schema changes, fall back to direct MaxDD
    if isinstance(m_base, pl.DataFrame) and "MaxDD" in m_base.columns:
        maxdd_base = float(m_base.get_column("MaxDD")[0])
    else:
        maxdd_base = _maxdd_from_equity(np.asarray(bt_base["equity"], dtype=float))

    # Scenario with a strong crash
    crash_day = 10
    crash_drop = -0.10  # -10%
    cfg = ScenarioConfig(
        name="CrashDown",
        B=0,
        block=10,
        seed=42,
        shock=ShockSpec(mean_shift=None, cov_scale=1.0, crash=(crash_day, crash_drop)),
    )

    results = run_scenarios(
        [cfg],
        df_ret_wide=df,
        allocator_factory=lambda: alloc,
        lookback=60,
        rebalance_freq="1mo",
        cost_bps=0.0,
        bench_weights=np.full(n_assets, 1.0 / max(n_assets, 1)),
    )

    assert len(results) == 1
    bt_scen = results[0].get("bt", {})
    m_scen = bt_metrics.compute_backtest_metrics(bt_scen)
    if isinstance(m_scen, pl.DataFrame) and "MaxDD" in m_scen.columns:
        maxdd_scen = float(m_scen.get_column("MaxDD")[0])
    else:
        maxdd_scen = _maxdd_from_equity(np.asarray(bt_scen.get("equity", []), dtype=float))

    assert np.isfinite(maxdd_base)
    assert np.isfinite(maxdd_scen)
    # The crash scenario should have worse (higher) max drawdown
    assert maxdd_scen >= maxdd_base - 1e-4
