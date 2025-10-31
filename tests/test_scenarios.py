import numpy as np

from portfolio.backtest.scenarios import (
    ScenarioConfig,
    ShockSpec,
    apply_shock,
    block_bootstrap_indices,
    historical_slice_returns,
    run_scenarios,
)


def test_apply_shock_basic(df_sample):
    s = ShockSpec(mean_shift=0.001, cov_scale=1.2, crash=(2, -0.02))
    out = apply_shock(df_sample, s)
    assert out.height == df_sample.height
    assert "A" in out.columns
    assert abs(float(out["A"][2])) > 0  # should be altered


def test_block_bootstrap_indices_properties():
    idx = block_bootstrap_indices(10, 3, 123)
    assert idx.shape == (10,)
    assert (idx >= 0).all() and (idx < 10).all()


def test_historical_slice_returns_inclusive(df_sample):
    sliced = historical_slice_returns(df_sample, "2024-01-02", "2024-01-04")
    assert sliced.height == 3
    assert sliced["date"][0].date().isoformat() == "2024-01-02"
    assert sliced["date"][-1].date().isoformat() == "2024-01-04"


def test_run_scenarios_smoke(df_sample, allocator_factory):
    cfg = ScenarioConfig(name="test_scenario", B=0, shock=None)
    bench = np.zeros(len([c for c in df_sample.columns if c != "date"]))
    result = run_scenarios(
        cfgs=[cfg],
        df_ret_wide=df_sample,
        allocator_factory=allocator_factory,
        lookback=3,
        rebalance_freq="1D",
        cost_bps=0.0,
        bench_weights=bench,
    )
    assert len(result) == 1
    assert "metrics" in result[0]
