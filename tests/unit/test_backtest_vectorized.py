# tests/unit/test_backtest_vectorized.py
import numpy as np
import polars as pl

from portfolio.backtest.engine import backtest_rebalanced, backtest_vectorized


def simple_allocator(df):
    # Equal Weight allocator
    # df es (Time, N+1) con date
    cols = [c for c in df.columns if c != "date"]
    N = len(cols)
    return np.ones(N) / N


def test_vectorized_vs_classic_match():
    # Crear datos sinteticos
    dates = [f"2023-01-{i:02d}" for i in range(1, 32)] + [f"2023-02-{i:02d}" for i in range(1, 29)]
    dates = sorted(dates)

    # 2 activos: Uno sube constante, otro baja constante
    reits_a = np.linspace(100, 110, len(dates))  # +10%
    reits_b = np.linspace(100, 90, len(dates))  # -10%

    df_prices = pl.DataFrame({"date": dates, "A": reits_a, "B": reits_b}).with_columns(
        pl.col("date").str.to_date()
    )

    # Calcular retornos
    df_ret = df_prices.select(
        [
            pl.col("date"),
            (pl.col("A") / pl.col("A").shift(1) - 1).alias("A"),
            (pl.col("B") / pl.col("B").shift(1) - 1).alias("B"),
        ]
    ).drop_nulls()

    # Parametros
    lookback = 10
    cfg = {"rebalance_freq": "1w", "lookback": lookback, "cost_bps": 0.0}

    # Run Classic
    res_classic = backtest_rebalanced(df_ret, allocator=simple_allocator, **cfg)

    # Run Vectorized
    res_vec = backtest_vectorized(df_ret, allocator=simple_allocator, **cfg)

    # Compare Final Equity
    eq_c = res_classic["equity"][-1]
    eq_v = res_vec["equity"][-1]

    # Tolerancia: Vectorized usa aproximaciones de bloque vs Classic dia a dia?
    # En teoria con fees=0 y logica alineada deberian ser identicos.
    # Pero el manejo de fronteras de rebalanceo puede variar en 1 dia.

    assert abs(eq_c - eq_v) / eq_c < 0.01, f"Classic {eq_c} != Vectorized {eq_v}"

    # Compare Turnover
    # Classic engine (legacy) NO drifta los pesos diariamente (Constant Mix implícito),
    # por lo que si el allocator da pesos fijos, el turnover sale 0.
    # El Vectorized SI drifta (Buy & Hold), por lo que DEBE haber turnover al rebalancear.

    # Assert Vectorized behaves correctly (Turnover > 0 due to price drift)
    to_v = res_vec["turnover"]["turnover"].sum()
    assert to_v > 0.5, f"Vectorized should have turnover due to drift, got {to_v}"

    # Assert similarity in equity (Drift impact is small in short horizon)
    assert abs(eq_c - eq_v) / eq_c < 0.02
