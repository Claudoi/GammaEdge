# tests/attribution/test_integration.py
import polars as pl
import pytest

from portfolio.attribution import brinson, euler


def test_brinson_euler_integration():
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "asset": ["A", "B"],
            "w": [0.6, 0.4],
            "r": [0.02, -0.01],
        }
    )

    brinson_out = brinson.run_brinson_engine(df)
    euler_out = euler.run_euler_engine(df)

    assert "contribution" in brinson_out.columns
    # La suma debe igualar el retorno de cartera: 0.6*0.02 + 0.4*(-0.01) = 0.008
    assert euler_out["contribution"].sum() == pytest.approx(0.008, abs=1e-3)
