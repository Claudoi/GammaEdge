# tests/attribution/test_engine.py
import polars as pl

from portfolio.attribution.engine import (
    AttributionResult,
    compute_portfolio_contributions,
)


def test_compute_portfolio_contributions_basic():
    # 2 fechas, 2 assets, pesos que suman 1
    weights = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "A": [0.6, 0.5],
            "B": [0.4, 0.5],
        }
    )
    returns = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "A": [0.01, -0.02],
            "B": [0.03, 0.04],
        }
    )

    res = compute_portfolio_contributions(
        weights=weights,
        returns=returns,
        method="generic",
    )

    # Tipo y metadatos básicos
    assert isinstance(res, AttributionResult)
    assert res.method == "generic"
    contribs = res.contributions

    # Forma y columnas esperadas
    assert contribs.shape == (2, 3)
    assert contribs.columns == ["date", "A", "B"]

    # Cálculo manual: contrib = peso * retorno
    expected_A = [0.6 * 0.01, 0.5 * -0.02]
    expected_B = [0.4 * 0.03, 0.5 * 0.04]

    assert contribs["A"].to_list() == expected_A
    assert contribs["B"].to_list() == expected_B

    # Meta básica
    assert res.meta is not None
    assert res.meta["asset_cols"] == ["A", "B"]
