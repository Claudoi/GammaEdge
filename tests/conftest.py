from datetime import date

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def df_sample():
    dates = pl.date_range(start=date(2024, 1, 1), end=date(2024, 1, 5), interval="1d", eager=True)
    return pl.DataFrame(
        {
            "date": dates,
            "A": [0.01, 0.02, -0.01, 0.0, 0.03],
            "B": [0.00, -0.01, 0.02, 0.01, 0.0],
        }
    )


@pytest.fixture
def allocator_factory():
    def factory():
        def alloc(df_ret_wide: pl.DataFrame) -> np.ndarray:
            n_assets = len([c for c in df_ret_wide.columns if c != "date"])
            return np.full(n_assets, 1.0 / n_assets)

        return alloc

    return factory
