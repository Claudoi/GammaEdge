# Return calculations
from __future__ import annotations
import numpy as np
import pandas as pd

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change().dropna(how="all")

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.sort_index()).diff().dropna(how="all")

def to_frequency(returns: pd.DataFrame, freq: str) -> pd.DataFrame:
    if returns.empty:
        return returns
    # Composición correcta: log → suma; simple → producto - 1
    is_log_like = (returns.abs().median() > 0.25).any()
    if is_log_like:
        out = returns.resample(freq).sum(min_count=1)
    else:
        out = (1.0 + returns).resample(freq).prod(min_count=1) - 1.0
    return out.dropna(how="all")

def winsorize(returns: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Recorta colas por percentiles (default 1%)."""
    lo = returns.quantile(q)
    hi = returns.quantile(1 - q)
    return returns.clip(lower=lo, upper=hi, axis=1)

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    miss = df.isna().sum()
    pct = (miss / total * 100.0).round(2)
    last = df.ffill().iloc[-1].isna()
    return pd.DataFrame({"missing_rows": miss, "missing_pct": pct, "ends_missing": last})
