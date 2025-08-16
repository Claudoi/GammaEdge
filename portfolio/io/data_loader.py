# Data loading utilities
from __future__ import annotations
from typing import Iterable, List, Dict, Any
import pandas as pd
import yfinance as yf
from .cache import load_df, save_df

def get_prices(
    tickers: Iterable[str], start, end, adjust: bool = True, force_refresh: bool = False
) -> pd.DataFrame:
    tks: List[str] = [str(t).upper() for t in tickers if str(t).strip()]
    cfg: Dict[str, Any] = {"tickers": ",".join(sorted(tks)), "start": str(start), "end": str(end), "adj": adjust}
    if not force_refresh:
        cached = load_df("prices", cfg)
        if cached is not None:
            return cached

    df = yf.download(tks, start=start, end=end, auto_adjust=adjust, progress=False)["Close"]
    df = df.dropna(how="all")
    save_df("prices", cfg, df)
    return df
