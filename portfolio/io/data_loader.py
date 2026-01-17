# portfolio/io/data_loader.py
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import polars as pl
import yfinance as yf

from portfolio.core.compat import dataclass_compat as dataclass

from .cache import load_df, save_df

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True, slots=True)
class PriceLoadConfig:
    tickers: tuple[str, ...]
    start: str
    end: str
    interval: str
    adjust: bool
    schema_version: str = "prices_v1"

    def cache_key(self) -> dict[str, Any]:
        return {
            "tickers": ",".join(self.tickers),
            "start": self.start,
            "end": self.end,
            "interval": self.interval,
            "adjust": self.adjust,
            "schema": self.schema_version,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    tks = tuple(sorted({str(t).strip().upper() for t in tickers if str(t).strip()}))
    if not tks:
        raise ValueError("No tickers provided after normalization.")
    return tks


def _to_polars_long_from_pandas_close(
    df_close: pd.DataFrame | pd.Series,
) -> pl.DataFrame:
    """
    Espera un DataFrame de pandas con columnas = tickers y una columna temporal 'Date' en el índice
    o ya reseteada como 'date'. Devuelve Polars largo: [date, ticker, price].
    """
    if not isinstance(df_close, pd.DataFrame):
        # single-ticker puede venir como Series
        df_close = df_close.to_frame()

    if "Date" in df_close.columns:
        df_close = df_close.rename(columns={"Date": "date"})
    if "date" not in df_close.columns:
        df_close = df_close.reset_index().rename(columns={"Date": "date"})

    # Asegurar dtype tiempo y float
    df_close["date"] = pd.to_datetime(df_close["date"], utc=False)
    for c in df_close.columns:
        if c != "date":
            df_close[c] = pd.to_numeric(df_close[c], errors="coerce")

    pl_df = pl.from_pandas(df_close)

    # melt → largo
    value_cols = [c for c in pl_df.columns if c != "date"]
    df_long = (
        pl_df.melt(
            id_vars=["date"], value_vars=value_cols, variable_name="ticker", value_name="price"
        )
        .drop_nulls()
        .with_columns(
            [
                pl.col("date").cast(pl.Datetime),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("price").cast(pl.Float64),
            ]
        )
        .sort(["ticker", "date"])
    )
    return df_long


def _fetch_yfinance_close(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    interval: str,
    adjust: bool,
) -> pd.DataFrame:
    """
    Descarga precios con yfinance y devuelve un pandas.DataFrame con columnas = tickers (Close).
    Maneja edge-cases de índice y columnas jerárquicas.
    """
    # Nota: group_by='ticker' hace más predecible el slicing 'Close'
    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=adjust,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    # Asegura DataFrame (yfinance puede devolver Series en algunos casos)
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Casos:
    # 1) Múltiples tickers → df["Close"] es DataFrame columnas=tickers
    # 2) Un solo ticker    → df["Close"] es Series; convertimos a DataFrame
    # 3) A veces ya no hay "Close": si el slicing falla, intentamos localizarlo
    try:
        close_obj = df["Close"]
        if isinstance(close_obj, pd.Series):
            close_df = close_obj.to_frame(name=tickers[0])
        else:
            close_df = close_obj  # ya es DataFrame
    except Exception:  # pragma: no cover
        # fallback robusto: buscar columnas que terminen en 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df.xs("Close", axis=1, level=-1, drop_level=True)
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=tickers[0])
        else:
            # último recurso: si ya viene en forma simple, asumimos que son cierres
            close_df = df if isinstance(df, pd.DataFrame) else df.to_frame()

    # Limpieza básica
    close_df = close_df.dropna(how="all")
    close_df.index.name = "Date"
    return close_df.reset_index()


def _fetch_yfinance_ohlcv(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Descarga datos OHLCV completos con yfinance (ajustados por dividendos/splits).
    Devuelve un DataFrame en formato largo: [date, ticker, open, high, low, close, volume].
    """
    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,  # Siempre ajustado para métricas cuantitativas
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    rows = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                # Solo un ticker: columnas son directas (Open, High, Low, Close, Volume)
                ticker_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                # Múltiples tickers: MultiIndex (ticker, field)
                ticker_df = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()

            ticker_df = ticker_df.reset_index()
            ticker_df.columns = ["date", "open", "high", "low", "close", "volume"]
            ticker_df["ticker"] = ticker
            rows.append(ticker_df)
        except KeyError:
            logger.warning("Ticker %s not found in download; skipping.", ticker)
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], utc=False)
    result = result[["date", "ticker", "open", "high", "low", "close", "volume"]]
    return result.dropna(subset=["close"])


def _validate_long_prices(df_long: pl.DataFrame) -> None:
    """
    Valida condiciones mínimas: no duplicados, fechas crecientes por ticker.
    Lanza ValueError si hay inconsistencias graves.
    """
    # Duplicados exactos
    total_rows = df_long.height
    unique_rows = df_long.unique(subset=["date", "ticker", "price"]).height
    dups = total_rows - unique_rows
    if dups > 0:
        logger.warning("Found %d duplicate rows in [date,ticker,price]; removing duplicates.", dups)

    # Monotonicidad de fechas por ticker
    sample = (
        df_long.sort(["ticker", "date"])
        .group_by("ticker")
        .agg((pl.col("date").diff().dt.total_milliseconds() < 0).sum().alias("backwards_steps"))
    )
    backwards_total = int(sample["backwards_steps"].sum())
    if backwards_total > 0:
        raise ValueError(f"Detected {backwards_total} non-monotonic date steps across tickers.")

    # Precios no positivos
    negs = df_long.filter(pl.col("price") <= 0).height
    if negs > 0:
        logger.warning("Detected %d non-positive prices; check data quality.", negs)


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────


def get_prices_long(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    interval: str = "1d",
    adjust: bool = True,
    force_refresh: bool = False,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Carga precios de cierre ajustados y devuelve un DataFrame **Polars** en formato largo.
    """
    t0 = time.perf_counter()
    tks = _normalize_tickers(tickers)
    cfg = PriceLoadConfig(
        tickers=tks, start=str(start), end=str(end), interval=interval, adjust=adjust
    )
    cache_key = cfg.cache_key()

    if use_cache and not force_refresh:
        cached = load_df("prices_long", cache_key)  # probablemente devuelve pandas
        if cached is not None:
            try:
                pl_cached = pl.from_pandas(cached)
                # Validar esquema mínimo
                if set(pl_cached.columns) >= {"date", "ticker", "price"}:
                    logger.info("Loaded prices_long from cache: %s", cache_key)
                    _validate_long_prices(pl_cached)
                    logger.info("get_prices_long OK (cache) in %.3fs", time.perf_counter() - t0)
                    return pl_cached
            except Exception as e:  # pragma: no cover
                logger.warning("Cache exists but couldn't be parsed to Polars: %s", e)

    # Descarga
    raw_close_pd = _fetch_yfinance_close(tks, cfg.start, cfg.end, cfg.interval, cfg.adjust)
    df_long = _to_polars_long_from_pandas_close(raw_close_pd)

    # Validación y limpieza final (elimina duplicados exactos si existieran)
    df_long = df_long.unique(subset=["date", "ticker"], keep="last")
    _validate_long_prices(df_long)

    # Guardar en caché como pandas para no romper compatibilidad
    if use_cache:
        try:
            save_df("prices_long", cache_key, df_long.to_pandas())
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to save prices_long to cache: %s", e)

    logger.info("get_prices_long OK (fresh) in %.3fs", time.perf_counter() - t0)
    return df_long


def get_prices_wide(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    interval: str = "1d",
    adjust: bool = True,
    force_refresh: bool = False,
    use_cache: bool = True,
    as_pandas: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """
    Igual que `get_prices_long` pero devuelve una matriz **ancha** (columnas por ticker).
    """
    df_long = get_prices_long(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        adjust=adjust,
        force_refresh=force_refresh,
        use_cache=use_cache,
    )

    df_wide = df_long.pivot(values="price", index="date", on="ticker").sort("date")

    if as_pandas:
        return df_wide.to_pandas()
    return df_wide


def get_ohlcv_long(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    interval: str = "1d",
) -> pl.DataFrame:
    """
    Descarga datos OHLCV completos (ajustados por dividendos/splits) en formato largo.

    Devuelve un DataFrame Polars con columnas:
    - date: Datetime
    - ticker: Utf8
    - open, high, low, close: Float64 (ajustados por dividendos)
    - volume: Float64

    Args:
        tickers: Símbolos de acciones
        start: Fecha inicio (YYYY-MM-DD)
        end: Fecha fin (YYYY-MM-DD)
        interval: Frecuencia de datos ('1d', '1w', etc.)

    Returns:
        DataFrame Polars en formato largo ordenado por [ticker, date]
    """
    t0 = time.perf_counter()
    tks = _normalize_tickers(tickers)

    # Descarga via yfinance
    raw_df = _fetch_yfinance_ohlcv(tks, start, end, interval)

    if raw_df.empty:
        logger.warning("No OHLCV data returned for tickers: %s", tks)
        return pl.DataFrame(
            schema={
                "date": pl.Datetime,
                "ticker": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    # Convertir a Polars
    pl_df = (
        pl.from_pandas(raw_df)
        .with_columns(
            [
                pl.col("date").cast(pl.Datetime),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        .sort(["ticker", "date"])
    )

    logger.info(
        "get_ohlcv_long OK: %d rows, %d tickers in %.3fs",
        pl_df.height,
        pl_df.select(pl.col("ticker").n_unique()).item(),
        time.perf_counter() - t0,
    )
    return pl_df
