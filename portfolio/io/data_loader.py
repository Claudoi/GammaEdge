# portfolio/io/data_loader.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

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
    schema_version: str = "prices_v1"   # bump si cambias el esquema

    def cache_key(self) -> dict[str, Any]:
        # clave determinista para cache
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


def _to_polars_long_from_pandas_close(df_close: pd.DataFrame) -> pl.DataFrame:
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
        pl_df.melt(id_vars=["date"], value_vars=value_cols, variable_name="ticker", value_name="price")
             .drop_nulls()
             .with_columns([
                 pl.col("date").cast(pl.Datetime),
                 pl.col("ticker").cast(pl.Utf8),
                 pl.col("price").cast(pl.Float64),
             ])
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

    # Casos:
    # 1) Múltiples tickers → df["Close"] es DataFrame columnas=tickers
    # 2) Un solo ticker   → df["Close"] es Series; convertimos a DataFrame
    # 3) A veces ya no hay "Close": si el slicing falla, intentamos localizarlo
    try:
        close = df["Close"]
        if isinstance(close, pd.Series):
            # Nombre de la serie puede ser 'Close'; necesitamos el ticker real
            # Si es un ticker único, podemos usar tickers[0]
            close = close.to_frame(name=tickers[0])
    except Exception:  # pragma: no cover
        # fallback robusto: buscar columnas que terminen en 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            close = df.xs("Close", axis=1, level=-1, drop_level=True)
        else:
            # último recurso: si ya viene en forma simple, asumimos que son cierres
            close = df

    # Limpieza básica
    close = close.dropna(how="all")
    close.index.name = "Date"
    return close.reset_index()


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
               .agg(
                   (pl.col("date").diff().dt.total_milliseconds() < 0)
                   .sum()
                   .alias("backwards_steps")
               )
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

    Parameters
    ----------
    tickers : Iterable[str]
        Lista/iterable de símbolos (se normalizan a mayúsculas y se deduplican).
    start, end : str
        Fechas (YYYY-MM-DD o ISO). Se pasan tal cual a yfinance.
    interval : str
        '1d', '1wk', '1mo', etc.
    adjust : bool
        Usa precios ajustados (dividendos/splits). Recomendado True.
    force_refresh : bool
        Ignora caché y re-descarga.
    use_cache : bool
        Usa caché (load_df/save_df) con clave determinista.

    Returns
    -------
    pl.DataFrame
        Columnas: ['date' (Datetime), 'ticker' (Utf8), 'price' (Float64)]
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

    # Validación y limpieza final
    # (elimina duplicados exactos si existieran)
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

    Returns
    -------
    pl.DataFrame | pd.DataFrame
        Si `as_pandas=True`, devuelve pandas (ancho) para compatibilidad con librerías legacy.
        Por defecto devuelve Polars (ancho) con columnas: ['date', T1, T2, ...]
    """
    cfg = {
        "interval": interval,
        "adjust": adjust,
        "force_refresh": force_refresh,
        "use_cache": use_cache,
    }
    df_long = get_prices_long(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        adjust=adjust,
        force_refresh=force_refresh,
        use_cache=use_cache,
    )

    df_wide = (
        df_long.pivot(values="price", index="date", columns="ticker")
               .sort("date")
    )

    if as_pandas:
        return df_wide.to_pandas()
    return df_wide
