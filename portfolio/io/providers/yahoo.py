# Yahoo Finance Data Provider
# ===========================
"""
Implementación del proveedor Yahoo Finance usando yfinance.

Este es el proveedor de fallback principal. Aunque es gratuito y tiene
amplia cobertura, no debe usarse como fuente única en producción debido
a cambios de comportamiento frecuentes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import polars as pl
import yfinance as yf

from portfolio.core.compat import UTC
from portfolio.io.providers.base import (
    BaseDataProvider,
    DataNotFoundError,
    ProviderError,
)
from portfolio.io.schemas import (
    create_empty_bars_df,
    create_empty_corporate_actions_df,
)

logger = logging.getLogger(__name__)


class YahooProvider(BaseDataProvider):
    """
    Proveedor de datos Yahoo Finance.

    Características:
    - Gratuito, sin API key requerida
    - Amplia cobertura de equities, ETFs, índices
    - Datos ajustados por dividendos y splits
    - NO provee corporate actions explícitas
    - NO soporta datos intraday confiables

    Limitaciones:
    - Rate limits no documentados (~2000 req/hr estimado)
    - Cambios de API sin previo aviso
    - Sin soporte oficial
    """

    name = "yahoo"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Inicializa Yahoo provider.

        Args:
            api_key: No requerido para Yahoo
            base_url: No usado
        """
        super().__init__(api_key=api_key, base_url=base_url)

    @property
    def rate_limit(self) -> tuple[int, int]:
        """Yahoo tiene rate limits no documentados, estimamos conservador."""
        return (100, 60)  # ~100 requests/min conservador

    @property
    def supports_intraday(self) -> bool:
        """Yahoo soporta intraday pero es poco confiable."""
        return False

    @property
    def supports_corporate_actions(self) -> bool:
        """Yahoo NO provee corporate actions como endpoint separado."""
        return False

    def get_bars_daily(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Descarga barras diarias OHLCV desde Yahoo Finance.

        Los datos vienen ajustados por dividendos y splits cuando
        se usa auto_adjust=True.
        """
        t0 = time.perf_counter()
        tks = self._normalize_tickers(tickers)
        self._validate_date_range(start, end)
        self._log_request("get_bars_daily", tks, start, end)

        try:
            df = yf.download(
                list(tks),
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            raise ProviderError(self.name, f"Download failed: {e}", e) from e

        if isinstance(df, pd.Series):
            df = df.to_frame()

        if df.empty:
            logger.warning("[%s] No data returned for tickers: %s", self.name, tks)
            return create_empty_bars_df()

        # Parsear estructura de yfinance a formato largo
        # yfinance con group_by='ticker' siempre devuelve MultiIndex (ticker, field)
        rows = []
        for ticker in tks:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    # MultiIndex: (ticker, field)
                    if ticker in df.columns.get_level_values(0):
                        ticker_df = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        logger.warning("[%s] Ticker %s not found in MultiIndex", self.name, ticker)
                        continue
                else:
                    # Columnas simples (raro pero por si acaso)
                    ticker_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

                ticker_df = ticker_df.reset_index()
                ticker_df.columns = ["date", "open", "high", "low", "close", "volume"]
                ticker_df["ticker"] = ticker
                rows.append(ticker_df)
            except KeyError as e:
                logger.warning("[%s] Ticker %s not found in response: %s", self.name, ticker, e)
                continue

        if not rows:
            raise DataNotFoundError(self.name, tks, start, end)

        result_pd = pd.concat(rows, ignore_index=True)
        result_pd["date"] = pd.to_datetime(result_pd["date"], utc=False)
        result_pd = result_pd.dropna(subset=["close"])

        # Convertir a Polars con schema correcto
        pl_df = pl.from_pandas(result_pd).with_columns(
            [
                pl.col("date").cast(pl.Date),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.lit(1.0).alias("adj_factor"),  # Yahoo ya ajusta, factor = 1.0
            ]
        )

        # Añadir metadata
        pl_df = self._add_metadata_columns(pl_df)

        # Ordenar
        pl_df = pl_df.sort(["ticker", "date"])

        logger.info(
            "[%s] get_bars_daily OK: %d rows, %d tickers in %.3fs",
            self.name,
            pl_df.height,
            pl_df.select(pl.col("ticker").n_unique()).item(),
            time.perf_counter() - t0,
        )

        return pl_df

    def get_corporate_actions(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Yahoo NO provee corporate actions como endpoint separado.

        Los ajustes ya están aplicados en los precios cuando auto_adjust=True.
        Este método intenta extraer info de dividendos/splits del Ticker object.
        """
        t0 = time.perf_counter()
        tks = self._normalize_tickers(tickers)
        self._log_request("get_corporate_actions", tks, start, end)

        rows = []
        for ticker in tks:
            try:
                tkr = yf.Ticker(ticker)

                # Dividendos
                dividends = tkr.dividends
                if dividends is not None and len(dividends) > 0:
                    div_df = dividends.reset_index()
                    div_df.columns = ["ex_date", "amount"]
                    div_df["ticker"] = ticker
                    div_df["action_type"] = "dividend"
                    div_df["factor"] = 1.0
                    div_df["currency"] = "USD"  # Yahoo no provee currency
                    rows.append(div_df)

                # Splits
                splits = tkr.splits
                if splits is not None and len(splits) > 0:
                    split_df = splits.reset_index()
                    split_df.columns = ["ex_date", "factor"]
                    split_df["ticker"] = ticker
                    split_df["action_type"] = "split"
                    split_df["amount"] = None
                    split_df["currency"] = None
                    rows.append(split_df)

            except Exception as e:
                logger.warning("[%s] Failed to get corp actions for %s: %s", self.name, ticker, e)
                continue

        if not rows:
            return create_empty_corporate_actions_df()

        result_pd = pd.concat(rows, ignore_index=True)
        result_pd["ex_date"] = pd.to_datetime(result_pd["ex_date"], utc=False)

        # Filtrar por rango de fechas
        result_pd = result_pd[(result_pd["ex_date"] >= start) & (result_pd["ex_date"] <= end)]

        if result_pd.empty:
            return create_empty_corporate_actions_df()

        pl_df = pl.from_pandas(result_pd).with_columns(
            [
                pl.col("ex_date").cast(pl.Date),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("action_type").cast(pl.Utf8),
                pl.col("factor").cast(pl.Float64),
                pl.col("amount").cast(pl.Float64),
                pl.col("currency").cast(pl.Utf8),
            ]
        )

        pl_df = self._add_metadata_columns(pl_df)

        logger.info(
            "[%s] get_corporate_actions OK: %d actions in %.3fs",
            self.name,
            pl_df.height,
            time.perf_counter() - t0,
        )

        return pl_df.sort(["ticker", "ex_date"])

    def get_reference_data(
        self,
        tickers: Iterable[str],
    ) -> pl.DataFrame:
        """
        Obtiene metadata básica de tickers desde Yahoo Finance.

        Nota: La información de Yahoo es limitada y puede estar incompleta.
        """
        t0 = time.perf_counter()
        tks = self._normalize_tickers(tickers)
        self._log_request("get_reference_data", tks)

        rows = []
        for ticker in tks:
            try:
                tkr = yf.Ticker(ticker)
                info = tkr.info

                rows.append(
                    {
                        "ticker": ticker,
                        "name": info.get("longName") or info.get("shortName"),
                        "exchange": info.get("exchange"),
                        "asset_class": self._infer_asset_class(info),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "figi": None,  # Yahoo no provee FIGI
                        "cik": None,  # Yahoo no provee CIK
                        "ipo_date": None,  # Parsearlo es complejo
                        "delisted_date": None,
                        "is_active": info.get("regularMarketPrice") is not None,
                    }
                )
            except Exception as e:
                logger.warning("[%s] Failed to get reference for %s: %s", self.name, ticker, e)
                rows.append(
                    {
                        "ticker": ticker,
                        "name": None,
                        "exchange": None,
                        "asset_class": None,
                        "sector": None,
                        "industry": None,
                        "figi": None,
                        "cik": None,
                        "ipo_date": None,
                        "delisted_date": None,
                        "is_active": None,
                    }
                )

        pl_df = pl.DataFrame(rows).with_columns(
            [
                pl.col("ticker").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8),
                pl.col("exchange").cast(pl.Utf8),
                pl.col("asset_class").cast(pl.Utf8),
                pl.col("sector").cast(pl.Utf8),
                pl.col("industry").cast(pl.Utf8),
                pl.col("figi").cast(pl.Utf8),
                pl.col("cik").cast(pl.Utf8),
                pl.col("ipo_date").cast(pl.Date),
                pl.col("delisted_date").cast(pl.Date),
                pl.col("is_active").cast(pl.Boolean),
                pl.lit(datetime.now(UTC)).alias("last_updated"),
            ]
        )

        logger.info(
            "[%s] get_reference_data OK: %d tickers in %.3fs",
            self.name,
            pl_df.height,
            time.perf_counter() - t0,
        )

        return pl_df

    def _infer_asset_class(self, info: dict) -> str | None:
        """Infiere asset class desde info de Yahoo."""
        quote_type = info.get("quoteType", "").upper()

        if quote_type == "ETF":
            return "etf"
        elif quote_type == "EQUITY":
            return "equity"
        elif quote_type == "INDEX":
            return "index"
        elif "ADR" in info.get("longName", "").upper():
            return "adr"

        return "equity"  # default
