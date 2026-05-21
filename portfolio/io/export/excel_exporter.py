# Golden Excel Dataset Exporter
# =============================
"""
Exportador de datasets "golden" a Excel para ML offline.

REGLAS:
- Excel es vista congelada del snapshot
- No se modifica manualmente
- Si algo cambia → nuevo snapshot, nuevo Excel

Estructura:
- Sheet 1: DATA (OHLCV + features)
- Sheet 2: METADATA (hashes, versiones)
- Sheet 3: INSTRUMENTS (catálogo)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Schema
# =============================================================================


@dataclass
class GoldenDatasetConfig:
    """Configuración del dataset golden."""

    # Identity
    dataset_id: str = "qqq_voo_bil_20y_v1"
    description: str = "QQQ/VOO/BIL 20-year dataset for ML training"

    # Universe
    tickers: list[str] | None = None

    # Time range
    start_date: date | None = None
    end_date: date | None = None

    # Source
    provider: str = "polygon"

    # Versions
    adjustment_version: str = "2.0.0"
    feature_set_version: str = "1.0.0"
    calendar_id: str = "NYSE"

    # Features to include
    features: list[str] | None = None

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = ["QQQ", "VOO", "BIL"]
        if self.features is None:
            self.features = [
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "realized_vol_20d",
                "momentum_12_1",
                "drawdown_20d",
                "dollar_volume",
            ]


# =============================================================================
# Excel Exporter
# =============================================================================


class GoldenExcelExporter:
    """
    Exporta dataset golden a Excel con estructura profesional.

    Sheets:
    - DATA: OHLCV + features (long format)
    - METADATA: hashes, versiones, info
    - INSTRUMENTS: catálogo de tickers

    Example:
        >>> exporter = GoldenExcelExporter(config)
        >>> exporter.export(df_data, df_instruments, "output.xlsx")
    """

    # Columnas core (siempre presentes)
    CORE_COLUMNS = [
        "date",
        "instrument_id",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    # Columnas de quality (si existen)
    QUALITY_COLUMNS = [
        "quality_flag",
        "adj_factor",
        "split_factor",
        "dividend_factor",
    ]

    def __init__(self, config: GoldenDatasetConfig):
        self.config = config
        self._content_hash: str | None = None

    def export(
        self,
        df_data: pl.DataFrame,
        df_instruments: pl.DataFrame | None = None,
        output_path: str | Path = "golden_dataset.xlsx",
        include_unadjusted: bool = False,
    ) -> dict:
        """
        Exporta dataset a Excel.

        Args:
            df_data: DataFrame con OHLCV + features
            df_instruments: DataFrame con catálogo de instruments
            output_path: Path del Excel
            include_unadjusted: Si incluir columnas unadjusted

        Returns:
            Metadata del export
        """
        output_path = Path(output_path)

        # Preparar data
        df_data = self._prepare_data(df_data)

        # Calcular hash
        self._content_hash = self._compute_content_hash(df_data)

        # Preparar instruments
        if df_instruments is None:
            df_instruments = self._create_instruments_from_data(df_data)

        # Preparar metadata
        metadata = self._create_metadata(df_data)

        # Exportar
        try:
            import xlsxwriter  # noqa: F401  # availability check; used by _export_with_xlsxwriter

            self._export_with_xlsxwriter(df_data, df_instruments, metadata, output_path)
        except ImportError:
            # Fallback a openpyxl
            self._export_with_polars(df_data, df_instruments, metadata, output_path)

        logger.info("Exported golden Excel to %s", output_path)
        logger.info("Content hash: %s", self._content_hash)

        return metadata

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepara y ordena el DataFrame para export."""
        # Seleccionar columnas
        columns_to_include = []

        # Core columns
        for col in self.CORE_COLUMNS:
            if col in df.columns:
                columns_to_include.append(col)

        # Features
        for feat in self.config.features or []:
            if feat in df.columns:
                columns_to_include.append(feat)

        # Quality columns
        for col in self.QUALITY_COLUMNS:
            if col in df.columns:
                columns_to_include.append(col)

        # Seleccionar y ordenar
        df = df.select(columns_to_include)

        # Sort canónico (importante para hash)
        sort_cols = []
        if "ticker" in df.columns:
            sort_cols.append("ticker")
        if "instrument_id" in df.columns:
            sort_cols.append("instrument_id")
        if "date" in df.columns:
            sort_cols.append("date")

        if sort_cols:
            df = df.sort(sort_cols)

        return df

    def _compute_content_hash(self, df: pl.DataFrame) -> str:
        """Calcula hash del contenido para reproducibilidad."""
        # Serializar a bytes
        buffer = df.write_csv().encode("utf-8")
        return hashlib.sha256(buffer).hexdigest()[:16]

    def _create_instruments_from_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Crea catálogo de instruments desde data."""
        ticker_col = "ticker" if "ticker" in df.columns else "instrument_id"

        instruments = []
        for ticker in df[ticker_col].unique().to_list():
            df_ticker = df.filter(pl.col(ticker_col) == ticker)

            instruments.append(
                {
                    "instrument_id": ticker,
                    "ticker": ticker,
                    "name": self._get_ticker_name(ticker),
                    "exchange": "NYSE_ARCA",
                    "inception_date": df_ticker["date"].min() if "date" in df.columns else None,
                    "last_date": df_ticker["date"].max() if "date" in df.columns else None,
                    "n_observations": df_ticker.height,
                }
            )

        return pl.DataFrame(instruments)

    def _get_ticker_name(self, ticker: str) -> str:
        """Nombre descriptivo del ticker."""
        names = {
            "QQQ": "Invesco QQQ Trust (NASDAQ-100)",
            "VOO": "Vanguard S&P 500 ETF",
            "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
            "SPY": "SPDR S&P 500 ETF Trust",
            "IWM": "iShares Russell 2000 ETF",
        }
        return names.get(ticker, ticker)

    def _create_metadata(self, df: pl.DataFrame) -> dict:
        """Crea diccionario de metadata."""
        return {
            "dataset_id": self.config.dataset_id,
            "description": self.config.description,
            "content_hash": self._content_hash,
            "provider": self.config.provider,
            "adjustment_version": self.config.adjustment_version,
            "feature_set_version": self.config.feature_set_version,
            "calendar_id": self.config.calendar_id,
            "tickers": self.config.tickers,
            "features": self.config.features,
            "start_date": str(df["date"].min()) if "date" in df.columns else None,
            "end_date": str(df["date"].max()) if "date" in df.columns else None,
            "n_rows": df.height,
            "n_columns": len(df.columns),
            "columns": df.columns,
            "created_at": datetime.utcnow().isoformat(),
            "gammaedge_version": "1.0.0",
        }

    def _export_with_xlsxwriter(
        self,
        df_data: pl.DataFrame,
        df_instruments: pl.DataFrame,
        metadata: dict[str, Any],
        output_path: Path,
    ) -> None:
        """Exporta usando xlsxwriter (mejor formato)."""
        import xlsxwriter

        workbook = xlsxwriter.Workbook(str(output_path))

        # Formatos
        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#1a1a2e",
                "font_color": "white",
                "border": 1,
            }
        )
        number_format = workbook.add_format({"num_format": "#,##0.00"})
        pct_format = workbook.add_format({"num_format": "0.00%"})

        # Sheet 1: DATA
        ws_data = workbook.add_worksheet("DATA")

        # Headers
        for col_idx, col in enumerate(df_data.columns):
            ws_data.write(0, col_idx, col, header_format)

        # Data
        for row_idx, row in enumerate(df_data.iter_rows(named=True), start=1):
            for col_idx, col in enumerate(df_data.columns):
                value = row[col]
                if value is None:
                    ws_data.write(row_idx, col_idx, "")
                elif col == "date":
                    ws_data.write(row_idx, col_idx, str(value))
                elif isinstance(value, (int, float)):
                    if "ret" in col or "momentum" in col or "drawdown" in col:
                        ws_data.write(row_idx, col_idx, value, pct_format)
                    else:
                        ws_data.write(row_idx, col_idx, value, number_format)
                else:
                    ws_data.write(row_idx, col_idx, str(value))

        # Freeze header
        ws_data.freeze_panes(1, 0)

        # Sheet 2: METADATA
        ws_meta = workbook.add_worksheet("METADATA")
        ws_meta.write(0, 0, "Key", header_format)
        ws_meta.write(0, 1, "Value", header_format)

        meta_row = 1
        for key, value in metadata.items():
            ws_meta.write(meta_row, 0, key)
            if isinstance(value, list):
                ws_meta.write(meta_row, 1, ", ".join(map(str, value)))
            else:
                ws_meta.write(meta_row, 1, str(value) if value else "")
            meta_row += 1

        ws_meta.set_column(0, 0, 25)
        ws_meta.set_column(1, 1, 80)

        # Sheet 3: INSTRUMENTS
        ws_inst = workbook.add_worksheet("INSTRUMENTS")

        for col_idx, col in enumerate(df_instruments.columns):
            ws_inst.write(0, col_idx, col, header_format)

        for row_idx, row in enumerate(df_instruments.iter_rows(named=True), start=1):
            for col_idx, col in enumerate(df_instruments.columns):
                value = row[col]
                ws_inst.write(row_idx, col_idx, str(value) if value else "")

        ws_inst.freeze_panes(1, 0)

        workbook.close()

    def _export_with_polars(
        self,
        df_data: pl.DataFrame,
        df_instruments: pl.DataFrame,
        metadata: dict[str, Any],
        output_path: Path,
    ) -> None:
        """Fallback export usando Polars → Excel."""
        # Polars puede escribir Excel directamente
        # pl.ExcelWriter is not part of public polars typing stubs but exists at runtime.
        with pl.ExcelWriter(str(output_path)) as writer:  # type: ignore[attr-defined]
            df_data.write_excel(writer, worksheet="DATA")
            df_instruments.write_excel(writer, worksheet="INSTRUMENTS")

            # Metadata como DataFrame
            df_meta = pl.DataFrame(
                [{"key": k, "value": str(v) if v else ""} for k, v in metadata.items()]
            )
            df_meta.write_excel(writer, worksheet="METADATA")


# =============================================================================
# Full Pipeline Export
# =============================================================================


def export_golden_dataset(
    df_raw: pl.DataFrame,
    output_path: str | Path = "golden_dataset.xlsx",
    dataset_id: str = "qqq_voo_bil_v1",
    features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Pipeline completo: raw → features → Excel golden.

    Args:
        df_raw: DataFrame OHLCV raw (long format)
        output_path: Path del Excel
        dataset_id: ID único del dataset
        features: Lista de features a incluir

    Returns:
        Metadata del export
    """
    from portfolio.trading.v1_features import V1FeatureBuilder

    # 1. Build features
    logger.info("Building features...")
    builder = V1FeatureBuilder()
    df_features = builder.build(df_raw)

    # 2. Merge con raw (para tener OHLCV)
    df_merged = df_raw.join(df_features, on="date", how="inner")

    # 3. Config
    config = GoldenDatasetConfig(
        dataset_id=dataset_id,
        features=features
        or [
            "ret_1d_QQQ",
            "ret_5d_QQQ",
            "ret_20d_QQQ",
            "vol_20d_QQQ",
            "mom_60d_QQQ",
            "dd_20d_QQQ",
            "ret_1d_VOO",
            "ret_5d_VOO",
            "ret_20d_VOO",
            "vol_20d_VOO",
            "mom_60d_VOO",
            "dd_20d_VOO",
            "ret_1d_BIL",
            "ret_5d_BIL",
            "spread_QQQ_VOO",
            "vol_ratio_QQQ_VOO",
        ],
    )

    # 4. Export
    exporter = GoldenExcelExporter(config)
    metadata = exporter.export(df_merged, output_path=output_path)

    return metadata


# =============================================================================
# Standalone function for quick export
# =============================================================================


def quick_export_from_yahoo(
    tickers: list[str] | None = None,
    years: int = 3,
    output_path: str = "golden_dataset.xlsx",
) -> dict[str, Any]:
    """
    Descarga datos de Yahoo y exporta Excel golden.

    Para testing rápido. Para producción usar Polygon.
    """
    if tickers is None:
        tickers = ["QQQ", "VOO", "BIL"]

    from datetime import timedelta

    import pandas as pd
    import yfinance as yf

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    logger.info("Downloading %s from Yahoo...", tickers)

    all_dfs = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df["ticker"] = ticker
        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df = pl.from_pandas(df_combined)

    # Normalize columns
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "date":
            col_map[col] = "date"
        elif col_lower == "open":
            col_map[col] = "open"
        elif col_lower == "high":
            col_map[col] = "high"
        elif col_lower == "low":
            col_map[col] = "low"
        elif col_lower == "close":
            col_map[col] = "close"
        elif col_lower == "volume":
            col_map[col] = "volume"
        elif col_lower == "ticker":
            col_map[col] = "ticker"

    df = df.rename(col_map)

    if df["date"].dtype == pl.Datetime:
        df = df.with_columns(pl.col("date").dt.date())

    # Add instrument_id
    df = df.with_columns(pl.col("ticker").alias("instrument_id"))

    return export_golden_dataset(df, output_path)
