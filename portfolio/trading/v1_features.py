# V1 Features Builder
# ===================
"""
Features específicos para V1 Allocation (QQQ/VOO/BIL).

Features robustos y macro-oriented:
- Returns multi-horizon por activo
- Volatilidad
- Momentum
- Drawdown
- Cross-features (spreads, ratios)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class V1FeatureConfig:
    """Configuración de features para V1."""

    assets: list[str] = None

    # Returns
    return_horizons: list[int] = None

    # Volatility
    vol_window: int = 20

    # Momentum
    momentum_window: int = 60

    # Drawdown
    drawdown_window: int = 20

    def __post_init__(self):
        if self.assets is None:
            self.assets = ["QQQ", "VOO", "BIL"]
        if self.return_horizons is None:
            self.return_horizons = [1, 5, 20]


class V1FeatureBuilder:
    """
    Construye features para V1 Allocation.

    Features por activo:
    - ret_{h}d: Returns de h días
    - vol_20d: Volatilidad realizada
    - mom_60d: Momentum (cumulative return)
    - dd_20d: Drawdown desde máximo

    Cross-features:
    - spread_QQQ_VOO: ret_QQQ - ret_VOO
    - vol_ratio_QQQ_VOO: vol_QQQ / vol_VOO
    - regime_risk_off: vol > threshold → risk-off signal
    """

    def __init__(self, config: V1FeatureConfig | None = None):
        self.config = config or V1FeatureConfig()

    def build(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
        close_col: str = "close",
    ) -> pl.DataFrame:
        """
        Construye todos los features V1.

        Input: Long format (date, ticker, close, ...)
        Output: Wide format (date, feature_1, feature_2, ...)
        """
        # Calcular features por ticker
        df = self._add_returns(df, ticker_col, date_col, close_col)
        df = self._add_volatility(df, ticker_col, date_col)
        df = self._add_momentum(df, ticker_col, date_col, close_col)
        df = self._add_drawdown(df, ticker_col, date_col, close_col)

        # Pivot a wide format
        df_wide = self._pivot_to_wide(df, date_col, ticker_col)

        # Añadir cross-features
        df_wide = self._add_cross_features(df_wide)

        return df_wide

    def _add_returns(
        self,
        df: pl.DataFrame,
        ticker_col: str,
        date_col: str,
        close_col: str,
    ) -> pl.DataFrame:
        """Añade returns multi-horizon."""
        df = df.sort([ticker_col, date_col])

        for h in self.config.return_horizons:
            df = df.with_columns(
                (pl.col(close_col) / pl.col(close_col).shift(h).over(ticker_col) - 1).alias(
                    f"ret_{h}d"
                )
            )

        return df

    def _add_volatility(
        self,
        df: pl.DataFrame,
        ticker_col: str,
        date_col: str,
    ) -> pl.DataFrame:
        """Añade volatilidad realizada."""
        window = self.config.vol_window

        df = df.with_columns(
            pl.col("ret_1d").rolling_std(window).over(ticker_col).alias(f"vol_{window}d")
        )

        return df

    def _add_momentum(
        self,
        df: pl.DataFrame,
        ticker_col: str,
        date_col: str,
        close_col: str,
    ) -> pl.DataFrame:
        """Añade momentum."""
        window = self.config.momentum_window

        df = df.with_columns(
            (pl.col(close_col) / pl.col(close_col).shift(window).over(ticker_col) - 1).alias(
                f"mom_{window}d"
            )
        )

        return df

    def _add_drawdown(
        self,
        df: pl.DataFrame,
        ticker_col: str,
        date_col: str,
        close_col: str,
    ) -> pl.DataFrame:
        """Añade drawdown desde máximo rolling."""
        window = self.config.drawdown_window

        df = df.with_columns(
            pl.col(close_col).rolling_max(window).over(ticker_col).alias("_rolling_max")
        )

        df = df.with_columns(
            ((pl.col(close_col) - pl.col("_rolling_max")) / pl.col("_rolling_max")).alias(
                f"dd_{window}d"
            )
        ).drop("_rolling_max")

        return df

    def _pivot_to_wide(
        self,
        df: pl.DataFrame,
        date_col: str,
        ticker_col: str,
    ) -> pl.DataFrame:
        """Pivota features a formato wide."""
        # Seleccionar columnas de features
        feature_cols = [c for c in df.columns if c.startswith(("ret_", "vol_", "mom_", "dd_"))]

        # Crear DataFrame por asset
        dfs = []
        for asset in self.config.assets:
            df_asset = df.filter(pl.col(ticker_col) == asset).select([date_col] + feature_cols)
            # Renombrar columnas con sufijo de activo
            renames = {col: f"{col}_{asset}" for col in feature_cols}
            df_asset = df_asset.rename(renames)
            dfs.append(df_asset)

        # Merge por fecha
        if len(dfs) == 0:
            return pl.DataFrame()

        df_wide = dfs[0]
        for _i, df_other in enumerate(dfs[1:], start=1):
            df_wide = df_wide.join(
                df_other,
                on=date_col,
                how="outer_coalesce",
            )

        return df_wide.sort(date_col)

    def _add_cross_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Añade cross-features entre activos."""
        # Spread QQQ - VOO
        if "ret_1d_QQQ" in df.columns and "ret_1d_VOO" in df.columns:
            df = df.with_columns(
                (pl.col("ret_1d_QQQ") - pl.col("ret_1d_VOO")).alias("spread_QQQ_VOO")
            )

        if "ret_5d_QQQ" in df.columns and "ret_5d_VOO" in df.columns:
            df = df.with_columns(
                (pl.col("ret_5d_QQQ") - pl.col("ret_5d_VOO")).alias("spread_5d_QQQ_VOO")
            )

        # Vol ratio
        if "vol_20d_QQQ" in df.columns and "vol_20d_VOO" in df.columns:
            df = df.with_columns(
                (pl.col("vol_20d_QQQ") / pl.col("vol_20d_VOO")).alias("vol_ratio_QQQ_VOO")
            )

        # Risk-off signal (high vol regime)
        if "vol_20d_QQQ" in df.columns:
            # Si vol > 2x median → risk-off
            vol_median = df["vol_20d_QQQ"].median()
            if vol_median and vol_median > 0:
                df = df.with_columns(
                    (pl.col("vol_20d_QQQ") > 2 * vol_median).alias("regime_high_vol")
                )

        return df

    def get_feature_names(self) -> list[str]:
        """Lista de nombres de features generados."""
        features = []

        for asset in self.config.assets:
            for h in self.config.return_horizons:
                features.append(f"ret_{h}d_{asset}")
            features.append(f"vol_{self.config.vol_window}d_{asset}")
            features.append(f"mom_{self.config.momentum_window}d_{asset}")
            features.append(f"dd_{self.config.drawdown_window}d_{asset}")

        # Cross-features
        features.extend(
            [
                "spread_QQQ_VOO",
                "spread_5d_QQQ_VOO",
                "vol_ratio_QQQ_VOO",
                "regime_high_vol",
            ]
        )

        return features


def build_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Función helper para construir features V1.

    >>> df_features = build_v1_features(df_ohlcv)
    """
    builder = V1FeatureBuilder()
    return builder.build(df)
