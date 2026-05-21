# Cross-Provider Reconciliation
# =============================
"""
Capa de reconciliación entre proveedores de datos.

Detecta divergencias entre providers y marca datos disputados.
Permite:
- Identificar problemas de calidad
- Elegir "truth" según política
- Excluir períodos corruptos de training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import polars as pl

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


TruthPolicy = Literal["primary_wins", "secondary_wins", "average", "conservative"]


@dataclass
class ReconciliationConfig:
    """Configuración de reconciliación."""

    # Umbrales para marcar como disputado
    price_diff_threshold: float = 0.01  # 1% de diferencia
    volume_ratio_threshold: float = 0.20  # 20% de diferencia

    # Política de resolución
    truth_policy: TruthPolicy = "primary_wins"

    # Cuántos días consecutivos de divergencia = problema grave
    consecutive_divergence_alert: int = 3


@dataclass
class ReconciliationResult:
    """Resultado de reconciliación entre dos providers."""

    df_reconciled: pl.DataFrame  # Datos con divergencias marcadas
    df_divergences: pl.DataFrame  # Solo las divergencias
    summary: dict

    @property
    def has_divergences(self) -> bool:
        return self.df_divergences.height > 0

    @property
    def divergence_rate(self) -> float:
        """Porcentaje de filas con divergencias."""
        if self.df_reconciled.height == 0:
            return 0.0
        return self.df_divergences.height / self.df_reconciled.height


class CrossProviderReconciler:
    """
    Reconcilia datos de múltiples providers.

    Características:
    - Compara precios close, open, high, low
    - Compara volúmenes
    - Detecta días faltantes en cada provider
    - Detecta split/dividend mismatches
    - Marca disputed y resuelve según policy

    Example:
        >>> reconciler = CrossProviderReconciler(
        ...     primary="polygon",
        ...     secondary="yahoo",
        ... )
        >>> result = reconciler.reconcile(df_primary, df_secondary)
        >>>
        >>> if result.divergence_rate > 0.05:
        ...     print("Warning: >5% divergence detected")
    """

    def __init__(
        self,
        primary: str,
        secondary: str,
        config: ReconciliationConfig | None = None,
    ):
        """
        Args:
            primary: Nombre del provider primario
            secondary: Nombre del provider secundario
            config: Configuración de reconciliación
        """
        self.primary = primary
        self.secondary = secondary
        self.config = config or ReconciliationConfig()

    def reconcile(
        self,
        df_primary: pl.DataFrame,
        df_secondary: pl.DataFrame,
    ) -> ReconciliationResult:
        """
        Reconcilia datos de dos providers.

        Args:
            df_primary: Datos del provider primario
            df_secondary: Datos del provider secundario

        Returns:
            ReconciliationResult con divergencias marcadas
        """
        logger.info(
            "Reconciling %s (%d rows) vs %s (%d rows)",
            self.primary,
            df_primary.height,
            self.secondary,
            df_secondary.height,
        )

        # Normalizar columnas
        df_p = self._normalize_columns(df_primary, suffix="_p")
        df_s = self._normalize_columns(df_secondary, suffix="_s")

        # Outer join por date + ticker
        df_merged = df_p.join(
            df_s,
            on=["date", "ticker"],
            how="outer",
            suffix="_s",
        )

        # Calcular divergencias
        df_with_divergences = self._calculate_divergences(df_merged)

        # Marcar disputados
        df_reconciled = self._mark_disputed(df_with_divergences)

        # Aplicar policy de resolución
        df_reconciled = self._apply_truth_policy(df_reconciled)

        # Extraer solo divergencias
        df_divergences = df_reconciled.filter(pl.col("is_disputed"))

        # Detectar problemas por ticker
        ticker_issues = self._analyze_by_ticker(df_divergences)

        # Summary
        divergence_rate = df_divergences.height / max(df_reconciled.height, 1)
        summary: dict[str, Any] = {
            "primary": self.primary,
            "secondary": self.secondary,
            "total_rows": df_reconciled.height,
            "divergent_rows": df_divergences.height,
            "divergence_rate": divergence_rate,
            "missing_in_primary": df_merged.filter(pl.col("close_p").is_null()).height,
            "missing_in_secondary": df_merged.filter(pl.col("close_s").is_null()).height,
            "tickers_with_issues": len(ticker_issues),
            "ticker_issues": ticker_issues,
            "reconciled_at": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Reconciliation complete: %d/%d divergent (%.2f%%)",
            df_divergences.height,
            df_reconciled.height,
            divergence_rate * 100,
        )

        return ReconciliationResult(
            df_reconciled=df_reconciled,
            df_divergences=df_divergences,
            summary=summary,
        )

    def _normalize_columns(
        self,
        df: pl.DataFrame,
        suffix: str,
    ) -> pl.DataFrame:
        """Normaliza nombres de columnas."""
        # Identificar columnas de precio/volumen
        price_cols = ["open", "high", "low", "close", "volume"]

        # Para columnas _raw, usar esas
        if "close_raw" in df.columns:
            price_cols = ["open_raw", "high_raw", "low_raw", "close_raw", "volume_raw"]

        # Renombrar solo las columnas de datos
        renames = {}
        for col in df.columns:
            if col in price_cols or col.replace("_raw", "") in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                # Normalizar nombre base
                base = col.replace("_raw", "")
                renames[col] = f"{base}{suffix}"

        return df.rename(renames).select(["date", "ticker"] + list(renames.values()))

    def _calculate_divergences(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcula métricas de divergencia."""
        return df.with_columns(
            [
                # Diferencia de precios close
                pl.when(pl.col("close_p").is_not_null() & pl.col("close_s").is_not_null())
                .then((pl.col("close_p") - pl.col("close_s")).abs() / pl.col("close_s"))
                .otherwise(None)
                .alias("close_diff_pct"),
                # Ratio de volumen
                pl.when(
                    pl.col("volume_p").is_not_null()
                    & pl.col("volume_s").is_not_null()
                    & (pl.col("volume_s") > 0)
                )
                .then(pl.col("volume_p") / pl.col("volume_s"))
                .otherwise(None)
                .alias("volume_ratio"),
                # Missing flags
                pl.col("close_p").is_null().alias("missing_in_primary"),
                pl.col("close_s").is_null().alias("missing_in_secondary"),
            ]
        )

    def _mark_disputed(self, df: pl.DataFrame) -> pl.DataFrame:
        """Marca filas con divergencias significativas."""
        return df.with_columns(
            [
                (
                    # Precio difiere más del umbral
                    (pl.col("close_diff_pct") > self.config.price_diff_threshold)
                    |
                    # Volumen difiere más del umbral
                    (
                        (
                            (pl.col("volume_ratio") < (1 - self.config.volume_ratio_threshold))
                            | (pl.col("volume_ratio") > (1 + self.config.volume_ratio_threshold))
                        )
                        & pl.col("volume_ratio").is_not_null()
                    )
                    |
                    # Missing en uno de los dos
                    pl.col("missing_in_primary")
                    | pl.col("missing_in_secondary")
                ).alias("is_disputed"),
                # Razón de la disputa
                pl.when(pl.col("missing_in_primary"))
                .then(pl.lit("missing_primary"))
                .when(pl.col("missing_in_secondary"))
                .then(pl.lit("missing_secondary"))
                .when(pl.col("close_diff_pct") > self.config.price_diff_threshold)
                .then(pl.lit("price_mismatch"))
                .when(
                    (pl.col("volume_ratio") < (1 - self.config.volume_ratio_threshold))
                    | (pl.col("volume_ratio") > (1 + self.config.volume_ratio_threshold))
                )
                .then(pl.lit("volume_mismatch"))
                .otherwise(pl.lit("none"))
                .alias("dispute_reason"),
            ]
        )

    def _apply_truth_policy(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplica la política de resolución para crear valores finales."""
        policy = self.config.truth_policy

        if policy == "primary_wins":
            return df.with_columns(
                [
                    pl.coalesce(pl.col("close_p"), pl.col("close_s")).alias("close_final"),
                    pl.coalesce(pl.col("volume_p"), pl.col("volume_s")).alias("volume_final"),
                ]
            )

        elif policy == "secondary_wins":
            return df.with_columns(
                [
                    pl.coalesce(pl.col("close_s"), pl.col("close_p")).alias("close_final"),
                    pl.coalesce(pl.col("volume_s"), pl.col("volume_p")).alias("volume_final"),
                ]
            )

        elif policy == "average":
            return df.with_columns(
                [
                    pl.when(pl.col("close_p").is_not_null() & pl.col("close_s").is_not_null())
                    .then((pl.col("close_p") + pl.col("close_s")) / 2)
                    .otherwise(pl.coalesce(pl.col("close_p"), pl.col("close_s")))
                    .alias("close_final"),
                    pl.when(pl.col("volume_p").is_not_null() & pl.col("volume_s").is_not_null())
                    .then((pl.col("volume_p") + pl.col("volume_s")) / 2)
                    .otherwise(pl.coalesce(pl.col("volume_p"), pl.col("volume_s")))
                    .alias("volume_final"),
                ]
            )

        elif policy == "conservative":
            # Usar el valor más conservador (lower close, lower volume)
            return df.with_columns(
                [
                    pl.when(pl.col("close_p").is_not_null() & pl.col("close_s").is_not_null())
                    .then(pl.min_horizontal(pl.col("close_p"), pl.col("close_s")))
                    .otherwise(pl.coalesce(pl.col("close_p"), pl.col("close_s")))
                    .alias("close_final"),
                    pl.when(pl.col("volume_p").is_not_null() & pl.col("volume_s").is_not_null())
                    .then(pl.min_horizontal(pl.col("volume_p"), pl.col("volume_s")))
                    .otherwise(pl.coalesce(pl.col("volume_p"), pl.col("volume_s")))
                    .alias("volume_final"),
                ]
            )

        # All TruthPolicy literal values handled above; this is a safety net.
        raise ValueError(f"Unknown truth_policy: {policy!r}")  # pragma: no cover

    def _analyze_by_ticker(self, df_divergences: pl.DataFrame) -> dict[str, dict]:
        """Analiza divergencias por ticker."""
        if df_divergences.height == 0:
            return {}

        issues = {}

        for ticker in df_divergences.select(pl.col("ticker").unique()).to_series():
            ticker_divergences = df_divergences.filter(pl.col("ticker") == ticker)

            reasons = ticker_divergences["dispute_reason"].value_counts()

            issues[ticker] = {
                "divergent_days": ticker_divergences.height,
                "reasons": reasons.to_dicts() if hasattr(reasons, "to_dicts") else {},
                "max_price_diff": ticker_divergences["close_diff_pct"].max(),
            }

        return issues

    def detect_split_mismatch(
        self,
        df_primary: pl.DataFrame,
        df_secondary: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Detecta splits que aparecen en un provider pero no en otro.

        Un split mismatch se detecta cuando:
        - close_p / close_s cambia bruscamente de un día a otro
        - Y luego se mantiene en el nuevo ratio
        """
        # Normalizar
        df_p = self._normalize_columns(df_primary, "_p")
        df_s = self._normalize_columns(df_secondary, "_s")

        df = df_p.join(df_s, on=["date", "ticker"], how="inner").sort(["ticker", "date"])

        # Calcular ratio y cambio de ratio
        df = df.with_columns(
            [
                (pl.col("close_p") / pl.col("close_s")).alias("price_ratio"),
            ]
        ).with_columns(
            [
                (pl.col("price_ratio") / pl.col("price_ratio").shift(1).over("ticker")).alias(
                    "ratio_change"
                ),
            ]
        )

        # Detectar cambios bruscos (típico de split no aplicado)
        split_candidates = df.filter(
            (pl.col("ratio_change") < 0.6) | (pl.col("ratio_change") > 1.5)
        )

        return split_candidates.select(
            ["date", "ticker", "close_p", "close_s", "price_ratio", "ratio_change"]
        )


def reconcile_providers(
    df_primary: pl.DataFrame,
    df_secondary: pl.DataFrame,
    primary_name: str = "primary",
    secondary_name: str = "secondary",
    **config_kwargs: Any,
) -> ReconciliationResult:
    """
    Función helper para reconciliación rápida.

    Example:
        >>> result = reconcile_providers(df_polygon, df_yahoo)
        >>> print(f"Divergence rate: {result.divergence_rate:.2%}")
    """
    config = ReconciliationConfig(**config_kwargs)
    reconciler = CrossProviderReconciler(primary_name, secondary_name, config)
    return reconciler.reconcile(df_primary, df_secondary)
