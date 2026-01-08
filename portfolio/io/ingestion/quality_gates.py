# Quality Gates for Data Ingestion
# =================================
"""
Validaciones automáticas para datos de mercado.

Los Quality Gates aseguran que solo datos que cumplen estándares mínimos
pasen de raw/ a clean/, evitando contaminar el pipeline con datos malos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

import polars as pl

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Resultado de una validación individual."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    severity: Literal["error", "warning", "info"] = "error"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass
class QualityReport:
    """Reporte completo de quality gates."""

    report_id: str
    passed: bool
    checks: list[CheckResult]
    timestamp: datetime
    summary: str

    @classmethod
    def create(cls, checks: list[CheckResult]) -> QualityReport:
        """Crea un reporte desde una lista de checks."""
        # Passed = todos los checks de severidad "error" pasaron
        errors = [c for c in checks if c.severity == "error"]
        passed = all(c.passed for c in errors)

        failed_count = sum(1 for c in checks if not c.passed)
        warning_count = sum(1 for c in checks if not c.passed and c.severity == "warning")

        summary = f"{len(checks)} checks: {len(checks) - failed_count} passed"
        if failed_count > 0:
            summary += f", {failed_count - warning_count} failed, {warning_count} warnings"

        return cls(
            report_id=str(uuid4()),
            passed=passed,
            checks=checks,
            timestamp=datetime.now(UTC),
            summary=summary,
        )

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "passed": self.passed,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class QualityGates:
    """
    Validaciones automáticas para datos de mercado.

    Ejecuta una serie de checks y genera un QualityReport.
    Los datos que no pasan los checks de severidad "error" no deben
    pasar a clean/.
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        max_daily_return: float = 0.50,
        max_unchanged_days: int = 5,
    ):
        """
        Configura los umbrales de quality gates.

        Args:
            max_missing_pct: Máximo porcentaje de missing permitido (default 5%)
            max_daily_return: Máximo retorno diario absoluto aceptable (default 50%)
            max_unchanged_days: Máximo días consecutivos sin cambio (default 5)
        """
        self.max_missing_pct = max_missing_pct
        self.max_daily_return = max_daily_return
        self.max_unchanged_days = max_unchanged_days

    def validate_bars(self, df: pl.DataFrame) -> QualityReport:
        """
        Ejecuta todos los quality gates para datos de barras.

        Args:
            df: DataFrame con datos de barras (schema SCHEMA_BARS_1D)

        Returns:
            QualityReport con resultados de todas las validaciones
        """
        checks = []

        # 1. Schema validation
        checks.append(self._check_schema(df))

        # 2. Empty data check
        checks.append(self._check_not_empty(df))

        if df.height == 0:
            return QualityReport.create(checks)

        # 3. Missing data per column
        checks.append(self._check_missing_data(df))

        # 4. Impossible values
        checks.append(self._check_impossible_values(df))

        # 5. Price relationships (high >= low, close in range)
        checks.append(self._check_price_relationships(df))

        # 6. Extreme outliers
        checks.append(self._check_outliers(df))

        # 7. Stale data (unchanged prices)
        checks.append(self._check_stale_data(df))

        # 8. Duplicates
        checks.append(self._check_duplicates(df))

        report = QualityReport.create(checks)

        if report.passed:
            logger.info("Quality gates PASSED: %s", report.summary)
        else:
            logger.warning("Quality gates FAILED: %s", report.summary)

        return report

    def _check_schema(self, df: pl.DataFrame) -> CheckResult:
        """Valida que el DataFrame tenga el schema esperado."""
        # Solo validamos columnas críticas (sin metadata)
        required_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        actual_cols = set(df.columns)
        missing = required_cols - actual_cols

        if missing:
            return CheckResult(
                name="schema",
                passed=False,
                message=f"Missing required columns: {missing}",
                details={"missing_columns": list(missing)},
            )

        return CheckResult(
            name="schema",
            passed=True,
            message="Schema validation passed",
        )

    def _check_not_empty(self, df: pl.DataFrame) -> CheckResult:
        """Verifica que haya datos."""
        if df.height == 0:
            return CheckResult(
                name="not_empty",
                passed=False,
                message="DataFrame is empty",
            )

        return CheckResult(
            name="not_empty",
            passed=True,
            message=f"DataFrame has {df.height} rows",
            details={"row_count": df.height},
        )

    def _check_missing_data(self, df: pl.DataFrame) -> CheckResult:
        """Verifica que no haya demasiados missing values."""
        price_cols = ["open", "high", "low", "close"]
        total = df.height

        missing_pcts = {}
        max_pct = 0.0

        for col in price_cols:
            if col in df.columns:
                missing = df.filter(pl.col(col).is_null()).height
                pct = missing / total if total > 0 else 0
                missing_pcts[col] = round(pct * 100, 2)
                max_pct = max(max_pct, pct)

        passed = max_pct <= self.max_missing_pct

        return CheckResult(
            name="missing_data",
            passed=passed,
            message=f"Max missing: {max_pct * 100:.2f}% (threshold: {self.max_missing_pct * 100:.0f}%)",
            details={"missing_pct_by_column": missing_pcts},
            severity="error" if not passed else "info",
        )

    def _check_impossible_values(self, df: pl.DataFrame) -> CheckResult:
        """Detecta valores imposibles (negativos, infinitos)."""
        issues = []

        # Precios no positivos
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                non_positive = df.filter(pl.col(col) <= 0).height
                if non_positive > 0:
                    issues.append(f"{col}: {non_positive} non-positive values")

        # Volumen negativo
        if "volume" in df.columns:
            negative_vol = df.filter(pl.col("volume") < 0).height
            if negative_vol > 0:
                issues.append(f"volume: {negative_vol} negative values")

        # Infinitos
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                inf_count = df.filter(pl.col(col).is_infinite()).height
                if inf_count > 0:
                    issues.append(f"{col}: {inf_count} infinite values")

        passed = len(issues) == 0

        return CheckResult(
            name="impossible_values",
            passed=passed,
            message="All values valid" if passed else f"Found issues: {'; '.join(issues)}",
            details={"issues": issues},
        )

    def _check_price_relationships(self, df: pl.DataFrame) -> CheckResult:
        """Verifica relaciones lógicas entre OHLC."""
        issues = []

        # high >= low
        if {"high", "low"}.issubset(df.columns):
            invalid = df.filter(pl.col("high") < pl.col("low")).height
            if invalid > 0:
                issues.append(f"{invalid} rows with high < low")

        # close between low and high
        if {"high", "low", "close"}.issubset(df.columns):
            invalid = df.filter(
                (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
            ).height
            if invalid > 0:
                issues.append(f"{invalid} rows with close outside [low, high]")

        # open between low and high
        if {"high", "low", "open"}.issubset(df.columns):
            invalid = df.filter(
                (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
            ).height
            if invalid > 0:
                issues.append(f"{invalid} rows with open outside [low, high]")

        passed = len(issues) == 0

        return CheckResult(
            name="price_relationships",
            passed=passed,
            message="All price relationships valid" if passed else f"Issues: {'; '.join(issues)}",
            details={"issues": issues},
            severity=(
                "warning" if issues else "info"
            ),  # Warning porque puede ser datos raros pero no críticos
        )

    def _check_outliers(self, df: pl.DataFrame) -> CheckResult:
        """Detecta retornos extremos (posibles errores de datos)."""
        if "close" not in df.columns or "ticker" not in df.columns:
            return CheckResult(
                name="outliers",
                passed=True,
                message="Skipped (missing columns)",
                severity="info",
            )

        # Calcular retornos por ticker
        df_with_ret = df.sort(["ticker", "date"]).with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret")
        )

        # Encontrar outliers extremos
        outliers = df_with_ret.filter(pl.col("ret").abs() > self.max_daily_return)
        outlier_count = outliers.height

        passed = outlier_count == 0

        details = {"outlier_count": outlier_count}
        if outlier_count > 0 and outlier_count <= 10:
            # Mostrar los outliers si son pocos
            details["outliers"] = outliers.select(["date", "ticker", "close", "ret"]).to_dicts()

        return CheckResult(
            name="outliers",
            passed=passed,
            message=f"Found {outlier_count} extreme returns (>{self.max_daily_return * 100:.0f}%)",
            details=details,
            severity="warning",  # Warning porque puede ser legítimo (earnings, etc)
        )

    def _check_stale_data(self, df: pl.DataFrame) -> CheckResult:
        """Detecta datos stale (mismo precio por muchos días)."""
        if "close" not in df.columns or "ticker" not in df.columns:
            return CheckResult(
                name="stale_data",
                passed=True,
                message="Skipped (missing columns)",
                severity="info",
            )

        # Contar días consecutivos sin cambio por ticker
        df_sorted = df.sort(["ticker", "date"])

        df_with_unchanged = df_sorted.with_columns(
            (pl.col("close") == pl.col("close").shift(1).over("ticker")).alias("unchanged")
        )

        # Contar rachas de unchanged
        stale_tickers = []
        for ticker in df_with_unchanged.select(pl.col("ticker").unique()).to_series():
            ticker_df = df_with_unchanged.filter(pl.col("ticker") == ticker)
            unchanged = ticker_df["unchanged"].to_list()

            max_streak = 0
            current_streak = 0
            for is_unchanged in unchanged:
                if is_unchanged:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            if max_streak >= self.max_unchanged_days:
                stale_tickers.append((ticker, max_streak))

        passed = len(stale_tickers) == 0

        return CheckResult(
            name="stale_data",
            passed=passed,
            message=f"Found {len(stale_tickers)} tickers with stale data",
            details={"stale_tickers": stale_tickers},
            severity="warning",
        )

    def _check_duplicates(self, df: pl.DataFrame) -> CheckResult:
        """Detecta filas duplicadas (mismo date+ticker)."""
        if {"date", "ticker"}.issubset(df.columns):
            unique_count = df.unique(subset=["date", "ticker"]).height
            dup_count = df.height - unique_count

            passed = dup_count == 0

            return CheckResult(
                name="duplicates",
                passed=passed,
                message=f"Found {dup_count} duplicate (date, ticker) pairs",
                details={"duplicate_count": dup_count},
            )

        return CheckResult(
            name="duplicates",
            passed=True,
            message="Skipped (missing columns)",
            severity="info",
        )


def run_quality_gates(
    df: pl.DataFrame,
    **config,
) -> tuple[bool, QualityReport]:
    """
    Función de conveniencia para ejecutar quality gates.

    Returns:
        (passed, report) - passed indica si los datos pueden pasar a clean/
    """
    gates = QualityGates(**config)
    report = gates.validate_bars(df)
    return report.passed, report
