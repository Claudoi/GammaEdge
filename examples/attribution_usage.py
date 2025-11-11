from __future__ import annotations

import os
import sys

import pandas as pd
import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from portfolio.attribution.euler import euler_risk_contributions
from portfolio.backtest.attribution_reporting import build_brinson_attribution_report


def demo_euler() -> None:
    """
    Pequeño ejemplo de contribuciones de riesgo tipo Euler
    para una cartera de 2 activos.
    """
    # Pesos de la cartera
    weights = pd.Series([0.6, 0.4], index=["A", "B"])

    # Matriz de covarianzas simple
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    rc = euler_risk_contributions(weights, cov)

    print("=== Euler risk contributions example ===")
    print("Weights:")
    print(weights)
    print("\nCovariance matrix:")
    print(cov)
    print("\nRisk contributions (sum = portfolio volatility):")
    print(rc)
    print(f"\nTotal σ_p (from contributions): {rc.sum():.4f}")
    print("-" * 60)


def demo_brinson() -> None:
    """
    Ejemplo mínimo de atribución tipo Brinson usando el helper
    de reporting de alto nivel.
    """
    # DataFrame en formato "ancho" como en los tests de Brinson:
    # 2 fechas × 2 grupos (0 y 1) con métricas alloc/select/interact/total.
    df = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "alloc_0": [0.1, 0.2],
            "alloc_1": [0.3, 0.4],
            "select_0": [0.5, 0.6],
            "select_1": [0.7, 0.8],
            "interact_0": [0.0, 0.0],
            "interact_1": [0.1, 0.2],
            "total_0": [0.6, 0.8],
            "total_1": [0.9, 1.0],
        }
    )

    report = build_brinson_attribution_report(df, how="sum")

    print("=== Brinson attribution example ===")
    print("\nNormalized long timeseries (timeseries):")
    print(report["timeseries"])

    print("\nAggregated by group (by_group):")
    print(report["by_group"])

    print("\nTotal metrics over the full period (total):")
    print(report["total"])
    print("-" * 60)


def main() -> None:
    demo_euler()
    demo_brinson()


if __name__ == "__main__":
    main()
