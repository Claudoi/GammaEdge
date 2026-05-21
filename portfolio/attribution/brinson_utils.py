# portfolio/attribution/brinson_utils.py
"""
Brinson coercion utilities.

These helpers live under ``portfolio.attribution`` because Brinson is an
attribution concept. Keeping them here breaks the historical import cycle
between ``portfolio.backtest`` and ``portfolio.attribution``:

    backtest.attribution_reporting -> attribution.brinson -> attribution.brinson_utils

The previous home was ``portfolio.backtest.brinson_utils``, which still
re-exports these names for backward compatibility.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl

__all__ = [
    "ensure_datetime",
    "coerce_brinson_timeseries_to_long",
]


# ────────────────────────────────────────────────────────────────────
# ensure_datetime: robusto a Object/NaT, con tipado limpio
# ────────────────────────────────────────────────────────────────────
def ensure_datetime(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    """Garantiza que df[col] sea pl.Datetime con valores válidos (no todo null)."""
    if col not in df.columns:
        return df

    dt = df.schema.get(col)
    if dt == pl.Datetime:
        return df

    # 1) Si viene como string, intenta strptime primero (mejor que cast).
    if dt == pl.Utf8:
        try:
            df2 = df.with_columns(pl.col(col).str.strptime(pl.Datetime, strict=False))
            # Acepta solo si realmente hay algún valor no nulo
            if (
                df2.schema.get(col) == pl.Datetime
                and df2.select(pl.col(col).is_not_null().any()).item()
            ):
                return df2
        except Exception:
            pass

    # 2) Intento de cast directo; acepta solo si no queda todo a null.
    try:
        df1 = df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
        if (
            df1.schema.get(col) == pl.Datetime
            and df1.select(pl.col(col).is_not_null().any()).item()
        ):
            return df1
    except Exception:
        pass

    # 3) Fallback robusto vía pandas
    try:
        s = df.get_column(col)
        pd_dt = pd.to_datetime(list(s), errors="coerce")
        py_vals: list[datetime | None] = [
            x.to_pydatetime() if not pd.isna(x) else None for x in pd_dt
        ]
        ser = pl.Series(col, py_vals).cast(pl.Datetime, strict=False)
        df3 = df.with_columns(ser)
        return df3
    except Exception:
        return df


# ────────────────────────────────────────────────────────────────────
# Brinson: coerción a formato largo estándar
# ────────────────────────────────────────────────────────────────────
def coerce_brinson_timeseries_to_long(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normaliza a ['date', 'group_id', 'alloc', 'select', 'interact', 'total'].
    Soporta:
      - Formato largo con 'group_id'
      - Formato largo con 'group' (etiquetas -> ids)
      - Formato ancho con columnas tipo 'alloc_0', 'alloc_1', etc.
    Regla clave: sólo conserva pares (date, group_id) realmente presentes en los datos.
    """
    # Asegura que 'date' sea Datetime desde el inicio (evita NaT/None posteriores)
    if "date" in df.columns:
        df = ensure_datetime(df, "date")

    cols = set(df.columns)
    metrics = ["alloc", "select", "interact", "total"]
    need = {"date", *metrics}

    # 0) Global-only (inyecta group_id = 0)
    if need.issubset(cols) and "group_id" not in cols and "group" not in cols:
        out0 = (
            df.select(list(need))
            .with_columns(pl.lit(0).alias("group_id"))
            .select(["date", "group_id"] + metrics)
        )
        return out0.with_columns(pl.col("group_id").cast(pl.Int64, strict=False))

    # 1) Largo con group_id
    if {"group_id", *need}.issubset(cols):
        out1 = df.select(["date", "group_id"] + metrics)
        return out1.with_columns(pl.col("group_id").cast(pl.Int64, strict=False))

    # 2) Largo con group (labels string -> ids)
    if {"group", *need}.issubset(cols):
        uniq = sorted([g for g in df.get_column("group").unique().to_list() if g is not None])
        mapping: dict[str, int] = {g: i for i, g in enumerate(uniq)}
        df2 = df.with_columns(
            pl.col("group").map_elements(lambda g: mapping.get(g, -1)).alias("group_id")
        ).select(["date", "group_id"] + metrics)
        return df2.with_columns(pl.col("group_id").cast(pl.Int64, strict=False))

    # 3) Ancho metric_{gid}: melt por métrica, concat y pivot (sin joins)
    pattern_cols: dict[str, list[str]] = {}
    for m in metrics:
        cs = [c for c in df.columns if c.startswith(m + "_") or c.startswith(m + "_g")]
        if not cs:
            cs = [
                c for c in df.columns if c.startswith(m) and any(ch.isdigit() for ch in c[len(m) :])
            ]
        pattern_cols[m] = cs

    if any(pattern_cols[m] for m in metrics):

        def _parse_gid(name: str) -> int:
            # 'm_g0', 'm_0', 'm0' -> 0
            tail = name.split("_", 1)[-1] if "_" in name else name[len(name.rstrip("0123456789")) :]
            if tail.startswith(("g", "G")):
                tail = tail[1:]
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else -1

        long_rows: list[pl.DataFrame] = []
        for m in metrics:
            cs = pattern_cols[m]
            if not cs:
                continue

            melted = (
                df.select(["date"] + cs)
                .melt(id_vars="date", variable_name="col", value_name="value")
                .with_columns(
                    pl.col("value").cast(pl.Float64, strict=False),
                    pl.col("col").map_elements(_parse_gid).alias("group_id"),
                    pl.lit(m).alias("metric"),
                )
                .drop("col")
            )
            # Conserva sólo observaciones reales
            melted = melted.filter(pl.col("value").is_not_null())

            # Compacta por seguridad (si hubiera duplicados accidentales)
            melted = melted.group_by(["date", "group_id", "metric"]).agg(
                pl.col("value").first().alias("value")
            )

            long_rows.append(melted)

        if not long_rows:
            raise ValueError("No wide metric_* columns found.")

        stacked = pl.concat(long_rows, how="vertical_relaxed").with_columns(
            pl.col("group_id").cast(pl.Int64, strict=False),
            pl.col("metric").cast(pl.Utf8, strict=False),
        )

        # Pivot a ancho: una fila por (date, group_id), columnas = métricas
        out = stacked.pivot(
            values="value",
            index=["date", "group_id"],
            on="metric",
            aggregate_function="first",
        ).sort(["date", "group_id"])

        # Garantiza que existan todas las métricas como columnas
        for m in metrics:
            if m not in out.columns:
                out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(m))

        return out.select(["date", "group_id"] + metrics)

    # Si no encaja en ninguno de los formatos soportados:
    raise ValueError(
        "Unsupported Brinson timeseries format: neither {group_id, group} nor wide metric_* columns found. "
        f"Available columns: {sorted(df.columns)}"
    )
