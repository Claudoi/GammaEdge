from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl


# ────────────────────────────────────────────────────────────────────
# ensure_datetime: robusto a Object/NaT, con tipado limpio
# ────────────────────────────────────────────────────────────────────
def ensure_datetime(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    dt = df.schema.get(col)
    if dt == pl.Datetime:
        return df

    # Intento directo
    try:
        df1 = df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
        if df1.schema.get(col) == pl.Datetime:
            return df1
    except Exception:
        pass

    # Parse desde string
    try:
        df2 = df.with_columns(
            pl.col(col).cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, strict=False)
        )
        if df2.schema.get(col) == pl.Datetime:
            return df2
    except Exception:
        pass

    # Fallback robusto vía pandas
    try:
        s = df.get_column(col)
        pd_dt = pd.to_datetime(list(s), errors="coerce")
        py_vals: list[datetime | None] = [
            x.to_pydatetime() if not pd.isna(x) else None for x in pd_dt
        ]
        ser = pl.Series(col, py_vals).cast(pl.Datetime, strict=False)
        return df.with_columns(ser)
    except Exception:
        return df


# ────────────────────────────────────────────────────────────────────
# coerce_brinson_timeseries_to_long: soporta long/group_id, long/group,
# wide metric_{gid}, e “inyecta” group_id=0 cuando es global-only.
# Además: joins robustos para evitar date_right por discrepancias.
# ────────────────────────────────────────────────────────────────────
def coerce_brinson_timeseries_to_long(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normaliza el timeseries de Brinson al formato largo estándar:
      ['date', 'group_id', 'alloc', 'select', 'interact', 'total']
    """
    cols = set(df.columns)
    metrics = ["alloc", "select", "interact", "total"]
    need = {"date", *metrics}

    # 0) global-only (inyecta group_id=0)
    if need.issubset(cols) and "group_id" not in cols and "group" not in cols:
        out0 = (
            df.select(list(need))
            .with_columns(pl.lit(0).alias("group_id"))
            .select(["date", "group_id"] + metrics)
        )
        return ensure_datetime(out0, "date").with_columns(
            pl.col("group_id").cast(pl.Int64, strict=False)
        )

    # 1) long con group_id
    if {"group_id", *need}.issubset(cols):
        out1 = df.select(["date", "group_id"] + metrics)
        out1 = ensure_datetime(out1, "date").with_columns(
            pl.col("group_id").cast(pl.Int64, strict=False)
        )
        return out1

    # 2) long con group (labels string)
    if {"group", *need}.issubset(cols):
        uniq = sorted([g for g in df.get_column("group").unique().to_list() if g is not None])
        mapping: dict[str, int] = {g: i for i, g in enumerate(uniq)}
        df2 = df.with_columns(
            pl.col("group").map_elements(lambda g: mapping.get(g, -1)).alias("group_id")
        ).select(["date", "group_id"] + metrics)
        df2 = ensure_datetime(df2, "date").with_columns(
            pl.col("group_id").cast(pl.Int64, strict=False)
        )
        return df2

    # 3) wide metric_{gid}
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
            # sufijo tras '_' o trailing digits; soporta 'm_g0', 'm_0', 'm0'
            tail = name.split("_", 1)[-1] if "_" in name else name[len(name.rstrip("0123456789")) :]
            if tail.startswith(("g", "G")):
                tail = tail[1:]
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else -1

        long_parts: list[pl.DataFrame] = []
        for m in metrics:
            cs = pattern_cols[m]
            if not cs:
                continue
            melted = df.select(["date"] + cs).melt(
                id_vars="date", variable_name="col", value_name=m
            )
            melted = melted.with_columns(
                pl.col(m).cast(pl.Float64, strict=False),
                pl.col("col").map_elements(_parse_gid).alias("group_id"),
            ).drop("col")
            # Normaliza claves antes de juntar
            melted = ensure_datetime(melted, "date").with_columns(
                pl.col("group_id").cast(pl.Int64, strict=False)
            )
            long_parts.append(melted.select(["date", "group_id", m]))

        # Ensambla con joins robustos: coalesce y drop de sufijos si aparecen
        out = long_parts[0]
        for part in long_parts[1:]:
            out = out.join(part, on=["date", "group_id"], how="outer", suffix="_r")
            # saneo defensivo de claves duplicadas
            if "date_r" in out.columns:
                out = out.with_columns(
                    pl.coalesce([pl.col("date"), pl.col("date_r")]).alias("date")
                ).drop("date_r")
            if "date_right" in out.columns:
                out = out.with_columns(
                    pl.coalesce([pl.col("date"), pl.col("date_right")]).alias("date")
                ).drop("date_right")
            if "group_id_r" in out.columns:
                out = out.with_columns(
                    pl.coalesce([pl.col("group_id"), pl.col("group_id_r")]).alias("group_id")
                ).drop("group_id_r")
            if "group_id_right" in out.columns:
                out = out.with_columns(
                    pl.coalesce([pl.col("group_id"), pl.col("group_id_right")]).alias("group_id")
                ).drop("group_id_right")

        out = out.select(["date", "group_id"] + [m for m in metrics if m in out.columns])
        out = ensure_datetime(out, "date").with_columns(
            pl.col("group_id").cast(pl.Int64, strict=False)
        )
        # Asegura que todas las métricas existan, aunque vengan ausentes del wide
        for m in metrics:
            if m not in out.columns:
                out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(m))
        return out.select(["date", "group_id"] + metrics)

    raise ValueError(
        "Unsupported Brinson timeseries format: neither {group_id, group} nor wide metric_* columns found. "
        f"Available columns: {sorted(df.columns)}"
    )
