# portfolio/features/returns.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

ReturnKind = Literal["log", "simple"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: long ↔ wide
# ──────────────────────────────────────────────────────────────────────────────

def long_to_wide(
    df_long: pl.DataFrame,
    value_col: str,
    index: str = "date",
    columns: str = "ticker",
) -> pl.DataFrame:
    """
    Convierte un DF largo (date|ticker|value_col) a ancho (date + 1 col por ticker).
    """
    required = {index, columns, value_col}
    if not required.issubset(df_long.columns):
        missing = required - set(df_long.columns)
        raise ValueError(f"Missing columns in long df: {missing}")
    return df_long.pivot(values=value_col, index=index, columns=columns).sort(index)


def wide_to_long(
    df_wide: pl.DataFrame, value_name: str, index: str = "date"
) -> pl.DataFrame:
    """
    Convierte un DF ancho (date + tickers) a largo (date|ticker|value_name).
    """
    if index not in df_wide.columns:
        raise ValueError(f"'{index}' column not found in wide df")
    value_cols = [c for c in df_wide.columns if c != index]
    return (
        df_wide.melt(
            id_vars=[index],
            value_vars=value_cols,
            variable_name="ticker",
            value_name=value_name,
        )
        .sort(["ticker", index])
    )

# ──────────────────────────────────────────────────────────────────────────────
# Internos: garantías de dtype/orden temporal
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dt_sorted(df: pl.DataFrame, time_col: str = "date") -> pl.DataFrame:
    """
    Fuerza dtype temporal y orden por fecha.
    Acepta pl.Date / pl.Datetime; convierte a pl.Datetime si es necesario.
    """
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' column missing")
    col = df[time_col]
    if col.dtype not in (pl.Date, pl.Datetime):
        df = df.with_columns(pl.col(time_col).cast(pl.Datetime))
    return df.sort([time_col])

# ──────────────────────────────────────────────────────────────────────────────
# Resample de precios (largo) y retornos
# ──────────────────────────────────────────────────────────────────────────────

def resample_prices_last(df_prices_long: pl.DataFrame, every: str = "1w") -> pl.LazyFrame:
    """
    Re-muestrea precios en formato largo usando el último valor de cada ventana por ticker.
    - df_prices_long: ['date', 'ticker', 'price']
    - every: '1d', '1w', '1mo', ...
    """
    required = {"date", "ticker", "price"}
    if not required.issubset(df_prices_long.columns):
        missing = required - set(df_prices_long.columns)
        raise ValueError(f"Missing columns in prices long df: {missing}")

    df = _ensure_dt_sorted(
        df_prices_long.with_columns(
            [
                pl.col("date").cast(pl.Datetime),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("price").cast(pl.Float64),
            ]
        ),
        "date",
    )

    # Compatibilidad: usar by=["ticker"] y label="right"
    lf = (
        df.lazy()
        .group_by_dynamic(
            index_column="date",
            every=every,
            by=["ticker"],
            closed="right",
            label="right",
        )
        .agg(pl.col("price").last().alias("price"))
        .sort(["ticker", "date"])
    )
    return lf


def compute_returns_from_prices_long(
    df_prices_long: pl.DataFrame,
    *,
    freq: str = "1w",
    kind: ReturnKind = "log",
    drop_first: bool = True,
) -> pl.LazyFrame:
    """
    Retornos a partir de precios largos, con resample determinista.
    """
    lf = resample_prices_last(df_prices_long, every=freq)

    if kind == "log":
        ret_expr = (pl.col("price") / pl.col("price").shift(1)).log()
    elif kind == "simple":
        ret_expr = (pl.col("price") / pl.col("price").shift(1) - 1.0)
    else:
        raise ValueError("kind must be 'log' or 'simple'")

    out = lf.with_columns(ret=ret_expr)
    if drop_first:
        out = out.filter(pl.col("ret").is_not_null())
    return out.select(["date", "ticker", "ret"])


def compute_returns_from_prices_wide(
    df_prices_wide: pl.DataFrame, *, kind: ReturnKind = "log"
) -> pl.DataFrame:
    """
    Retornos a partir de precios anchos (date + tickers). Sin resample (usa el índice tal cual).
    """
    if "date" not in df_prices_wide.columns:
        raise ValueError("'date' column is required.")
    tickers = [c for c in df_prices_wide.columns if c != "date"]
    if not tickers:
        raise ValueError("No ticker columns found.")
    lf = df_prices_wide.lazy().sort("date")
    exprs = []
    for t in tickers:
        expr = (
            (pl.col(t) / pl.col(t).shift(1)).log()
            if kind == "log"
            else (pl.col(t) / pl.col(t).shift(1) - 1.0)
        )
        exprs.append(expr.alias(t))
    out = lf.with_columns(exprs)
    out = out.drop_nulls(subset=[tickers[0]])  # descarta la primera fila de referencia
    return out.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Winsorización por ticker (largo y ancho)
# ──────────────────────────────────────────────────────────────────────────────

def winsorize_long(df_ret_long: pl.DataFrame, ret_col: str = "ret", q: float = 0.01) -> pl.DataFrame:
    """
    Recorta colas por percentiles por ticker (default 1%).
    Devuelve columnas: ['date','ticker','ret_w'].
    """
    if not {"date", "ticker", ret_col}.issubset(df_ret_long.columns):
        raise ValueError("df_ret_long must contain ['date','ticker', ret_col]")

    qlow, qhigh = float(q), 1.0 - float(q)
    base = df_ret_long.lazy()
    qs = base.group_by("ticker").agg(
        [
            pl.col(ret_col).quantile(qlow).alias("q_low"),
            pl.col(ret_col).quantile(qhigh).alias("q_high"),
        ]
    )
    out = (
        qs.join(base, on="ticker", how="right")
        .with_columns(
            pl.when(pl.col(ret_col) < pl.col("q_low"))
            .then(pl.col("q_low"))
            .when(pl.col(ret_col) > pl.col("q_high"))
            .then(pl.col("q_high"))
            .otherwise(pl.col(ret_col))
            .alias("ret_w")
        )
        .select(["date", "ticker", "ret_w"])
        .sort(["ticker", "date"])
        .collect()
    )
    return out


def winsorize_wide(df_ret_wide: pl.DataFrame, q: float = 0.01) -> pl.DataFrame:
    """
    Winsoriza columnas de retornos (ancho) de forma independiente (por columna).
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("'date' column required.")
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    lf = df_ret_wide.lazy()
    exprs = []
    for t in tickers:
        lo = pl.col(t).quantile(q)
        hi = pl.col(t).quantile(1 - q)
        exprs.append(pl.col(t).clip(lo, hi).alias(t))
    return lf.with_columns(exprs).collect()

# ──────────────────────────────────────────────────────────────────────────────
# Conversión de frecuencia de retornos (log vs simple)
# ──────────────────────────────────────────────────────────────────────────────

def infer_return_kind(df_ret_wide: pl.DataFrame, sample_cols: Optional[int] = 5) -> ReturnKind:
    """
    Heurística robusta:
    - si la mayoría de |r| <= 0.3 y no hay valores <-1, asumimos 'simple' si min > -1
    - si hay valores <= -1 es imposible en 'simple' ⇒ 'log'
    - si las magnitudes son grandes (|r|>0.5 frecuente), probablemente 'log'
    """
    cols = [c for c in df_ret_wide.columns if c != "date"]
    if sample_cols is not None:
        cols = cols[: max(1, min(sample_cols, len(cols)))]
    X = df_ret_wide.select(cols).to_numpy()
    if np.isneginf(X).any() or (X <= -1.0).any():
        return "log"
    med_abs = np.nanmedian(np.abs(X))
    return "simple" if med_abs < 0.3 else "log"


def returns_to_frequency_wide(
    df_ret_wide: pl.DataFrame,
    *,
    freq: str,
    kind: Optional[ReturnKind] = None,
    min_count: int = 1,  # reservado; no se usa en esta implementación
) -> pl.DataFrame:
    """
    Agrega retornos a frecuencia 'freq' respetando la composición:
    - log: suma
    - simple: (1+r).prod - 1
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("'date' column required.")
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    if not tickers:
        return df_ret_wide

    if kind is None:
        kind = infer_return_kind(df_ret_wide)

    lf = df_ret_wide.lazy().sort("date")
    if kind == "log":
        agg_exprs = [pl.col(t).sum().alias(t) for t in tickers]
    else:
        agg_exprs = [((1.0 + pl.col(t)).product() - 1.0).alias(t) for t in tickers]

    # Compat: evitamos all_horizontal con generadores (polars antiguos)
    out = (
        lf.group_by_dynamic("date", every=freq, closed="right", label="right")
        .agg(agg_exprs)
        .sort("date")
    )
    return out.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Estadísticos, annualización y reportes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Periodicity:
    """Escalas típicas."""
    per_year: int  # 252, 52, 12 ...


def annualize_mean(mu_periodic: np.ndarray | pl.Series, period: Periodicity) -> np.ndarray:
    return np.asarray(mu_periodic) * float(period.per_year)


def annualize_vol(sigma_periodic: np.ndarray | pl.Series, period: Periodicity) -> np.ndarray:
    return np.asarray(sigma_periodic) * np.sqrt(float(period.per_year))


def summary_stats(
    df_ret_wide: pl.DataFrame,
    *,
    risk_free: float = 0.0,  # rf periodic (misma frecuencia que df_ret_wide)
    ddof: int = 1,
) -> pl.DataFrame:
    """
    Estadísticos por activo: mean, std, skew, kurt, Sharpe, n_obs, missing_pct.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("'date' column required.")
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    if not tickers:
        return pl.DataFrame(
            rows,
            schema=["ticker", "mean", "std", "skew", "kurt", "sharpe", "n_obs", "missing_pct"],
            orient="row",   
        )


    # Agregaciones de toda la tabla -> .select() (no .agg() en LazyFrame)
    aggs = []
    for t in tickers:
        r = pl.col(t)
        aggs.extend(
            [
                r.mean().alias(f"{t}__mean"),
                r.std(ddof=ddof).alias(f"{t}__std"),
                r.skew().alias(f"{t}__skew"),
                r.kurtosis().alias(f"{t}__kurt"),
                pl.len().alias(f"{t}__n_total"),
                r.is_null().sum().alias(f"{t}__n_na"),
            ]
        )
    tmp = df_ret_wide.select(aggs)  # ← clave

    rows = []
    total_rows = df_ret_wide.height
    for t in tickers:
        mean = tmp.select(f"{t}__mean").item()
        stdv = tmp.select(f"{t}__std").item()
        skew = tmp.select(f"{t}__skew").item()
        kurt = tmp.select(f"{t}__kurt").item()
        n_total = tmp.select(f"{t}__n_total").item()
        n_na = tmp.select(f"{t}__n_na").item()
        n_obs = (n_total - n_na) if (n_total is not None and n_na is not None) else None
        sharpe = ((mean - risk_free) / stdv) if (stdv and stdv > 0 and mean is not None) else None
        missing_pct = (n_na / n_total * 100.0) if n_total else None
        rows.append((t, mean, stdv, skew, kurt, sharpe, n_obs, missing_pct))

    return pl.DataFrame(
        rows, schema=["ticker", "mean", "std", "skew", "kurt", "sharpe", "n_obs", "missing_pct"]
    )


def missing_report_wide(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reporte de missing y si la serie termina en missing (útil para detectar fallos de ingestión por ticker).
    """
    tickers = [c for c in df.columns if c != "date"]
    lf = df.lazy()
    aggs = []
    for t in tickers:
        aggs.extend(
            [
                pl.col(t).is_null().sum().alias(f"{t}__miss"),
                pl.last(pl.col(t)).is_null().cast(pl.Int8).alias(f"{t}__ends_missing"),
            ]
        )
    tmp = lf.agg(aggs).collect()
    rows = []
    total = df.height
    for t in tickers:
        miss = tmp.select(f"{t}__miss").item()
        endm = tmp.select(f"{t}__ends_missing").item()
        pct = (miss / total * 100.0) if total else None
        rows.append((t, miss, pct, bool(endm)))
    return pl.DataFrame(
        rows,
        schema=["ticker", "missing_rows", "missing_pct", "ends_missing"],
        orient="row",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Compatibilidad pandas (por si tienes celdas/notebooks antiguos)
# ──────────────────────────────────────────────────────────────────────────────

def simple_returns_pd(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Equivalente a tu función original, implementado con Polars bajo el capó cuando es útil.
    """
    df = prices.sort_index()
    out = df.pct_change().iloc[1:]
    return out

def log_returns_pd(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_index()
    out = np.log(df).diff().iloc[1:]
    return out

def to_frequency_pd(returns: pd.DataFrame, freq: str) -> pd.DataFrame:
    if returns.empty:
        return returns
    # Inferencia simple: si hay valores <= -1 → log
    is_log_like = (returns <= -1.0).any().any() or (returns.abs().median() > 0.25).any()
    if is_log_like:
        out = returns.resample(freq).sum(min_count=1)
    else:
        out = (1.0 + returns).resample(freq).prod(min_count=1) - 1.0
    return out.dropna(how="all")

def winsorize_pd(returns: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    lo = returns.quantile(q)
    hi = returns.quantile(1 - q)
    return returns.clip(lower=lo, upper=hi, axis=1)

def missing_report_wide(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reporte de missing y si la serie termina en missing (útil para detectar fallos por ticker).
    """
    if "date" not in df.columns:
        raise ValueError("'date' column required.")
    tickers = [c for c in df.columns if c != "date"]
    if not tickers:
        return pl.DataFrame(
            {"ticker": [], "missing_rows": [], "missing_pct": [], "ends_missing": []}
        )

    # Agregaciones a nivel de toda la tabla → usar .select(), no .agg()
    aggs = []
    for t in tickers:
        aggs.extend(
            [
                pl.col(t).is_null().sum().alias(f"{t}__miss"),
                pl.col(t).last().is_null().cast(pl.Int8).alias(f"{t}__ends_missing"),
            ]
        )

    tmp = df.select(aggs)  # <— cambio clave: select en DF (o lf.select si quisieras lazy)

    total = df.height
    rows = []
    for t in tickers:
        miss = tmp.select(f"{t}__miss").item()
        endm = tmp.select(f"{t}__ends_missing").item()
        pct = (miss / total * 100.0) if total else None
        rows.append((t, miss, pct, bool(endm)))

    return pl.DataFrame(
        rows, schema=["ticker", "missing_rows", "missing_pct", "ends_missing"]
    )

