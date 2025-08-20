# portfolio/optim/exposures.py
from __future__ import annotations
from typing import Sequence, List, Tuple, Optional, Dict
import numpy as np
import polars as pl

def build_onehot_exposure(
    names: Sequence[str],
    meta: pl.DataFrame,
    *,
    cols: Sequence[str] = ("sector", "country"),
) -> Tuple[np.ndarray, List[str]]:
    """
    Convierte columnas categóricas (sector, country, ...) en una matriz one-hot F x N
    alineada con 'names'. meta debe tener columnas: ['ticker'] + cols.
    """
    if meta is None or meta.height == 0:
        return np.zeros((0, len(names))), []

    df = meta.select(["ticker", *cols])
    # normaliza strings
    for c in cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.to_uppercase())
    # pivot wide por cada columna y acumula
    mats = []
    labels: List[str] = []
    N = len(names)
    names_upper = [s.upper() for s in names]

    for c in cols:
        if c not in df.columns:
            continue
        # valores únicos
        cats = df.select(pl.col(c).unique().drop_nulls()).to_series().to_list()
        cats = [str(x).upper() for x in cats if x is not None]
        cats_sorted = sorted(set(cats))
        F = len(cats_sorted)
        M = np.zeros((F, N), dtype=float)
        # mapa por ticker
        m = {row["ticker"].upper(): str(row[c]).upper() if row[c] is not None else None for row in df.to_dicts()}
        for j, t in enumerate(names_upper):
            v = m.get(t)
            if v is None:
                continue
            if v in cats_sorted:
                i = cats_sorted.index(v)
                M[i, j] = 1.0
        mats.append(M)
        labels.extend([f"{c}:{v}" for v in cats_sorted])

    if not mats:
        return np.zeros((0, N), dtype=float), []
    X = np.vstack(mats)
    return X, labels
