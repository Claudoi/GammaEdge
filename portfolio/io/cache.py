# portfolio/io/cache.py
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Versión de esquema de artefactos: si cambias estructura/semántica, súbela
SCHEMA_VERSION = "v1"

# Compresión “global” preferida (rápida y con buen ratio)
# Nota: esta lista incluye valores que Polars acepta y pandas podría no aceptar.
ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]
PARQUET_COMPRESSION: ParquetCompression = "zstd"

# Subconjunto que pandas (pyarrow/fastparquet) reconoce por tipo en mypy
PandasParquetCompression = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: hashing, paths, atomic writes, file locks
# ──────────────────────────────────────────────────────────────────────────────



def _normalize_for_hash(obj: Any) -> Any:
    """
    Convierte objetos no serializables en representaciones estables.
    - Dicts y listas se ordenan/normalizan
    - Objetos se pasan a str()
    """
    if isinstance(obj, Mapping):
        return {
            str(k): _normalize_for_hash(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
        }
    if isinstance(obj, (list, tuple, set)):  # noqa: UP038
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:  # noqa: UP038
        return obj
    # fechas, paths, enums, numpy scalars, etc.
    return str(obj)


def _hash_config(cfg: Mapping[str, Any], schema_version: str = SCHEMA_VERSION) -> str:
    payload = {"schema": schema_version, **_normalize_for_hash(cfg)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_path(kind: str, cfg: Mapping[str, Any], ext: str = "parquet") -> Path:
    """
    Devuelve la ruta del artefacto en caché para (kind, cfg).
    """
    h = _hash_config(cfg)
    return CACHE_DIR / f"{kind}_{h}.{ext}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic rename en la mayoría de FS


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout: float = 10.0, poll: float = 0.05) -> Iterator[None]:
    """
    Lock muy simple basado en archivo .lock.
    Evita condiciones de carrera en escrituras concurrentes.
    """
    deadline = time.time() + timeout
    fd: int | None = None
    while True:
        try:
            # O_EXCL + O_CREAT aseguran exclusión
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if time.time() > deadline:
                # Si no se obtiene el lock, continuamos sin bloquear (fail-open)
                fd = None
                break
            time.sleep(poll)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                os.remove(lock_path)


def exists(kind: str, cfg: Mapping[str, Any], ext: str = "parquet") -> bool:
    return cache_path(kind, cfg, ext).exists()


def age_seconds(kind: str, cfg: Mapping[str, Any], ext: str = "parquet") -> float | None:
    p = cache_path(kind, cfg, ext)
    if not p.exists():
        return None
    return time.time() - p.stat().st_mtime


def invalidate(kind: str, cfg: Mapping[str, Any], ext: str = "parquet") -> None:
    p = cache_path(kind, cfg, ext)
    with contextlib.suppress(FileNotFoundError):
        os.remove(p)


# ──────────────────────────────────────────────────────────────────────────────
# Mapeo de compresión global → compresión aceptada por pandas
# ──────────────────────────────────────────────────────────────────────────────


def _pandas_compression_from(
    comp: ParquetCompression,
) -> PandasParquetCompression | None:
    """
    Pandas/pyarrow no acepta "uncompressed" ni "lzo" como literales de compresión.
    - Para "uncompressed"/"lzo", devolvemos None (sin compresión).
    - Para el resto, mapeamos 1:1.
    """
    if comp in ("snappy", "gzip", "brotli", "lz4", "zstd"):
        # cast garantiza a mypy que el valor pertenece al literal reducido
        return cast(PandasParquetCompression, comp)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pandas API (retro-compat)
# ──────────────────────────────────────────────────────────────────────────────


def save_df(kind: str, cfg: Mapping[str, Any], df: pd.DataFrame) -> Path:
    """
    Guarda pandas.DataFrame a parquet (pyarrow) con escritura atómica.
    Usa el mapeo de compresión compatible con pandas.
    """
    p = cache_path(kind, cfg, ext="parquet")
    lock = p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock):
        tmp = p.with_suffix(p.suffix + ".tmp")
        p.parent.mkdir(parents=True, exist_ok=True)
        comp_pd = _pandas_compression_from(PARQUET_COMPRESSION)
        df.to_parquet(tmp, engine="pyarrow", compression=comp_pd, index=False)
        os.replace(tmp, p)
    return p


def load_df(kind: str, cfg: Mapping[str, Any]) -> pd.DataFrame | None:
    """
    Carga pandas.DataFrame desde parquet si existe, None si no.
    """
    p = cache_path(kind, cfg, ext="parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ──────────────────────────────────────────────────────────────────────────────
# Polars API (nativa y rápida)
# ──────────────────────────────────────────────────────────────────────────────


def save_pl(
    kind: str,
    cfg: Mapping[str, Any],
    df: pl.DataFrame,
    compression: ParquetCompression = PARQUET_COMPRESSION,
) -> Path:
    """
    Guarda Polars DataFrame a Parquet; Polars acepta más valores de compresión
    (p. ej., 'uncompressed', 'lzo') que pandas.
    """
    p = cache_path(kind, cfg, ext="parquet")
    lock = p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock):
        tmp = p.with_suffix(p.suffix + ".tmp")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(tmp, compression=compression)
        os.replace(tmp, p)
    return p


def load_pl(kind: str, cfg: Mapping[str, Any]) -> pl.DataFrame | None:
    p = cache_path(kind, cfg, ext="parquet")
    if not p.exists():
        return None
    return pl.read_parquet(p)


# ──────────────────────────────────────────────────────────────────────────────
# NumPy matrices/arrays (ideal para Σ, fronteras, etc.)
# ──────────────────────────────────────────────────────────────────────────────


def save_np(kind: str, cfg: Mapping[str, Any], arr: np.ndarray, compressed: bool = True) -> Path:
    ext = "npz" if compressed else "npy"
    p = cache_path(kind, cfg, ext=ext)
    lock = p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock):
        tmp = p.with_suffix(p.suffix + ".tmp")
        p.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            np.savez_compressed(tmp, data=arr)
        else:
            np.save(tmp, arr)
        os.replace(tmp, p)
    return p


def load_np(kind: str, cfg: Mapping[str, Any]) -> np.ndarray | None:
    p_npz = cache_path(kind, cfg, ext="npz")
    p_npy = cache_path(kind, cfg, ext="npy")
    if p_npz.exists():
        with np.load(p_npz) as f:
            return cast(np.ndarray, f["data"])
    if p_npy.exists():
        return cast(np.ndarray, np.load(p_npy))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# JSON (metadatos ligeros)
# ──────────────────────────────────────────────────────────────────────────────


def save_json(kind: str, cfg: Mapping[str, Any], obj: Mapping[str, Any]) -> Path:
    p = cache_path(kind, cfg, ext="json")
    lock = p.with_suffix(p.suffix + ".lock")
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    with _file_lock(lock):
        _atomic_write_bytes(p, data)
    return p


def load_json(kind: str, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    p = cache_path(kind, cfg, ext="json")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return cast(dict[str, Any], json.loads(f.read().decode("utf-8")))
