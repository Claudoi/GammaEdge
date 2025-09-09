# portfolio/io/cache.py
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

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

# Compresión por defecto para Parquet (rápida y con buen ratio)
PARQUET_COMPRESSION = "zstd"  # requiere pyarrow instalado


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
        return {str(k): _normalize_for_hash(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # fechas, paths, enums, etc.
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
def _file_lock(lock_path: Path, timeout: float = 10.0, poll: float = 0.05):
    """
    Lock muy simple basado en archivo .lock.
    Evita condiciones de carrera en escrituras concurrentes.
    """
    deadline = time.time() + timeout
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


def age_seconds(kind: str, cfg: Mapping[str, Any], ext: str = "parquet") -> Optional[float]:
    p = cache_path(kind, cfg, ext)
    if not p.exists():
        return None
    return time.time() - p.stat().st_mtime


def invalidate(kind: str, cfg: Mapping[str, Any], ext: str = "parquet") -> None:
    p = cache_path(kind, cfg, ext)
    with contextlib.suppress(FileNotFoundError):
        os.remove(p)


# ──────────────────────────────────────────────────────────────────────────────
# Pandas API (retro-compat)
# ──────────────────────────────────────────────────────────────────────────────

def save_df(kind: str, cfg: Mapping[str, Any], df: pd.DataFrame) -> Path:
    """
    Guarda pandas.DataFrame a parquet (pyarrow) con compresión Zstd usando escritura atómica.
    """
    p = cache_path(kind, cfg, ext="parquet")
    lock = p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock):
        # usamos to_parquet en buffer para conservar atomicidad
        # pandas no expone fácilmente escritura a bytes; usamos pyarrow vía to_parquet en fichero temporal
        # así que persistimos a .tmp y renombramos
        tmp = p.with_suffix(p.suffix + ".tmp")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp, engine="pyarrow", compression=PARQUET_COMPRESSION, index=False)
        os.replace(tmp, p)
    return p


def load_df(kind: str, cfg: Mapping[str, Any]) -> Optional[pd.DataFrame]:
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

def save_pl(kind: str, cfg: Mapping[str, Any], df: pl.DataFrame) -> Path:
    """
    Guarda polars.DataFrame a parquet con compresión Zstd y escritura atómica.
    """
    p = cache_path(kind, cfg, ext="parquet")
    lock = p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock):
        tmp = p.with_suffix(p.suffix + ".tmp")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(tmp, compression=PARQUET_COMPRESSION)
        os.replace(tmp, p)
    return p


def load_pl(kind: str, cfg: Mapping[str, Any]) -> Optional[pl.DataFrame]:
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


def load_np(kind: str, cfg: Mapping[str, Any]) -> Optional[np.ndarray]:
    p_npz = cache_path(kind, cfg, ext="npz")
    p_npy = cache_path(kind, cfg, ext="npy")
    if p_npz.exists():
        with np.load(p_npz) as f:
            return f["data"]
    if p_npy.exists():
        return np.load(p_npy)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# JSON (metadatos ligeros)
# ──────────────────────────────────────────────────────────────────────────────

def save_json(kind: str, cfg: Mapping[str, Any], obj: Mapping[str, Any]) -> Path:
    p = cache_path(kind, cfg, ext="json")
    lock = p.with_suffix(p.suffix + ".lock")
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with _file_lock(lock):
        _atomic_write_bytes(p, data)
    return p


def load_json(kind: str, cfg: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    p = cache_path(kind, cfg, ext="json")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
