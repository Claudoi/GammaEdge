# Caching utilities
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _hash_config(cfg: Dict[str, Any]) -> str:
    s = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def cache_path(kind: str, cfg: Dict[str, Any]) -> Path:
    h = _hash_config(cfg)
    return CACHE_DIR / f"{kind}_{h}.parquet"

def save_df(kind: str, cfg: Dict[str, Any], df: pd.DataFrame) -> Path:
    p = cache_path(kind, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return p

def load_df(kind: str, cfg: Dict[str, Any]) -> pd.DataFrame | None:
    p = cache_path(kind, cfg)
    if p.exists():
        return pd.read_parquet(p)
    return None

def invalidate(kind: str, cfg: Dict[str, Any]) -> None:
    p = cache_path(kind, cfg)
    if p.exists():
        os.remove(p)
