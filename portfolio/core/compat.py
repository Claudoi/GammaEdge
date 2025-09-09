# portfolio/core/compat.py
from __future__ import annotations

from dataclasses import dataclass as _dataclass

# Decorador compatible: permite usar frozen+slots aunque Python <3.10 no soporte slots
def dataclass_compat(*args, **kwargs):
    kwargs.pop("slots", None)  # en 3.9 no existe
    return _dataclass(*args, **kwargs)

# UTC compatible en todas las versiones
try:
    from datetime import UTC  # Py 3.11+
except ImportError:
    from datetime import timezone as _tz  # Py 3.9/3.10
    UTC = _tz.utc