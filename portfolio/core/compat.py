# portfolio/core/compat.py
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass as _dataclass

# Decorador compatible con "slots" incluso en 3.9 (los ignora si no existen)
def dataclass_compat(*args, **kwargs):
    kwargs.pop("slots", None)  # en 3.9 no existe
    return _dataclass(*args, **kwargs)

# Syntactic sugar para "frozen+slots" donde se pueda
def dataclass_frozen_slots():
    def _wrap(cls):
        params = {"frozen": True}
        # "slots" solo si la implementación lo soporta
        try:
            return _dataclass(**params, slots=True)(cls)  # type: ignore[arg-type]
        except TypeError:
            return _dataclass(**params)(cls)
    return _wrap

# UTC compatible sin disparar UP017 (usa getattr)
UTC = getattr(_dt, "UTC", _dt.timezone.utc)