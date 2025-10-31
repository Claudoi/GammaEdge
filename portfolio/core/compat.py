# portfolio/core/compat.py
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass as _dataclass
from typing import Any, Callable, cast, dataclass_transform

try:
    from typing import dataclass_transform  # py311+
except Exception:  # py310 y anteriores
    from typing_extensions import dataclass_transform


# Decorador compatible con "slots" (los ignora si no existen) y visible para mypy
@dataclass_transform()
def dataclass_compat(*args: Any, **kwargs: Any) -> Callable[[type], type]:
    kwargs.pop("slots", None)  # en 3.9 no existe

    def _wrap(cls: type) -> type:
        wrapped = _dataclass(*args, **kwargs)(cls)
        return cast(type, wrapped)

    return _wrap


# Syntactic sugar para "frozen+slots" donde se pueda (visible para mypy)
@dataclass_transform(frozen_default=True)
def dataclass_frozen_slots() -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        params = {"frozen": True}
        try:
            return cast(type, _dataclass(**params, slots=True)(cls))
        except TypeError:
            return cast(type, _dataclass(**params)(cls))

    return _wrap


# UTC compatible sin disparar UP017 (usa getattr)
UTC = getattr(_dt, "UTC", _dt.timezone.utc)
