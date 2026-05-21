# Enhanced Trading Calendar
# =========================
"""
Calendario de trading mejorado con:
- Early closes
- DST-aware timestamps
- Session windows por exchange
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass
class TradingSession:
    """Representa una sesión de trading."""

    date: date
    exchange: str
    market_open: datetime  # UTC
    market_close: datetime  # UTC
    is_early_close: bool = False
    is_holiday: bool = False


class EnhancedCalendar:
    """
    Calendario de trading con soporte para:
    - Early closes (Navidad, etc.)
    - DST transitions
    - Múltiples exchanges

    Example:
        >>> cal = EnhancedCalendar("NYSE")
        >>> session = cal.get_session(date(2024, 12, 24))  # Christmas Eve
        >>> print(session.market_close)  # 13:00 ET = 18:00 UTC
        >>> print(session.is_early_close)  # True
    """

    # Horarios estándar por exchange (local time)
    EXCHANGE_HOURS = {
        "NYSE": (time(9, 30), time(16, 0)),
        "NASDAQ": (time(9, 30), time(16, 0)),
        "XNYS": (time(9, 30), time(16, 0)),
        "XNAS": (time(9, 30), time(16, 0)),
    }

    # Early close hour (1:00 PM ET)
    EARLY_CLOSE_HOUR = time(13, 0)

    # Early close dates (fechas fijas, ajustar por año)
    EARLY_CLOSE_DATES_2024 = {
        date(2024, 7, 3),  # Día antes de Independence Day
        date(2024, 11, 29),  # Día después de Thanksgiving
        date(2024, 12, 24),  # Christmas Eve
    }

    EARLY_CLOSE_DATES_2025 = {
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
    }

    # Holidays (mercado cerrado)
    HOLIDAYS_2024 = {
        date(2024, 1, 1): "New Year's Day",
        date(2024, 1, 15): "MLK Day",
        date(2024, 2, 19): "Presidents' Day",
        date(2024, 3, 29): "Good Friday",
        date(2024, 5, 27): "Memorial Day",
        date(2024, 6, 19): "Juneteenth",
        date(2024, 7, 4): "Independence Day",
        date(2024, 9, 2): "Labor Day",
        date(2024, 11, 28): "Thanksgiving",
        date(2024, 12, 25): "Christmas",
    }

    HOLIDAYS_2025 = {
        date(2025, 1, 1): "New Year's Day",
        date(2025, 1, 20): "MLK Day",
        date(2025, 2, 17): "Presidents' Day",
        date(2025, 4, 18): "Good Friday",
        date(2025, 5, 26): "Memorial Day",
        date(2025, 6, 19): "Juneteenth",
        date(2025, 7, 4): "Independence Day",
        date(2025, 9, 1): "Labor Day",
        date(2025, 11, 27): "Thanksgiving",
        date(2025, 12, 25): "Christmas",
    }

    # Timezone por exchange
    EXCHANGE_TZ = {
        "NYSE": "America/New_York",
        "NASDAQ": "America/New_York",
        "XNYS": "America/New_York",
        "XNAS": "America/New_York",
    }

    def __init__(self, exchange: str = "NYSE"):
        self.exchange = exchange
        self._tz = ZoneInfo(self.EXCHANGE_TZ.get(exchange, "America/New_York"))
        self._holidays = {**self.HOLIDAYS_2024, **self.HOLIDAYS_2025}
        self._early_closes = self.EARLY_CLOSE_DATES_2024 | self.EARLY_CLOSE_DATES_2025

        # Intentar cargar exchange_calendars para precisión
        self._xcal = None
        try:
            import exchange_calendars as xcals

            self._xcal = xcals.get_calendar(
                {"NYSE": "XNYS", "NASDAQ": "XNAS"}.get(exchange, exchange)
            )
        except ImportError:
            logger.debug("exchange_calendars not available, using fallback")

    def is_trading_day(self, d: date) -> bool:
        """Verifica si es día de trading."""
        if self._xcal:
            try:
                return bool(self._xcal.is_session(d))
            except Exception:
                pass

        # Fallback
        if d.weekday() >= 5:
            return False
        return d not in self._holidays

    def is_early_close(self, d: date) -> bool:
        """Verifica si es día con cierre temprano."""
        if self._xcal:
            try:
                if not self._xcal.is_session(d):
                    return False
                close = self._xcal.session_close(d)
                # Early close si cierra antes de las 16:00 ET
                close_local = close.astimezone(self._tz)
                return bool(close_local.hour < 16)
            except Exception:
                pass

        return d in self._early_closes

    def get_session(self, d: date) -> TradingSession:
        """
        Obtiene la sesión de trading para una fecha.

        Retorna timestamps UTC para open y close.
        """
        if not self.is_trading_day(d):
            return TradingSession(
                date=d,
                exchange=self.exchange,
                market_open=datetime.min,
                market_close=datetime.min,
                is_holiday=True,
            )

        # Obtener horarios
        if self._xcal:
            try:
                open_ts = self._xcal.session_open(d)
                close_ts = self._xcal.session_close(d)
                is_early = close_ts.astimezone(self._tz).hour < 16

                return TradingSession(
                    date=d,
                    exchange=self.exchange,
                    market_open=open_ts.to_pydatetime(),
                    market_close=close_ts.to_pydatetime(),
                    is_early_close=is_early,
                )
            except Exception:
                pass

        # Fallback
        hours = self.EXCHANGE_HOURS.get(self.exchange, (time(9, 30), time(16, 0)))
        open_time, close_time = hours

        is_early = d in self._early_closes
        if is_early:
            close_time = self.EARLY_CLOSE_HOUR

        # Crear datetimes en timezone local y convertir a UTC
        open_dt = datetime.combine(d, open_time, tzinfo=self._tz)
        close_dt = datetime.combine(d, close_time, tzinfo=self._tz)

        return TradingSession(
            date=d,
            exchange=self.exchange,
            market_open=open_dt.astimezone(ZoneInfo("UTC")),
            market_close=close_dt.astimezone(ZoneInfo("UTC")),
            is_early_close=is_early,
        )

    def get_decision_time(
        self,
        d: date,
        offset_minutes: int = 5,
    ) -> datetime:
        """
        Obtiene el decision time para una fecha.

        Decision time = market_close + offset (ej: +5 min)
        """
        session = self.get_session(d)
        if session.is_holiday:
            raise ValueError(f"{d} is not a trading day")

        return session.market_close + timedelta(minutes=offset_minutes)

    def get_execution_time(
        self,
        d: date,
        offset_minutes: int = 1,
    ) -> datetime:
        """
        Obtiene el execution time para una fecha.

        Execution time = market_open + offset (ej: +1 min)
        """
        session = self.get_session(d)
        if session.is_holiday:
            raise ValueError(f"{d} is not a trading day")

        return session.market_open + timedelta(minutes=offset_minutes)

    def get_next_trading_day(self, d: date) -> date:
        """Obtiene el siguiente día de trading."""
        next_d = d + timedelta(days=1)
        while not self.is_trading_day(next_d):
            next_d += timedelta(days=1)
            if (next_d - d).days > 30:
                raise ValueError("No trading day found in next 30 days")
        return next_d

    def get_trading_days_between(self, start: date, end: date) -> list[date]:
        """Lista de trading days en un rango."""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def count_trading_days(self, start: date, end: date) -> int:
        """Cuenta trading days en un rango."""
        return len(self.get_trading_days_between(start, end))


@lru_cache(maxsize=10)
def get_enhanced_calendar(exchange: str = "NYSE") -> EnhancedCalendar:
    """Obtiene un calendario (cached)."""
    return EnhancedCalendar(exchange)


# =============================================================================
# Calendar-Aware Availability
# =============================================================================


def get_availability_timestamp(
    observation_date: date,
    availability_type: str = "market_close",
    exchange: str = "NYSE",
    offset_minutes: int = 0,
) -> datetime:
    """
    Calcula timestamp de disponibilidad calendar-aware.

    Args:
        observation_date: Fecha de la observación
        availability_type: "market_close", "next_open", "next_close"
        exchange: Exchange para horarios
        offset_minutes: Offset adicional

    Returns:
        Datetime UTC cuando el dato está disponible
    """
    cal = get_enhanced_calendar(exchange)

    if availability_type == "market_close":
        return cal.get_decision_time(observation_date, offset_minutes)

    elif availability_type == "next_open":
        next_day = cal.get_next_trading_day(observation_date)
        return cal.get_execution_time(next_day, offset_minutes)

    elif availability_type == "next_close":
        next_day = cal.get_next_trading_day(observation_date)
        return cal.get_decision_time(next_day, offset_minutes)

    else:
        raise ValueError(f"Unknown availability type: {availability_type}")
