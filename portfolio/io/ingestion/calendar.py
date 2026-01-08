# Trading Calendar
# ================
"""
Trading calendar para validaciones calendar-aware.

Usa exchange_calendars (o pandas_market_calendars como fallback)
para determinar días de trading.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

ExchangeCode = Literal["NYSE", "NASDAQ", "XNYS", "XNAS"]


class TradingCalendar:
    """
    Calendario de trading para un exchange.

    Permite:
    - Verificar si una fecha es día de trading
    - Obtener lista de días de trading en un rango
    - Explicar gaps (fin de semana, feriado, etc.)

    Example:
        >>> cal = TradingCalendar("NYSE")
        >>> cal.is_trading_day(date(2024, 12, 25))  # Christmas
        False
        >>> cal.explain_gap(date(2024, 12, 24), date(2024, 12, 26))
        'holiday: Christmas Day'
    """

    # Mapeo de exchanges a nombres de exchange_calendars
    EXCHANGE_MAP = {
        "NYSE": "XNYS",
        "NASDAQ": "XNAS",
        "XNYS": "XNYS",
        "XNAS": "XNAS",
    }

    # Feriados conocidos (fallback si no hay librería)
    US_HOLIDAYS_2024 = {
        date(2024, 1, 1): "New Year's Day",
        date(2024, 1, 15): "Martin Luther King Jr. Day",
        date(2024, 2, 19): "Presidents' Day",
        date(2024, 3, 29): "Good Friday",
        date(2024, 5, 27): "Memorial Day",
        date(2024, 6, 19): "Juneteenth",
        date(2024, 7, 4): "Independence Day",
        date(2024, 9, 2): "Labor Day",
        date(2024, 11, 28): "Thanksgiving Day",
        date(2024, 12, 25): "Christmas Day",
    }

    US_HOLIDAYS_2025 = {
        date(2025, 1, 1): "New Year's Day",
        date(2025, 1, 20): "Martin Luther King Jr. Day",
        date(2025, 2, 17): "Presidents' Day",
        date(2025, 4, 18): "Good Friday",
        date(2025, 5, 26): "Memorial Day",
        date(2025, 6, 19): "Juneteenth",
        date(2025, 7, 4): "Independence Day",
        date(2025, 9, 1): "Labor Day",
        date(2025, 11, 27): "Thanksgiving Day",
        date(2025, 12, 25): "Christmas Day",
    }

    def __init__(self, exchange: str = "NYSE"):
        """
        Args:
            exchange: Código del exchange (NYSE, NASDAQ)
        """
        self.exchange = exchange
        self._calendar = None
        self._use_fallback = False

        # Intentar cargar exchange_calendars
        try:
            import exchange_calendars as xcals

            exchange_code = self.EXCHANGE_MAP.get(exchange, exchange)
            self._calendar = xcals.get_calendar(exchange_code)
            logger.debug("Loaded exchange_calendars for %s", exchange_code)
        except ImportError:
            logger.warning("exchange_calendars not installed, using fallback")
            self._use_fallback = True
        except Exception as e:
            logger.warning("Failed to load calendar: %s, using fallback", e)
            self._use_fallback = True

    def is_trading_day(self, d: date) -> bool:
        """Verifica si una fecha es día de trading."""
        if self._calendar is not None:
            try:
                return self._calendar.is_session(d)
            except Exception:
                pass

        # Fallback: fin de semana + feriados conocidos
        if d.weekday() >= 5:  # Sábado o domingo
            return False

        holidays = {**self.US_HOLIDAYS_2024, **self.US_HOLIDAYS_2025}
        return d not in holidays

    def get_trading_days(self, start: date, end: date) -> list[date]:
        """Obtiene lista de días de trading en un rango."""
        if self._calendar is not None:
            try:
                sessions = self._calendar.sessions_in_range(start, end)
                return [s.date() for s in sessions]
            except Exception:
                pass

        # Fallback
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def explain_gap(self, date1: date, date2: date) -> str:
        """
        Explica por qué hay gap entre dos fechas.

        Returns:
            'weekend', 'holiday: {name}', 'unexplained'
        """
        if date2 <= date1:
            return "no_gap"

        # Si son consecutivos en calendario, no hay gap
        if (date2 - date1).days == 1:
            return "no_gap"

        # Verificar cada día entre date1 y date2
        reasons = []
        current = date1 + timedelta(days=1)

        while current < date2:
            if current.weekday() >= 5:
                reasons.append("weekend")
            else:
                holidays = {**self.US_HOLIDAYS_2024, **self.US_HOLIDAYS_2025}
                if current in holidays:
                    reasons.append(f"holiday: {holidays[current]}")
                elif not self.is_trading_day(current):
                    reasons.append("market_closed")
                else:
                    reasons.append("unexplained")
            current += timedelta(days=1)

        # Resumir
        if not reasons:
            return "no_gap"

        if all(r == "weekend" for r in reasons):
            return "weekend"

        holidays = [r for r in reasons if r.startswith("holiday")]
        if holidays:
            return holidays[0]  # Retornar el primer feriado

        if "unexplained" in reasons:
            return "unexplained"

        return reasons[0]

    def count_trading_days(self, start: date, end: date) -> int:
        """Cuenta días de trading en un rango."""
        return len(self.get_trading_days(start, end))

    def next_trading_day(self, d: date) -> date:
        """Obtiene el siguiente día de trading después de d."""
        current = d + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
            if (current - d).days > 30:  # Safety limit
                break
        return current

    def prev_trading_day(self, d: date) -> date:
        """Obtiene el día de trading anterior a d."""
        current = d - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
            if (d - current).days > 30:  # Safety limit
                break
        return current


@lru_cache(maxsize=10)
def get_calendar(exchange: str = "NYSE") -> TradingCalendar:
    """Obtiene un calendario (cached)."""
    return TradingCalendar(exchange)


def market_close_time(d: date, exchange: str = "NYSE") -> datetime:
    """
    Obtiene el timestamp de cierre de mercado para una fecha.

    NYSE/NASDAQ cierran a las 16:00 ET.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    close_time = datetime(d.year, d.month, d.day, 16, 0, 0, tzinfo=et)
    return close_time


def market_open_time(d: date, exchange: str = "NYSE") -> datetime:
    """
    Obtiene el timestamp de apertura de mercado para una fecha.

    NYSE/NASDAQ abren a las 9:30 ET.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    open_time = datetime(d.year, d.month, d.day, 9, 30, 0, tzinfo=et)
    return open_time
