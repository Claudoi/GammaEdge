# Provider Package Exports
"""
Data providers for market data ingestion.

Available providers:
- yahoo: Yahoo Finance (free, fallback)
- polygon: Polygon.io (primary, paid)
- alpaca: Alpaca Markets (execution + data)
- tiingo: Tiingo (EOD/fundamentals)
"""

from portfolio.io.providers.base import (
    AuthenticationError,
    BaseDataProvider,
    DataNotFoundError,
    ProviderError,
    RateLimitError,
)
from portfolio.io.providers.yahoo import YahooProvider

# Optional providers (require API keys)
# Import these conditionally based on available keys

__all__ = [
    # Base class
    "BaseDataProvider",
    # Exceptions
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "DataNotFoundError",
    # Providers
    "YahooProvider",
]
