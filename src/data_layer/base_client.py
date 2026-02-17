"""Abstract base class for prediction market API clients."""

from abc import ABC, abstractmethod
from typing import Optional

from src.data_layer.models import UnifiedEvent, UnifiedMarket, PriceHistory


class BaseMarketClient(ABC):
    """Interface that all platform clients must implement."""

    @abstractmethod
    def get_active_events(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UnifiedEvent]:
        """Fetch currently active (open) events."""
        ...

    @abstractmethod
    def get_closed_events(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UnifiedEvent]:
        """Fetch closed (resolved) events for historical analysis."""
        ...

    @abstractmethod
    def get_markets_for_event(self, event_id: str) -> list[UnifiedMarket]:
        """Get all markets within an event."""
        ...

    @abstractmethod
    def get_price_history(
        self,
        market_id: str,
        token_id: str,
        interval: str = "max",
        fidelity: Optional[int] = None,
    ) -> PriceHistory:
        """Fetch historical price timeseries for a market outcome token."""
        ...
