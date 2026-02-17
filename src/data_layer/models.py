"""Unified data models for prediction market data across platforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class MarketStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"


@dataclass
class Tag:
    """Market category tag."""
    id: int
    label: str
    slug: str


@dataclass
class PricePoint:
    """Single price observation."""
    timestamp: datetime
    price: float       # Implied probability 0.0–1.0
    raw_price: float   # Original value from source


@dataclass
class UnifiedMarket:
    """
    Platform-agnostic market representation.
    All prices are normalized to implied probability [0.0, 1.0].
    """

    # Identity
    platform: Platform
    market_id: str
    condition_id: Optional[str]
    slug: str
    question: str
    description: str

    # Parent event
    event_id: Optional[str] = None
    event_title: Optional[str] = None
    event_slug: Optional[str] = None

    # Outcomes  (e.g. ["Yes", "No"] for binary markets)
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])
    outcome_prices: list[float] = field(default_factory=list)  # [0.65, 0.35]

    # Polymarket CLOB token IDs (needed for price history)
    clob_token_ids: list[str] = field(default_factory=list)

    # Kalshi ticker (needed for Kalshi API calls)
    ticker: Optional[str] = None

    # Financial
    volume: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0

    # Pricing snapshot
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    last_trade_price: Optional[float] = None

    # Status & resolution
    status: MarketStatus = MarketStatus.ACTIVE
    result: Optional[str] = None  # "yes" / "no" / None (Kalshi resolved result)

    # Timestamps
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Tags / categories
    tags: list[Tag] = field(default_factory=list)
    category: str = ""

    # Raw API response (for debugging / AI classification)
    raw_data: Optional[dict] = None

    @property
    def yes_price(self) -> float:
        if self.outcome_prices and len(self.outcome_prices) > 0:
            return self.outcome_prices[0]
        return 0.0

    @property
    def no_price(self) -> float:
        if self.outcome_prices and len(self.outcome_prices) > 1:
            return self.outcome_prices[1]
        return 1.0 - self.yes_price

    @property
    def hours_to_close(self) -> Optional[float]:
        if self.end_date:
            delta = self.end_date - datetime.now(timezone.utc)
            return delta.total_seconds() / 3600
        return None

    @property
    def resolved_yes(self) -> Optional[bool]:
        """Did this market resolve YES?  None if unresolved."""
        if self.result is not None:
            return self.result.lower() == "yes"
        # Polymarket: check final outcome_prices
        if self.status == MarketStatus.CLOSED and self.outcome_prices:
            return self.outcome_prices[0] > 0.5
        return None


@dataclass
class UnifiedEvent:
    """Platform-agnostic event (groups multiple markets)."""
    platform: Platform
    event_id: str
    title: str
    slug: str
    description: str
    markets: list[UnifiedMarket] = field(default_factory=list)
    tags: list[Tag] = field(default_factory=list)
    category: str = ""
    volume: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    closed: bool = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    raw_data: Optional[dict] = None

    # Kalshi event ticker
    event_ticker: Optional[str] = None


@dataclass
class PriceHistory:
    """Historical price series for a single market/outcome token."""
    market_id: str
    platform: Platform
    token_id: str       # CLOB token ID (Polymarket) or ticker (Kalshi)
    question: str
    outcome_label: str  # "Yes" or "No"
    data_points: list[PricePoint] = field(default_factory=list)

    def to_dataframe(self):
        """Convert to pandas DataFrame with datetime index."""
        import pandas as pd

        records = [
            {"timestamp": p.timestamp, "price": p.price}
            for p in self.data_points
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("timestamp").sort_index()
        return df
