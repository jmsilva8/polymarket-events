"""Insider Alpha trading strategy with configurable thresholds."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.data_layer.models import PriceHistory


@dataclass
class StrategyParams:
    """All tunable parameters for the insider alpha strategy."""
    price_threshold: float = 0.65      # Buy YES if price crosses this
    hours_before_close: float = 24.0   # Only consider signals within this window
    min_leak_score: int = 7            # Minimum insider risk score to trade
    min_volume: float = 10_000.0       # Minimum market volume


@dataclass
class TradeSignal:
    """A detected trading signal."""
    market_id: str
    market_title: str
    platform: str
    insider_risk_score: int
    entry_price: float              # Price when signal triggered
    signal_time: datetime           # When the signal was detected
    end_date: datetime              # Market close/resolution time
    hours_before_close_actual: float  # How many hours before close the signal fired
    resolved_yes: Optional[bool] = None  # Did it resolve YES?
    pnl: Optional[float] = None         # Profit/loss per $1 bet


class InsiderAlphaStrategy:
    """
    Detects potential insider trading signals in price history.

    Logic: If a market with high insider risk score shows the YES price
    crossing above `price_threshold` within `hours_before_close` of the
    market's end date, that's a potential insider signal. We buy YES at
    that price.

    P&L:
    - If resolved YES:  profit = 1.0 - entry_price
    - If resolved NO:   loss   = -entry_price
    """

    def __init__(self, params: StrategyParams):
        self.params = params

    def evaluate(
        self,
        price_history: PriceHistory,
        market_id: str,
        market_title: str,
        platform: str,
        insider_risk_score: int,
        end_date: Optional[datetime],
        volume: float,
        resolved_yes: Optional[bool],
    ) -> Optional[TradeSignal]:
        """
        Evaluate a single market for insider trading signals.

        Returns a TradeSignal if a signal is detected, None otherwise.
        """
        # Filter: score must meet threshold
        if insider_risk_score < self.params.min_leak_score:
            return None

        # Filter: volume must meet threshold
        if volume < self.params.min_volume:
            return None

        # Need end_date and resolution to backtest
        if end_date is None or resolved_yes is None:
            return None

        # Need price data
        if not price_history.data_points:
            return None

        # Ensure end_date is timezone-aware
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Define the signal window: [end_date - hours_before_close, end_date]
        window_start = end_date - timedelta(hours=self.params.hours_before_close)

        # Look for the FIRST time the price crosses the threshold within the window
        for dp in price_history.data_points:
            ts = dp.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if ts < window_start:
                continue
            if ts > end_date:
                break

            if dp.price >= self.params.price_threshold:
                hours_before = (end_date - ts).total_seconds() / 3600.0

                # Calculate P&L
                if resolved_yes:
                    pnl = 1.0 - dp.price  # Payout $1, paid entry_price
                else:
                    pnl = -dp.price  # Lost the entry price

                return TradeSignal(
                    market_id=market_id,
                    market_title=market_title,
                    platform=platform,
                    insider_risk_score=insider_risk_score,
                    entry_price=dp.price,
                    signal_time=ts,
                    end_date=end_date,
                    hours_before_close_actual=hours_before,
                    resolved_yes=resolved_yes,
                    pnl=pnl,
                )

        return None
