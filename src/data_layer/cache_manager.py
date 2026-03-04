"""Persistent caching layer for market data (JSON metadata, CSV timeseries)."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import CACHE_DIR, EXPORTS_DIR
from src.data_layer.models import UnifiedEvent, UnifiedMarket, PriceHistory

logger = logging.getLogger(__name__)


class CacheManager:
    """Save/load events JSON and export market data + price history to CSV."""

    def __init__(
        self, cache_dir: Path = CACHE_DIR, exports_dir: Path = EXPORTS_DIR
    ):
        self.cache_dir = cache_dir
        self.exports_dir = exports_dir

    # ── Events JSON ────────────────────────────────────────────────

    def cache_events(self, events: list[UnifiedEvent], filename: str) -> Path:
        path = self.cache_dir / "events" / f"{filename}.json"
        data = [e.raw_data for e in events if e.raw_data]
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Cached %d events → %s", len(events), path)
        return path

    def load_cached_events(self, filename: str) -> list[dict] | None:
        path = self.cache_dir / "events" / f"{filename}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ── Market exports ─────────────────────────────────────────────

    def _market_rows(self, markets: list[UnifiedMarket]) -> list[dict]:
        return [
            {
                "platform": m.platform.value,
                "market_id": m.market_id,
                "ticker": m.ticker or "",
                "event_id": m.event_id,
                "event_title": m.event_title,
                "question": m.question,
                "slug": m.slug,
                "category": m.category,
                "yes_price": m.yes_price,
                "last_trade_price": m.last_trade_price,
                "volume": m.volume,
                "volume_24h": m.volume_24h,
                "liquidity": m.liquidity,
                "open_interest": m.open_interest,
                "status": m.status.value,
                "result": m.result or "",
                "resolved_yes": m.resolved_yes,
                "start_date": m.start_date,
                "end_date": m.end_date,
                "tags": "|".join(t.label for t in m.tags),
            }
            for m in markets
        ]

    def export_markets_parquet(
        self, markets: list[UnifiedMarket], filename: str
    ) -> Path:
        df = pd.DataFrame(self._market_rows(markets))
        path = self.exports_dir / f"{filename}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Exported %d markets → %s", len(markets), path)
        return path

    def load_markets_parquet(self, filename: str) -> pd.DataFrame | None:
        path = self.exports_dir / f"{filename}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def export_markets_csv(
        self, markets: list[UnifiedMarket], filename: str
    ) -> Path:
        df = pd.DataFrame(self._market_rows(markets))
        path = self.exports_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
        logger.info("Exported %d markets → %s", len(markets), path)
        return path

    # ── Price History exports ──────────────────────────────────────

    def _price_history_rows(self, histories: list[PriceHistory]) -> list[dict]:
        return [
            {
                "timestamp": dp.timestamp.isoformat(),
                "market_id": ph.market_id,
                "token_id": ph.token_id,
                "question": ph.question,
                "outcome": ph.outcome_label,
                "price": dp.price,
            }
            for ph in histories
            for dp in ph.data_points
        ]

    def export_price_history_parquet(
        self, histories: list[PriceHistory], filename: str
    ) -> Path:
        df = pd.DataFrame(self._price_history_rows(histories))
        path = self.exports_dir / f"{filename}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Exported %d price points → %s", len(df), path)
        return path

    def export_price_history_csv(
        self, histories: list[PriceHistory], filename: str
    ) -> Path:
        df = pd.DataFrame(self._price_history_rows(histories))
        path = self.exports_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
        logger.info("Exported %d price points → %s", len(df), path)
        return path
