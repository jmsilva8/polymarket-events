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

    # ── Market CSV ─────────────────────────────────────────────────

    def export_markets_csv(
        self, markets: list[UnifiedMarket], filename: str
    ) -> Path:
        rows = []
        for m in markets:
            rows.append(
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
            )

        df = pd.DataFrame(rows)
        path = self.exports_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
        logger.info("Exported %d markets → %s", len(rows), path)
        return path

    # ── Price History CSV ──────────────────────────────────────────

    def export_price_history_csv(
        self, histories: list[PriceHistory], filename: str
    ) -> Path:
        rows = []
        for ph in histories:
            for dp in ph.data_points:
                rows.append(
                    {
                        "timestamp": dp.timestamp.isoformat(),
                        "market_id": ph.market_id,
                        "token_id": ph.token_id,
                        "question": ph.question,
                        "outcome": ph.outcome_label,
                        "price": dp.price,
                    }
                )

        df = pd.DataFrame(rows)
        path = self.exports_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
        logger.info("Exported %d price points → %s", len(rows), path)
        return path
