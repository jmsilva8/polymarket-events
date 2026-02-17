"""
Polymarket API client wrapping Gamma API (metadata) and CLOB API (prices).

Gamma API (no auth): https://gamma-api.polymarket.com
  GET /events       – list/filter events
  GET /events/{id}  – single event with nested markets
  GET /markets      – list/filter markets
  GET /tags         – all available tags

CLOB API (no auth for reads): https://clob.polymarket.com
  GET /prices-history?market={token_id}&interval={interval}&fidelity={minutes}
  GET /price?token_id={token_id}&side=buy
  GET /book?token_id={token_id}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    POLYMARKET_GAMMA_BASE_URL,
    POLYMARKET_CLOB_BASE_URL,
    POLYMARKET_RATE,
    POLYMARKET_BURST,
    CACHE_DIR,
    ENTERTAINMENT_TAG_LABELS,
)
from src.data_layer.base_client import BaseMarketClient
from src.data_layer.models import (
    Platform,
    MarketStatus,
    Tag,
    PricePoint,
    UnifiedMarket,
    UnifiedEvent,
    PriceHistory,
)
from src.data_layer.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class PolymarketClient(BaseMarketClient):
    """
    Client for Polymarket Gamma + CLOB APIs.
    No authentication required for all read operations.
    """

    def __init__(self, cache_enabled: bool = True):
        self.gamma_base = POLYMARKET_GAMMA_BASE_URL
        self.clob_base = POLYMARKET_CLOB_BASE_URL
        self.cache_enabled = cache_enabled
        self.cache_dir = CACHE_DIR
        self.rate_limiter = TokenBucketRateLimiter(
            rate=POLYMARKET_RATE, burst=POLYMARKET_BURST
        )
        self.http = httpx.Client(timeout=30.0, follow_redirects=True)

    # ── Low-level HTTP ─────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _gamma_get(self, endpoint: str, params: Optional[dict] = None):
        self.rate_limiter.acquire()
        url = f"{self.gamma_base}{endpoint}"
        logger.debug("GET %s params=%s", url, params)
        resp = self.http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _clob_get(self, endpoint: str, params: Optional[dict] = None):
        self.rate_limiter.acquire()
        url = f"{self.clob_base}{endpoint}"
        logger.debug("GET %s params=%s", url, params)
        resp = self.http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Events ─────────────────────────────────────────────────────

    def get_active_events(
        self, limit: int = 100, offset: int = 0, tag_id: Optional[int] = None
    ) -> list[UnifiedEvent]:
        params: dict = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        if tag_id is not None:
            params["tag_id"] = tag_id
        raw = self._gamma_get("/events", params)
        return [self._parse_event(e) for e in raw]

    def get_closed_events(
        self,
        limit: int = 100,
        offset: int = 0,
        tag_id: Optional[int] = None,
        end_date_min: Optional[str] = None,
        end_date_max: Optional[str] = None,
    ) -> list[UnifiedEvent]:
        params: dict = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "order": "endDate",
            "ascending": "false",
        }
        if tag_id is not None:
            params["tag_id"] = tag_id
        if end_date_min:
            params["end_date_min"] = end_date_min
        if end_date_max:
            params["end_date_max"] = end_date_max
        raw = self._gamma_get("/events", params)
        return [self._parse_event(e) for e in raw]

    def get_all_active_events(
        self, tag_id: Optional[int] = None, max_pages: int = 50
    ) -> list[UnifiedEvent]:
        """Auto-paginate through all active events."""
        return self._paginate(self.get_active_events, tag_id=tag_id, max_pages=max_pages)

    def get_all_closed_events(
        self,
        tag_id: Optional[int] = None,
        end_date_min: Optional[str] = None,
        end_date_max: Optional[str] = None,
        max_pages: int = 100,
    ) -> list[UnifiedEvent]:
        """Auto-paginate through all closed events."""
        return self._paginate(
            self.get_closed_events,
            tag_id=tag_id,
            end_date_min=end_date_min,
            end_date_max=end_date_max,
            max_pages=max_pages,
        )

    def _paginate(self, fetch_fn, max_pages: int = 50, **kwargs) -> list[UnifiedEvent]:
        all_events: list[UnifiedEvent] = []
        page_size = 100
        for page in range(max_pages):
            events = fetch_fn(limit=page_size, offset=page * page_size, **kwargs)
            all_events.extend(events)
            if len(events) < page_size:
                break
        logger.info("Fetched %d events total", len(all_events))
        return all_events

    # ── Markets ────────────────────────────────────────────────────

    def get_markets_for_event(self, event_id: str) -> list[UnifiedMarket]:
        raw = self._gamma_get(f"/events/{event_id}")
        event = self._parse_event(raw)
        return event.markets

    # ── Tags ───────────────────────────────────────────────────────

    def get_tags(self, limit: int = 500) -> list[dict]:
        cache_path = self.cache_dir / "tags" / "all_tags.json"
        if self.cache_enabled and cache_path.exists():
            return json.loads(cache_path.read_text())

        tags = self._gamma_get("/tags", params={"limit": limit})
        if self.cache_enabled:
            cache_path.write_text(json.dumps(tags, indent=2))
        return tags

    def discover_entertainment_tags(
        self, keywords: Optional[list[str]] = None
    ) -> list[Tag]:
        """Find tags matching entertainment/pop-culture keywords."""
        kws = keywords or ENTERTAINMENT_TAG_LABELS
        all_tags = self.get_tags()
        matching: list[Tag] = []
        for t in all_tags:
            label = t.get("label", "").lower()
            slug = t.get("slug", "").lower()
            if any(kw in label or kw in slug for kw in kws):
                matching.append(Tag(id=t["id"], label=t.get("label", ""), slug=t.get("slug", "")))
        logger.info("Found %d entertainment-related tags", len(matching))
        return matching

    def filter_entertainment_events(
        self, events: list[UnifiedEvent], keywords: Optional[list[str]] = None
    ) -> list[UnifiedEvent]:
        """
        Filter events by checking if any of their tags match entertainment keywords.
        This is the primary filtering method since Polymarket tags are per-event,
        not a query parameter.
        """
        kws = [k.lower() for k in (keywords or ENTERTAINMENT_TAG_LABELS)]
        result: list[UnifiedEvent] = []
        for event in events:
            event_tags = [t.label.lower() for t in event.tags]
            # Also check raw_data tags (more complete than parsed tags)
            if event.raw_data:
                for t in event.raw_data.get("tags", []):
                    event_tags.append(t.get("label", "").lower())
            if any(kw in tag for tag in event_tags for kw in kws):
                result.append(event)
        return result

    # ── Price History (CLOB) ───────────────────────────────────────

    def get_price_history(
        self,
        market_id: str,
        token_id: str,
        interval: str = "max",
        fidelity: Optional[int] = None,
    ) -> PriceHistory:
        """
        Fetch historical prices from CLOB API.

        Args:
            market_id: For metadata reference.
            token_id:  CLOB token ID (from UnifiedMarket.clob_token_ids).
            interval:  "max", "1w", "1d", "6h", "1h"
            fidelity:  Resolution in minutes (60=hourly, 1440=daily).
        """
        cache_key = f"{token_id}_{interval}_{fidelity or 'auto'}"
        cache_path = self.cache_dir / "price_history" / f"{cache_key}.json"

        if self.cache_enabled and cache_path.exists():
            raw = json.loads(cache_path.read_text())
        else:
            params: dict = {"market": token_id, "interval": interval}
            if fidelity is not None:
                params["fidelity"] = fidelity
            raw = self._clob_get("/prices-history", params)
            if self.cache_enabled:
                cache_path.write_text(json.dumps(raw))

        return self._parse_price_history(raw, market_id, token_id)

    def get_current_price(self, token_id: str, side: str = "buy") -> float:
        raw = self._clob_get("/price", params={"token_id": token_id, "side": side})
        return float(raw.get("price", 0.0))

    # ── Parsing Helpers ────────────────────────────────────────────

    def _parse_event(self, raw: dict) -> UnifiedEvent:
        markets = [
            self._parse_market(m, event_context=raw)
            for m in raw.get("markets", [])
        ]
        tags = [
            Tag(id=t.get("id", 0), label=t.get("label", ""), slug=t.get("slug", ""))
            for t in raw.get("tags", [])
        ]
        return UnifiedEvent(
            platform=Platform.POLYMARKET,
            event_id=str(raw.get("id", "")),
            title=raw.get("title", ""),
            slug=raw.get("slug", ""),
            description=raw.get("description", ""),
            markets=markets,
            tags=tags,
            category=raw.get("category", ""),
            volume=_safe_float(raw.get("volume")),
            liquidity=_safe_float(raw.get("liquidity")),
            active=raw.get("active", False),
            closed=raw.get("closed", False),
            start_date=_parse_dt(raw.get("startDate")),
            end_date=_parse_dt(raw.get("endDate")),
            raw_data=raw,
        )

    def _parse_market(
        self, raw: dict, event_context: Optional[dict] = None
    ) -> UnifiedMarket:
        outcome_prices = _parse_json_list_floats(raw.get("outcomePrices", "[]"))
        outcomes = _parse_json_list_strings(raw.get("outcomes", '["Yes","No"]'))
        clob_token_ids = _parse_json_list_strings(raw.get("clobTokenIds", "[]"))

        tags = [
            Tag(id=t.get("id", 0), label=t.get("label", ""), slug=t.get("slug", ""))
            for t in raw.get("tags", [])
        ]

        if raw.get("closed"):
            status = MarketStatus.CLOSED
        elif raw.get("archived"):
            status = MarketStatus.ARCHIVED
        else:
            status = MarketStatus.ACTIVE

        return UnifiedMarket(
            platform=Platform.POLYMARKET,
            market_id=str(raw.get("id", "")),
            condition_id=raw.get("conditionId", ""),
            slug=raw.get("slug", ""),
            question=raw.get("question", raw.get("groupItemTitle", "")),
            description=raw.get("description", ""),
            event_id=str(
                raw.get("eventId", (event_context or {}).get("id", ""))
            ),
            event_title=(event_context or {}).get("title", ""),
            event_slug=(event_context or {}).get("slug", ""),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            volume=_safe_float(raw.get("volumeNum", raw.get("volume"))),
            liquidity=_safe_float(raw.get("liquidityNum", raw.get("liquidity"))),
            volume_24h=_safe_float(raw.get("volume24hr")),
            best_bid=_safe_float_or_none(raw.get("bestBid")),
            best_ask=_safe_float_or_none(raw.get("bestAsk")),
            spread=_safe_float_or_none(raw.get("spread")),
            last_trade_price=_safe_float_or_none(raw.get("lastTradePrice")),
            status=status,
            start_date=_parse_dt(raw.get("startDate")),
            end_date=_parse_dt(raw.get("endDate")),
            created_at=_parse_dt(raw.get("createdAt")),
            closed_at=_parse_dt(raw.get("closedTime")),
            tags=tags,
            category=raw.get("category", ""),
            raw_data=raw,
        )

    def _parse_price_history(
        self, raw: dict, market_id: str, token_id: str
    ) -> PriceHistory:
        data_points: list[PricePoint] = []
        for pt in raw.get("history", []):
            ts = datetime.fromtimestamp(pt["t"], tz=timezone.utc)
            price = float(pt["p"])
            data_points.append(PricePoint(timestamp=ts, price=price, raw_price=price))
        return PriceHistory(
            market_id=market_id,
            platform=Platform.POLYMARKET,
            token_id=token_id,
            question="",
            outcome_label="",
            data_points=data_points,
        )

    # ── Cleanup ────────────────────────────────────────────────────

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Module-level helpers ───────────────────────────────────────────


def _parse_json_list_floats(val) -> list[float]:
    """Parse a value that may be a JSON-encoded string list of numbers."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(val, list):
        return [float(x) for x in val]
    return []


def _parse_json_list_strings(val) -> list[str]:
    """Parse a value that may be a JSON-encoded string list."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def _parse_dt(val) -> Optional[datetime]:
    if not val:
        return None
    try:
        if isinstance(val, str):
            val = val.replace("Z", "+00:00")
            return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None
    return None


def _safe_float(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_float_or_none(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
