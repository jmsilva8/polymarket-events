"""
Kalshi API client.

Base URL: https://api.elections.kalshi.com/trade-api/v2
Docs:     https://docs.kalshi.com

Many market-data endpoints are PUBLIC (no auth needed).
Auth uses RSA key pair: sign(timestamp + method + path) with RSA-PSS SHA-256.
"""

import base64
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    KALSHI_API_BASE_URL,
    KALSHI_API_KEY_ID,
    KALSHI_PRIVATE_KEY_PATH,
    KALSHI_RATE,
    KALSHI_BURST,
    CACHE_DIR,
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


class KalshiClient(BaseMarketClient):
    """
    Client for the Kalshi prediction market API.

    Public endpoints (no auth): GET /markets, GET /events
    Auth endpoints: order placement, portfolio, etc.
    """

    def __init__(self, use_auth: bool = False):
        self.base_url = KALSHI_API_BASE_URL
        self.use_auth = use_auth
        self.rate_limiter = TokenBucketRateLimiter(rate=KALSHI_RATE, burst=KALSHI_BURST)
        self.http = httpx.Client(timeout=30.0, follow_redirects=True)

        self._private_key = None
        if use_auth and KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH:
            self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key for request signing."""
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            key_path = Path(KALSHI_PRIVATE_KEY_PATH)
            if key_path.exists():
                self._private_key = load_pem_private_key(
                    key_path.read_bytes(), password=None
                )
                logger.info("Kalshi RSA private key loaded")
            else:
                logger.warning("Kalshi private key file not found: %s", key_path)
        except ImportError:
            logger.warning(
                "cryptography package not installed; Kalshi auth unavailable. "
                "Install with: pip install cryptography"
            )

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """Generate auth headers for a Kalshi API request."""
        if not self._private_key or not KALSHI_API_KEY_ID:
            return {}

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}"

        signature = self._private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    # ── Low-level HTTP ─────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    def _api_get(self, path: str, params: Optional[dict] = None) -> dict:
        self.rate_limiter.acquire()
        url = f"{self.base_url}{path}"
        headers = self._sign_request("GET", path) if self.use_auth else {}
        logger.debug("GET %s params=%s auth=%s", url, params, bool(headers))
        resp = self.http.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ── Events ─────────────────────────────────────────────────────

    def get_active_events(
        self, limit: int = 100, offset: int = 0, cursor: Optional[str] = None
    ) -> list[UnifiedEvent]:
        params: dict = {"limit": limit, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        raw = self._api_get("/events", params)
        events = raw.get("events", [])
        return [self._parse_event(e) for e in events]

    def get_closed_events(
        self, limit: int = 100, offset: int = 0, cursor: Optional[str] = None
    ) -> list[UnifiedEvent]:
        params: dict = {"limit": limit, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        raw = self._api_get("/events", params)
        events = raw.get("events", [])
        return [self._parse_event(e) for e in events]

    def get_all_active_events(self, max_pages: int = 50) -> list[UnifiedEvent]:
        return self._paginate_events("open", max_pages)

    def get_all_closed_events(self, max_pages: int = 100) -> list[UnifiedEvent]:
        return self._paginate_events("settled", max_pages)

    def _paginate_events(self, status: str, max_pages: int) -> list[UnifiedEvent]:
        all_events: list[UnifiedEvent] = []
        cursor: Optional[str] = None
        for _ in range(max_pages):
            params: dict = {"limit": 100, "status": status}
            if cursor:
                params["cursor"] = cursor
            raw = self._api_get("/events", params)
            events = raw.get("events", [])
            all_events.extend(self._parse_event(e) for e in events)
            cursor = raw.get("cursor")
            if not cursor or not events:
                break
        logger.info("Fetched %d Kalshi events (status=%s)", len(all_events), status)
        return all_events

    # ── Markets ────────────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[UnifiedMarket], Optional[str]]:
        """Fetch markets with pagination. Returns (markets, next_cursor)."""
        params: dict = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        raw = self._api_get("/markets", params)
        markets = [self._parse_market(m) for m in raw.get("markets", [])]
        return markets, raw.get("cursor")

    def get_markets_for_event(self, event_id: str) -> list[UnifiedMarket]:
        """Fetch all markets for a given event ticker."""
        all_markets: list[UnifiedMarket] = []
        cursor: Optional[str] = None
        for _ in range(50):
            markets, cursor = self.get_markets(
                event_ticker=event_id, cursor=cursor
            )
            all_markets.extend(markets)
            if not cursor or not markets:
                break
        return all_markets

    def get_market_by_ticker(self, ticker: str) -> UnifiedMarket:
        raw = self._api_get(f"/markets/{ticker}")
        return self._parse_market(raw.get("market", raw))

    def get_all_closed_markets(
        self, max_pages: int = 100
    ) -> list[UnifiedMarket]:
        """Paginate through all settled markets."""
        all_markets: list[UnifiedMarket] = []
        cursor: Optional[str] = None
        for _ in range(max_pages):
            markets, cursor = self.get_markets(status="settled", cursor=cursor)
            all_markets.extend(markets)
            if not cursor or not markets:
                break
        logger.info("Fetched %d settled Kalshi markets", len(all_markets))
        return all_markets

    # ── Price History ──────────────────────────────────────────────

    def get_price_history(
        self,
        market_id: str,
        token_id: str,
        interval: str = "max",
        fidelity: Optional[int] = None,
    ) -> PriceHistory:
        """
        Kalshi does NOT have a dedicated timeseries endpoint.
        Returns an empty PriceHistory.
        For Kalshi, use last_trade_price and result from get_market_by_ticker().
        """
        logger.warning(
            "Kalshi has no historical timeseries API. Returning empty PriceHistory."
        )
        return PriceHistory(
            market_id=market_id,
            platform=Platform.KALSHI,
            token_id=token_id,
            question="",
            outcome_label="",
            data_points=[],
        )

    # ── Parsing ────────────────────────────────────────────────────

    def _parse_event(self, raw: dict) -> UnifiedEvent:
        status = (raw.get("status") or "").lower()
        return UnifiedEvent(
            platform=Platform.KALSHI,
            event_id=raw.get("event_ticker", ""),
            event_ticker=raw.get("event_ticker", ""),
            title=raw.get("title", ""),
            slug=raw.get("event_ticker", "").lower(),
            description=raw.get("sub_title") or "",
            category=raw.get("category", ""),
            active=status not in ("settled", "closed", "finalized"),
            closed=status in ("settled", "closed", "finalized"),
            raw_data=raw,
        )

    def _parse_market(self, raw: dict) -> UnifiedMarket:
        # Determine status
        api_status = raw.get("status", "").lower()
        if api_status in ("closed", "determined", "finalized", "settled"):
            status = MarketStatus.CLOSED
        elif api_status in ("active", "open"):
            status = MarketStatus.ACTIVE
        else:
            status = MarketStatus.ARCHIVED

        # Prices: Kalshi uses dollar strings like "0.6500"
        yes_bid = _safe_float_or_none(raw.get("yes_bid"))
        yes_ask = _safe_float_or_none(raw.get("yes_ask"))
        last_price = _safe_float_or_none(raw.get("last_price"))

        # Build outcome prices from last_price if available
        outcome_prices: list[float] = []
        if last_price is not None:
            outcome_prices = [last_price, 1.0 - last_price]

        # Volume: Kalshi uses fixed-point integers
        volume = _safe_float(raw.get("volume"))
        volume_24h = _safe_float(raw.get("volume_24h"))
        open_interest = _safe_float(raw.get("open_interest"))

        return UnifiedMarket(
            platform=Platform.KALSHI,
            market_id=raw.get("ticker", ""),
            ticker=raw.get("ticker", ""),
            condition_id=None,
            slug=raw.get("ticker", "").lower(),
            question=raw.get("title", ""),
            description=raw.get("yes_sub_title", ""),
            event_id=raw.get("event_ticker", ""),
            event_title="",
            event_slug=raw.get("event_ticker", "").lower(),
            outcomes=["Yes", "No"],
            outcome_prices=outcome_prices,
            volume=volume,
            liquidity=0.0,
            volume_24h=volume_24h,
            open_interest=open_interest,
            best_bid=yes_bid,
            best_ask=yes_ask,
            last_trade_price=last_price,
            status=status,
            result=raw.get("result", None) or None,
            start_date=_parse_kalshi_dt(raw.get("open_time")),
            end_date=_parse_kalshi_dt(raw.get("close_time")),
            created_at=_parse_kalshi_dt(raw.get("open_time")),
            closed_at=_parse_kalshi_dt(raw.get("close_time")),
            category=raw.get("category", ""),
            raw_data=raw,
        )

    # ── Cleanup ────────────────────────────────────────────────────

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Module-level helpers ───────────────────────────────────────────


def _parse_kalshi_dt(val) -> Optional[datetime]:
    if not val:
        return None
    try:
        if isinstance(val, str):
            val = val.replace("Z", "+00:00")
            return datetime.fromisoformat(val)
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
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
