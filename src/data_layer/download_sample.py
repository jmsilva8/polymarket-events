"""
Download all closed market data from both platforms (all categories).
Generates a cost estimation report before any LLM calls.

Usage:
    python -m src.data_layer.download_sample
"""

import sys
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import EXPORTS_DIR, MIN_VOLUME_USD
from src.data_layer.polymarket_client import PolymarketClient
from src.data_layer.kalshi_client import KalshiClient
from src.data_layer.cache_manager import CacheManager
from src.data_layer.models import UnifiedMarket, UnifiedEvent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Only include markets that closed on or after this date.
# Kalshi only got retail approval in late 2023; pre-2024 data is thin on both platforms.
CUTOFF_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def download_polymarket_all(client: PolymarketClient) -> list[UnifiedEvent]:
    """Fetch closed events (all categories) since CUTOFF_DATE."""
    logger.info("Fetching closed Polymarket events since %s...", CUTOFF_DATE.date())
    all_closed = client.get_all_closed_events(
        max_pages=100,
        end_date_min=CUTOFF_DATE.strftime("%Y-%m-%dT%H:%M:%SZ"),
        volume_min=MIN_VOLUME_USD,
        cache_key="polymarket_closed",
    )
    logger.info("Total closed events: %d", len(all_closed))
    return all_closed


def download_kalshi_all(client: KalshiClient) -> list[UnifiedMarket]:
    """
    Fetch settled Kalshi markets (all categories) since CUTOFF_DATE.
    The Kalshi events endpoint has no date filter, so we fetch all settled
    events, then filter markets by closed_at after fetching.
    """
    logger.info("Fetching settled Kalshi events...")
    all_events = client.get_all_closed_events(max_pages=50)
    logger.info("Total settled Kalshi events: %d", len(all_events))

    # Fetch markets for all events, then filter by close date
    all_markets: list[UnifiedMarket] = []
    for event in all_events:
        try:
            markets = client.get_markets_for_event(event.event_id)
            # Tag each market with its parent event's category
            for m in markets:
                m.category = event.category
                m.event_title = event.title
            # Drop markets that closed before the cutoff (closed_at=None means keep)
            markets = [
                m for m in markets
                if m.closed_at is None or m.closed_at >= CUTOFF_DATE
            ]
            all_markets.extend(markets)
        except Exception as e:
            logger.warning("Failed to fetch markets for %s: %s", event.event_id, e)

    logger.info("Total Kalshi markets since %s: %d", CUTOFF_DATE.date(), len(all_markets))
    return all_markets


def estimate_llm_costs(
    unique_market_count: int,
    archetype_match_rate: float = 0.3,
) -> dict:
    """
    Estimate LLM classification costs.

    Assumptions:
      - ~500 input tokens per classification (system prompt + market context)
      - ~200 output tokens per classification (JSON response)
      - archetype_match_rate: fraction that will match existing archetypes (free)
    """
    markets_needing_llm = int(unique_market_count * (1 - archetype_match_rate))
    input_tokens_per = 500
    output_tokens_per = 200

    total_input = markets_needing_llm * input_tokens_per
    total_output = markets_needing_llm * output_tokens_per

    # Pricing per 1M tokens (as of Feb 2026)
    models = {
        "GPT-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
        "Claude Haiku 4.5": {"input_per_m": 0.80, "output_per_m": 4.00},
        "Claude Sonnet 4.5": {"input_per_m": 3.00, "output_per_m": 15.00},
    }

    costs = {}
    for model, pricing in models.items():
        input_cost = (total_input / 1_000_000) * pricing["input_per_m"]
        output_cost = (total_output / 1_000_000) * pricing["output_per_m"]
        total = input_cost + output_cost
        batch_total = total * 0.5  # 50% batch discount
        costs[model] = {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total": total,
            "batch_total": batch_total,
        }

    return {
        "total_unique_markets": unique_market_count,
        "archetype_matches_est": int(unique_market_count * archetype_match_rate),
        "markets_needing_llm": markets_needing_llm,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "costs_by_model": costs,
    }


def print_cost_report(
    poly_events: list[UnifiedEvent],
    poly_markets: list[UnifiedMarket],
    kalshi_markets: list[UnifiedMarket],
    cost_est: dict,
):
    """Print a detailed cost estimation report."""
    print("\n" + "=" * 70)
    print("  INSIDER ALPHA — DATA DOWNLOAD & COST ESTIMATION REPORT")
    print("=" * 70)

    print("\n[DATA] DATA SUMMARY")
    print(f"  Date range:  {CUTOFF_DATE.date()} → present (all categories)")
    print(f"  Polymarket events:  {len(poly_events)}")
    print(f"  Polymarket markets: {len(poly_markets)}")
    print(f"  Kalshi markets:     {len(kalshi_markets)}")
    print(f"  Total markets across platforms:   {len(poly_markets) + len(kalshi_markets)}")

    # Volume filtering
    poly_above_vol = [m for m in poly_markets if m.volume >= MIN_VOLUME_USD]
    kalshi_above_vol = [m for m in kalshi_markets if m.volume >= MIN_VOLUME_USD]
    print(f"\n  Markets above ${MIN_VOLUME_USD:,.0f} volume threshold:")
    print(f"    Polymarket: {len(poly_above_vol)}")
    print(f"    Kalshi:     {len(kalshi_above_vol)}")
    print(f"    Total:      {len(poly_above_vol) + len(kalshi_above_vol)}")

    # Show category breakdown
    print("\n[TAGS] POLYMARKET EVENT TAG BREAKDOWN")
    tag_counts: Counter = Counter()
    for e in poly_events:
        if e.raw_data:
            for t in e.raw_data.get("tags", []):
                tag_counts[t.get("label", "")] += 1
    for tag, count in tag_counts.most_common(15):
        print(f"    {count:>4}x  {tag}")

    print("\n[TAGS] KALSHI CATEGORY BREAKDOWN")
    kalshi_cats: Counter = Counter()
    for m in kalshi_markets:
        kalshi_cats[m.category or "uncategorized"] += 1
    for cat, count in kalshi_cats.most_common():
        print(f"    {count:>4}x  {cat}")

    # Price history availability (Polymarket only)
    poly_with_tokens = [m for m in poly_markets if m.clob_token_ids]
    print(f"\n[PRICE] PRICE HISTORY (Polymarket only)")
    print(f"  Markets with CLOB token IDs: {len(poly_with_tokens)}")
    print(f"  (Each needs 1 API call to fetch timeseries for backtesting)")

    # Cost estimation
    ce = cost_est
    print(f"\n[COST] LLM CLASSIFICATION COST ESTIMATE")
    print(f"  Unique markets to classify:    {ce['total_unique_markets']}")
    print(f"  Est. archetype matches (free): {ce['archetype_matches_est']}")
    print(f"  Markets needing LLM call:      {ce['markets_needing_llm']}")
    print(f"  Est. input tokens total:       {ce['total_input_tokens']:,}")
    print(f"  Est. output tokens total:      {ce['total_output_tokens']:,}")

    print(f"\n  {'Model':<20} {'Standard':>10} {'Batch (50% off)':>16}")
    print(f"  {'-'*20} {'-'*10} {'-'*16}")
    for model, c in ce["costs_by_model"].items():
        print(f"  {model:<20} ${c['total']:>8.4f} ${c['batch_total']:>14.4f}")

    print(f"\n  Running GPT-4o-mini + Haiku on all markets:")
    mini_cost = ce["costs_by_model"]["GPT-4o-mini"]["total"]
    haiku_cost = ce["costs_by_model"]["Claude Haiku 4.5"]["total"]
    print(f"    Standard: ${mini_cost + haiku_cost:.4f}")
    print(f"    Batch:    ${(mini_cost + haiku_cost) * 0.5:.4f}")

    print("\n" + "=" * 70)
    print("  >>  Review the numbers above. No LLM calls have been made yet.")
    print("  >>  Proceed to Phase 2 (AI Classifier) when ready.")
    print("=" * 70 + "\n")


def main():
    cache = CacheManager()

    # ── Polymarket ──────────────────────────────────────────────
    logger.info("=== POLYMARKET ===")
    with PolymarketClient() as poly:
        poly_events = download_polymarket_all(poly)
        poly_markets = [m for e in poly_events for m in e.markets]

        # Cache and export
        cache.cache_events(poly_events, "polymarket_all_closed")
        if poly_markets:
            cache.export_markets_parquet(poly_markets, "polymarket_all_sample")

    # ── Kalshi ──────────────────────────────────────────────────
    logger.info("\n=== KALSHI ===")
    with KalshiClient(use_auth=False) as kalshi:
        kalshi_markets = download_kalshi_all(kalshi)

        if kalshi_markets:
            cache.export_markets_parquet(kalshi_markets, "kalshi_all_sample")

    # ── Combine & estimate costs ────────────────────────────────
    all_markets = poly_markets + kalshi_markets
    above_volume = [m for m in all_markets if m.volume >= MIN_VOLUME_USD]
    cost_est = estimate_llm_costs(len(above_volume))

    print_cost_report(poly_events, poly_markets, kalshi_markets, cost_est)


if __name__ == "__main__":
    main()
