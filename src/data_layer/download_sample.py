"""
Download closed Polymarket events by curated tag slugs, deduplicate,
and export to parquet. Generates a cost estimation report before any LLM calls.

Usage:
    python -m src.data_layer.download_sample
"""

import sys
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import EXPORTS_DIR, MIN_VOLUME_USD, DOWNLOAD_TAG_SLUGS
from src.data_layer.polymarket_client import PolymarketClient
from src.data_layer.cache_manager import CacheManager
from src.data_layer.models import UnifiedMarket, UnifiedEvent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Only include markets that closed on or after this date.
CUTOFF_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def download_polymarket_by_tags(
    client: PolymarketClient,
    tag_slugs: list[str] = DOWNLOAD_TAG_SLUGS,
) -> tuple[list[UnifiedEvent], dict[str, int]]:
    """Fetch closed events for each tag slug, deduplicate by event_id.

    Returns:
        (deduplicated events, per-tag event counts before dedup)
    """
    seen_ids: set[str] = set()
    all_events: list[UnifiedEvent] = []
    tag_counts: dict[str, int] = {}

    for slug in tag_slugs:
        logger.info("--- Fetching tag: %s ---", slug)
        events = client.get_all_closed_events(
            tag_slug=slug,
            end_date_min=CUTOFF_DATE.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ascending=True,
            max_pages=200,
            cache_key=f"polymarket_{slug}",
        )
        tag_counts[slug] = len(events)

        new = 0
        for ev in events:
            if ev.event_id not in seen_ids:
                seen_ids.add(ev.event_id)
                all_events.append(ev)
                new += 1
        logger.info(
            "  %s: %d events fetched, %d new (after dedup)", slug, len(events), new
        )

    logger.info(
        "Total: %d unique events from %d tag pulls", len(all_events), len(tag_slugs)
    )
    return all_events, tag_counts


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


def print_report(
    events: list[UnifiedEvent],
    all_markets: list[UnifiedMarket],
    filtered_markets: list[UnifiedMarket],
    tag_counts: dict[str, int],
    cost_est: dict,
):
    """Print download summary and LLM cost estimation report."""
    print("\n" + "=" * 70)
    print("  INSIDER ALPHA -- DATA DOWNLOAD & COST ESTIMATION REPORT")
    print("=" * 70)

    print("\n[DATA] DATA SUMMARY")
    print(f"  Date range:  {CUTOFF_DATE.date()} -> present")
    print(f"  Unique events:  {len(events):,}")
    print(f"  Total markets (all volumes):  {len(all_markets):,}")
    print(f"  Markets >= ${MIN_VOLUME_USD:,.0f} volume:  {len(filtered_markets):,}")

    print("\n[TAGS] PER-TAG EVENT COUNTS (before dedup)")
    for slug, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:>5}x  {slug}")

    # Top tags across deduplicated events
    print("\n[TAGS] TOP TAGS ACROSS EVENTS (after dedup)")
    label_counts: Counter = Counter()
    for e in events:
        if e.raw_data:
            for t in e.raw_data.get("tags", []):
                label_counts[t.get("label", "")] += 1
    for tag, count in label_counts.most_common(20):
        print(f"    {count:>5}x  {tag}")

    # Price history availability
    with_tokens = [m for m in filtered_markets if m.clob_token_ids]
    print(f"\n[PRICE] PRICE HISTORY")
    print(f"  Markets with CLOB token IDs: {len(with_tokens):,}")

    # Cost estimation
    ce = cost_est
    print(f"\n[COST] LLM CLASSIFICATION COST ESTIMATE")
    print(f"  Markets to classify:           {ce['total_unique_markets']:,}")
    print(f"  Est. archetype matches (free): {ce['archetype_matches_est']:,}")
    print(f"  Markets needing LLM call:      {ce['markets_needing_llm']:,}")
    print(f"  Est. input tokens total:       {ce['total_input_tokens']:,}")
    print(f"  Est. output tokens total:      {ce['total_output_tokens']:,}")

    print(f"\n  {'Model':<20} {'Standard':>10} {'Batch (50% off)':>16}")
    print(f"  {'-'*20} {'-'*10} {'-'*16}")
    for model, c in ce["costs_by_model"].items():
        print(f"  {model:<20} ${c['total']:>8.4f} ${c['batch_total']:>14.4f}")

    print("\n" + "=" * 70)
    print("  >>  Review the numbers above. No LLM calls have been made yet.")
    print("  >>  Proceed to Phase 2 (AI Classifier) when ready.")
    print("=" * 70 + "\n")


def main():
    cache = CacheManager()

    # ── Download by tags ──────────────────────────────────────────
    logger.info("=== POLYMARKET (tag-based download) ===")
    with PolymarketClient() as poly:
        events, tag_counts = download_polymarket_by_tags(poly)

    # ── Flatten & filter ──────────────────────────────────────────
    all_markets = [m for e in events for m in e.markets]
    filtered_markets = [m for m in all_markets if m.volume >= MIN_VOLUME_USD]
    logger.info(
        "Markets: %d total, %d above $%s volume threshold",
        len(all_markets),
        len(filtered_markets),
        f"{MIN_VOLUME_USD:,.0f}",
    )

    # ── Cache & export ────────────────────────────────────────────
    cache.cache_events(events, "polymarket_tagged_closed")
    if filtered_markets:
        cache.export_markets_parquet(filtered_markets, "polymarket_tagged_sample")
        cache.export_markets_csv(filtered_markets, "polymarket_tagged_sample")

    # ── Report ────────────────────────────────────────────────────
    cost_est = estimate_llm_costs(len(filtered_markets))
    print_report(events, all_markets, filtered_markets, tag_counts, cost_est)


if __name__ == "__main__":
    main()
