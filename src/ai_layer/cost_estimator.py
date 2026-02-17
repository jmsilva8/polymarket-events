"""
Pre-run cost estimation for LLM classification.
Run this BEFORE any LLM calls to get user approval on projected spend.
"""

from src.data_layer.models import UnifiedMarket
from src.ai_layer.archetypes import ArchetypeLibrary

# Pricing per 1M tokens (as of Feb 2026)
MODEL_PRICING = {
    "GPT-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "Claude Haiku 4.5": {"input_per_m": 0.80, "output_per_m": 4.00},
    "Claude Sonnet 4.5": {"input_per_m": 3.00, "output_per_m": 15.00},
}

# Estimated tokens per classification
EST_INPUT_TOKENS = 500  # system prompt + market context
EST_OUTPUT_TOKENS = 200  # JSON response


def estimate_classification_costs(
    markets: list[UnifiedMarket],
    archetypes: ArchetypeLibrary | None = None,
) -> dict:
    """
    Estimate costs before running the classifier.

    Returns a dict with:
      - total markets
      - archetype matches (free)
      - markets needing LLM
      - cost per model (standard + batch)
    """
    archetypes = archetypes or ArchetypeLibrary()

    archetype_matches = 0
    llm_needed = 0

    for m in markets:
        match = archetypes.match(m.question, m.description)
        if match and match.confidence == "high":
            archetype_matches += 1
        else:
            llm_needed += 1

    total_input = llm_needed * EST_INPUT_TOKENS
    total_output = llm_needed * EST_OUTPUT_TOKENS

    costs = {}
    for model, pricing in MODEL_PRICING.items():
        input_cost = (total_input / 1_000_000) * pricing["input_per_m"]
        output_cost = (total_output / 1_000_000) * pricing["output_per_m"]
        total = input_cost + output_cost
        costs[model] = {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total": round(total, 6),
            "batch_total": round(total * 0.5, 6),
        }

    return {
        "total_markets": len(markets),
        "archetype_matches": archetype_matches,
        "llm_needed": llm_needed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "costs_by_model": costs,
    }


def print_cost_estimate(est: dict) -> None:
    """Pretty-print a cost estimation report."""
    print("\n[COST] LLM CLASSIFICATION COST ESTIMATE")
    print(f"  Total markets:               {est['total_markets']}")
    print(f"  Archetype matches (free):    {est['archetype_matches']}")
    print(f"  Markets needing LLM call:    {est['llm_needed']}")
    print(f"  Est. input tokens:           {est['total_input_tokens']:,}")
    print(f"  Est. output tokens:          {est['total_output_tokens']:,}")
    print()
    print(f"  {'Model':<20} {'Standard':>10} {'Batch (50% off)':>16}")
    print(f"  {'-'*20} {'-'*10} {'-'*16}")
    for model, c in est["costs_by_model"].items():
        print(f"  {model:<20} ${c['total']:>8.4f} ${c['batch_total']:>14.4f}")

    mini = est["costs_by_model"]["GPT-4o-mini"]["total"]
    haiku = est["costs_by_model"]["Claude Haiku 4.5"]["total"]
    print(f"\n  GPT-4o-mini + Haiku together:")
    print(f"    Standard: ${mini + haiku:.4f}")
    print(f"    Batch:    ${(mini + haiku) * 0.5:.4f}")
