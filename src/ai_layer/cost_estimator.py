"""
Pre-run cost estimation for the multi-agent pipeline.
Run this BEFORE any LLM calls to get user approval on projected spend.
"""

from src.data_layer.models import UnifiedMarket

# Pricing per 1M tokens (as of Feb 2026)
MODEL_PRICING = {
    "GPT-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "Claude Haiku 4.5": {"input_per_m": 0.80, "output_per_m": 4.00},
    "Claude Sonnet 4.5": {"input_per_m": 3.00, "output_per_m": 15.00},
}

# Estimated tokens per agent call
AGENT_TOKEN_ESTIMATES = {
    "agent_a": {"input": 500, "output": 200},   # text classification
    "agent_b": {"input": 800, "output": 300},   # quant analysis with tool outputs
    "revision": {"input": 1200, "output": 400}, # both reports in context
    "decision": {"input": 1500, "output": 500}, # full context + Bayesian reasoning
}


def estimate_pipeline_costs(markets: list[UnifiedMarket]) -> dict:
    """
    Estimate costs for running the full multi-agent pipeline.

    Returns a dict with per-agent and total cost breakdowns.
    """
    n = len(markets)
    costs_by_agent = {}

    for agent, tokens in AGENT_TOKEN_ESTIMATES.items():
        total_input = n * tokens["input"]
        total_output = n * tokens["output"]
        agent_costs = {}
        for model, pricing in MODEL_PRICING.items():
            input_cost = (total_input / 1_000_000) * pricing["input_per_m"]
            output_cost = (total_output / 1_000_000) * pricing["output_per_m"]
            total = input_cost + output_cost
            agent_costs[model] = {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total": round(total, 6),
                "batch_total": round(total * 0.5, 6),
            }
        costs_by_agent[agent] = {
            "markets": n,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "costs_by_model": agent_costs,
        }

    return {
        "total_markets": n,
        "costs_by_agent": costs_by_agent,
    }


def print_cost_estimate(est: dict) -> None:
    """Pretty-print a cost estimation report."""
    print("\n[COST] MULTI-AGENT PIPELINE COST ESTIMATE")
    print(f"  Total markets: {est['total_markets']}\n")

    for agent, data in est["costs_by_agent"].items():
        print(f"  {agent.upper()} ({data['total_input_tokens']:,} input / {data['total_output_tokens']:,} output tokens total)")
        print(f"  {'Model':<20} {'Standard':>10} {'Batch (50% off)':>16}")
        print(f"  {'-'*20} {'-'*10} {'-'*16}")
        for model, c in data["costs_by_model"].items():
            print(f"  {model:<20} ${c['total']:>8.4f} ${c['batch_total']:>14.4f}")
        print()
