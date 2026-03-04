"""Run the insider risk classification pipeline via LangGraph."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

from src.graph import build_classification_graph

# Build and invoke the classification graph
graph = build_classification_graph()

final_state = graph.invoke({
    "polymarket_csv": "data/exports/polymarket_all_sample.csv",
    "kalshi_csv": "data/exports/kalshi_all_sample.csv",
    "volume_threshold": 10_000,
    "primary_model": "gpt-4o-mini",
    "secondary_model": "haiku",
    "run_secondary": False,
    "export_path": "data/exports/classifications_gpt4omini.csv",
})

# ── Display results ───────────────────────────────────────────────────

summary = final_state["summary"]
results = final_state["results"]

print(f"\nClassified {summary['total_classified']} markets")

print("\nScore distribution:")
for score, count in sorted(summary["score_distribution"].items()):
    bar = "#" * count
    print(f"  {score:2d}: {count:4d} {bar}")

print("\nModel usage:")
for model, count in summary["model_usage"].items():
    print(f"  {model}: {count}")

print("\nConfidence:")
for conf, count in summary["confidence_distribution"].items():
    print(f"  {conf}: {count}")

high_risk = [r for r in results if r.insider_risk_score >= 7]
print(f"\nHigh-risk markets (score >= 7): {len(high_risk)}")
for r in sorted(high_risk, key=lambda x: -x.insider_risk_score)[:15]:
    print(f"  [{r.insider_risk_score}] {r.market_title[:70]}")
    print(f"      {r.reasoning[:90]}")

print(f"\nExported to {summary['export_path']}")
