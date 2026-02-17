"""Run full GPT-4o-mini classification on all entertainment markets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.WARNING)

from src.data_layer.models import UnifiedMarket, Platform, MarketStatus, Tag
from src.ai_layer.archetypes import ArchetypeLibrary
from src.ai_layer.classifier import MarketClassifier

# Load markets
poly_df = pd.read_csv("data/exports/polymarket_entertainment_sample.csv")
kalshi_df = pd.read_csv("data/exports/kalshi_entertainment_sample.csv")
combined = pd.concat([poly_df, kalshi_df], ignore_index=True)
above_threshold = combined[combined["volume"] >= 10_000]

markets = []
for _, row in above_threshold.iterrows():
    platform = Platform.POLYMARKET if row["platform"] == "polymarket" else Platform.KALSHI
    tags_list = []
    if pd.notna(row.get("tags", "")):
        for label in str(row["tags"]).split("|"):
            label = label.strip()
            if label:
                tags_list.append(Tag(id=0, label=label, slug=label.lower().replace(" ", "-")))
    m = UnifiedMarket(
        platform=platform,
        market_id=str(row["market_id"]),
        condition_id=None,
        slug=str(row.get("slug", "")),
        question=str(row["question"]),
        description="",
        category=str(row.get("category", "")),
        volume=float(row["volume"]),
        tags=tags_list,
        status=MarketStatus.CLOSED,
    )
    markets.append(m)

print(f"Total markets: {len(markets)}")

# Classify all
classifier = MarketClassifier(primary_model="gpt-4o-mini", cache_enabled=True)
results = classifier.classify_batch(markets)

print(f"\nClassified {len(results)} markets")

# Score distribution
scores = Counter(r.insider_risk_score for r in results)
print("\nScore distribution:")
for score in sorted(scores.keys()):
    bar = "#" * scores[score]
    print(f"  {score:2d}: {scores[score]:4d} {bar}")

# Model usage
models = Counter(r.model_used for r in results)
print("\nModel usage:")
for model, count in models.most_common():
    print(f"  {model}: {count}")

# Confidence distribution
confs = Counter(r.confidence for r in results)
print("\nConfidence:")
for conf, count in confs.most_common():
    print(f"  {conf}: {count}")

# High-risk markets (score >= 7)
high_risk = [r for r in results if r.insider_risk_score >= 7]
print(f"\nHigh-risk markets (score >= 7): {len(high_risk)}")
for r in sorted(high_risk, key=lambda x: -x.insider_risk_score)[:15]:
    print(f"  [{r.insider_risk_score}] {r.market_title[:70]}")
    print(f"      {r.reasoning[:90]}")

# Export results to CSV
rows = []
for r in results:
    rows.append({
        "market_id": r.market_id,
        "market_title": r.market_title,
        "platform": r.platform,
        "insider_risk_score": r.insider_risk_score,
        "confidence": r.confidence,
        "reasoning": r.reasoning,
        "info_holders": "|".join(r.info_holders),
        "leak_vectors": "|".join(r.leak_vectors),
        "model_used": r.model_used,
        "archetype_match": r.archetype_match or "",
    })
df = pd.DataFrame(rows)
df.to_csv("data/exports/classifications_gpt4omini.csv", index=False)
print(f"\nExported classifications to data/exports/classifications_gpt4omini.csv")
