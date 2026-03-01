"""
LangGraph pipeline for insider risk classification of prediction markets.

Graph topology:
    START -> load_markets -> filter_markets -> classify_markets -> export_results -> END
"""

import logging
from collections import Counter
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END

from src.ai_layer.classifier import MarketClassifier
from src.ai_layer.schemas import MarketClassification
from src.config import MIN_VOLUME_USD, EXPORTS_DIR
from src.data_layer.models import UnifiedMarket, Platform, MarketStatus, Tag

logger = logging.getLogger(__name__)


# ── Graph State ───────────────────────────────────────────────────────

class ClassificationState(TypedDict):
    """Typed state flowing through the classification graph."""

    # Inputs (set at invocation)
    polymarket_csv: str
    kalshi_csv: str
    volume_threshold: float
    primary_model: str
    secondary_model: Optional[str]
    run_secondary: bool
    export_path: str

    # Intermediate (populated by nodes)
    all_markets: list            # list[UnifiedMarket]
    filtered_markets: list       # list[UnifiedMarket]

    # Outputs (populated by final nodes)
    results: list                # list[MarketClassification]
    summary: dict


# ── Node Functions ────────────────────────────────────────────────────

def load_markets(state: ClassificationState) -> dict:
    """Load markets from CSV files and reconstruct UnifiedMarket objects."""
    dfs = []
    if state.get("polymarket_csv"):
        dfs.append(pd.read_csv(state["polymarket_csv"]))
    if state.get("kalshi_csv"):
        dfs.append(pd.read_csv(state["kalshi_csv"]))

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    markets = []
    for _, row in combined.iterrows():
        platform = (
            Platform.POLYMARKET
            if row["platform"] == "polymarket"
            else Platform.KALSHI
        )

        tags_list = []
        if pd.notna(row.get("tags", "")):
            for label in str(row["tags"]).split("|"):
                label = label.strip()
                if label:
                    tags_list.append(
                        Tag(id=0, label=label, slug=label.lower().replace(" ", "-"))
                    )

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

    logger.info("Loaded %d markets from CSVs", len(markets))
    return {"all_markets": markets}


def filter_markets(state: ClassificationState) -> dict:
    """Filter markets by volume threshold."""
    threshold = state.get("volume_threshold", MIN_VOLUME_USD)
    filtered = [m for m in state["all_markets"] if m.volume >= threshold]
    logger.info(
        "Filtered to %d markets (volume >= $%s)",
        len(filtered),
        f"{threshold:,.0f}",
    )
    return {"filtered_markets": filtered}


def classify_markets(state: ClassificationState) -> dict:
    """Run the two-layer classifier (archetype + LLM) on all filtered markets."""
    classifier = MarketClassifier(
        primary_model=state.get("primary_model", "gpt-4o-mini"),
        secondary_model=state.get("secondary_model", "haiku"),
        cache_enabled=True,
    )
    results = classifier.classify_batch(
        state["filtered_markets"],
        run_secondary=state.get("run_secondary", False),
    )
    return {"results": results}


def export_results(state: ClassificationState) -> dict:
    """Export classification results to CSV and compute summary stats."""
    results: list[MarketClassification] = state["results"]

    # Build export rows
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

    export_path = state.get(
        "export_path", str(EXPORTS_DIR / "classifications.csv")
    )
    df = pd.DataFrame(rows)
    df.to_csv(export_path, index=False)

    # Compute summary statistics
    scores = Counter(r.insider_risk_score for r in results)
    models = Counter(r.model_used for r in results)
    confs = Counter(r.confidence for r in results)

    summary = {
        "total_classified": len(results),
        "score_distribution": dict(sorted(scores.items())),
        "model_usage": dict(models.most_common()),
        "confidence_distribution": dict(confs.most_common()),
        "high_risk_count": sum(1 for r in results if r.insider_risk_score >= 7),
        "export_path": export_path,
    }

    logger.info(
        "Exported %d classifications to %s", len(results), export_path
    )
    return {"summary": summary}


# ── Graph Construction ────────────────────────────────────────────────

def build_classification_graph():
    """Build and compile the classification pipeline graph."""
    builder = StateGraph(ClassificationState)

    builder.add_node("load_markets", load_markets)
    builder.add_node("filter_markets", filter_markets)
    builder.add_node("classify_markets", classify_markets)
    builder.add_node("export_results", export_results)

    builder.add_edge(START, "load_markets")
    builder.add_edge("load_markets", "filter_markets")
    builder.add_edge("filter_markets", "classify_markets")
    builder.add_edge("classify_markets", "export_results")
    builder.add_edge("export_results", END)

    return builder.compile()
