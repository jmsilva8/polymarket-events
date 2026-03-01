"""
Two-layer market classifier for insider risk scoring.

Layer 1: Archetype lookup (free, instant)
Layer 2: LLM classification via LangChain (costs tokens, used for new market types)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import CACHE_DIR
from src.data_layer.models import UnifiedMarket
from src.ai_layer.archetypes import ArchetypeLibrary
from src.ai_layer.schemas import MarketClassification, LLMClassificationResponse
from src.ai_layer.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Cache directory for classifications
CLASSIFICATION_CACHE_DIR = CACHE_DIR / "classifications"
CLASSIFICATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_provider(model: str) -> str:
    """Map a model identifier to its LangChain model_provider string."""
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    raise ValueError(f"Cannot determine provider for model: {model!r}")


class MarketClassifier:
    """
    Classifies markets by insider risk using a two-layer approach:
      1. Check archetype library for pattern match (free)
      2. Fall back to LLM if no archetype matches

    Supports multiple LLM providers via LangChain init_chat_model.
    """

    # Model short-name aliases
    MODELS = {
        "gpt-4o-mini": "gpt-4o-mini",
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20250929",
    }

    def __init__(
        self,
        primary_model: str = "gpt-4o-mini",
        secondary_model: Optional[str] = "haiku",
        archetypes: Optional[ArchetypeLibrary] = None,
        cache_enabled: bool = True,
    ):
        self.primary_model_name = self.MODELS.get(primary_model, primary_model)
        self.secondary_model_name = (
            self.MODELS.get(secondary_model, secondary_model)
            if secondary_model
            else None
        )
        self.archetypes = archetypes or ArchetypeLibrary()
        self.cache_enabled = cache_enabled

        # Initialize LangChain chat models with structured output
        self.primary_llm = self._init_llm(self.primary_model_name)
        self.secondary_llm = (
            self._init_llm(self.secondary_model_name)
            if self.secondary_model_name
            else None
        )

    def _init_llm(self, model: str):
        """Create a LangChain chat model with structured output bound."""
        provider = _resolve_provider(model)
        base_llm = init_chat_model(model, temperature=0.1, model_provider=provider)
        return base_llm.with_structured_output(LLMClassificationResponse)

    def classify(self, market: UnifiedMarket) -> MarketClassification:
        """
        Classify a single market. Tries archetype match first, then LLM.
        Results are cached by market_id.
        """
        # Check cache
        if self.cache_enabled:
            cached = self._load_cached(market.market_id)
            if cached:
                return cached

        # Layer 1: Archetype match
        arch_match = self.archetypes.match(market.question, market.description)
        if arch_match and arch_match.confidence == "high":
            result = MarketClassification(
                market_id=market.market_id,
                market_title=market.question,
                platform=market.platform.value,
                archetype_match=arch_match.archetype_id,
                insider_risk_score=arch_match.score,
                confidence=arch_match.confidence,
                reasoning=arch_match.reasoning,
                info_holders=arch_match.info_holders,
                leak_vectors=[],
                model_used="archetype-lookup",
            )
            self._save_cached(result)
            return result

        # Layer 2: LLM classification
        result = self._classify_with_llm(
            market, self.primary_model_name, self.primary_llm
        )
        self._save_cached(result)
        return result

    def classify_batch(
        self,
        markets: list[UnifiedMarket],
        run_secondary: bool = False,
    ) -> list[MarketClassification]:
        """
        Classify a batch of markets with progress bar.

        If run_secondary=True, also runs the secondary model and logs
        disagreements where scores differ by >= 2 points.
        """
        results: list[MarketClassification] = []
        disagreements: list[dict] = []

        for market in tqdm(markets, desc="Classifying markets"):
            try:
                primary = self.classify(market)
                results.append(primary)

                if run_secondary and self.secondary_llm:
                    secondary = self._classify_with_llm(
                        market, self.secondary_model_name, self.secondary_llm
                    )
                    diff = abs(primary.insider_risk_score - secondary.insider_risk_score)
                    if diff >= 2:
                        disagreements.append({
                            "market_id": market.market_id,
                            "title": market.question[:80],
                            "primary_score": primary.insider_risk_score,
                            "primary_model": primary.model_used,
                            "secondary_score": secondary.insider_risk_score,
                            "secondary_model": secondary.model_used,
                            "diff": diff,
                        })

            except Exception as e:
                logger.error("Failed to classify %s: %s", market.market_id, e)

        if disagreements:
            logger.warning(
                "%d disagreements (score diff >= 2) between models:", len(disagreements)
            )
            for d in disagreements:
                logger.warning(
                    "  %s: %s=%d vs %s=%d (diff=%d) | %s",
                    d["market_id"], d["primary_model"], d["primary_score"],
                    d["secondary_model"], d["secondary_score"], d["diff"],
                    d["title"],
                )

        return results

    def _classify_with_llm(
        self, market: UnifiedMarket, model_name: str, llm
    ) -> MarketClassification:
        """Call the LLM to classify a market using structured output."""
        tags_str = ", ".join(t.label for t in market.tags) or market.category or "N/A"
        end_date_str = market.end_date.isoformat() if market.end_date else "Unknown"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=market.question,
            description=(market.description or "")[:500],
            platform=market.platform.value,
            tags=tags_str,
            end_date=end_date_str,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # .invoke() returns a validated LLMClassificationResponse directly
        parsed: LLMClassificationResponse = llm.invoke(messages)

        return MarketClassification(
            market_id=market.market_id,
            market_title=market.question,
            platform=market.platform.value,
            archetype_match=None,
            insider_risk_score=parsed.insider_risk_score,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            info_holders=parsed.info_holders,
            leak_vectors=parsed.leak_vectors,
            model_used=self._display_name(model_name),
        )

    @staticmethod
    def _display_name(model: str) -> str:
        """Convert a model identifier to a human-readable display name."""
        if "gpt-4o-mini" in model:
            return "gpt-4o-mini"
        elif "haiku" in model:
            return "claude-haiku-4.5"
        elif "sonnet" in model:
            return "claude-sonnet-4.5"
        return model

    # ── Cache ──────────────────────────────────────────────────────

    def _cache_path(self, market_id: str) -> Path:
        return CLASSIFICATION_CACHE_DIR / f"{market_id}.json"

    def _save_cached(self, result: MarketClassification) -> None:
        if not self.cache_enabled:
            return
        self._cache_path(result.market_id).write_text(
            result.model_dump_json(indent=2)
        )

    def _load_cached(self, market_id: str) -> Optional[MarketClassification]:
        path = self._cache_path(market_id)
        if path.exists():
            try:
                return MarketClassification(**json.loads(path.read_text()))
            except Exception:
                return None
        return None
