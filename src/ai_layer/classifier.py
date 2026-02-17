"""
Two-layer market classifier for insider risk scoring.

Layer 1: Archetype lookup (free, instant)
Layer 2: LLM classification via litellm (costs tokens, used for new market types)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import CACHE_DIR
from src.data_layer.models import UnifiedMarket
from src.ai_layer.archetypes import ArchetypeLibrary
from src.ai_layer.schemas import MarketClassification, LLMClassificationResponse
from src.ai_layer.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Cache directory for classifications
CLASSIFICATION_CACHE_DIR = CACHE_DIR / "classifications"
CLASSIFICATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MarketClassifier:
    """
    Classifies markets by insider risk using a two-layer approach:
      1. Check archetype library for pattern match (free)
      2. Fall back to LLM if no archetype matches

    Supports multiple LLM providers via litellm.
    """

    # litellm model identifiers
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
        self.primary_model = self.MODELS.get(primary_model, primary_model)
        self.secondary_model = self.MODELS.get(secondary_model, secondary_model) if secondary_model else None
        self.archetypes = archetypes or ArchetypeLibrary()
        self.cache_enabled = cache_enabled

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
        result = self._classify_with_llm(market, self.primary_model)
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

                if run_secondary and self.secondary_model:
                    secondary = self._classify_with_llm(market, self.secondary_model)
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
        self, market: UnifiedMarket, model: str
    ) -> MarketClassification:
        """Call the LLM to classify a market from scratch."""
        import litellm

        tags_str = ", ".join(t.label for t in market.tags) or market.category or "N/A"
        end_date_str = market.end_date.isoformat() if market.end_date else "Unknown"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=market.question,
            description=(market.description or "")[:500],
            platform=market.platform.value,
            tags=tags_str,
            end_date=end_date_str,
        )

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.1,
            response_format={"type": "json_object"} if "gpt" in model else None,
        )

        raw_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

        parsed = LLMClassificationResponse(**json.loads(raw_text))

        # Determine model display name
        if "gpt-4o-mini" in model:
            model_name = "gpt-4o-mini"
        elif "haiku" in model:
            model_name = "claude-haiku-4.5"
        elif "sonnet" in model:
            model_name = "claude-sonnet-4.5"
        else:
            model_name = model

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
            model_used=model_name,
        )

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
