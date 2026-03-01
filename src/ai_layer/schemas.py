"""Classification output schema for the insider risk grading system."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MarketClassification(BaseModel):
    """Structured output from the market classifier (Layer 1 or Layer 2)."""

    market_id: str
    market_title: str
    platform: str  # "polymarket" | "kalshi"

    # Classification
    archetype_match: Optional[str] = Field(
        None, description="Matched archetype ID, or null if LLM-classified from scratch"
    )
    insider_risk_score: int = Field(ge=1, le=10)
    confidence: str = Field(description="high | medium | low")
    reasoning: str = Field(description="Brief explanation for the score")

    # Insider detail
    info_holders: list[str] = Field(
        default_factory=list,
        description="Groups who may have advance knowledge (e.g. 'academy voters', 'show producers')",
    )
    leak_vectors: list[str] = Field(
        default_factory=list,
        description="How information could leak to markets (e.g. 'social media', 'betting line shifts')",
    )

    # Metadata
    model_used: str = Field(
        description="gpt-4o-mini | claude-haiku-4.5 | claude-sonnet-4.5 | archetype-lookup"
    )
    classified_at: datetime = Field(default_factory=datetime.utcnow)


class LLMClassificationResponse(BaseModel):
    """
    Structured classification output from the LLM.
    Used with LangChain's .with_structured_output() — field descriptions
    are passed to the model as part of the JSON Schema.
    """

    insider_risk_score: int = Field(
        ge=1, le=10,
        description="Insider risk score from 1 (no insider edge) to 10 (extreme insider advantage)",
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    reasoning: str = Field(
        description="1-2 sentence explanation for the assigned score",
    )
    info_holders: list[str] = Field(
        default_factory=list,
        description="Groups who may have advance knowledge (e.g. 'academy voters', 'show producers')",
    )
    leak_vectors: list[str] = Field(
        default_factory=list,
        description="How information could leak to markets (e.g. 'social media', 'betting line shifts')",
    )
    suggested_archetype: Optional[str] = Field(
        None,
        description="If this looks like a reusable category, suggest a snake_case archetype name (e.g. 'reality_tv_elimination'), else null",
    )
