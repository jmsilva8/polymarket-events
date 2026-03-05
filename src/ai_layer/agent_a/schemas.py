"""Input/output schemas for Agent A (insider risk classifier)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AgentAInputPackage(BaseModel):
    """All market text signals Agent A needs to assess insider risk."""

    market_id: str
    question: str
    description: str
    category: str
    tags: list[str]
    platform: str  # "polymarket" | "kalshi"
    end_date: Optional[datetime] = None


class AgentAReport(BaseModel):
    """Final output from agent_a_initial() — passed to Revision Agent."""

    market_id: str
    market_title: str
    platform: str
    insider_risk_score: int = Field(ge=1, le=10)
    confidence: str = Field(description="high | medium | low")
    reasoning: str = Field(description="2-3 sentence explanation citing specific factors")
    info_holders: list[str] = Field(
        default_factory=list,
        description="Specific groups who plausibly have advance knowledge",
    )
    leak_vectors: list[str] = Field(
        default_factory=list,
        description="How non-public information could plausibly reach markets",
    )
    model_used: str
    classified_at: datetime = Field(default_factory=datetime.utcnow)


class AgentARevisionResponse(BaseModel):
    """Output from agent_a_revise() after receiving Revision Agent feedback."""

    finding_changed: bool = Field(
        description="True if score or direction changed meaningfully from original"
    )
    updated_insider_risk_score: int = Field(ge=1, le=10)
    updated_confidence: str = Field(description="high | medium | low")
    delta_explanation: str = Field(
        description="What changed and why, or why the original score holds"
    )
    final_reasoning: str = Field(
        description="Full updated reasoning for the score"
    )
    updated_info_holders: list[str]
    updated_leak_vectors: list[str]


class _LLMClassificationResponse(BaseModel):
    """Internal structured output schema bound to the LLM via .with_structured_output()."""

    insider_risk_score: int = Field(
        ge=1, le=10,
        description="Insider risk score from 1 (no advance knowledge possible) to 10 (tiny group with definitive advance knowledge and massive incentive)",
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    reasoning: str = Field(
        description="2-3 sentence explanation citing the specific information access, lead time, incentive, and leak probability factors that drive this score",
    )
    info_holders: list[str] = Field(
        default_factory=list,
        description="Specific groups or roles who plausibly have advance knowledge (e.g. 'company executives', 'regulatory staff', 'show producers')",
    )
    leak_vectors: list[str] = Field(
        default_factory=list,
        description="Concrete mechanisms by which non-public information could reach prediction markets (e.g. 'social media posts by insiders', 'order flow from connected traders')",
    )
