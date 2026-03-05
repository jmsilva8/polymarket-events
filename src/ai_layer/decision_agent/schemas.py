"""Input and output schemas for the Decision Agent."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


@dataclass
class DecisionAgentInputPackage:
    """Complete input to the Decision Agent."""

    # From Revision Agent
    revision_flag: Literal[
        "NONE",
        "PUBLIC_INFO_ADJUSTED",
        "PRE_SIGNAL",
        "REVERSION",
        "INTERNAL_CONFLICT",
        "DIRECTIONAL_CONFLICT",
    ]
    flag_explanation: str
    agent_a_report: dict    # Full AgentAReport serialized
    agent_b_report: dict    # Full AgentBReport serialized
    revision_notes: str
    recommendation_to_decision_agent: Literal["GO_EVALUATE", "SKIP", "WATCH"]

    # Market context
    current_market_price: float     # 0.0–1.0 (YES implied probability)
    evaluation_date: datetime
    end_date: datetime
    market_id: str


class DecisionAgentOutput(BaseModel):
    """Final output — passed to export/portfolio layer."""

    decision: Literal["GO", "SKIP"]
    bet_direction: Literal["YES", "NO", "null"]

    analysis: dict = Field(default_factory=lambda: {
        "agent_a_score": None,
        "agent_b_score": None,
        "weight_a_percentage": None,
        "weight_b_percentage": None,
        "weighting_rationale": "",
        "weighted_score": None,
        "current_market_price": None,
        "adjusted_probability_of_win": None,
        "estimated_edge_pp": None,
        "edge_assessment": "",
    })

    full_reasoning: str
    revision_flag_applied: str

    recommendation: dict = Field(default_factory=lambda: {
        "action": None,   # "INVEST" | "PASS" | "WATCH"
        "bet": None,      # "YES" | "NO" | None
        "risk_grade": None,
        "current_price": None,
        "reasoning_summary": "",
    })

    evaluation_date: str
    market_id: str
