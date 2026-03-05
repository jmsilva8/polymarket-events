"""All Pydantic and dataclass models for Agent B."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from src.data_layer.models import PricePoint


# ── Input ─────────────────────────────────────────────────────────────────────

@dataclass
class VolumePoint:
    timestamp: datetime
    volume_usd: float


@dataclass
class AgentBInputPackage:
    """
    Everything Agent B receives. No market metadata — numbers only.
    All timeseries must be truncated at evaluation_date by the caller.
    """
    evaluation_date: datetime
    end_date: datetime
    price_history: list[PricePoint]     # sorted chronologically, truncated at T
    current_price: float

    # Volume — provide whatever is available; Agent B adapts
    volume_history: list[VolumePoint] = field(default_factory=list)
    volume_total_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    market_age_days: Optional[float] = None


# ── Input assessment ──────────────────────────────────────────────────────────

class InputAssessment(BaseModel):
    can_run_price_jump: bool
    can_run_momentum: bool
    can_run_volume: bool
    volume_mode: Literal["timeseries", "approximation", "unavailable"]
    price_point_count: int
    hours_to_close: float
    skipped_tools: list[str]
    data_quality_notes: list[str]


# ── Tool output models ────────────────────────────────────────────────────────

class PriceJumpResult(BaseModel):
    detected: bool
    largest_jump_pp: float
    direction: Literal["UP", "DOWN", "NONE"]
    best_window_hours: int
    from_price: float
    to_price: float
    hours_before_close: float
    is_sustained: bool
    move_shape: Literal["gradual", "sudden", "none"]
    # gradual = price built up over multiple hours (consistent with informed accumulation)
    # sudden  = price moved sharply in < 2 hours (consistent with public news reaction)
    all_windows: list[dict]


class VolumeResult(BaseModel):
    mode: Literal["timeseries", "approximation", "unavailable"]
    spike_detected: bool
    spike_ratio: Optional[float]
    baseline_avg: Optional[float]
    recent_volume: Optional[float]
    hours_before_close: Optional[float]
    pattern: Optional[Literal["burst", "sustained", "flat"]]
    note: str


class MomentumHorizon(BaseModel):
    horizon_hours: int
    slope_pp_per_hour: float
    direction: Literal["UP", "DOWN", "FLAT"]
    r_squared: float
    price_at_start: float
    price_at_end: float


class MomentumResult(BaseModel):
    dominant_direction: Literal["UP", "DOWN", "FLAT", "MIXED"]
    consistency: Literal["trending", "volatile", "reverting", "insufficient_data"]
    acceleration: Literal["increasing", "decreasing", "stable", "unknown"]
    by_horizon: list[MomentumHorizon]


class ConsistencyCheck(BaseModel):
    price_and_momentum_agree: bool
    volume_confirms_direction: bool
    signals_contradictory: bool
    dominant_direction: Literal["UP", "DOWN", "MIXED", "NONE"]
    conflicting_signals: list[str]


# ── Agent B output models ─────────────────────────────────────────────────────

class SignalBreakdown(BaseModel):
    detected: bool
    direction: Literal["UP", "DOWN", "FLAT", "NONE"]
    magnitude: Literal["none", "weak", "moderate", "strong", "extreme"]
    timing_quality: Literal["poor", "acceptable", "good", "excellent"]
    sustained: bool
    weight_assigned: Literal["low", "medium", "high"]
    note: str  # one sentence, numbers only


class AgentBReport(BaseModel):
    """Initial report — produced by agent_b_initial()."""
    signal_direction: Literal["YES", "NO", "SKIP"]
    behavior_score: int                              # 1–10 integer
    confidence: Literal["low", "medium", "high"]
    price_jump_assessment: SignalBreakdown
    volume_assessment: SignalBreakdown
    momentum_assessment: SignalBreakdown
    consistency: ConsistencyCheck
    key_findings: list[str]
    reasoning: str
    context_for_other_agents: str
    # Audit fields
    evaluation_date: str
    tools_run: list[str]
    tools_skipped: list[str]
    data_quality_notes: list[str]


class AgentBRevisionResponse(BaseModel):
    """Produced by agent_b_revise() when Revision Agent sends feedback."""
    tools_re_run: list[str]
    parameter_changes: dict
    finding_changed: bool
    updated_signal_direction: Literal["YES", "NO", "SKIP"]
    updated_behavior_score: int
    updated_confidence: Literal["low", "medium", "high"]
    delta_explanation: str
    final_reasoning: str
    context_for_other_agents: str
