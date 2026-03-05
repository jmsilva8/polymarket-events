from dataclasses import dataclass


@dataclass
class DecisionAgentParams:
    # LLM config (used by decision_agent — the LLM-based version)
    model_name: str = "gpt-4o-mini"  # fallback: claude-haiku-4-5-20251001
    temperature: float = 0.0  # Mandatory for reproducibility

    # ── Deterministic thresholds (used by decision_agent_deterministic) ──
    # Sweep these via backtesting to find optimal values.

    # Score gates: both agents must exceed these minimums to even consider GO
    min_a_score: int = 4
    min_b_score: int = 4

    # Weights for combining A and B scores into a single weighted_score
    # B gets higher default weight because it's evidence-based (price data)
    weight_a: float = 0.4
    weight_b: float = 0.6

    # Minimum weighted_score (1-10 scale) required to GO
    go_score_threshold: float = 6.5

    # Maximum estimated edge in percentage points (pp) at weighted_score = 10
    # Used in the linear edge estimation formula
    max_edge_pp: float = 20.0

    # Minimum edge (pp) required to GO — filters out marginal cases
    min_edge_pp: float = 5.0
