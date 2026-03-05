"""Configurable thresholds for Agent B tools — tune via backtesting sweep."""

from dataclasses import dataclass, field


@dataclass
class AgentBParams:
    """
    Configurable thresholds for Agent B tools.
    These are starting defaults — tune via backtesting sweep.
    Same pattern as StrategyParams in src/backtest_engine/strategy.py.

    Backtesting note: historical data arrives at 12h resolution
    (6 points: t-72h, t-60h, t-48h, t-36h, t-24h, t-12h before close).
    min_price_points is set to 5 to allow tools to run with this data.
    Sub-12h windows are excluded since no data exists at that resolution.
    """
    # Input assessment
    min_price_points: int = 5
    # If set, appended to data_quality_notes so LLM knows data resolution
    data_frequency_hours: int = 12

    # price_jump_detector — 6h window dropped (no data at that resolution)
    jump_windows_hours: list[int] = field(default_factory=lambda: [12, 24, 36, 48, 60, 72])
    min_jump_pp: float = 5.0
    sustained_revert_threshold: float = 0.5

    # volume_spike_checker
    spike_threshold_multiplier: float = 3.0

    # momentum_analyzer — 6h horizon dropped; lowered r_squared since fewer points
    momentum_horizons_hours: list[int] = field(default_factory=lambda: [24, 48, 72])
    min_r_squared: float = 0.5
    min_slope_pp_per_hour: float = 0.2
