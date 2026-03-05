"""Configurable thresholds for Agent B tools — tune via backtesting sweep."""

from dataclasses import dataclass, field


@dataclass
class AgentBParams:
    """
    Configurable thresholds for Agent B tools.
    These are starting defaults — tune via backtesting sweep.
    Same pattern as StrategyParams in src/backtest_engine/strategy.py.

    Backtesting note: historical data arrives at 12h resolution
    over a 5-day window (120h before close to 24h before close = 96h).
    At 12h resolution that yields ~8 data points.
    min_price_points is set to 3 to avoid filtering markets with sparse data.
    Sub-12h windows are excluded since no data exists at that resolution.
    """
    # Input assessment
    min_price_points: int = 3
    # If set, appended to data_quality_notes so LLM knows data resolution
    data_frequency_hours: int = 12

    # price_jump_detector — covers the full 96h data window
    jump_windows_hours: list[int] = field(default_factory=lambda: [12, 24, 36, 48, 60, 72, 84, 96])
    min_jump_pp: float = 5.0
    sustained_revert_threshold: float = 0.5

    # volume_spike_checker
    spike_threshold_multiplier: float = 3.0

    # momentum_analyzer — 6h horizon dropped; lowered r_squared since fewer points
    momentum_horizons_hours: list[int] = field(default_factory=lambda: [24, 48, 72])
    min_r_squared: float = 0.5
    min_slope_pp_per_hour: float = 0.2
