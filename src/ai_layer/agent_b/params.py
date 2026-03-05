"""Configurable thresholds for Agent B tools — tune via backtesting sweep."""

from dataclasses import dataclass, field


@dataclass
class AgentBParams:
    """
    Configurable thresholds for Agent B tools.
    These are starting defaults — tune via backtesting sweep.
    Same pattern as StrategyParams in src/backtest_engine/strategy.py.
    """
    # Input assessment
    min_price_points: int = 10

    # price_jump_detector
    jump_windows_hours: list[int] = field(default_factory=lambda: [6, 12, 24, 48, 72])
    min_jump_pp: float = 5.0
    sustained_revert_threshold: float = 0.5

    # volume_spike_checker
    spike_threshold_multiplier: float = 3.0

    # momentum_analyzer
    momentum_horizons_hours: list[int] = field(default_factory=lambda: [6, 12, 24, 48])
    min_r_squared: float = 0.6
    min_slope_pp_per_hour: float = 0.2
