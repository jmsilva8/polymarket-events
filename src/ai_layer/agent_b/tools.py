"""
Pure Python tool functions for Agent B.

All functions are deterministic: same inputs always produce same outputs.
No LLM calls, no I/O.
"""

from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np

from src.ai_layer.agent_b.params import AgentBParams
from src.ai_layer.agent_b.schemas import (
    AgentBInputPackage,
    ConsistencyCheck,
    MomentumHorizon,
    MomentumResult,
    PriceJumpResult,
    VolumeResult,
)
from src.data_layer.models import PricePoint


# ── Helpers ────────────────────────────────────────────────────────────────────

def _price_at(
    price_history: list[PricePoint],
    target: datetime,
) -> Optional[float]:
    """Return the price at or just before `target`. None if no points exist."""
    candidates = [p for p in price_history if p.timestamp <= target]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.timestamp).price


def _prices_in_window(
    price_history: list[PricePoint],
    start: datetime,
    end: datetime,
) -> list[PricePoint]:
    """Return all price points in [start, end] inclusive."""
    return [p for p in price_history if start <= p.timestamp <= end]


# ── price_jump_detector ────────────────────────────────────────────────────────

def price_jump_detector(
    price_history: list[PricePoint],
    end_date: datetime,
    evaluation_date: datetime,
    params: AgentBParams,
) -> PriceJumpResult:
    """
    Detect the largest price jump across multiple look-back windows.

    For each window_hours:
      - from_price = price at (evaluation_date - window_hours) or earliest available
      - to_price   = price at evaluation_date
      - delta_pp   = (to_price - from_price) * 100
      - sustained  = price never pulled back > threshold * jump after the move started
      - move_shape = "sudden" if jump happened within 2h sub-window, else "gradual"
    """
    hours_to_close = (end_date - evaluation_date).total_seconds() / 3600

    all_windows: list[dict] = []
    best: Optional[dict] = None

    for window_hours in params.jump_windows_hours:
        window_start = evaluation_date - timedelta(hours=window_hours)
        from_price = _price_at(price_history, window_start)
        to_price = _price_at(price_history, evaluation_date)

        if from_price is None or to_price is None:
            all_windows.append({
                "window_hours": window_hours,
                "detected": False,
                "reason": "insufficient data",
            })
            continue

        delta_pp = (to_price - from_price) * 100

        if abs(delta_pp) < params.min_jump_pp:
            all_windows.append({
                "window_hours": window_hours,
                "detected": False,
                "delta_pp": round(delta_pp, 2),
            })
            continue

        # Sustained check: did price pull back more than threshold * |jump| after moving?
        points_after = _prices_in_window(price_history, window_start, evaluation_date)
        is_sustained = True
        if points_after and abs(delta_pp) > 0:
            revert_limit = abs(delta_pp) * params.sustained_revert_threshold
            # Price after the move should not swing back by more than revert_limit pp
            peak = max(p.price for p in points_after) if delta_pp > 0 else min(p.price for p in points_after)
            if delta_pp > 0:
                worst_revert_pp = (peak - to_price) * 100  # how much it fell from peak
                is_sustained = worst_revert_pp <= revert_limit
            else:
                worst_revert_pp = (to_price - peak) * 100  # how much it rose from trough
                is_sustained = worst_revert_pp <= revert_limit

        # Move shape: check if jump happened within a 2h sub-window
        move_shape: Literal["gradual", "sudden", "none"] = "none"
        if abs(delta_pp) >= params.min_jump_pp:
            # Scan 2h sub-windows within the main window
            sudden = False
            for sub_offset_hours in range(int(window_hours)):
                sub_start = window_start + timedelta(hours=sub_offset_hours)
                sub_end = sub_start + timedelta(hours=2)
                p_start = _price_at(price_history, sub_start)
                p_end = _price_at(price_history, sub_end)
                if p_start is not None and p_end is not None:
                    sub_delta = abs((p_end - p_start) * 100)
                    # If a 2h sub-window accounts for > 70% of the total move → sudden
                    if sub_delta >= 0.7 * abs(delta_pp):
                        sudden = True
                        break
            move_shape = "sudden" if sudden else "gradual"

        entry = {
            "window_hours": window_hours,
            "detected": True,
            "delta_pp": round(delta_pp, 2),
            "from_price": round(from_price, 4),
            "to_price": round(to_price, 4),
            "is_sustained": is_sustained,
            "move_shape": move_shape,
        }
        all_windows.append(entry)

        if best is None or abs(delta_pp) > abs(best["delta_pp"]):
            best = entry

    if best is None:
        # No window exceeded min_jump_pp
        last_price = price_history[-1].price if price_history else 0.0
        return PriceJumpResult(
            detected=False,
            largest_jump_pp=0.0,
            direction="NONE",
            best_window_hours=0,
            from_price=last_price,
            to_price=last_price,
            hours_before_close=hours_to_close,
            is_sustained=False,
            move_shape="none",
            all_windows=all_windows,
        )

    direction: Literal["UP", "DOWN", "NONE"] = "UP" if best["delta_pp"] > 0 else "DOWN"
    return PriceJumpResult(
        detected=True,
        largest_jump_pp=best["delta_pp"],
        direction=direction,
        best_window_hours=best["window_hours"],
        from_price=best["from_price"],
        to_price=best["to_price"],
        hours_before_close=hours_to_close,
        is_sustained=best["is_sustained"],
        move_shape=best["move_shape"],
        all_windows=all_windows,
    )


# ── volume_spike_checker ───────────────────────────────────────────────────────

def volume_spike_checker(
    package: AgentBInputPackage,
    volume_mode: Literal["timeseries", "approximation"],
    params: AgentBParams,
) -> VolumeResult:
    """
    Check for abnormal volume relative to baseline.

    Timeseries mode: use per-period volume_history.
    Approximation mode: use volume_total_usd + volume_24h_usd + market_age_days.
    """
    hours_to_close = (
        package.end_date - package.evaluation_date
    ).total_seconds() / 3600

    if volume_mode == "timeseries":
        cutoff = package.evaluation_date - timedelta(hours=24)
        baseline_points = [
            v for v in package.volume_history
            if v.timestamp < cutoff
        ]
        recent_points = [
            v for v in package.volume_history
            if v.timestamp >= cutoff
        ]

        if not baseline_points:
            return VolumeResult(
                mode="timeseries",
                volume_source="timeseries",
                spike_detected=False,
                spike_ratio=None,
                baseline_avg=None,
                recent_volume=None,
                hours_before_close=hours_to_close,
                pattern=None,
                note="Insufficient baseline data for timeseries spike check.",
            )

        # Compute daily baseline average
        if baseline_points:
            earliest = min(v.timestamp for v in baseline_points)
            span_days = max(
                (cutoff - earliest).total_seconds() / 86400, 1.0
            )
            baseline_total = sum(v.volume_usd for v in baseline_points)
            baseline_avg = baseline_total / span_days
        else:
            baseline_avg = 0.0

        recent_volume = sum(v.volume_usd for v in recent_points)
        spike_ratio = recent_volume / baseline_avg if baseline_avg > 0 else None
        spike_detected = (
            spike_ratio is not None
            and spike_ratio >= params.spike_threshold_multiplier
        )

        # Pattern: sustained vs burst
        pattern: Optional[Literal["burst", "sustained", "flat"]] = None
        if spike_detected and recent_points:
            volumes = [v.volume_usd for v in recent_points]
            max_v = max(volumes)
            # burst = top 25% of periods account for > 80% of total
            if len(volumes) >= 2:
                threshold = max_v * 0.25
                concentrated = sum(v for v in volumes if v >= threshold)
                pattern = "burst" if concentrated / recent_volume > 0.8 else "sustained"
        elif not spike_detected and recent_volume > 0:
            pattern = "flat"

        return VolumeResult(
            mode="timeseries",
            volume_source="timeseries",
            spike_detected=spike_detected,
            spike_ratio=round(spike_ratio, 2) if spike_ratio is not None else None,
            baseline_avg=round(baseline_avg, 2),
            recent_volume=round(recent_volume, 2),
            hours_before_close=hours_to_close,
            pattern=pattern,
            note="Timeseries mode: baseline computed from all periods before last 24h.",
        )

    else:  # approximation
        if (
            package.volume_total_usd is None
            or package.volume_24h_usd is None
            or package.market_age_days is None
            or package.market_age_days <= 0
        ):
            return VolumeResult(
                mode="approximation",
                volume_source="proxy_total",
                spike_detected=False,
                spike_ratio=None,
                baseline_avg=None,
                recent_volume=None,
                hours_before_close=hours_to_close,
                pattern=None,
                note="Approximation mode: missing required fields.",
            )

        baseline_avg = package.volume_total_usd / package.market_age_days
        recent_volume = package.volume_24h_usd
        spike_ratio = recent_volume / baseline_avg if baseline_avg > 0 else None
        spike_detected = (
            spike_ratio is not None
            and spike_ratio >= params.spike_threshold_multiplier
        )

        return VolumeResult(
            mode="approximation",
            volume_source="proxy_total",
            spike_detected=spike_detected,
            spike_ratio=round(spike_ratio, 2) if spike_ratio is not None else None,
            baseline_avg=round(baseline_avg, 2),
            recent_volume=round(recent_volume, 2),
            hours_before_close=hours_to_close,
            pattern=None,  # cannot determine from single snapshot
            note=(
                "FALLBACK proxy_total: baseline = total_volume / market_age_days. "
                "In backtesting this reflects end-of-market totals, not volume at eval time. "
                "Pattern unavailable (single snapshot). Lower confidence."
            ),
        )


# ── momentum_analyzer ─────────────────────────────────────────────────────────

def momentum_analyzer(
    price_history: list[PricePoint],
    end_date: datetime,
    evaluation_date: datetime,
    params: AgentBParams,
) -> MomentumResult:
    """
    Fit linear regression over multiple look-back horizons.
    Uses numpy.polyfit only (no scipy).
    """
    by_horizon: list[MomentumHorizon] = []

    for horizon_hours in params.momentum_horizons_hours:
        horizon_start = evaluation_date - timedelta(hours=horizon_hours)
        window_points = _prices_in_window(price_history, horizon_start, evaluation_date)

        if len(window_points) < 2:
            continue

        # Convert to arrays: x = hours since horizon_start, y = price * 100 (pp)
        x = np.array([
            (p.timestamp - horizon_start).total_seconds() / 3600
            for p in window_points
        ])
        y = np.array([p.price * 100 for p in window_points])

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # pp per hour
        predicted = np.polyval(coeffs, x)

        # R-squared
        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Direction (only report if r² is sufficient)
        if r_squared >= params.min_r_squared:
            if slope >= params.min_slope_pp_per_hour:
                direction: Literal["UP", "DOWN", "FLAT"] = "UP"
            elif slope <= -params.min_slope_pp_per_hour:
                direction = "DOWN"
            else:
                direction = "FLAT"
        else:
            direction = "FLAT"  # insufficient fit quality

        by_horizon.append(MomentumHorizon(
            horizon_hours=horizon_hours,
            slope_pp_per_hour=round(float(slope), 4),
            direction=direction,
            r_squared=round(float(r_squared), 4),
            price_at_start=round(float(window_points[0].price), 4),
            price_at_end=round(float(window_points[-1].price), 4),
        ))

    # Aggregate across horizons
    valid = [h for h in by_horizon if h.r_squared >= params.min_r_squared]

    if len(valid) < 2:
        return MomentumResult(
            dominant_direction="FLAT",
            consistency="insufficient_data",
            acceleration="unknown",
            by_horizon=by_horizon,
        )

    directions = [h.direction for h in valid]
    up_count = directions.count("UP")
    down_count = directions.count("DOWN")

    if up_count > down_count:
        dominant: Literal["UP", "DOWN", "FLAT", "MIXED"] = "UP"
    elif down_count > up_count:
        dominant = "DOWN"
    elif up_count == down_count and up_count > 0:
        dominant = "MIXED"
    else:
        dominant = "FLAT"

    # Consistency
    unique_dirs = set(directions)
    if len(unique_dirs) == 1 and unique_dirs != {"FLAT"}:
        consistency: Literal["trending", "volatile", "reverting", "insufficient_data"] = "trending"
    elif len(unique_dirs) > 1:
        # Check reverting: short horizons point opposite to long horizons
        sorted_valid = sorted(valid, key=lambda h: h.horizon_hours)
        short_dir = sorted_valid[0].direction
        long_dir = sorted_valid[-1].direction
        if short_dir != long_dir and short_dir != "FLAT" and long_dir != "FLAT":
            consistency = "reverting"
        else:
            consistency = "volatile"
    else:
        consistency = "trending"  # all FLAT

    # Acceleration: compare slope of shortest vs longest valid horizon
    sorted_valid = sorted(valid, key=lambda h: h.horizon_hours)
    acceleration: Literal["increasing", "decreasing", "stable", "unknown"] = "unknown"
    if len(sorted_valid) >= 2:
        short_slope = abs(sorted_valid[0].slope_pp_per_hour)
        long_slope = abs(sorted_valid[-1].slope_pp_per_hour)
        if short_slope > long_slope * 1.2:
            acceleration = "increasing"
        elif long_slope > short_slope * 1.2:
            acceleration = "decreasing"
        else:
            acceleration = "stable"

    return MomentumResult(
        dominant_direction=dominant,
        consistency=consistency,
        acceleration=acceleration,
        by_horizon=by_horizon,
    )


# ── check_consistency ─────────────────────────────────────────────────────────

def check_consistency(
    price_result: PriceJumpResult,
    volume_result: VolumeResult,
    momentum_result: MomentumResult,
) -> ConsistencyCheck:
    """
    Cross-check whether price jump, volume, and momentum agree directionally.
    Pure Python — no LLM.
    """
    conflicting_signals: list[str] = []

    # price_and_momentum_agree
    price_dir = price_result.direction if price_result.detected else "NONE"
    momentum_dir = momentum_result.dominant_direction

    price_and_momentum_agree = (
        price_result.detected
        and momentum_dir not in ("FLAT", "MIXED", "INSUFFICIENT_DATA")
        and price_dir == momentum_dir
    )

    if (
        price_result.detected
        and momentum_dir not in ("FLAT", "MIXED")
        and price_dir != "NONE"
        and price_dir != momentum_dir
    ):
        conflicting_signals.append(
            f"Price direction ({price_dir}) conflicts with momentum ({momentum_dir})"
        )

    # volume_confirms_direction
    volume_confirms_direction = (
        price_result.detected
        and volume_result.spike_detected
        # volume spike + price jump in same overall direction
        and price_result.direction != "NONE"
    )

    # signals_contradictory: price and momentum both detected with opposite directions
    signals_contradictory = (
        price_result.detected
        and momentum_result.consistency not in ("insufficient_data",)
        and momentum_dir not in ("FLAT", "MIXED")
        and price_dir != "NONE"
        and price_dir != momentum_dir
    )

    # dominant direction from the available signals
    candidate_dirs = []
    if price_result.detected and price_result.direction != "NONE":
        candidate_dirs.append(price_result.direction)
    if momentum_dir not in ("FLAT", "MIXED"):
        candidate_dirs.append(momentum_dir)

    if not candidate_dirs:
        dominant: Literal["UP", "DOWN", "MIXED", "NONE"] = "NONE"
    elif all(d == candidate_dirs[0] for d in candidate_dirs):
        dominant = candidate_dirs[0]  # type: ignore[assignment]
    else:
        dominant = "MIXED"

    return ConsistencyCheck(
        price_and_momentum_agree=price_and_momentum_agree,
        volume_confirms_direction=volume_confirms_direction,
        signals_contradictory=signals_contradictory,
        dominant_direction=dominant,
        conflicting_signals=conflicting_signals,
    )
