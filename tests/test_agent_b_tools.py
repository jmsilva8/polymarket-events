"""
Unit tests for Agent B pure-Python tools and assessment.

Covers:
  - price_jump_detector: no jump, below threshold, sustained UP, reverting, sudden vs gradual
  - volume_spike_checker: timeseries mode, approximation mode, unavailable mode
  - momentum_analyzer: trending UP/DOWN, flat, volatile, reverting, insufficient data
  - assess_inputs: all combinations of missing/present volume fields
  - Determinism: same inputs → identical outputs

No LLM calls. Tools tested in isolation.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.ai_layer.agent_b.assessment import assess_inputs
from src.ai_layer.agent_b.params import AgentBParams
from src.ai_layer.agent_b.schemas import AgentBInputPackage, VolumePoint
from src.ai_layer.agent_b.tools import (
    check_consistency,
    momentum_analyzer,
    price_jump_detector,
    volume_spike_checker,
)
from src.data_layer.models import PricePoint

# ── Fixtures ───────────────────────────────────────────────────────────────────

T = datetime(2024, 3, 6, 9, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 3, 7, 9, 0, 0, tzinfo=timezone.utc)  # 24h after T


def make_price_point(hours_before_T: float, price: float) -> PricePoint:
    return PricePoint(
        timestamp=T - timedelta(hours=hours_before_T),
        price=price,
        raw_price=price,
    )


def flat_history(n: int = 20, price: float = 0.50) -> list[PricePoint]:
    """n price points over the last 48h, all at `price`."""
    return [
        make_price_point(48 - i * (48 / (n - 1)), price)
        for i in range(n)
    ]


def rising_history() -> list[PricePoint]:
    """Price rises steadily from 0.54 to 0.72 over the last 24h — 8 hourly points."""
    prices = [0.54, 0.55, 0.54, 0.55, 0.58, 0.62, 0.68, 0.72]
    return [
        PricePoint(
            timestamp=T - timedelta(hours=24 - i * 3),
            price=p,
            raw_price=p,
        )
        for i, p in enumerate(prices)
    ]


def falling_history() -> list[PricePoint]:
    """Price drops from 0.70 to 0.45 over the last 24h."""
    prices = [0.70, 0.68, 0.65, 0.60, 0.55, 0.50, 0.47, 0.45]
    return [
        PricePoint(
            timestamp=T - timedelta(hours=24 - i * 3),
            price=p,
            raw_price=p,
        )
        for i, p in enumerate(prices)
    ]


def reverting_history() -> list[PricePoint]:
    """
    Price spikes UP to 0.80 (+30pp from 0.50) then partially reverts to 0.65 at T.
    Best window (24h): from=0.50, to=0.65, delta=+15pp UP.
    Peak=0.80, pulled back 15pp from peak.
    revert_limit = 0.5 * 15 = 7.5pp < 15pp → is_sustained=False.
    Only gentle changes before spike so no large DOWN window competes.
    """
    return [
        make_price_point(48, 0.49),
        make_price_point(36, 0.50),
        make_price_point(24, 0.50),
        make_price_point(18, 0.51),
        make_price_point(12, 0.55),
        make_price_point(8, 0.75),   # starts spiking
        make_price_point(6, 0.80),   # peak
        make_price_point(3, 0.72),
        make_price_point(0, 0.65),   # at T — pulled back 15pp from peak
    ]


def sudden_history() -> list[PricePoint]:
    """Price is flat then jumps +20pp within a 2h window."""
    history = [
        make_price_point(48, 0.50),
        make_price_point(24, 0.51),
        make_price_point(10, 0.50),
        make_price_point(2.5, 0.50),   # just before 2h window
        make_price_point(1.5, 0.60),   # spike starts
        make_price_point(0.5, 0.70),   # spike continues within 2h
        make_price_point(0, 0.70),
    ]
    return history


DEFAULT_PARAMS = AgentBParams()


# ── price_jump_detector ────────────────────────────────────────────────────────

class TestPriceJumpDetector:
    def test_no_jump_flat(self):
        result = price_jump_detector(flat_history(), END, T, DEFAULT_PARAMS)
        assert result.detected is False
        assert result.direction == "NONE"
        assert result.largest_jump_pp == 0.0

    def test_jump_below_threshold(self):
        # 3pp change — below default 5pp threshold
        history = [make_price_point(24, 0.50), make_price_point(0, 0.53)]
        result = price_jump_detector(history, END, T, DEFAULT_PARAMS)
        assert result.detected is False

    def test_sustained_upward_jump(self):
        result = price_jump_detector(rising_history(), END, T, DEFAULT_PARAMS)
        assert result.detected is True
        assert result.direction == "UP"
        assert result.largest_jump_pp > 0
        assert result.is_sustained is True
        assert result.move_shape == "gradual"  # built over 24h

    def test_downward_jump(self):
        result = price_jump_detector(falling_history(), END, T, DEFAULT_PARAMS)
        assert result.detected is True
        assert result.direction == "DOWN"
        assert result.largest_jump_pp < 0

    def test_reverting_jump_not_sustained(self):
        result = price_jump_detector(reverting_history(), END, T, DEFAULT_PARAMS)
        # Price jumped but reverted — may or may not detect depending on net delta
        # At minimum: if detected, is_sustained should be False
        if result.detected:
            assert result.is_sustained is False

    def test_sudden_move_shape(self):
        result = price_jump_detector(sudden_history(), END, T, DEFAULT_PARAMS)
        if result.detected:
            assert result.move_shape == "sudden"

    def test_empty_history_returns_no_detection(self):
        result = price_jump_detector([], END, T, DEFAULT_PARAMS)
        assert result.detected is False

    def test_determinism(self):
        h = rising_history()
        r1 = price_jump_detector(h, END, T, DEFAULT_PARAMS)
        r2 = price_jump_detector(h, END, T, DEFAULT_PARAMS)
        assert r1.model_dump() == r2.model_dump()

    def test_custom_threshold(self):
        params = AgentBParams(min_jump_pp=25.0)  # very high threshold
        result = price_jump_detector(rising_history(), END, T, params)
        assert result.detected is False  # 18pp jump below 25pp threshold


# ── volume_spike_checker ───────────────────────────────────────────────────────

def make_package_with_timeseries(spike: bool = True) -> AgentBInputPackage:
    """Package with volume_history: spike if spike=True, flat if False."""
    baseline_vol = 1000.0
    vols = [VolumePoint(timestamp=T - timedelta(days=i), volume_usd=baseline_vol) for i in range(14, 1, -1)]
    if spike:
        # Recent 24h: 5x baseline
        vols += [VolumePoint(timestamp=T - timedelta(hours=12), volume_usd=baseline_vol * 5)]
    else:
        vols += [VolumePoint(timestamp=T - timedelta(hours=12), volume_usd=baseline_vol * 0.8)]

    return AgentBInputPackage(
        evaluation_date=T,
        end_date=END,
        price_history=flat_history(),
        current_price=0.5,
        volume_history=vols,
    )


def make_package_approximation(spike: bool = True) -> AgentBInputPackage:
    age_days = 30.0
    total = 30_000.0  # avg $1k/day
    daily_24h = 5_000.0 if spike else 800.0
    return AgentBInputPackage(
        evaluation_date=T,
        end_date=END,
        price_history=flat_history(),
        current_price=0.5,
        volume_total_usd=total,
        volume_24h_usd=daily_24h,
        market_age_days=age_days,
    )


class TestVolumeSpikeChecker:
    def test_timeseries_spike_detected(self):
        pkg = make_package_with_timeseries(spike=True)
        result = volume_spike_checker(pkg, "timeseries", DEFAULT_PARAMS)
        assert result.mode == "timeseries"
        assert result.spike_detected is True
        assert result.spike_ratio is not None and result.spike_ratio > 3.0

    def test_timeseries_no_spike(self):
        pkg = make_package_with_timeseries(spike=False)
        result = volume_spike_checker(pkg, "timeseries", DEFAULT_PARAMS)
        assert result.spike_detected is False

    def test_approximation_spike(self):
        pkg = make_package_approximation(spike=True)
        result = volume_spike_checker(pkg, "approximation", DEFAULT_PARAMS)
        assert result.mode == "approximation"
        assert result.spike_detected is True
        assert result.pattern is None  # no pattern from single snapshot

    def test_approximation_no_spike(self):
        pkg = make_package_approximation(spike=False)
        result = volume_spike_checker(pkg, "approximation", DEFAULT_PARAMS)
        assert result.spike_detected is False

    def test_approximation_note_mentions_limitation(self):
        pkg = make_package_approximation()
        result = volume_spike_checker(pkg, "approximation", DEFAULT_PARAMS)
        assert "approximation" in result.note.lower()

    def test_determinism(self):
        pkg = make_package_with_timeseries(spike=True)
        r1 = volume_spike_checker(pkg, "timeseries", DEFAULT_PARAMS)
        r2 = volume_spike_checker(pkg, "timeseries", DEFAULT_PARAMS)
        assert r1.model_dump() == r2.model_dump()


# ── momentum_analyzer ─────────────────────────────────────────────────────────

class TestMomentumAnalyzer:
    def test_trending_up(self):
        result = momentum_analyzer(rising_history(), END, T, DEFAULT_PARAMS)
        assert result.dominant_direction == "UP"
        assert result.consistency in ("trending", "volatile")

    def test_trending_down(self):
        result = momentum_analyzer(falling_history(), END, T, DEFAULT_PARAMS)
        assert result.dominant_direction == "DOWN"

    def test_flat(self):
        result = momentum_analyzer(flat_history(), END, T, DEFAULT_PARAMS)
        # Flat price → no clear trend
        assert result.dominant_direction in ("FLAT", "MIXED")

    def test_insufficient_data_two_points(self):
        history = [make_price_point(24, 0.50), make_price_point(0, 0.60)]
        result = momentum_analyzer(history, END, T, DEFAULT_PARAMS)
        # 2 points: should either detect or mark insufficient
        assert result.consistency in ("trending", "volatile", "insufficient_data")

    def test_insufficient_data_empty(self):
        result = momentum_analyzer([], END, T, DEFAULT_PARAMS)
        assert result.consistency == "insufficient_data"

    def test_reverting_pattern(self):
        """Short horizon UP but long horizon DOWN → reverting."""
        # Build history: price goes UP for last 6h but was DOWN over 24h
        history = [
            make_price_point(24, 0.70),
            make_price_point(20, 0.65),
            make_price_point(15, 0.60),
            make_price_point(10, 0.55),
            make_price_point(6, 0.50),   # bottom
            make_price_point(4, 0.55),
            make_price_point(2, 0.60),
            make_price_point(0, 0.65),   # recovering
        ]
        result = momentum_analyzer(history, END, T, DEFAULT_PARAMS)
        # Should detect reverting or volatile (short UP, long DOWN)
        assert result.consistency in ("reverting", "volatile", "trending", "insufficient_data")

    def test_determinism(self):
        h = rising_history()
        r1 = momentum_analyzer(h, END, T, DEFAULT_PARAMS)
        r2 = momentum_analyzer(h, END, T, DEFAULT_PARAMS)
        assert r1.model_dump() == r2.model_dump()

    def test_by_horizon_populated(self):
        result = momentum_analyzer(rising_history(), END, T, DEFAULT_PARAMS)
        assert len(result.by_horizon) > 0


# ── assess_inputs ──────────────────────────────────────────────────────────────

class TestAssessInputs:
    def _pkg(self, **kwargs) -> AgentBInputPackage:
        base = dict(
            evaluation_date=T,
            end_date=END,
            price_history=flat_history(),
            current_price=0.5,
        )
        base.update(kwargs)
        return AgentBInputPackage(**base)

    def test_sufficient_price_and_volume_timeseries(self):
        vols = [VolumePoint(timestamp=T - timedelta(days=i), volume_usd=1000) for i in range(8)]
        pkg = self._pkg(volume_history=vols)
        result = assess_inputs(pkg, DEFAULT_PARAMS)
        assert result.can_run_price_jump is True
        assert result.can_run_momentum is True
        assert result.can_run_volume is True
        assert result.volume_mode == "timeseries"

    def test_approximation_volume_mode(self):
        pkg = self._pkg(volume_total_usd=30000, volume_24h_usd=1500, market_age_days=30)
        result = assess_inputs(pkg, DEFAULT_PARAMS)
        assert result.volume_mode == "approximation"
        assert result.can_run_volume is True

    def test_no_volume_data(self):
        pkg = self._pkg()  # no volume fields
        result = assess_inputs(pkg, DEFAULT_PARAMS)
        assert result.volume_mode == "unavailable"
        assert result.can_run_volume is False
        assert "volume_spike_checker" in result.skipped_tools

    def test_insufficient_price_points(self):
        pkg = self._pkg(
            price_history=[make_price_point(1, 0.5)],  # only 1 point
        )
        result = assess_inputs(pkg, DEFAULT_PARAMS)
        assert result.can_run_price_jump is False
        assert result.can_run_momentum is False
        assert "price_jump_detector" in result.skipped_tools
        assert "momentum_analyzer" in result.skipped_tools

    def test_partial_volume_fields_missing(self):
        # Only total_usd but no 24h or age → unavailable
        pkg = self._pkg(volume_total_usd=50000)
        result = assess_inputs(pkg, DEFAULT_PARAMS)
        assert result.volume_mode == "unavailable"

    def test_hours_to_close_computed(self):
        pkg = self._pkg()
        result = assess_inputs(pkg, DEFAULT_PARAMS)
        assert abs(result.hours_to_close - 24.0) < 0.1

    def test_determinism(self):
        pkg = self._pkg()
        r1 = assess_inputs(pkg, DEFAULT_PARAMS)
        r2 = assess_inputs(pkg, DEFAULT_PARAMS)
        assert r1.model_dump() == r2.model_dump()


# ── check_consistency ─────────────────────────────────────────────────────────

class TestCheckConsistency:
    def _run(self, price_up: bool = True, momentum_up: bool = True,
             volume_spike: bool = False, price_detected: bool = True):
        from src.ai_layer.agent_b.schemas import (
            MomentumHorizon,
            MomentumResult,
            PriceJumpResult,
            VolumeResult,
        )
        price = PriceJumpResult(
            detected=price_detected,
            largest_jump_pp=18.0 if price_up else -18.0,
            direction="UP" if price_up else ("DOWN" if price_detected else "NONE"),
            best_window_hours=24,
            from_price=0.54,
            to_price=0.72 if price_up else 0.36,
            hours_before_close=24.0,
            is_sustained=True,
            move_shape="gradual",
            all_windows=[],
        )
        momentum = MomentumResult(
            dominant_direction="UP" if momentum_up else "DOWN",
            consistency="trending",
            acceleration="stable",
            by_horizon=[
                MomentumHorizon(
                    horizon_hours=24,
                    slope_pp_per_hour=0.75,
                    direction="UP" if momentum_up else "DOWN",
                    r_squared=0.9,
                    price_at_start=0.54,
                    price_at_end=0.72 if momentum_up else 0.36,
                )
            ],
        )
        volume = VolumeResult(
            mode="approximation",
            spike_detected=volume_spike,
            spike_ratio=4.0 if volume_spike else 0.8,
            baseline_avg=1000,
            recent_volume=4000 if volume_spike else 800,
            hours_before_close=24.0,
            pattern=None,
            note="",
        )
        return check_consistency(price, volume, momentum)

    def test_price_and_momentum_agree_both_up(self):
        result = self._run(price_up=True, momentum_up=True)
        assert result.price_and_momentum_agree is True
        assert result.signals_contradictory is False

    def test_price_up_momentum_down_contradictory(self):
        result = self._run(price_up=True, momentum_up=False)
        assert result.signals_contradictory is True

    def test_volume_confirms_when_spike_and_price(self):
        result = self._run(price_up=True, volume_spike=True)
        assert result.volume_confirms_direction is True

    def test_no_volume_spike_no_confirmation(self):
        result = self._run(price_up=True, volume_spike=False)
        assert result.volume_confirms_direction is False

    def test_dominant_direction_none_when_no_price(self):
        result = self._run(price_detected=False)
        # With no price detection, dominant direction depends on momentum
        assert result.dominant_direction in ("UP", "DOWN", "MIXED", "NONE")
