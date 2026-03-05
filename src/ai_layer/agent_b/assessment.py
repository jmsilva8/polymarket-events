"""Input assessment — runs before any tool or LLM call."""

from src.ai_layer.agent_b.params import AgentBParams
from src.ai_layer.agent_b.schemas import AgentBInputPackage, InputAssessment


def assess_inputs(
    package: AgentBInputPackage,
    params: AgentBParams,
) -> InputAssessment:
    """
    Inspect the input package and determine which tools can run
    and in what mode. Runs in pure Python before any tool or LLM.
    """
    notes: list[str] = []
    skipped: list[str] = []

    price_ok = len(package.price_history) >= params.min_price_points
    if not price_ok:
        notes.append(
            f"Only {len(package.price_history)} price points available "
            f"(minimum: {params.min_price_points}). "
            f"Price jump and momentum tools skipped."
        )
        skipped.extend(["price_jump_detector", "momentum_analyzer"])
    elif params.data_frequency_hours >= 12:
        notes.append(
            f"Sparse data: {len(package.price_history)} points at ~{params.data_frequency_hours}h intervals. "
            f"Jump detection is the primary signal — momentum regression has limited reliability."
        )

    if package.volume_history and len(package.volume_history) >= 7:
        volume_mode = "timeseries"
    elif (
        package.volume_24h_usd is not None
        and package.volume_total_usd is not None
        and package.market_age_days is not None
    ):
        volume_mode = "approximation"
        notes.append(
            "Volume: approximation mode (no per-period breakdown). "
            "Volume signal carries lower confidence."
        )
    else:
        volume_mode = "unavailable"
        skipped.append("volume_spike_checker")
        notes.append("Volume data unavailable. Volume signal omitted.")

    hours_to_close = (
        package.end_date - package.evaluation_date
    ).total_seconds() / 3600

    return InputAssessment(
        can_run_price_jump=price_ok,
        can_run_momentum=price_ok,
        can_run_volume=volume_mode != "unavailable",
        volume_mode=volume_mode,
        price_point_count=len(package.price_history),
        hours_to_close=hours_to_close,
        skipped_tools=skipped,
        data_quality_notes=notes,
    )
