"""
Agent B entry points: agent_b_initial() and agent_b_revise().

LLM config:
  - Model: claude-haiku-4-5-20251001 (primary)
  - temperature: 0 (mandatory for reproducibility)
  - structured output via .with_structured_output()
"""

import logging
from typing import Optional

from langchain.chat_models import init_chat_model

from src.ai_layer.agent_b.assessment import assess_inputs
from src.ai_layer.agent_b.params import AgentBParams
from src.ai_layer.agent_b.prompts import (
    AGENT_B_SYSTEM_PROMPT,
    build_agent_b_prompt,
    build_agent_b_revision_prompt,
)
from src.ai_layer.agent_b.schemas import (
    AgentBInputPackage,
    AgentBReport,
    AgentBRevisionResponse,
    ConsistencyCheck,
    MomentumResult,
    PriceJumpResult,
    VolumeResult,
)
from src.ai_layer.agent_b.tools import (
    check_consistency,
    momentum_analyzer,
    price_jump_detector,
    volume_spike_checker,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Fallback report for when the LLM or data fails entirely
_SKIP_REPORT_DEFAULTS = {
    "signal_direction": "SKIP",
    "behavior_score": 1,
    "confidence": "low",
    "key_findings": [],
    "reasoning": "Analysis failed — defaulting to SKIP.",
    "context_for_other_agents": "Agent B encountered an error. Treat as no signal.",
    "tools_run": [],
    "tools_skipped": [],
    "data_quality_notes": [],
}


def _make_llm(model: str = DEFAULT_MODEL):
    """Initialize LangChain LLM with structured output bound."""
    return init_chat_model(
        model, temperature=0, model_provider="anthropic"
    ).with_structured_output(AgentBReport)


def _make_revision_llm(model: str = DEFAULT_MODEL):
    return init_chat_model(
        model, temperature=0, model_provider="anthropic"
    ).with_structured_output(AgentBRevisionResponse)


def _empty_signal_breakdown(note: str = "Tool skipped — insufficient data."):
    from src.ai_layer.agent_b.schemas import SignalBreakdown
    return SignalBreakdown(
        detected=False,
        direction="NONE",
        magnitude="none",
        timing_quality="poor",
        sustained=False,
        weight_assigned="low",
        note=note,
    )


def _empty_consistency() -> ConsistencyCheck:
    return ConsistencyCheck(
        price_and_momentum_agree=False,
        volume_confirms_direction=False,
        signals_contradictory=False,
        dominant_direction="NONE",
        conflicting_signals=[],
    )


def agent_b_initial(
    package: AgentBInputPackage,
    params: AgentBParams,
    llm=None,
) -> AgentBReport:
    """
    Initial deterministic run. Execution order is fixed — never changes.

    1. assess_inputs()
    2. price_jump_detector (if can_run_price_jump)
    3. volume_spike_checker (if can_run_volume)
    4. momentum_analyzer (if can_run_momentum)
    5. check_consistency
    6. build_agent_b_prompt()
    7. Single LLM call → AgentBReport (structured output)
    8. Populate audit fields
    """
    if llm is None:
        llm = _make_llm()

    # 1. Input assessment
    assessment = assess_inputs(package, params)
    tools_run: list[str] = []
    tools_skipped: list[str] = list(assessment.skipped_tools)

    # 2. Price jump detector
    price_result: Optional[PriceJumpResult] = None
    if assessment.can_run_price_jump:
        try:
            price_result = price_jump_detector(
                package.price_history,
                package.end_date,
                package.evaluation_date,
                params,
            )
            tools_run.append("price_jump_detector")
        except Exception as e:
            logger.warning("price_jump_detector failed: %s", e)
            tools_skipped.append("price_jump_detector")

    # 3. Volume spike checker
    volume_result: Optional[VolumeResult] = None
    if assessment.can_run_volume:
        try:
            volume_result = volume_spike_checker(
                package, assessment.volume_mode, params
            )
            tools_run.append("volume_spike_checker")
        except Exception as e:
            logger.warning("volume_spike_checker failed: %s", e)
            tools_skipped.append("volume_spike_checker")

    # 4. Momentum analyzer
    momentum_result: Optional[MomentumResult] = None
    if assessment.can_run_momentum:
        try:
            momentum_result = momentum_analyzer(
                package.price_history,
                package.end_date,
                package.evaluation_date,
                params,
            )
            tools_run.append("momentum_analyzer")
        except Exception as e:
            logger.warning("momentum_analyzer failed: %s", e)
            tools_skipped.append("momentum_analyzer")

    # 5. Consistency check (only if at least one tool ran)
    consistency: Optional[ConsistencyCheck] = None
    if price_result is not None or momentum_result is not None:
        try:
            consistency = check_consistency(
                price_result or PriceJumpResult(
                    detected=False, largest_jump_pp=0.0, direction="NONE",
                    best_window_hours=0, from_price=0.0, to_price=0.0,
                    hours_before_close=assessment.hours_to_close,
                    is_sustained=False, move_shape="none", all_windows=[],
                ),
                volume_result or VolumeResult(
                    mode="unavailable", spike_detected=False, spike_ratio=None,
                    baseline_avg=None, recent_volume=None,
                    hours_before_close=assessment.hours_to_close,
                    pattern=None, note="Volume unavailable.",
                ),
                momentum_result or MomentumResult(
                    dominant_direction="FLAT",
                    consistency="insufficient_data",
                    acceleration="unknown",
                    by_horizon=[],
                ),
            )
            tools_run.append("check_consistency")
        except Exception as e:
            logger.warning("check_consistency failed: %s", e)

    # 6–7. Build prompt and call LLM
    user_prompt = build_agent_b_prompt(
        assessment, price_result, volume_result, momentum_result, consistency, package
    )

    try:
        report: AgentBReport = llm.invoke([
            {"role": "system", "content": AGENT_B_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.error("Agent B LLM call failed: %s", e)
        # Return a safe default rather than crashing the pipeline
        return AgentBReport(
            **_SKIP_REPORT_DEFAULTS,
            price_jump_assessment=_empty_signal_breakdown("LLM call failed."),
            volume_assessment=_empty_signal_breakdown("LLM call failed."),
            momentum_assessment=_empty_signal_breakdown("LLM call failed."),
            consistency=_empty_consistency(),
            evaluation_date=package.evaluation_date.isoformat(),
            tools_run=tools_run,
            tools_skipped=tools_skipped,
            data_quality_notes=assessment.data_quality_notes,
        )

    # 8. Patch audit fields (LLM may not fill these correctly)
    report.evaluation_date = package.evaluation_date.isoformat()
    report.tools_run = tools_run
    report.tools_skipped = tools_skipped
    report.data_quality_notes = assessment.data_quality_notes

    logger.debug(
        "Agent B initial: market=%s score=%d direction=%s confidence=%s",
        package.evaluation_date.isoformat(),
        report.behavior_score,
        report.signal_direction,
        report.confidence,
    )
    return report


def agent_b_revise(
    original_report: AgentBReport,
    revision_feedback: str,
    package: AgentBInputPackage,
    params: AgentBParams,
    llm=None,
) -> AgentBRevisionResponse:
    """
    Called when Revision Agent sends targeted feedback to Agent B.
    Single LLM call — the revision loop is managed externally (max 5 iterations).

    Tool re-runs (if needed by LLM reasoning) happen within this function
    in Python — no second LLM call.
    """
    if llm is None:
        llm = _make_revision_llm()

    # Re-run tools so LLM can re-examine with original inputs
    assessment = assess_inputs(package, params)

    price_result: Optional[PriceJumpResult] = None
    if assessment.can_run_price_jump:
        try:
            price_result = price_jump_detector(
                package.price_history, package.end_date,
                package.evaluation_date, params,
            )
        except Exception:
            pass

    volume_result: Optional[VolumeResult] = None
    if assessment.can_run_volume:
        try:
            volume_result = volume_spike_checker(
                package, assessment.volume_mode, params
            )
        except Exception:
            pass

    momentum_result: Optional[MomentumResult] = None
    if assessment.can_run_momentum:
        try:
            momentum_result = momentum_analyzer(
                package.price_history, package.end_date,
                package.evaluation_date, params,
            )
        except Exception:
            pass

    user_prompt = build_agent_b_revision_prompt(
        original_report,
        revision_feedback,
        assessment,
        price_result,
        volume_result,
        momentum_result,
        package,
    )

    try:
        response: AgentBRevisionResponse = llm.invoke([
            {"role": "system", "content": AGENT_B_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.error("Agent B revision LLM call failed: %s", e)
        return AgentBRevisionResponse(
            tools_re_run=[],
            parameter_changes={},
            finding_changed=False,
            updated_signal_direction=original_report.signal_direction,
            updated_behavior_score=original_report.behavior_score,
            updated_confidence=original_report.confidence,
            delta_explanation="LLM call failed — original report unchanged.",
            final_reasoning=original_report.reasoning,
            context_for_other_agents=original_report.context_for_other_agents,
        )

    logger.debug(
        "Agent B revision: finding_changed=%s new_score=%d",
        response.finding_changed,
        response.updated_behavior_score,
    )
    return response
