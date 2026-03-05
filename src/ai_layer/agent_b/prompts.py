"""LLM prompts for Agent B — initial run and revision."""

import json
from typing import Optional

from src.ai_layer.agent_b.schemas import (
    AgentBInputPackage,
    AgentBReport,
    InputAssessment,
    MomentumResult,
    PriceJumpResult,
    VolumeResult,
)

AGENT_B_SYSTEM_PROMPT = """
You are a quantitative signal analyst for a prediction market trading system.

You receive structured outputs from mathematical tools that analyzed the price
timeseries of a binary prediction market (YES/NO outcome).

CRITICAL CONSTRAINTS:
- You do not know what the market is about. Do not speculate about it.
- Do not use any external knowledge about events, people, or outcomes.
- Reason exclusively from the numbers provided.
- If you reference anything not in the tool outputs, your response is invalid.

VOLUME DATA NOTE:
Volume data is frequently unavailable for historical markets.
If volume tools were skipped due to missing data: DO NOT reduce the behavior_score.
Only reduce confidence (to "medium" or "low") when working price-only.
A strong price/momentum signal without volume confirmation is still a valid signal —
volume absence is a data gap, not evidence of no signal.

When volume_source="timeseries": baseline is real per-period data — treat as high confidence.
When volume_source="proxy_total": baseline = total_volume / market_age_days, which in
backtesting reflects end-of-market totals rather than volume at eval time. Treat the
volume signal as directional context only — do not treat spike_ratio as a precise measure,
and weight it lower than timeseries evidence.

YOUR TASK:
1. Assess the magnitude and quality of each signal independently
2. Evaluate whether signals are consistent with each other
3. Determine the directional recommendation: YES, NO, or SKIP
4. Assign a behavior_score from 1 to 10 (integer only)
5. Explain your reasoning clearly for downstream agents that will challenge it

INVESTMENT ALTERNATIVES:
- YES:  evidence suggests price will converge toward 1.0 — buy YES
- NO:   evidence suggests price will converge toward 0.0 — buy NO
- SKIP: signals absent, contradictory, or insufficient to act on

SCORING GUIDE (behavior_score 1–10):
1–2:  No meaningful signals. Price flat, momentum absent.
3–4:  Weak. Minor movement within normal variance. Low confidence.
5–6:  Moderate. One meaningful signal present, not confirmed by others.
7–8:  Strong. Two or more signals agree in direction. Sustained movement.
9–10: Very strong. All available signals converge, large magnitude, late timing, sustained.
      Reserve for exceptional cases only — do not inflate.

MANDATORY RULES:
- Volume spike with no directional price confirmation → SKIP
- Price jump that is NOT sustained (is_sustained=false) → reduce score by at least 2
- Contradictory signals (price UP, momentum DOWN) → SKIP regardless of magnitude
- Signals far from close (>72h) carry less weight than signals within 24h of close
- If volume tools were skipped (missing data): reduce confidence, NOT score
- Always report move_shape in your assessment — it is relevant to downstream agents
- Price-only analysis (no volume) is valid — do not artificially cap score for it

Respond in the exact JSON schema provided.
"""


def build_agent_b_prompt(
    assessment: InputAssessment,
    price_result: Optional[PriceJumpResult],
    volume_result: Optional[VolumeResult],
    momentum_result: Optional[MomentumResult],
    consistency,  # Optional[ConsistencyCheck]
    package: AgentBInputPackage,
) -> str:
    """
    Assemble the user message for the Agent B LLM call.

    Includes only numbers — no market metadata, no text about what the bet is.
    Agent B must remain blind to market content.
    """
    sections = []

    sections.append(
        f"MARKET CONTEXT (numbers only):\n"
        f"  Current price:     {package.current_price:.4f}\n"
        f"  Hours to close:    {assessment.hours_to_close:.1f}\n"
        f"  Price data points: {assessment.price_point_count}"
    )

    if assessment.data_quality_notes:
        sections.append(
            "DATA QUALITY NOTES:\n"
            + "\n".join(f"  - {n}" for n in assessment.data_quality_notes)
        )

    if price_result:
        sections.append(
            "PRICE JUMP TOOL OUTPUT:\n"
            + json.dumps(price_result.model_dump(), indent=2)
        )

    if volume_result:
        sections.append(
            "VOLUME TOOL OUTPUT:\n"
            + json.dumps(volume_result.model_dump(), indent=2)
        )

    if momentum_result:
        sections.append(
            "MOMENTUM TOOL OUTPUT:\n"
            + json.dumps(momentum_result.model_dump(), indent=2)
        )

    if consistency:
        sections.append(
            "CONSISTENCY CHECK:\n"
            + json.dumps(consistency.model_dump(), indent=2)
        )

    if assessment.skipped_tools:
        sections.append(
            "TOOLS SKIPPED (insufficient data):\n"
            + "\n".join(f"  - {t}" for t in assessment.skipped_tools)
        )

    return "\n\n".join(sections)


def build_agent_b_revision_prompt(
    original_report: AgentBReport,
    revision_feedback: str,
    assessment: InputAssessment,
    price_result: Optional[PriceJumpResult],
    volume_result: Optional[VolumeResult],
    momentum_result: Optional[MomentumResult],
    package: AgentBInputPackage,
) -> str:
    """
    Prompt for Agent B when responding to Revision Agent feedback.

    Includes original report, feedback verbatim, and all original tool outputs
    so the LLM can re-examine them if needed.
    """
    original_section = build_agent_b_prompt(
        assessment, price_result, volume_result, None, None, package
    )

    return (
        f"YOUR PREVIOUS REPORT:\n"
        f"{json.dumps(original_report.model_dump(), indent=2)}\n\n"
        f"REVISION AGENT FEEDBACK:\n"
        f"{revision_feedback}\n\n"
        f"ORIGINAL TOOL OUTPUTS (for re-examination):\n"
        f"{original_section}\n\n"
        f"INSTRUCTION:\n"
        f"If the feedback identifies a specific issue you can address by re-examining\n"
        f"the tool outputs above with different parameters, do so and explain:\n"
        f"  - Which tool you re-examined\n"
        f"  - What parameter you changed and why\n"
        f"  - Whether your conclusion changed\n"
        f"  - Updated score, direction, and confidence if changed\n\n"
        f"If your original conclusion holds under scrutiny, explain why the\n"
        f"original reasoning is correct despite the feedback.\n\n"
        f"Do not speculate about the market content. Reason only from numbers.\n"
    )
