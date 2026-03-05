"""
Decision Agent — main entry point.

Flow:
1. Validate input
2. Early exit for SKIP/WATCH (Revision Agent already decided)
3. Single LLM call with structured reasoning
4. Parse key decisions from free-text LLM response
5. Return DecisionAgentOutput
"""

import logging
import re

from langchain.chat_models import init_chat_model

from src.ai_layer.decision_agent.params import DecisionAgentParams
from src.ai_layer.decision_agent.prompts import (
    DECISION_AGENT_SYSTEM_PROMPT,
    build_decision_agent_prompt,
)
from src.ai_layer.decision_agent.schemas import (
    DecisionAgentInputPackage,
    DecisionAgentOutput,
)

logger = logging.getLogger(__name__)


def decision_agent(
    package: DecisionAgentInputPackage,
    params: DecisionAgentParams,
) -> DecisionAgentOutput:
    """Run the Decision Agent on one market's full signal package."""

    # 1. Basic validation
    if not (0.0 <= package.current_market_price <= 1.0):
        logger.warning(
            "Invalid market price %s for %s — skipping",
            package.current_market_price, package.market_id,
        )
        return DecisionAgentOutput(
            decision="SKIP",
            bet_direction="null",
            full_reasoning="Invalid market price. Defaulting to SKIP.",
            revision_flag_applied="VALIDATION_ERROR",
            market_id=package.market_id,
            evaluation_date=package.evaluation_date.isoformat(),
            recommendation={
                "action": "PASS",
                "bet": None,
                "risk_grade": 0,
                "current_price": package.current_market_price,
                "reasoning_summary": "Invalid market price.",
            },
        )

    # 2. Honour Revision Agent's hard decisions immediately
    if package.recommendation_to_decision_agent in ("SKIP", "WATCH"):
        action = (
            "WATCH"
            if package.recommendation_to_decision_agent == "WATCH"
            else "PASS"
        )
        return DecisionAgentOutput(
            decision="SKIP",
            bet_direction="null",
            full_reasoning=(
                f"Revision Agent recommendation: {package.recommendation_to_decision_agent}. "
                f"Reason: {package.flag_explanation}"
            ),
            revision_flag_applied=package.revision_flag,
            market_id=package.market_id,
            evaluation_date=package.evaluation_date.isoformat(),
            recommendation={
                "action": action,
                "bet": None,
                "risk_grade": 0,
                "current_price": package.current_market_price,
                "reasoning_summary": package.flag_explanation,
            },
        )

    # 3. LLM call for GO_EVALUATE cases
    llm = init_chat_model(
        params.model_name,
        temperature=params.temperature,
        model_provider="anthropic",
    )
    user_prompt = build_decision_agent_prompt(package)

    try:
        response = llm.invoke([
            {"role": "system", "content": DECISION_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        reasoning_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error("Decision Agent LLM call failed for %s: %s", package.market_id, e)
        return DecisionAgentOutput(
            decision="SKIP",
            bet_direction="null",
            full_reasoning=f"LLM call failed: {e}",
            revision_flag_applied=package.revision_flag,
            market_id=package.market_id,
            evaluation_date=package.evaluation_date.isoformat(),
            recommendation={
                "action": "PASS",
                "bet": None,
                "risk_grade": 0,
                "current_price": package.current_market_price,
                "reasoning_summary": "LLM error — defaulting to SKIP.",
            },
        )

    # 4. Extract structured decisions from free-text reasoning
    parsed = _extract_decision(reasoning_text, package.current_market_price)

    a_score = package.agent_a_report.get("insider_risk_score", 5)
    b_score = package.agent_b_report.get("behavior_score", 5)

    return DecisionAgentOutput(
        decision=parsed["decision"],
        bet_direction=parsed["bet_direction"],
        analysis={
            "agent_a_score": a_score,
            "agent_b_score": b_score,
            "weight_a_percentage": parsed.get("weight_a"),
            "weight_b_percentage": parsed.get("weight_b"),
            "weighting_rationale": parsed.get("weighting_rationale", ""),
            "weighted_score": parsed.get("weighted_score"),
            "current_market_price": package.current_market_price,
            "adjusted_probability_of_win": parsed.get("adjusted_probability"),
            "estimated_edge_pp": parsed.get("edge_pp"),
            "edge_assessment": parsed.get("edge_assessment", ""),
        },
        full_reasoning=reasoning_text,
        revision_flag_applied=package.revision_flag,
        market_id=package.market_id,
        evaluation_date=package.evaluation_date.isoformat(),
        recommendation={
            "action": "INVEST" if parsed["decision"] == "GO" else "PASS",
            "bet": parsed["bet_direction"] if parsed["decision"] == "GO" else None,
            "risk_grade": int(round(parsed.get("weighted_score") or 5)),
            "current_price": package.current_market_price,
            "reasoning_summary": _summarize(parsed),
        },
    )


def _extract_decision(llm_response: str, current_price: float) -> dict:
    """
    Extract structured data from the LLM's free-text reasoning.
    Looks for key patterns in the output.
    """
    result = {
        "decision": "SKIP",
        "bet_direction": "null",
        "weight_a": 50,
        "weight_b": 50,
        "weighting_rationale": "",
        "weighted_score": 5.0,
        "adjusted_probability": current_price,
        "edge_pp": 0.0,
        "edge_assessment": "not meaningful",
    }

    text = llm_response

    # Decision: look for explicit GO signal
    if re.search(r'\bdecision\s*[=:]\s*["\']?GO["\']?', text, re.IGNORECASE):
        result["decision"] = "GO"
    elif re.search(r'\bfinal decision\b.{0,30}\bGO\b', text, re.IGNORECASE):
        result["decision"] = "GO"
    elif re.search(r'\bRECOMMEND\s+(?:INVEST|GO)\b', text, re.IGNORECASE):
        result["decision"] = "GO"

    # Bet direction — only meaningful if GO
    if result["decision"] == "GO":
        yes_pos = text.upper().find("BET_DIRECTION: YES") if "BET_DIRECTION" in text.upper() else -1
        no_pos = text.upper().find("BET_DIRECTION: NO") if "BET_DIRECTION" in text.upper() else -1
        if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
            result["bet_direction"] = "YES"
        elif no_pos >= 0:
            result["bet_direction"] = "NO"
        else:
            # Fall back: look for "bet on YES / bet on NO"
            if re.search(r'\bbet\s+(?:direction[:\s]+)?YES\b', text, re.IGNORECASE):
                result["bet_direction"] = "YES"
            elif re.search(r'\bbet\s+(?:direction[:\s]+)?NO\b', text, re.IGNORECASE):
                result["bet_direction"] = "NO"
            elif re.search(r'\bbuy YES\b', text, re.IGNORECASE):
                result["bet_direction"] = "YES"
            elif re.search(r'\bbuy NO\b', text, re.IGNORECASE):
                result["bet_direction"] = "NO"

    # Weights
    m = re.search(r'[Ww]eight\s+[Aa]gent\s+A\s+(?:at\s+)?(\d+)\s*%', text)
    if m:
        result["weight_a"] = int(m.group(1))
        result["weight_b"] = 100 - result["weight_a"]

    m = re.search(r'[Ww]eight\s+[Aa]gent\s+B\s+(?:at\s+)?(\d+)\s*%', text)
    if m:
        result["weight_b"] = int(m.group(1))
        if "weight_a" not in result or result["weight_a"] == 50:
            result["weight_a"] = 100 - result["weight_b"]

    # Weighted score
    m = re.search(r'[Ww]eighted\s+score[:\s=]+([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        result["weighted_score"] = float(m.group(1))

    # Adjusted probability
    m = re.search(r'[Aa]djusted\s+probabilit[yi][:\s=~]+([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        val = float(m.group(1))
        # Normalise: could be expressed as 0.75 or 75
        result["adjusted_probability"] = val if val <= 1.0 else val / 100.0

    result["edge_pp"] = abs(result["adjusted_probability"] - current_price) * 100

    # Edge assessment
    if re.search(r'\bnot meaningful\b', text, re.IGNORECASE):
        result["edge_assessment"] = "not meaningful"
    elif re.search(r'\bmeaningful\b', text, re.IGNORECASE):
        result["edge_assessment"] = "meaningful"

    return result


def _summarize(parsed: dict) -> str:
    if parsed["decision"] == "GO":
        return (
            f"Recommend INVEST on {parsed['bet_direction']} "
            f"with ~{parsed['edge_pp']:.1f}pp estimated edge. "
            f"Weighted score: {parsed.get('weighted_score', '?'):.1f}."
        )
    return (
        f"Recommend PASS. Edge not meaningful "
        f"(~{parsed['edge_pp']:.1f}pp). Risk/reward insufficient."
    )
