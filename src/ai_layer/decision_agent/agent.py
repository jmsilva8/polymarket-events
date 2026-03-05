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


def _provider(model: str) -> str:
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    raise ValueError(f"Cannot determine provider for model: {model!r}")


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
        model_provider=_provider(params.model_name),
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


# ── Deterministic decision combiner ───────────────────────────────────────────

def decision_agent_deterministic(
    package: DecisionAgentInputPackage,
    params: DecisionAgentParams,
) -> DecisionAgentOutput:
    """
    Deterministic GO/SKIP decision — no LLM call.

    Combines Agent A and B scores with fixed weights and thresholds.
    All parameters are tunable via DecisionAgentParams for backtesting sweeps.

    Edge formula: edge_pp = (weighted_score - 5) / 5 * max_edge_pp
    This gives 0pp at score=5 (no signal), max_edge_pp at score=10.
    Bet direction always comes from Agent B's signal_direction.
    """
    eval_date = package.evaluation_date.isoformat()

    def _skip(reason: str, action: str = "PASS") -> DecisionAgentOutput:
        return DecisionAgentOutput(
            decision="SKIP",
            bet_direction="null",
            full_reasoning=reason,
            revision_flag_applied=package.revision_flag,
            market_id=package.market_id,
            evaluation_date=eval_date,
            recommendation={
                "action": action,
                "bet": None,
                "risk_grade": 0,
                "current_price": package.current_market_price,
                "reasoning_summary": reason,
            },
        )

    # 1. Validate price
    if not (0.0 <= package.current_market_price <= 1.0):
        return _skip("Invalid market price. Defaulting to SKIP.")

    # 2. Honour Revision Agent hard decisions
    if package.recommendation_to_decision_agent == "SKIP":
        return _skip(
            f"Revision Agent: SKIP. {package.flag_explanation}", action="PASS"
        )
    if package.recommendation_to_decision_agent == "WATCH":
        return _skip(
            f"Revision Agent: WATCH. {package.flag_explanation}", action="WATCH"
        )

    # 3. Extract scores and confidence
    a_score = package.agent_a_report.get("insider_risk_score", 1)
    b_score = package.agent_b_report.get("behavior_score", 1)
    a_confidence = package.agent_a_report.get("confidence", "medium")
    b_confidence = package.agent_b_report.get("confidence", "medium")
    signal_direction = package.agent_b_report.get("signal_direction", "SKIP")

    # 4. Score gates (applied to raw scores before confidence adjustment)
    if a_score < params.min_a_score:
        return _skip(
            f"Agent A score {a_score} below minimum {params.min_a_score}. SKIP."
        )
    if b_score < params.min_b_score:
        return _skip(
            f"Agent B score {b_score} below minimum {params.min_b_score}. SKIP."
        )

    # 5. B must have a directional signal
    if signal_direction == "SKIP":
        return _skip("Agent B has no directional signal. SKIP.")

    # 6. Apply confidence multipliers to effective scores
    _conf_mult = {
        "high": params.confidence_high,
        "medium": params.confidence_medium,
        "low": params.confidence_low,
    }
    eff_a = a_score * _conf_mult.get(a_confidence, params.confidence_medium)
    eff_b = b_score * _conf_mult.get(b_confidence, params.confidence_medium)

    # 7. Weighted score
    weighted_score = round(params.weight_a * eff_a + params.weight_b * eff_b, 2)

    # 8. Score threshold
    if weighted_score < params.go_score_threshold:
        return _skip(
            f"Weighted score {weighted_score:.2f} below threshold "
            f"{params.go_score_threshold}. SKIP."
        )

    # 8. Edge estimate: linear from 0pp at score=5 to max_edge_pp at score=10
    raw_edge = max(0.0, (weighted_score - 5.0) / 5.0) * params.max_edge_pp
    direction_sign = 1 if signal_direction == "YES" else -1
    adjusted_prob = package.current_market_price + direction_sign * raw_edge / 100
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))
    edge_pp = abs(adjusted_prob - package.current_market_price) * 100

    # 9. Edge threshold
    if edge_pp < params.min_edge_pp:
        return _skip(
            f"Estimated edge {edge_pp:.1f}pp below minimum {params.min_edge_pp}pp. SKIP."
        )

    # 10. GO
    reasoning = (
        f"GO: A={a_score}({a_confidence})→eff={eff_a:.2f} "
        f"B={b_score}({b_confidence})→eff={eff_b:.2f} "
        f"weighted={weighted_score:.2f} (threshold={params.go_score_threshold}). "
        f"Signal={signal_direction}. Edge≈{edge_pp:.1f}pp "
        f"(price={package.current_market_price:.2f} → adj={adjusted_prob:.2f})."
    )
    return DecisionAgentOutput(
        decision="GO",
        bet_direction=signal_direction,
        analysis={
            "agent_a_score": a_score,
            "agent_b_score": b_score,
            "weight_a_percentage": int(params.weight_a * 100),
            "weight_b_percentage": int(params.weight_b * 100),
            "weighting_rationale": (
                f"Fixed weights A={params.weight_a:.0%} B={params.weight_b:.0%} "
                f"× confidence multipliers (A:{a_confidence}={_conf_mult.get(a_confidence):.2f}, "
                f"B:{b_confidence}={_conf_mult.get(b_confidence):.2f})"
            ),
            "weighted_score": weighted_score,
            "current_market_price": package.current_market_price,
            "adjusted_probability_of_win": adjusted_prob,
            "estimated_edge_pp": round(edge_pp, 2),
            "edge_assessment": "meaningful",
        },
        full_reasoning=reasoning,
        revision_flag_applied=package.revision_flag,
        market_id=package.market_id,
        evaluation_date=eval_date,
        recommendation={
            "action": "INVEST",
            "bet": signal_direction,
            "risk_grade": int(round(weighted_score)),
            "current_price": package.current_market_price,
            "reasoning_summary": (
                f"INVEST {signal_direction} — weighted score {weighted_score:.1f}, "
                f"~{edge_pp:.1f}pp estimated edge."
            ),
        },
    )
