"""
Revision Agent — LLM-powered QA and cross-pattern detector.

Single LLM call per market. Receives Agent A and Agent B reports,
validates coherence, detects cross-patterns, routes to Decision Agent.

Design decisions (non-negotiable):
  - temperature=0 for backtesting reproducibility
  - Structured output binding via .with_structured_output()
  - Max 5 feedback iterations per market (managed by graph node)
  - PUBLIC_INFO_ADJUSTED and PRE_SIGNAL → autonomous SKIP/WATCH decisions
  - DIRECTIONAL_CONFLICT → GO_EVALUATE (Decision Agent resolves via weighting)
"""

import logging
from typing import Literal, Optional

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── Output schema ──────────────────────────────────────────────────────────────

class FeedbackMessage(BaseModel):
    """Feedback to be sent to an agent for revision."""
    recipient: Literal["A", "B"]
    message: str


class RevisionAgentOutput(BaseModel):
    """Final output passed to Decision Agent."""

    revision_flag: Literal[
        "NONE",
        "PUBLIC_INFO_ADJUSTED",
        "PRE_SIGNAL",
        "REVERSION",
        "INTERNAL_CONFLICT",
        "DIRECTIONAL_CONFLICT",
    ]
    flag_explanation: str

    # Pass-through for Decision Agent context
    agent_a_report: dict
    agent_b_report: dict

    # Narrative combining coherence assessment + pattern evidence + feedback decisions
    revision_notes: str

    # Feedback routing (empty if no feedback needed)
    feedback_to_send: list[FeedbackMessage] = []

    recommendation_to_decision_agent: Literal["GO_EVALUATE", "SKIP", "WATCH"]
    iterations_used: int = 0
    llm_reasoning_summary: Optional[str] = None


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a QA validator for a multi-agent prediction market trading system.

TWO SPECIALIST AGENTS report to you:

AGENT A: Classifies markets by insider risk (1-10 scale)
  Sees: market title, description, category, platform
  Blind to: price/volume data
  Output: insider_risk_score + reasoning + info_holders + leak_vectors

AGENT B: Detects anomalous price/volume behavior (1-10 scale)
  Sees: price history timeseries (volume often unavailable for historical data)
  Blind to: market context, what the bet is about
  Output: behavior_score + signal_direction (YES/NO/SKIP)

IMPORTANT — VOLUME DATA:
Agent B frequently runs without volume data (especially on historical markets).
This is expected. Do NOT flag missing volume as an incoherence in Agent B's report.
A price/momentum-only analysis is valid. Absence of volume confirmation should
lower your confidence in the signal but does not invalidate it.

YOUR RESPONSIBILITIES:

1. VALIDATE COHERENCE (within each agent independently)

   Agent A coherence check:
   ✓ Is the insider_risk_score supported by the reasoning?
   ✓ Are the info_holders credible for this risk level?
   ✓ Does high score (7+) have detailed reasoning and named holders?
   ✓ Does low score (1-3) avoid high-risk category claims?

   Agent B coherence check:
   ✓ Is behavior_score consistent with price_jump_assessment results?
   ✓ Is behavior_score consistent with momentum_assessment results?
   ✓ If signals_contradictory=true, does high score still make sense?
   ✓ If is_sustained=false, can score remain high?
   ✓ Do NOT flag volume tools being skipped as incoherence.

2. DETECT CROSS-PATTERNS (between agents)

   Pattern: PUBLIC_INFO_ADJUSTED
   Trigger: B behavior_score >= 7 AND A insider_risk_score <= 3
   Meaning: B found real market anomaly, but A sees low insider risk
   Interpretation: Market already adjusted to public information. No edge.
   Decision: SKIP (autonomous — do not ask Decision Agent)

   Pattern: PRE_SIGNAL
   Trigger: A insider_risk_score >= 7 AND B signal_direction = "SKIP"
   Meaning: High insider risk but no market movement yet
   Interpretation: Signal is premature. Market may move later.
   Decision: WATCH

   Pattern: REVERSION
   Trigger: B behavior_score >= 7 AND B is_sustained = false
   Meaning: B scored high but price jump didn't hold
   Interpretation: Likely noise — send feedback to Agent B
   Decision: SKIP (pending revision)

   Pattern: INTERNAL_CONFLICT
   Trigger: B behavior_score >= 7 AND B signals_contradictory = true
   Meaning: B's own signals disagree (e.g. price UP, momentum DOWN)
   Interpretation: Agent B should clarify which signal is more reliable
   Decision: Send feedback to Agent B

   Pattern: DIRECTIONAL_CONFLICT
   Trigger: A insider_risk_score >= 7 AND B behavior_score >= 7 AND
            B signal_direction != "SKIP" AND A/B imply opposite outcomes
   Meaning: Both high confidence but opposing signals
   Interpretation: Genuinely ambiguous — Decision Agent resolves via weighting
   Decision: GO_EVALUATE

   Pattern: NONE
   Trigger: All other cases (including both high + aligned, or both low)
   Decision: GO_EVALUATE

3. GENERATE FEEDBACK (only for specific, actionable issues)

   Send to Agent A if: score/reasoning mismatch, no named holders for high score
   Send to Agent B if: behavior_score/tool mismatch, contradictory signals

   Good feedback (specific and actionable):
   - "Your score is 7 but is_sustained=false. Does score hold given reversion?"
   - "Momentum horizons conflict — re-examine with tighter windows."

   Bad feedback (do NOT send):
   - "I'm not sure about this" — not actionable
   - "Please re-run your analysis" — produces identical output

   Do NOT send feedback for PUBLIC_INFO_ADJUSTED — Agent B's output is correct.
   The issue is the cross-agent interpretation.

4. MAKE AUTONOMOUS DECISIONS
   PUBLIC_INFO_ADJUSTED → SKIP
   PRE_SIGNAL → WATCH
   REVERSION / INTERNAL_CONFLICT → SKIP + send feedback
   DIRECTIONAL_CONFLICT → GO_EVALUATE (Decision Agent resolves)
   NONE → GO_EVALUATE

Output JSON only, exactly as specified in the schema.
"""


def _build_user_prompt(agent_a_report: dict, agent_b_report: dict) -> str:
    import json
    return (
        "Analyze these two agent reports for coherence and cross-patterns:\n\n"
        f"AGENT A REPORT:\n{json.dumps(agent_a_report, indent=2, default=str)}\n\n"
        f"AGENT B REPORT:\n{json.dumps(agent_b_report, indent=2, default=str)}\n\n"
        "Your analysis:\n"
        "1. Is Agent A's report coherent? Assess reasoning quality and evidence.\n"
        "2. Is Agent B's report coherent? Assess tool outputs vs score.\n"
        "3. What cross-pattern (if any) emerges?\n"
        "4. Should feedback be sent? To whom? What message?\n"
        "5. What is your recommendation to the Decision Agent?\n\n"
        "Respond in JSON format only."
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def revision_agent(
    agent_a_report: dict,
    agent_b_report: dict,
    model: str = DEFAULT_MODEL,
    llm=None,
) -> RevisionAgentOutput:
    """
    Run the Revision Agent on one pair of Agent A + B reports.

    Returns RevisionAgentOutput with revision_flag, recommendation,
    and any feedback messages to route back to agents.
    """
    if llm is None:
        llm = init_chat_model(
            model, temperature=0, model_provider="anthropic"
        ).with_structured_output(RevisionAgentOutput)

    user_prompt = _build_user_prompt(agent_a_report, agent_b_report)

    try:
        result: RevisionAgentOutput = llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        # Ensure pass-through fields are populated
        result.agent_a_report = agent_a_report
        result.agent_b_report = agent_b_report
    except Exception as e:
        logger.error("Revision Agent LLM call failed: %s", e)
        # Conservative fallback: skip on error
        result = RevisionAgentOutput(
            revision_flag="NONE",
            flag_explanation=f"Revision Agent error: {e}. Defaulting to SKIP.",
            agent_a_report=agent_a_report,
            agent_b_report=agent_b_report,
            revision_notes="Revision Agent encountered an error. Conservative SKIP applied.",
            feedback_to_send=[],
            recommendation_to_decision_agent="SKIP",
            iterations_used=0,
        )

    logger.debug(
        "Revision Agent: flag=%s recommendation=%s feedback_count=%d",
        result.revision_flag,
        result.recommendation_to_decision_agent,
        len(result.feedback_to_send),
    )
    return result
