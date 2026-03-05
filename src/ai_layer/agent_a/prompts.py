"""Prompts for Agent A (insider risk classifier)."""

import json
from typing import Optional

from src.ai_layer.agent_a.schemas import AgentAInputPackage, AgentAReport


AGENT_A_SYSTEM_PROMPT = """\
You are an expert analyst specializing in information asymmetry in prediction markets.

Your task: Given a prediction market's title, description, category, and tags, assign an \
insider risk score from 1 to 10 indicating how much non-public information could plausibly \
influence trading on this market.

## Scoring Framework

Analyze these five dimensions for the specific market in front of you:

1. **Information access**: Who has advance knowledge of the outcome? How many people are in \
that group? A smaller, more identifiable group means higher risk.

2. **Lead time**: How far in advance is the outcome knowable to insiders? Longer windows \
create more opportunity to trade profitably before the market adjusts.

3. **Financial incentive**: How large is the potential trading edge? Outcomes with large, \
binary price swings and deep markets create stronger motivation to act on inside information.

4. **Leak probability**: Through what realistic channels could this information reach \
prediction markets? Consider direct trading by insiders, social media disclosures, \
tipping off connected traders, or information seeping through supply chains.

5. **Enforcement barriers**: Do NDAs, regulatory prohibitions (e.g. securities law, \
professional codes), or reputational consequences constrain leaking? These reduce \
probability of leakage but do not eliminate the underlying information asymmetry.

## Score Scale

- **1–2**: Outcome determined by millions of independent actors or fundamentally unpredictable \
processes. No individual or group has meaningful advance knowledge.
- **3–4**: Outcome known to a larger group (dozens to hundreds of people), with strong legal \
protections and professional norms that make leaks uncommon in practice.
- **5–6**: Outcome known to a moderate-size group; financial incentives exist but meaningful \
barriers reduce the probability of actionable leakage.
- **7–8**: Small, identifiable group with clear advance knowledge; significant financial \
incentive; barriers exist but leaks are realistic and have occurred in comparable situations.
- **9–10**: Very small group (single digits to low tens) holds definitive advance knowledge \
weeks or months ahead; massive financial incentive; barriers are insufficient deterrent \
given the potential reward.

## Critical Instructions

- Reason from first principles about the specific market structure.
- Identify the actual decision-makers or knowledge-holders for this outcome.
- Do not anchor on surface-level category labels — analyze the underlying information dynamics.
- A high score requires both (a) identifiable insiders AND (b) realistic leak pathways.
- A low score requires genuinely diffuse or unpredictable outcomes, not merely legal protections.
- temperature=0 is set — your output must be fully deterministic and evidence-based.

Output JSON only, exactly as specified in the schema.
"""


def build_agent_a_prompt(package: AgentAInputPackage) -> str:
    """Build the user message for agent_a_initial()."""
    tags_str = ", ".join(package.tags) if package.tags else "None"
    end_date_str = package.end_date.isoformat() if package.end_date else "Unknown"

    return (
        f"Classify this prediction market for insider risk:\n\n"
        f"**Title:** {package.question}\n"
        f"**Description:** {(package.description or '')[:500]}\n"
        f"**Platform:** {package.platform}\n"
        f"**Category:** {package.category or 'N/A'}\n"
        f"**Tags:** {tags_str}\n"
        f"**End Date:** {end_date_str}\n\n"
        f"Respond in JSON format only."
    )


def build_agent_a_revision_prompt(
    original_report: AgentAReport,
    revision_feedback: str,
    package: AgentAInputPackage,
) -> str:
    """Build the user message for agent_a_revise()."""
    tags_str = ", ".join(package.tags) if package.tags else "None"
    end_date_str = package.end_date.isoformat() if package.end_date else "Unknown"

    return (
        f"You previously classified this market and the Revision Agent has flagged a coherence issue.\n\n"
        f"YOUR ORIGINAL ANALYSIS:\n"
        f"{json.dumps(original_report.model_dump(mode='json'), indent=2)}\n\n"
        f"REVISION AGENT FEEDBACK:\n"
        f"{revision_feedback}\n\n"
        f"MARKET:\n"
        f"**Title:** {package.question}\n"
        f"**Description:** {(package.description or '')[:500]}\n"
        f"**Platform:** {package.platform}\n"
        f"**Category:** {package.category or 'N/A'}\n"
        f"**Tags:** {tags_str}\n"
        f"**End Date:** {end_date_str}\n\n"
        f"Re-examine your analysis in light of the feedback. "
        f"If the feedback reveals a genuine error in your reasoning, update your score and explanation. "
        f"If your original analysis was correct, maintain your score with a stronger justification.\n\n"
        f"Respond in JSON format only."
    )
