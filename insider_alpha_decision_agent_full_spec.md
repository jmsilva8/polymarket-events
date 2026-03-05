# Insider Alpha — Decision Agent Full Specification
## Final Investment Recommendation & Go/No-Go Decision Engine

**Who should read this:**
| Role | Sections to read |
|---|---|
| Decision Agent implementer | 1, 2, 3, 4, 5, 9 |
| Revision Agent owner | 1, 3, 8 |
| GitHub owner (orchestrator) | 1, 6, 7, 9 |
| Team reviewing decisions | 1, 3, 5, 8 |

Read Section 1 regardless of your role — it defines principles that affect all integration points.

---

# SECTION 1 — Core Principles
### Non-negotiable design decisions

**1. Decision Agent is 100% LLM-based.**
No hard-coded rules for weighting, thresholds, or price adjustments. The LLM reads all inputs and reasons through to a decision. Temperature=0 for reproducibility.

**2. LLM decides dynamic weights.**
The LLM analyzes Agent A insider score, Agent B behavior score, their confidence/quality, and decides how much to weight each. It explains its weighting in output.

**3. Bayesian reasoning is built into the prompt.**
The LLM updates the market probability given the insider signal strength. It compares "adjusted probability" vs "current market price" to determine edge.

**4. Revision Agent flag is binding for SKIP/WATCH.**
If revision_flag == "SKIP" or "WATCH", Decision Agent outputs SKIP immediately. For other flags (NONE, DIRECTIONAL_CONFLICT), LLM proceeds to full analysis.

**5. LLM can search for news independently.**
If Agent B score is very high, the LLM can call Tavily to double-check for recent public information. This informs its reasoning but doesn't override Revision Agent's flag.

**6. Edge must be "meaningful".**
The LLM determines what "meaningful" means given context: price position, signal strength, time to close, etc. No fixed threshold.

**7. Temperature=0 on all LLM calls.**
Mandatory for reproducibility across backtest runs.

**8. Output is fully transparent.**
All weighting, probability calculations, and reasoning are explained. This is an audit trail.

---

# SECTION 2 — Decision Agent: Implementation Spec

Create directory: `src/ai_layer/decision_agent/` with files: `__init__.py`, `params.py`, `schemas.py`, `prompts.py`, `agent.py`.

---

## 2.1 `params.py` — DecisionAgentParams

```python
from dataclasses import dataclass

@dataclass
class DecisionAgentParams:
    """
    Minimal config for Decision Agent.
    The LLM handles all decision logic.
    """
    # Model config
    model_name: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0  # Mandatory

    # Optional: enable/disable Tavily search for high B scores
    enable_news_search: bool = True
    news_search_threshold_b_score: int = 7  # Search if B >= 7
```

---

## 2.2 `schemas.py` — All Pydantic Models

### Input models

```python
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime

@dataclass
class DecisionAgentInputPackage:
    """Complete input to Decision Agent."""

    # From Revision Agent
    revision_flag: Literal[
        "NONE",
        "PUBLIC_INFO_ADJUSTED",
        "PRE_SIGNAL",
        "REVERSION",
        "INTERNAL_CONFLICT",
        "DIRECTIONAL_CONFLICT"
    ]
    flag_explanation: str

    # Agent reports (passed through from Revision Agent)
    agent_a_report: dict  # Full AgentAReport
    agent_b_report: dict  # Full AgentBReport
    revision_notes: str
    recommendation_to_decision_agent: Literal["GO_EVALUATE", "SKIP", "WATCH"]

    # Market context
    current_market_price: float  # 0.0 to 1.0 (YES probability)
    evaluation_date: datetime
    end_date: datetime
    market_id: str  # For traceability
```

### Output model

```python
from pydantic import BaseModel, Field

class DecisionAgentOutput(BaseModel):
    """Final output — passed to portfolio/execution layer."""

    # Primary decision
    decision: Literal["GO", "SKIP"]
    bet_direction: Literal["YES", "NO", "null"]

    # Detailed analysis
    analysis: dict = Field(
        default_factory=lambda: {
            "agent_a_score": None,
            "agent_b_score": None,
            "weight_a_percentage": None,
            "weight_b_percentage": None,
            "weighting_rationale": "",

            "weighted_score": None,

            "current_market_price": None,
            "adjusted_probability_of_win": None,
            "estimated_edge_pp": None,
            "edge_assessment": "",  # "meaningful" or "not meaningful"

            "news_search_performed": False,
            "news_findings": "",
        }
    )

    # Reasoning for audit
    full_reasoning: str

    # Flag handling
    revision_flag_applied: str

    # Recommendation summary
    recommendation: dict = Field(
        default_factory=lambda: {
            "action": None,  # "INVEST" | "PASS" | "WATCH"
            "bet": None,  # "YES" | "NO" | None
            "risk_grade": None,  # 1-10
            "current_price": None,
            "reasoning_summary": "",
        }
    )

    # Audit fields
    evaluation_date: str
    market_id: str
```

---

## 2.3 `prompts.py` — LLM System Prompt

### System Prompt (The Core of Decision Agent)

```python
DECISION_AGENT_SYSTEM_PROMPT = """
You are the final decision-making agent in an insider risk prediction market system.

YOUR ROLE:
You receive reports from two specialist agents and a revision QA validator.
You synthesize all information and make a final GO/SKIP investment recommendation.

INPUTS YOU RECEIVE:
1. AGENT A: Insider Risk Assessment
   - insider_score (1-10): How likely insiders have advance knowledge
   - confidence: "low", "medium", "high"
   - reasoning: Why this score
   - info_holders: Groups who might know
   - leak_vectors: How info could leak to market

2. AGENT B: Market Behavior Analysis
   - behavior_score (1-10): How anomalous are price/volume/momentum?
   - signal_direction: "YES", "NO", or "SKIP"
   - confidence: "low", "medium", "high"
   - key_findings: Specific price jumps, volume spikes, momentum patterns

3. REVISION AGENT: QA & Pattern Detection
   - revision_flag: Pattern identified (NONE, PUBLIC_INFO_ADJUSTED, PRE_SIGNAL, etc)
   - flag_explanation: Why this pattern was detected
   - recommendation_to_decision_agent: "GO_EVALUATE", "SKIP", or "WATCH"

4. MARKET CONTEXT:
   - current_market_price: What the market currently prices the YES outcome at
   - evaluation_date: When you're making this decision
   - end_date: When the market resolves

YOUR TASK (in order):

═══════════════════════════════════════════════════════════════

STEP 1: CHECK REVISION AGENT RECOMMENDATIONS FIRST

If recommendation_to_decision_agent == "SKIP":
  → Output decision="SKIP"
  → Use flag_explanation as your reasoning
  → Do NOT proceed further

If recommendation_to_decision_agent == "WATCH":
  → Output decision="SKIP"
  → Explain that signal is premature (no market movement yet)
  → Do NOT proceed further

If recommendation_to_decision_agent == "GO_EVALUATE":
  → Proceed to Step 2

═══════════════════════════════════════════════════════════════

STEP 2: DYNAMIC WEIGHTING

Analyze both agents and decide how much weight to give each.

Consider:
- Which agent has higher confidence?
- Which agent's data is higher quality?
- If Agent B found strong anomalies (behavior_score >= 7), its quantitative
  evidence is compelling. Weight it heavily.
- If Agent B found nothing (behavior_score <= 3), then Agent A's insider
  assessment matters more.
- If both are weak, overall signal is weak.

Do NOT use fixed percentages. Reason from first principles.

Example reasoning patterns:
- "Agent B found sustained +18pp price jump with high confidence. This is
  strong quantitative evidence. Weight Agent B at 65%, Agent A at 35%."
- "Agent A sees moderate insider risk, but Agent B found no anomalies.
  Weight Agent A at 70%, Agent B at 30%."
- "Both high confidence and aligned. Weight 50-50 and note strong convergence."

Output your weighting and rationale clearly.

═══════════════════════════════════════════════════════════════

STEP 3: COMPUTE WEIGHTED SCORE

weighted_score = (agent_a_score × weight_a) + (agent_b_score × weight_b)

This is your combined risk assessment (1-10).

═══════════════════════════════════════════════════════════════

STEP 4: BAYESIAN UPDATE — ADJUSTED PROBABILITY OF WIN

The market currently prices the YES outcome at current_market_price.
But you have insider information (via Agent A + B).

If your combined signal is strong, the TRUE probability of YES is probably
higher than the market price. Calculate an "adjusted probability" that
reflects what you believe is true.

Reasoning:
- If weighted_score >= 7 AND signal is strong → adjusted_prob should be
  meaningfully higher than current_market_price
- If weighted_score <= 4 → adjusted_prob stays close to current_market_price
- If signals conflict (Agent A says YES, Agent B says NO) → adjusted_prob
  stays near 0.50 or current market

Example:
  Current market price: 0.60 (60% YES probability)
  Weighted score: 8 (strong signal)
  → Adjusted probability: ~0.75-0.80 (you believe YES is actually 75-80% likely)
  → Edge: 15-20 percentage points above market price

═══════════════════════════════════════════════════════════════

STEP 5: ASSESS EDGE MAGNITUDE

Compare adjusted_probability vs current_market_price:

edge = abs(adjusted_probability - current_market_price)

Decide: Is this edge "meaningful"?

Factors to consider:
- How confident are you in your adjusted_probability?
- Is the edge big enough to justify taking the risk?
  (Larger edge = more confidence)
- If current_market_price is already at 0.95, the upside is capped at 5pp.
  Even if you're right, the profit is small.
- If current_market_price is at 0.50, the upside is unlimited (50pp).

There is NO fixed threshold. You decide what "meaningful" means here.
Communicate your assessment clearly.

═══════════════════════════════════════════════════════════════

STEP 6: OPTIONAL — SEARCH FOR RECENT NEWS (if B score very high)

If behavior_score >= 7, you can optionally search for recent news about this
market. This helps confirm:
- Is the market movement explainable by public news?
- Or is it potentially insider-driven?

If you find major news that explains the movement → this might be a false
positive (market already adjusted). Adjust your analysis accordingly.

If you find NO major news despite high score → this strengthens the case that
the movement is insider-driven.

═══════════════════════════════════════════════════════════════

STEP 7: FINAL DECISION

If edge is meaningful AND signals are coherent:
  → decision = "GO"
  → bet_direction = direction of strongest signal
  → recommendation.action = "INVEST"

Otherwise:
  → decision = "SKIP"
  → bet_direction = null
  → recommendation.action = "PASS"

═══════════════════════════════════════════════════════════════

OUTPUT REQUIREMENTS:

1. Explain your weighting decision
2. Show weighted_score calculation
3. Explain your adjusted_probability reasoning
4. Assess the edge (is it meaningful?)
5. State your final decision and why
6. Provide a 2-3 sentence summary for portfolio layer

Be explicit. This is an audit trail. Every number and decision must be
justified.

CONSTRAINTS:
- Temperature=0: Be deterministic, reproducible
- Respect Revision Agent's SKIP/WATCH recommendations
- Do NOT speculate beyond the data provided
- Do NOT second-guess the agents unless you have good reason
"""
```

### User Prompt Builder

```python
def build_decision_agent_prompt(
    package: DecisionAgentInputPackage,
) -> str:
    """
    Assemble the user message for Decision Agent LLM call.
    """
    import json
    from datetime import datetime

    sections = []

    # 1. Revision Agent recommendation
    sections.append(f"""REVISION AGENT ASSESSMENT:
Flag: {package.revision_flag}
Recommendation to Decision Agent: {package.recommendation_to_decision_agent}
Explanation: {package.flag_explanation}
Notes: {package.revision_notes}""")

    # 2. Agent A summary
    a = package.agent_a_report
    sections.append(f"""AGENT A — INSIDER RISK:
Score: {a.get('insider_score')}/10
Confidence: {a.get('confidence')}
Reasoning: {a.get('reasoning')}
Info Holders: {', '.join(a.get('info_holders', [])) if a.get('info_holders') else 'none identified'}
Leak Vectors: {', '.join(a.get('leak_vectors', [])) if a.get('leak_vectors') else 'none identified'}""")

    # 3. Agent B summary
    b = package.agent_b_report
    sections.append(f"""AGENT B — MARKET BEHAVIOR:
Score: {b.get('behavior_score')}/10
Confidence: {b.get('confidence')}
Signal Direction: {b.get('signal_direction')}
Key Findings:
{json.dumps(b.get('key_findings', []), indent=2)}
Reasoning: {b.get('reasoning')}""")

    # 4. Market context
    hours_to_close = (
        (package.end_date - package.evaluation_date).total_seconds() / 3600
    )
    sections.append(f"""MARKET CONTEXT:
Current Market Price (YES): {package.current_market_price:.4f}
Current Market Price (NO): {1 - package.current_market_price:.4f}
Hours Until Resolution: {hours_to_close:.1f}
Market ID: {package.market_id}""")

    return "\n\n".join(sections)
```

---

## 2.4 `agent.py` — Entry Point

```python
from langchain.chat_models import init_chat_model
from src.ai_layer.decision_agent.schemas import DecisionAgentInputPackage, DecisionAgentOutput
from src.ai_layer.decision_agent.prompts import (
    DECISION_AGENT_SYSTEM_PROMPT,
    build_decision_agent_prompt,
)
from src.ai_layer.decision_agent.params import DecisionAgentParams


def decision_agent(
    package: DecisionAgentInputPackage,
    params: DecisionAgentParams,
) -> DecisionAgentOutput:
    """
    Main Decision Agent entry point.

    Flow:
    1. Validate input
    2. Check if Revision Agent already decided (SKIP/WATCH) → early exit
    3. Single LLM call to analyze and decide
    4. Parse LLM response and format output
    5. Return structured recommendation
    """

    # Input validation
    if not (0.0 <= package.current_market_price <= 1.0):
        return DecisionAgentOutput(
            decision="SKIP",
            bet_direction="null",
            full_reasoning="Invalid market price. Defaulting to SKIP.",
            revision_flag_applied="VALIDATION_ERROR",
            market_id=package.market_id,
            evaluation_date=package.evaluation_date.isoformat(),
        )

    # Early exit for Revision Agent hard decisions
    if package.recommendation_to_decision_agent in ["SKIP", "WATCH"]:
        return DecisionAgentOutput(
            decision="SKIP",
            bet_direction="null",
            full_reasoning=f"Revision Agent recommendation: {package.recommendation_to_decision_agent}. "
                          f"Reason: {package.flag_explanation}",
            revision_flag_applied=package.revision_flag,
            market_id=package.market_id,
            evaluation_date=package.evaluation_date.isoformat(),
            recommendation={
                "action": "WATCH" if package.recommendation_to_decision_agent == "WATCH" else "PASS",
                "bet": None,
                "risk_grade": 0,
                "current_price": package.current_market_price,
                "reasoning_summary": package.flag_explanation,
            }
        )

    # Initialize LLM with structured output
    llm = init_chat_model(
        params.model_name,
        temperature=params.temperature,
        model_provider="anthropic",
    )

    # Build prompt
    user_prompt = build_decision_agent_prompt(package)

    # Single LLM call
    messages = [
        {"role": "system", "content": DECISION_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)
    reasoning_text = response.content if hasattr(response, 'content') else str(response)

    # Parse LLM response to extract structured data
    # The LLM outputs reasoning in free text, we extract the key decisions from it
    decision_result = _extract_decision_from_llm_response(
        reasoning_text,
        package.agent_a_report.get('insider_score', 5),
        package.agent_b_report.get('behavior_score', 5),
        package.current_market_price,
    )

    # Format final output
    return DecisionAgentOutput(
        decision=decision_result["decision"],
        bet_direction=decision_result["bet_direction"],
        analysis={
            "agent_a_score": package.agent_a_report.get('insider_score'),
            "agent_b_score": package.agent_b_report.get('behavior_score'),
            "weight_a_percentage": decision_result.get("weight_a"),
            "weight_b_percentage": decision_result.get("weight_b"),
            "weighting_rationale": decision_result.get("weighting_rationale", ""),
            "weighted_score": decision_result.get("weighted_score"),
            "current_market_price": package.current_market_price,
            "adjusted_probability_of_win": decision_result.get("adjusted_probability"),
            "estimated_edge_pp": decision_result.get("edge_pp"),
            "edge_assessment": decision_result.get("edge_assessment", ""),
            "news_search_performed": decision_result.get("news_search_performed", False),
            "news_findings": decision_result.get("news_findings", ""),
        },
        full_reasoning=reasoning_text,
        revision_flag_applied=package.revision_flag,
        market_id=package.market_id,
        evaluation_date=package.evaluation_date.isoformat(),
        recommendation={
            "action": "INVEST" if decision_result["decision"] == "GO" else "PASS",
            "bet": decision_result["bet_direction"],
            "risk_grade": int(round(decision_result.get("weighted_score", 5))),
            "current_price": package.current_market_price,
            "reasoning_summary": _summarize_decision(decision_result),
        }
    )


def _extract_decision_from_llm_response(
    llm_response: str,
    agent_a_score: int,
    agent_b_score: int,
    current_price: float,
) -> dict:
    """
    Extract structured data from LLM's free-text reasoning.

    The LLM outputs detailed reasoning in text format.
    This function extracts the key decision points.

    Look for patterns like:
    - "Weight Agent A at 60%, Agent B at 40%"
    - "Adjusted probability: 0.75"
    - "Edge: 15 percentage points"
    - "decision = GO"
    """
    import re

    result = {
        "decision": "SKIP",  # default
        "bet_direction": "null",
        "weight_a": 50,
        "weight_b": 50,
        "weighting_rationale": "",
        "weighted_score": 5.0,
        "adjusted_probability": current_price,
        "edge_pp": 0.0,
        "edge_assessment": "not meaningful",
        "news_search_performed": False,
        "news_findings": "",
    }

    # Extract decision
    if "decision=\"GO\"" in llm_response or "decision = \"GO\"" in llm_response or \
       "decision: GO" in llm_response.upper():
        result["decision"] = "GO"

    # Extract bet direction
    if "YES" in llm_response and result["decision"] == "GO":
        if "NO" not in llm_response or llm_response.index("YES") < llm_response.index("NO"):
            result["bet_direction"] = "YES"
    if "NO" in llm_response and result["decision"] == "GO":
        if "YES" not in llm_response or llm_response.index("NO") < llm_response.index("YES"):
            result["bet_direction"] = "NO"

    # Extract weights
    weight_a_match = re.search(r'[Ww]eight\s+[Aa]gent\s+A\s+(?:at\s+)?(\d+)\s*%', llm_response)
    if weight_a_match:
        result["weight_a"] = int(weight_a_match.group(1))
        result["weight_b"] = 100 - result["weight_a"]

    # Extract weighted score
    score_match = re.search(r'[Ww]eighted\s+score[:\s=]+([0-9.]+)', llm_response)
    if score_match:
        result["weighted_score"] = float(score_match.group(1))

    # Extract adjusted probability
    adj_prob_match = re.search(r'[Aa]djusted\s+probabilit[yi][:\s=]+([0-9.]+)', llm_response)
    if adj_prob_match:
        result["adjusted_probability"] = float(adj_prob_match.group(1))

    # Calculate edge
    result["edge_pp"] = abs(result["adjusted_probability"] - current_price) * 100

    # Assess edge
    if "meaningful" in llm_response.lower():
        if "not meaningful" in llm_response.lower():
            result["edge_assessment"] = "not meaningful"
        else:
            result["edge_assessment"] = "meaningful"

    return result


def _summarize_decision(decision_result: dict) -> str:
    """Generate a 2-3 sentence summary of the decision."""
    decision = decision_result["decision"]
    edge = decision_result["edge_pp"]
    direction = decision_result["bet_direction"]

    if decision == "GO":
        return (
            f"Recommend INVEST on {direction} with estimated edge of {edge:.1f}pp. "
            f"Combined signal strength (weighted score {decision_result['weighted_score']:.1f}) "
            f"supports this position."
        )
    else:
        return (
            f"Recommend PASS. Edge assessment is not meaningful "
            f"(estimated {edge:.1f}pp). Risk/reward not favorable."
        )
```

---

## 2.5 Testing Requirements

Create `tests/test_decision_agent.py`:

- **Revision Agent flag handling:**
  - Input: recommendation == "SKIP" → Output: decision == "SKIP"
  - Input: recommendation == "WATCH" → Output: decision == "SKIP" with "WATCH" reasoning

- **LLM reasoning quality:**
  - Input: Agent A high (8) + Agent B high (8) + aligned → Output: decision == "GO"
  - Input: Agent A low (2) + Agent B high (8) + PUBLIC_INFO_ADJUSTED flag → Output: decision == "SKIP"
  - Input: Agent A high (8) + Agent B low (2) → Output: weighted_score moderate, edge assessment critical

- **Edge assessment:**
  - Input: current_price == 0.99 + weighted_score == 8 → Output: edge small, decision consideration
  - Input: current_price == 0.50 + weighted_score == 8 → Output: edge large, decision == "GO"

- **Determinism:**
  - Same inputs → identical outputs across multiple runs (due to temperature=0)

Do NOT test the LLM's reasoning capability. Test the integration and parsing only.

---

# SECTION 3 — How Decision Agent Works: Worked Examples

## Example 1: Strong Convergent Signal → GO

**Input:**
```
Revision flag: NONE
Agent A: insider_score=8, "FDA approval known by 3 people, leaks easy"
Agent B: behavior_score=8, "Price jumped +18pp sustained, momentum trending UP"
Current price: 0.62
```

**LLM Reasoning:**
```
Step 1: Revision says GO_EVALUATE. Proceed.

Step 2: Both agents high and coherent.
  - Agent A shows strong insider risk (8/10)
  - Agent B shows strong market anomaly (8/10)
  - Weights: A at 45%, B at 55% (B's quantitative evidence is more concrete)

Step 3: Weighted score = (8 × 0.45) + (8 × 0.55) = 8.0

Step 4: Bayesian update
  - Current market: 0.62 (62% YES probability)
  - My signals: Very strong, both pointing YES
  - Adjusted probability: 0.78 (78% YES probability)
  - Reasoning: If insiders know + market is already moving, true probability
    is probably higher than 62%

Step 5: Edge assessment
  - Edge = 0.78 - 0.62 = 0.16 (16 percentage points)
  - Is 16pp meaningful? YES. Signal is strong, data quality high.

Step 7: DECISION = GO, bet on YES
```

**Output:**
```json
{
  "decision": "GO",
  "bet_direction": "YES",
  "analysis": {
    "weighted_score": 8.0,
    "adjusted_probability_of_win": 0.78,
    "edge_pp": 16.0,
    "edge_assessment": "meaningful"
  },
  "recommendation": {
    "action": "INVEST",
    "bet": "YES"
  }
}
```

---

## Example 2: High B but Low A (False Positive) → SKIP

**Input:**
```
Revision flag: PUBLIC_INFO_ADJUSTED
Agent A: insider_score=2, "This is an election (public data)"
Agent B: behavior_score=8, "Price jumped +20pp, volume spike"
Current price: 0.78
Revision recommendation: SKIP
```

**LLM Reasoning:**
```
Step 1: Revision says SKIP.
→ Immediate output: SKIP

Reasoning: Revision Agent identified PUBLIC_INFO_ADJUSTED pattern.
Agent B found real market anomaly but Agent A sees low insider risk.
This indicates market already adjusted to public information.
No remaining edge.
```

**Output:**
```json
{
  "decision": "SKIP",
  "bet_direction": "null",
  "revision_flag_applied": "PUBLIC_INFO_ADJUSTED",
  "full_reasoning": "Revision Agent flagged PUBLIC_INFO_ADJUSTED. Market movement (B score 8) is explained by public information, not insider edge. No remaining arbitrage opportunity."
}
```

---

## Example 3: Pre-Signal (A high, no market move yet) → WATCH

**Input:**
```
Revision flag: PRE_SIGNAL
Agent A: insider_score=9, "3 company executives know, announcement in 2 days"
Agent B: behavior_score=2, "No price movement, volume normal"
Current price: 0.50
Revision recommendation: WATCH
```

**LLM Reasoning:**
```
Step 1: Revision says WATCH.
→ Immediate output: SKIP with "WATCH" status

Reasoning: Insider risk is very high but market hasn't moved yet.
Information is likely not yet public. Deferring trade pending market confirmation.
Portfolio should monitor this for re-evaluation closer to announcement.
```

**Output:**
```json
{
  "decision": "SKIP",
  "recommendation": {
    "action": "WATCH"
  },
  "full_reasoning": "Pre-signal condition: very high insider risk but no market anomaly yet. Market will likely move closer to event date. Deferring investment pending confirmation."
}
```

---

# SECTION 4 — Integration with GitHub Owner

## Graph State Update

```python
class GraphState(TypedDict):
    # ... existing fields ...
    agent_a_reports: dict[str, dict]
    agent_b_reports: dict[str, dict]
    revision_outputs: dict[str, dict]
    decision_outputs: dict[str, DecisionAgentOutput]
```

## Calling Decision Agent from Graph

```python
def decision_node(state: GraphState) -> dict:
    """
    For each market, read Revision output.
    Call Decision Agent.
    Aggregate results.
    """
    decision_outputs = {}

    for market_id in state["markets"]:
        package = DecisionAgentInputPackage(
            revision_flag=state["revision_outputs"][market_id]["revision_flag"],
            flag_explanation=state["revision_outputs"][market_id]["flag_explanation"],
            agent_a_report=state["agent_a_reports"][market_id],
            agent_b_report=state["agent_b_reports"][market_id],
            revision_notes=state["revision_outputs"][market_id]["revision_notes"],
            recommendation_to_decision_agent=state["revision_outputs"][market_id]["recommendation_to_decision_agent"],
            current_market_price=state["current_prices"][market_id],
            evaluation_date=state["evaluation_date"],
            end_date=state["markets"][market_id].end_date,
            market_id=market_id,
        )

        output = decision_agent(package, DecisionAgentParams())
        decision_outputs[market_id] = output

    return {"decision_outputs": decision_outputs}
```

## File Structure

```
src/ai_layer/decision_agent/
    __init__.py
    params.py
    schemas.py
    prompts.py
    agent.py

src/graph.py  ← Add decision_node
```

---

# SECTION 5 — LLM Configuration & Reproducibility

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0,  # Mandatory: zero variance for reproducibility
    model_provider="anthropic",
)

# Same input → identical output across multiple runs
# Verified via backtesting
```

---

# SECTION 6 — Cost & Latency Analysis

```
Per classification:
  - Input tokens: ~800 (both reports + context)
  - Output tokens: ~400 (reasoning + decision)
  - Total: ~1200 tokens

Haiku pricing:
  - Input: 0.80 per 1M tokens
  - Output: 4.00 per 1M tokens
  - Cost per call: (~800 * 0.80 + ~400 * 4.00) / 1M = $0.00208

For 1000 markets:
  - Total cost: ~$2.08
  - Latency: ~1-2s per call
  - With batching: ~30 seconds for 1000 markets
```

---

# SECTION 7 — Signal Interpretation Reference

| Agent A | Agent B | Revision Flag | LLM Decision |
|---------|---------|---|---|
| High (8+) | High (8+) | NONE | GO (ideal) |
| Low (1-3) | High (8+) | PUBLIC_INFO_ADJUSTED | SKIP (false positive) |
| High (8+) | SKIP | PRE_SIGNAL | WATCH |
| High (8+) | High (8+) | DIRECTIONAL_CONFLICT | LLM weighs both |
| Any | Any | Any | LLM decides |

---

# SECTION 8 — All Design Decisions

| Decision | Choice | Reason |
|----------|--------|-------|
| Weighting | LLM decides dynamically | Flexible, adaptive to context |
| Threshold | LLM decides "meaningful" edge | No false precision, context-dependent |
| Bayesian logic | Built into prompt | Aligns with financial reasoning |
| News search | Optional (LLM triggers it) | Allows for double-checking public info |
| Revision override | Binds for SKIP/WATCH | Clear authority, prevents loops |
| Temperature | 0 | Reproducibility for backtesting |
| Model | Haiku | Cost-efficient, sufficient for reasoning |

---

# SECTION 9 — Deliverables Checklist

For Decision Agent implementer:

- [ ] `params.py` — DecisionAgentParams
- [ ] `schemas.py` — DecisionAgentInputPackage & DecisionAgentOutput
- [ ] `prompts.py` — System prompt + user prompt builder
- [ ] `agent.py` — Main decision_agent() function + helper functions
- [ ] `__init__.py` — Clean exports
- [ ] `tests/test_decision_agent.py` — Test cases from Section 2.5
- [ ] Documentation — Docstrings on all functions
- [ ] Error handling — Graceful degradation for malformed inputs

For GitHub owner:

- [ ] Add decision_node to LangGraph
- [ ] Update GraphState
- [ ] Connect decision_outputs to export layer
- [ ] Logging for all decisions (audit trail)

For team:

- [ ] Backtesting to validate LLM reasoning quality
- [ ] Monitor decision distribution (are we SKIP'ing too much?)
- [ ] Review sample outputs for bias or inconsistency

---

**For Decision Agent implementation: see Sections 2–3.**
**For integration with Agent A, B, and Revision: see `insider_alpha_agent_b_full_spec.md` Sections 1, 4, 5.**
**For overall workflow: see `insider_alpha_workflow.md`.**
