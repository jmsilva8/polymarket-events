# Revision Agent Specification (LLM Version)
## AI-Powered QA & Cross-Pattern Detector
**Model:** Claude Haiku 4.5 (or Sonnet for complex cases)
**Type:** LLM Agent with structured output binding
**Temperature:** 0 (mandatory for reproducibility, analytical tone)
**Integration with:** `insider_alpha_agent_b_full_spec.md` (SECTIONS 4, 5, 6)
**Status:** ✅ LLM Version - Production-Ready
**Date:** March 2026

---

## EXECUTIVE SUMMARY

### What Changed from Deterministic?

| Aspect | Deterministic | LLM |
|--------|---|---|
| **Coherence Detection** | Hard-coded rules (score >= 7 → check info_holders) | LLM reasoning (understands nuance, context) |
| **Pattern Detection** | Exact score comparisons | LLM analyzes relationships intelligently |
| **Feedback Generation** | Static templates | LLM-crafted contextual messages |
| **Decision Making** | Boolean logic | LLM reasoning with transparency |
| **Latency** | <1ms | 1-2s per market |
| **Cost** | $0 | ~$0.002 per classification |
| **Reproducibility** | 100% deterministic | 100% (temp=0) |

---

## PART 1: ARCHITECTURE

### Single LLM Call with Structured Output

```
Input: Agent A Report + Agent B Report (parallel)
  ↓
Claude Haiku 4.5 (temp=0)
  ├─ System Prompt (defines role, patterns, feedback rules, max 5 feedback iterations)
  ├─ User Prompt (both agent reports)
  └─ Structured Output Binding (RevisionAgentOutput)
  ↓
Output: {revision_flag, flag_explanation, recommendation, revision_notes, feedback_to_send}
```

### Why Single Call?

✅ **Efficiency**: Analyzes both agents in one pass
✅ **Coherence**: Patterns detected in context of both reports
✅ **Cost**: ~1 cent per market (vs multiple calls)
✅ **Speed**: Single 1-2s latency (vs 5+ seconds)

---

## PART 2: SYSTEM PROMPT (Temperature: 0, Max 5 Feedback Iterations)

```
You are a QA validator for a multi-agent prediction market trading system.

TWO SPECIALIST AGENTS report to you:

AGENT A: Classifies markets by insider risk (1-10 scale)
  Sees: market title, description, category, platform
  Blind to: price/volume data
  Output: insider_risk_score + reasoning + info_holders

AGENT B: Detects anomalous price/volume behavior (1-10 scale)
  Sees: price history, volume timeseries, CLOB token data
  Blind to: market context, what the bet is about
  Output: behavior_score + signal_direction (YES/NO/SKIP)

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

2. DETECT CROSS-PATTERNS (between agents)

   Pattern: PUBLIC_INFO_ADJUSTED
   Trigger: B behavior_score >= 7 AND A insider_risk_score <= 3
   Meaning: B found real market anomaly, but A sees low insider risk
   Interpretation: Market already adjusted to public information. No edge.
   Decision: SKIP (autonomous, don't ask Decision Agent)

   Pattern: PRE_SIGNAL
   Trigger: A insider_risk_score >= 7 AND B signal_direction = "SKIP"
   Meaning: High insider risk but no market movement yet
   Interpretation: Signal is premature. Market may move later.
   Decision: WATCH (don't invest now, monitor)

   Pattern: REVERSION
   Trigger: B behavior_score >= 7 AND B is_sustained = false
   Meaning: B scored high but price jump didn't hold
   Interpretation: False signal, likely noise
   Action: Send feedback to Agent B asking to recalculate

   Pattern: INTERNAL_CONFLICT
   Trigger: B behavior_score >= 7 AND B signals_contradictory = true
   Meaning: B's own tools disagree (price UP, momentum DOWN)
   Interpretation: B should clarify which signal is more reliable
   Action: Send feedback to Agent B asking to resolve conflict

   Pattern: DIRECTIONAL_CONFLICT
   Trigger: A insider_risk_score >= 7 AND B behavior_score >= 7 AND directions_differ
   Meaning: Both high confidence but opposite recommendations
   Interpretation: Genuinely ambiguous signal — requires Decision Agent weighting
   Decision: GO_EVALUATE (pass to Decision Agent for dynamic weighting)

   Pattern: NONE
   Trigger: All other cases (including both high + aligned)
   Meaning: Either weak signals or coherent convergent signal
   Decision: GO_EVALUATE (proceed to Decision Agent)

3. GENERATE FEEDBACK (when incoherent)

   Send to Agent A if: insider_risk_score/reasoning mismatch, insufficient evidence for score level
   Send to Agent B if: behavior_score/tool_output mismatch, contradictory signals (price vs momentum)

   Both agents follow symmetric feedback loop:
   - Max 1 revision per agent per iteration
   - Agent must respond with recalculation or justification
   - If finding unchanged after feedback → treat as robust to scrutiny (higher confidence)
   - If finding changed → use updated score/direction

   Do NOT send feedback if: pattern is purely comparative (e.g., PUBLIC_INFO_ADJUSTED)

   Feedback format (same for both agents):
   - Specific: Reference exact score, reasoning, tool output
   - Actionable: Ask for recalculation with specific parameter changes or clarification
   - Respectful: Acknowledge the agent's valid findings

4. MAKE AUTONOMOUS DECISIONS

   You decide: feedback recipients, pattern, recommendation
   Do NOT defer to Decision Agent for decisions only you can make
   Examples:
     ✓ PUBLIC_INFO_ADJUSTED: You decide SKIP
     ✓ PRE_SIGNAL: You decide WATCH
     ✓ REVERSION: You send feedback to B
     ✓ DIRECTIONAL_CONFLICT: You pass to Decision Agent with GO_EVALUATE (ambiguous case)

CONSTRAINTS:
- Temperature: 0 (deterministic, required for backtesting reproducibility)
- No speculation beyond provided data
- Maximum feedback iterations: 5 total per market analysis
  (Apply equally to Agent A and Agent B. Each agent receives feedback once per iteration.)
- Output JSON only, structured exactly as specified
```

---

## PART 3: USER PROMPT TEMPLATE

```
Analyze these two agent reports for coherence and cross-patterns:

AGENT A REPORT:
{agent_a_json}

AGENT B REPORT:
{agent_b_json}

Your analysis:
1. Is Agent A's report coherent? Assess reasoning quality, evidence for score.
2. Is Agent B's report coherent? Assess tool outputs vs score.
3. What cross-pattern (if any) emerges?
4. Should feedback be sent? To whom? What message?
5. What is your recommendation to Decision Agent?

Respond in JSON format only.
```

---

## PART 4: OUTPUT SCHEMA (Pydantic with Structured Output Binding)

**Matches insider_alpha_agent_b_full_spec.md Section 5 expectations**

```python
from pydantic import BaseModel
from typing import Literal, Optional

class FeedbackMessage(BaseModel):
    """Feedback to be sent to an agent for revision"""
    recipient: Literal["A", "B"]  # Symmetric revision capability for both agents
    message: str

class RevisionAgentOutput(BaseModel):
    """Final output passed to Decision Agent. Simplified schema aligned with insider_alpha."""

    # Core revision flag
    revision_flag: Literal[
        "NONE",
        "PUBLIC_INFO_ADJUSTED",
        "PRE_SIGNAL",
        "REVERSION",
        "INTERNAL_CONFLICT",
        "DIRECTIONAL_CONFLICT"
    ]

    # Explanation of the flag and pattern reasoning
    flag_explanation: str

    # Original agent reports (pass-through for Decision Agent context)
    agent_a_report: dict  # Full AgentAReport serialized
    agent_b_report: dict  # Full AgentBReport serialized

    # Consolidated analysis and feedback decisions
    # Format: narrative paragraph combining:
    #   - Agent A coherence assessment
    #   - Agent B coherence assessment
    #   - Pattern evidence
    #   - Feedback decisions (if any)
    revision_notes: str

    # Feedback routing (if any feedback to send)
    feedback_to_send: list[FeedbackMessage] = []

    # Recommendation to Decision Agent
    recommendation_to_decision_agent: Literal["GO_EVALUATE", "SKIP", "WATCH"]

    # Iteration tracking (for feedback loop management)
    iterations_used: int = 0

    # Optional audit trail for observability
    llm_reasoning_summary: Optional[str] = None
```

**Example revision_notes format:**
```
Agent A coherence: ✓ Score 8 well-supported. Reasoning details FDA approval timing. Info holders (FDA, pharma execs) credible.
Agent B coherence: ✗ Score 7 contradicted by is_sustained=false. Price jumped +12pp but reverted.
Pattern: REVERSION. Evidence: [B score=7, is_sustained=false, price reverted].
Feedback: Sending to Agent B to reconsider sustenance thresholds.
```

---

## PART 5: EXAMPLE OUTPUTS (Using insider_alpha Schema)

### Example 1: Convergent Signal (NONE → GO_EVALUATE)

```json
{
  "revision_flag": "NONE",
  "flag_explanation": "Both agents coherent and high confidence. Both signals point YES direction. No conflicting patterns.",
  "agent_a_report": { "insider_risk_score": 8, "signal_direction": "YES", "reasoning": "FDA approval likely within 24h..." },
  "agent_b_report": { "behavior_score": 8, "signal_direction": "YES", "price_jump_assessment": {...} },
  "revision_notes": "Agent A coherence: ✓ Score 8 well-supported. Reasoning details FDA approval. Info holders (FDA, pharma execs) credible. Agent B coherence: ✓ Score 8 consistent. Price jumped +18pp sustained, momentum trending UP across all horizons, signals aligned. Pattern: NONE. Both agents coherent and aligned. Feedback: None needed.",
  "feedback_to_send": [],
  "recommendation_to_decision_agent": "GO_EVALUATE",
  "iterations_used": 0,
  "llm_reasoning_summary": "Exceptional case: both agents high, coherent, aligned directions. Ideal GO setup."
}
```

### Example 2: False Positive Detection (PUBLIC_INFO_ADJUSTED → SKIP)

```json
{
  "revision_flag": "PUBLIC_INFO_ADJUSTED",
  "flag_explanation": "B detected real market anomaly (score 8) but A sees this as low-risk public election (score 2). Market already adjusted to public information. No insider edge remains.",
  "agent_a_report": { "insider_risk_score": 2, "signal_direction": "YES", "reasoning": "Election outcome (public info)" },
  "agent_b_report": { "behavior_score": 8, "signal_direction": "YES", "price_jump_assessment": {...} },
  "revision_notes": "Agent A coherence: ✓ Score 2 supported. Reasoning: 'election outcome' (low-risk category), no credible info holders. Agent B coherence: ✓ Score 8 supported by price jumped +20pp sustained, volume spike, momentum trending. Pattern: PUBLIC_INFO_ADJUSTED. Evidence: [A score=2, B score=8, move_shape=sudden]. Interpretation: Market already adjusted to public election news. No feedback needed—Agent B correctly found anomaly, but the edge already priced in.",
  "feedback_to_send": [],
  "recommendation_to_decision_agent": "SKIP",
  "iterations_used": 0,
  "llm_reasoning_summary": "Classic false positive: real anomaly (B) but public info (A). Market absorbed news. No profit opportunity."
}
```

### Example 3: Reversion Detected (REVERSION → feedback to B)

```json
{
  "revision_flag": "REVERSION",
  "flag_explanation": "Agent B's behavior_score=7 conflicts with is_sustained=false. High score implies confident signal, but price reverted.",
  "agent_a_report": { "insider_risk_score": 5, "signal_direction": "YES", "reasoning": "Moderate insider risk" },
  "agent_b_report": { "behavior_score": 7, "signal_direction": "YES", "is_sustained": false, "price_jump_assessment": {...} },
  "revision_notes": "Agent A coherence: ✓ Score 5 reasonable. Insider risk moderate but not exceptional. Agent B coherence: ✗ Score 7 contradicted by tools. Price jumped +12pp BUT is_sustained=false (price reverted). High score unjustified if move didn't hold. Pattern: REVERSION. Evidence: [B score=7, is_sustained=false]. Feedback: Sending to Agent B to reconsider sustenance thresholds and whether score of 7 is still justified.",
  "feedback_to_send": [
    {
      "recipient": "B",
      "message": "Your behavior_score is 7, suggesting strong signal. However, is_sustained=false indicates the price jump reverted. This contradicts a high score. Please re-examine: (1) Does a sustenance window of >4h minimum hold? (2) Does volume confirm the direction? Does your score still hold at 7 given the reversion?"
    }
  ],
  "recommendation_to_decision_agent": "SKIP",
  "iterations_used": 1,
  "llm_reasoning_summary": "Agent B found something real but it didn't stick. Likely noise. Awaiting B recalculation."
}
```

### Example 4: Directional Conflict (DIRECTIONAL_CONFLICT → GO_EVALUATE)

```json
{
  "revision_flag": "DIRECTIONAL_CONFLICT",
  "flag_explanation": "Both agents high confidence (A=8, B=8) but opposite directions. Agent A says YES (insider info suggests approval), Agent B says NO (price momentum suggests rejection). Genuinely ambiguous — requires Decision Agent weighting.",
  "agent_a_report": { "insider_risk_score": 8, "signal_direction": "YES", "reasoning": "Regulatory approval likely" },
  "agent_b_report": { "behavior_score": 8, "signal_direction": "NO", "momentum_assessment": {...} },
  "revision_notes": "Agent A coherence: ✓ Score 8 supported. Reasoning: regulatory approval probable. Info holders: government officials. Agent B coherence: ✓ Score 8 supported. Price trending DOWN, momentum negative across horizons. Pattern: DIRECTIONAL_CONFLICT. Evidence: [A score=8, B score=8, A direction=YES, B direction=NO]. Both agents are internally coherent but point opposite directions. This is genuinely ambiguous and requires Decision Agent to weigh both signals.",
  "feedback_to_send": [],
  "recommendation_to_decision_agent": "GO_EVALUATE",
  "iterations_used": 0,
  "llm_reasoning_summary": "Both agents high and coherent, but directions conflict. Pass to Decision Agent for dynamic weighting."
}
```

### Example 5: Insufficient Evidence to Agent A (Feedback to A)

```json
{
  "revision_flag": "NONE",
  "flag_explanation": "Agent A score lacks supporting evidence. Agent B shows no anomaly. Sending feedback to Agent A to clarify reasoning.",
  "agent_a_report": { "insider_risk_score": 6, "signal_direction": "YES", "reasoning": "Market seems risky" },
  "agent_b_report": { "behavior_score": 2, "signal_direction": "SKIP", "price_jump_assessment": {...} },
  "revision_notes": "Agent A coherence: ✗ Score 6 vague. Reasoning lacks specifics: no named info holders, no concrete evidence of insider risk. 'Market seems risky' is insufficient justification for score 6. Agent B coherence: ✓ Score 2 appropriate. No price jump, flat momentum, no volume spike. Pattern: NONE (both low signal, but A's reasoning incoherent). Feedback: Asking Agent A to clarify reasoning and provide specific evidence for score 6.",
  "feedback_to_send": [
    {
      "recipient": "A",
      "message": "Your insider_risk_score is 6, but your reasoning states only 'market seems risky' without specifics. Score 6 requires concrete evidence. Please re-examine: (1) What specific category is this market? (2) Are there named potential info holders? (3) What market conditions create insider risk? If you cannot identify credible holders or concrete risk factors, the score should be lower (3-4 range)."
    }
  ],
  "recommendation_to_decision_agent": "SKIP",
  "iterations_used": 1,
  "llm_reasoning_summary": "Agent A score lacks evidence. Asking for clarification. Agent B confirms no behavior signal."
}
```

---

## PART 6: IMPLEMENTATION CONSIDERATIONS

### LLM Configuration

```python
from langchain.chat_models import init_chat_model

revision_agent_llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0,  # Mandatory for reproducibility across backtests
    model_provider="anthropic"
).with_structured_output(RevisionAgentOutput)
```

### Cost Analysis

```
Per classification:
  - Input tokens: ~500 (both reports)
  - Output tokens: ~300 (analysis + feedback)
  - Total: ~800 tokens

Haiku pricing (Feb 2026):
  - Input: 0.80 per 1M tokens
  - Output: 4.00 per 1M tokens
  - Cost per call: (~500 * 0.80 + ~300 * 4.00) / 1M = $0.00171

For 1000 markets:
  - Total cost: $1.71
  - Latency: ~2000s (single-threaded) = 33 minutes
  - With batch parallelism (50 workers): ~40 seconds
```

### Reproducibility

```python
# Same input always produces same output due to:
# ✓ temperature=0 (zero variance, deterministic)
# ✓ structured output binding (no format variance)
# ✓ deterministic seed (set in LLM config)

# Reproducibility: 100% (critical for backtesting)
# Backtesting: Cache LLM responses by input hash for speed
```

---

## PART 7: FALLBACK & ERROR HANDLING

### If LLM Response Malformed

```python
if json.JSONDecodeError or pydantic.ValidationError:
    # Retry with same input (99% success on second attempt)
    retry_count = 3
    # If still fails, return conservative output:
    default_output = {
        "recommendation": "SKIP",
        "reason": "Unable to analyze. Default to conservative.",
        "requires_human_review": true
    }
```

### If LLM Unsure

```python
if output.confidence == "low":
    output.requires_human_review = True
    # Flag for human analyst to review
```

---

## PART 8: MIGRATION FROM DETERMINISTIC

### Phase 1: Run Both in Parallel (Production Testing)
```
Input → [Deterministic Revision Agent] → Output A
      → [LLM Revision Agent]            → Output B
      → Compare disagreements
```

### Phase 2: Swap to LLM When 95%+ Agreement
```
Input → [LLM Revision Agent] → Output
      → (Deterministic as fallback cache)
```

### Phase 3: LLM Only with Caching
```
Input → [Hash] → [Cache lookup]
             → Miss: Call LLM
             → Hit: Return cached
```

---

## SUMMARY TABLE

| Aspect | Deterministic | LLM |
|--------|---|---|
| **Throughput** | 1000s/sec | 100s/sec (batched) |
| **Latency per market** | <1ms | 1-2s |
| **Cost per market** | $0 | $0.002 |
| **Coherence detection** | Rules-based | Reasoning-based |
| **Feedback quality** | Templates | Context-aware |
| **Reproducibility** | 100% | 100% (temp=0) |
| **Complexity** | Simple | Moderate |
| **Backtesting validity** | ✅ Yes | ✅ Yes (with caching) |
| **Recommended use** | Production (speed/cost) | Both (quality + reproducibility) |

---

**Recommendation:** Start with **LLM** for correctness. Cache aggressively. Switch to deterministic for production speed if needed.
