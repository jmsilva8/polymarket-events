# Insider Alpha — Agent B Full Specification
## Multi-Agent Pipeline: Implementation & Integration Guide

**Who should read this:**
| Role | Sections to read |
|---|---|
| Agent B implementer | 1, 2, 3, 9 |
| Revision Agent owner | 1, 4, 5, 8, 9 |
| Decision Agent owner | 1, 4, 6, 8, 9 |
| GitHub owner (orchestrator) | 1, 4, 7, 8, 9 |

Read Section 1 regardless of your role — it defines principles that affect all integration points.

---

---

# SECTION 1 — Core Principles
### Non-negotiable design decisions that affect all roles

**1. Agent B is blind.**
It receives price/volume timeseries and timestamps only. It does not receive market title, description, category, platform, or any text about what the bet is. This is intentional and must not be changed by any team member.

**2. Agent B calls no external APIs.**
It works exclusively with data delivered in its input package. If data is missing, it degrades gracefully — it does not fetch anything.

**3. No forward-looking bias.**
All price and volume data passed to Agent B must be truncated at evaluation date T. This is enforced by the orchestrator (GitHub owner), not by Agent B itself.

**4. Tools are pure Python functions.**
Deterministic: same inputs always produce same outputs. No LLM calls inside tools.

**5. The LLM receives only tool outputs.**
It never sees raw price data, raw volume data, or any market metadata.

**6. temperature=0 on all LLM calls.**
Mandatory for reproducibility across backtest runs.

**7. behavior_score is an integer (1–10).**
Same scale as Agent A's insider_score. No floats — false precision.

**8. AgentBParams thresholds are starting defaults only.**
All 8 tool parameters will be tuned via backtesting sweep. Do not hardcode values in logic.

---

---

# SECTION 2 — Agent B: Implementation Spec
### For: Agent B implementer

Create directory: `src/ai_layer/agent_b/` with files: `__init__.py`, `params.py`, `schemas.py`, `tools.py`, `assessment.py`, `prompts.py`, `agent.py`.

Follow the same code style as existing `src/ai_layer/` files.

---

## 2.1 `params.py` — AgentBParams

```python
from dataclasses import dataclass, field

@dataclass
class AgentBParams:
    """
    Configurable thresholds for Agent B tools.
    These are starting defaults — tune via backtesting sweep.
    Same pattern as StrategyParams in src/backtest_engine/strategy.py.
    """
    # Input assessment
    min_price_points: int = 10

    # price_jump_detector
    jump_windows_hours: list[int] = field(default_factory=lambda: [6, 12, 24, 48, 72])
    min_jump_pp: float = 5.0
    sustained_revert_threshold: float = 0.5

    # volume_spike_checker
    spike_threshold_multiplier: float = 3.0

    # momentum_analyzer
    momentum_horizons_hours: list[int] = field(default_factory=lambda: [6, 12, 24, 48])
    min_r_squared: float = 0.6
    min_slope_pp_per_hour: float = 0.2
```

---

## 2.2 `schemas.py` — All Pydantic Models

### Input package

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from src.data_layer.models import PricePoint

@dataclass
class VolumePoint:
    timestamp: datetime
    volume_usd: float

@dataclass
class AgentBInputPackage:
    """
    Everything Agent B receives. No market metadata — numbers only.
    All timeseries must be truncated at evaluation_date by the caller.
    """
    evaluation_date: datetime
    end_date: datetime
    price_history: list[PricePoint]     # sorted chronologically, truncated at T
    current_price: float

    # Volume — provide whatever is available; Agent B adapts
    volume_history: list[VolumePoint] = field(default_factory=list)
    volume_total_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    market_age_days: Optional[float] = None
```

### Input assessment

```python
from pydantic import BaseModel
from typing import Literal

class InputAssessment(BaseModel):
    can_run_price_jump: bool
    can_run_momentum: bool
    can_run_volume: bool
    volume_mode: Literal["timeseries", "approximation", "unavailable"]
    price_point_count: int
    hours_to_close: float
    skipped_tools: list[str]
    data_quality_notes: list[str]
```

### Tool output models

```python
class PriceJumpResult(BaseModel):
    detected: bool
    largest_jump_pp: float
    direction: Literal["UP", "DOWN", "NONE"]
    best_window_hours: int
    from_price: float
    to_price: float
    hours_before_close: float
    is_sustained: bool
    move_shape: Literal["gradual", "sudden", "none"]
    # gradual = price built up over multiple hours (consistent with informed accumulation)
    # sudden  = price moved sharply in < 2 hours (consistent with public news reaction)
    all_windows: list[dict]

class VolumeResult(BaseModel):
    mode: Literal["timeseries", "approximation", "unavailable"]
    spike_detected: bool
    spike_ratio: Optional[float]
    baseline_avg: Optional[float]
    recent_volume: Optional[float]
    hours_before_close: Optional[float]
    pattern: Optional[Literal["burst", "sustained", "flat"]]
    note: str

class MomentumHorizon(BaseModel):
    horizon_hours: int
    slope_pp_per_hour: float
    direction: Literal["UP", "DOWN", "FLAT"]
    r_squared: float
    price_at_start: float
    price_at_end: float

class MomentumResult(BaseModel):
    dominant_direction: Literal["UP", "DOWN", "FLAT", "MIXED"]
    consistency: Literal["trending", "volatile", "reverting", "insufficient_data"]
    acceleration: Literal["increasing", "decreasing", "stable", "unknown"]
    by_horizon: list[MomentumHorizon]

class ConsistencyCheck(BaseModel):
    price_and_momentum_agree: bool
    volume_confirms_direction: bool
    signals_contradictory: bool
    dominant_direction: Literal["UP", "DOWN", "MIXED", "NONE"]
    conflicting_signals: list[str]
```

### Agent B output models

```python
class SignalBreakdown(BaseModel):
    detected: bool
    direction: Literal["UP", "DOWN", "FLAT", "NONE"]
    magnitude: Literal["none", "weak", "moderate", "strong", "extreme"]
    timing_quality: Literal["poor", "acceptable", "good", "excellent"]
    sustained: bool
    weight_assigned: Literal["low", "medium", "high"]
    note: str   # one sentence, numbers only

class AgentBReport(BaseModel):
    """Initial report — produced by agent_b_initial()."""
    signal_direction: Literal["YES", "NO", "SKIP"]
    behavior_score: int                             # 1–10 integer
    confidence: Literal["low", "medium", "high"]
    price_jump_assessment: SignalBreakdown
    volume_assessment: SignalBreakdown
    momentum_assessment: SignalBreakdown
    consistency: ConsistencyCheck
    key_findings: list[str]
    reasoning: str
    context_for_other_agents: str
    # Audit fields
    evaluation_date: str
    tools_run: list[str]
    tools_skipped: list[str]
    data_quality_notes: list[str]

class AgentBRevisionResponse(BaseModel):
    """Produced by agent_b_revise() when Revision Agent sends feedback."""
    tools_re_run: list[str]
    parameter_changes: dict
    finding_changed: bool
    updated_signal_direction: Literal["YES", "NO", "SKIP"]
    updated_behavior_score: int
    updated_confidence: Literal["low", "medium", "high"]
    delta_explanation: str
    final_reasoning: str
    context_for_other_agents: str
```

---

## 2.3 `tools.py` — Pure Python Tool Functions

### `price_jump_detector`

```python
def price_jump_detector(
    price_history: list[PricePoint],
    end_date: datetime,
    evaluation_date: datetime,
    params: AgentBParams,
) -> PriceJumpResult:
```

**Logic:**
- For each `window_hours` in `params.jump_windows_hours`:
  - Find price at `evaluation_date - window_hours` (or earliest point available)
  - Compute delta in percentage points vs price at `evaluation_date`
  - If `abs(delta) >= params.min_jump_pp`: check sustained
- **Sustained check:** after the move started, did price ever pull back more than `sustained_revert_threshold * jump_pp`? If yes → `is_sustained=False`
- **Move shape:** if the jump happened within a 2h sub-window → `"sudden"`; if it built over the full window → `"gradual"`; if not detected → `"none"`
- Return the window with the largest absolute jump
- If no window exceeds `min_jump_pp`: `detected=False`

---

### `volume_spike_checker`

```python
def volume_spike_checker(
    package: AgentBInputPackage,
    volume_mode: Literal["timeseries", "approximation"],
    params: AgentBParams,
) -> VolumeResult:
```

**Timeseries mode** (volume_history available, len >= 7):
- Baseline = average daily volume from earliest date to `(evaluation_date - 24h)`
- Recent = sum of volume_history in last 24h before evaluation_date
- `spike_ratio = recent / baseline`
- `pattern = "sustained"` if spike persists across multiple periods, `"burst"` if concentrated in one

**Approximation mode** (only volume_total + volume_24h available):
- Baseline = `volume_total_usd / market_age_days`
- Recent = `volume_24h_usd`
- `spike_ratio = recent / baseline`
- `pattern = None` (cannot determine from single snapshot)
- Note limitation clearly in the `note` field

`spike_detected = True` if `spike_ratio >= params.spike_threshold_multiplier`

---

### `momentum_analyzer`

```python
def momentum_analyzer(
    price_history: list[PricePoint],
    end_date: datetime,
    evaluation_date: datetime,
    params: AgentBParams,
) -> MomentumResult:
```

**Logic:**
- For each `horizon_hours` in `params.momentum_horizons_hours`:
  - Take all price points from `(evaluation_date - horizon_hours)` to `evaluation_date`
  - Fit linear regression using `numpy.polyfit` (do not import scipy)
  - `slope` = pp per hour; `r_squared` = goodness of fit
  - `direction`: UP if slope >= min_slope_pp_per_hour; DOWN if <= -min_slope_pp_per_hour; FLAT otherwise
  - Only report a direction if `r_squared >= params.min_r_squared`
- `dominant_direction`: direction shared by majority of horizons
- `consistency`:
  - `"trending"`: all horizons agree, r² consistently high
  - `"volatile"`: direction changes across horizons
  - `"reverting"`: short horizons point opposite to long horizons
  - `"insufficient_data"`: fewer than 2 valid horizons
- `acceleration`: compare slope of shortest vs longest valid horizon

---

### `check_consistency`

```python
def check_consistency(
    price_result: PriceJumpResult,
    volume_result: VolumeResult,
    momentum_result: MomentumResult,
) -> ConsistencyCheck:
```

Pure Python — no LLM. Checks whether the three tool outputs directionally agree.
- `price_and_momentum_agree`: both detected and point the same direction
- `volume_confirms_direction`: spike detected AND price jump detected in same direction
- `signals_contradictory`: price direction != momentum dominant_direction (both with sufficient confidence)

---

## 2.4 `assessment.py` — Input Assessment

```python
def assess_inputs(
    package: AgentBInputPackage,
    params: AgentBParams,
) -> InputAssessment:
    """
    Inspect the input package and determine which tools can run
    and in what mode. Runs in pure Python before any tool or LLM.
    """
    notes = []
    skipped = []

    price_ok = len(package.price_history) >= params.min_price_points
    if not price_ok:
        notes.append(
            f"Only {len(package.price_history)} price points available "
            f"(minimum: {params.min_price_points}). "
            f"Price jump and momentum tools skipped."
        )
        skipped.extend(["price_jump_detector", "momentum_analyzer"])

    if package.volume_history and len(package.volume_history) >= 7:
        volume_mode = "timeseries"
    elif (package.volume_24h_usd is not None
          and package.volume_total_usd is not None
          and package.market_age_days is not None):
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
```

---

## 2.5 `prompts.py` — LLM Prompts

### System prompt

```python
AGENT_B_SYSTEM_PROMPT = """
You are a quantitative signal analyst for a prediction market trading system.

You receive structured outputs from mathematical tools that analyzed the price
and volume timeseries of a binary prediction market (YES/NO outcome).

CRITICAL CONSTRAINTS:
- You do not know what the market is about. Do not speculate about it.
- Do not use any external knowledge about events, people, or outcomes.
- Reason exclusively from the numbers provided.
- If you reference anything not in the tool outputs, your response is invalid.

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
1–2:  No meaningful signals. Price flat, volume normal, momentum absent.
3–4:  Weak. Minor movement within normal variance. Low confidence.
5–6:  Moderate. One meaningful signal present, not confirmed by others.
7–8:  Strong. Two or more signals agree in direction. Sustained movement.
9–10: Very strong. All signals converge, large magnitude, late timing, sustained.
      Reserve for exceptional cases only — do not inflate.

MANDATORY RULES:
- Volume spike with no directional price confirmation → SKIP
- Price jump that is NOT sustained (is_sustained=false) → reduce score by at least 2
- Contradictory signals (price UP, momentum DOWN) → SKIP regardless of magnitude
- Signals far from close (>72h) carry less weight than signals within 24h of close
- If tools were skipped due to missing data: reduce confidence, not score
- Always report move_shape in your assessment — it is relevant to downstream agents

Respond in the exact JSON schema provided.
"""
```

### User prompt builder

```python
def build_agent_b_prompt(
    assessment: InputAssessment,
    price_result: Optional[PriceJumpResult],
    volume_result: Optional[VolumeResult],
    momentum_result: Optional[MomentumResult],
    consistency: Optional[ConsistencyCheck],
    package: AgentBInputPackage,
) -> str:
    """
    Assemble the user message for the Agent B LLM call.

    Include:
    - Market context: current_price, hours_to_close, price_point_count (numbers only)
    - Data quality notes from assessment
    - Tool outputs serialized as JSON
    - Consistency check result

    DO NOT include: market_id, title, description, platform, category,
    or any text field from the market. Agent B must remain blind.
    """
    import json

    sections = []

    sections.append(f"""MARKET CONTEXT (numbers only):
  Current price:     {package.current_price:.4f}
  Hours to close:    {assessment.hours_to_close:.1f}
  Price data points: {assessment.price_point_count}""")

    if assessment.data_quality_notes:
        sections.append("DATA QUALITY NOTES:\n" +
                        "\n".join(f"  - {n}" for n in assessment.data_quality_notes))

    if price_result:
        sections.append("PRICE JUMP TOOL OUTPUT:\n" +
                        json.dumps(price_result.model_dump(), indent=2))

    if volume_result:
        sections.append("VOLUME TOOL OUTPUT:\n" +
                        json.dumps(volume_result.model_dump(), indent=2))

    if momentum_result:
        sections.append("MOMENTUM TOOL OUTPUT:\n" +
                        json.dumps(momentum_result.model_dump(), indent=2))

    if consistency:
        sections.append("CONSISTENCY CHECK:\n" +
                        json.dumps(consistency.model_dump(), indent=2))

    if assessment.skipped_tools:
        sections.append("TOOLS SKIPPED (insufficient data):\n" +
                        "\n".join(f"  - {t}" for t in assessment.skipped_tools))

    return "\n\n".join(sections)
```

### Revision prompt builder

```python
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

    Structure:
    1. Original AgentBReport (so LLM remembers its previous output)
    2. Revision Agent feedback verbatim
    3. All original tool outputs (so LLM can re-examine them)
    4. Instruction: if re-running a tool with different parameters,
       state what changed and why. If conclusion unchanged, explain why.
    """
    import json

    return f"""YOUR PREVIOUS REPORT:
{json.dumps(original_report.model_dump(), indent=2)}

REVISION AGENT FEEDBACK:
{revision_feedback}

ORIGINAL TOOL OUTPUTS (for re-examination):
{build_agent_b_prompt(assessment, price_result, volume_result, None, None, package)}

INSTRUCTION:
If the feedback identifies a specific issue you can address by re-examining
the tool outputs above with different parameters, do so and explain:
  - Which tool you re-examined
  - What parameter you changed and why
  - Whether your conclusion changed
  - Updated score, direction, and confidence if changed

If your original conclusion holds under scrutiny, explain why the
original reasoning is correct despite the feedback.

Do not speculate about the market content. Reason only from numbers.
"""
```

---

## 2.6 `agent.py` — Entry Points

```python
def agent_b_initial(
    package: AgentBInputPackage,
    params: AgentBParams,
    llm,
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

def agent_b_revise(
    original_report: AgentBReport,
    revision_feedback: str,
    package: AgentBInputPackage,
    params: AgentBParams,
    llm,
) -> AgentBRevisionResponse:
    """
    Called when Revision Agent sends targeted feedback to Agent B.
    Single LLM call — the loop is managed externally (max 5 iterations total).

    If the LLM determines a tool needs re-running with different parameters:
    - Call the tool in Python within this function with the new params
    - Pass updated results back into the same LLM call via the prompt
    - Do NOT make a second LLM call
    """
```

**LLM config:**
- Model: `claude-haiku-4-5` (primary). Use same initialization pattern as `src/ai_layer/classifier.py`
- temperature: 0
- Structured output: `.with_structured_output(AgentBReport)` / `.with_structured_output(AgentBRevisionResponse)`
- No tools bound to the LLM

---

## 2.7 Testing Requirements

Create `tests/test_agent_b_tools.py`:

- `price_jump_detector`: no jump, jump below threshold, large sustained jump, large reverting jump, sudden vs gradual shape
- `volume_spike_checker`: timeseries mode, approximation mode, unavailable mode
- `momentum_analyzer`: trending up, trending down, flat, volatile, reverting, insufficient data
- `assess_inputs`: all combinations of missing/present volume fields
- **Determinism check**: same inputs → identical outputs on repeated calls for all tools

Do not test the LLM call. Test tools and assessment in isolation only.

---

---

# SECTION 3 — How Agent B Works: Worked Example
### For: All roles — read this to understand what Agent B produces and why

## The raw Polymarket data

When price history is fetched for a market, each data point is:
```json
{"t": 1709683200, "p": 0.54}
```
`t` = Unix timestamp, `p` = YES probability (0.0 to 1.0).

## Example price series (human-readable)

```
Market resolves: Mar 7, 9:00am
Evaluation date T: Mar 6, 9:00am  (24h before close)

Mar 5, 12:00pm → 0.54
Mar 5,  3:00pm → 0.55
Mar 5,  6:00pm → 0.54
Mar 5,  9:00pm → 0.55
Mar 6, 12:00am → 0.58
Mar 6,  3:00am → 0.62
Mar 6,  6:00am → 0.68
Mar 6,  9:00am → 0.72   ← T (evaluation date, data truncated here)
```

## What price_jump_detector computes

For the 24h window (Mar 5 9am → Mar 6 9am):
- Price at start: 0.54 → Price at T: 0.72 → **delta: +18pp UP**
- Price moved gradually over 8 periods → `move_shape: "gradual"`
- Price never pulled back after moving → `is_sustained: true`

## What momentum_analyzer computes

Fits a straight line through the price points in each horizon:
```
24h slope: +0.75pp/hour  r²=0.88  direction=UP
12h slope: +0.85pp/hour  r²=0.93  direction=UP
 6h slope: +1.00pp/hour  r²=0.99  direction=UP
```
All horizons agree → `dominant_direction: UP`, `consistency: trending`, `acceleration: increasing`

## What the LLM receives (no raw data — only tool outputs)

```
MARKET CONTEXT:
  Current price:    0.7200
  Hours to close:   24.0
  Price data points: 8

PRICE JUMP TOOL: detected=true, jump=+18pp UP, window=24h,
                 sustained=true, move_shape=gradual, hours_before_close=24.0

MOMENTUM TOOL: dominant_direction=UP, consistency=trending, acceleration=increasing
  6h:  slope=+1.00pp/hr  r²=0.99  UP
  12h: slope=+0.85pp/hr  r²=0.93  UP
  24h: slope=+0.75pp/hr  r²=0.88  UP

CONSISTENCY: price_and_momentum_agree=true, signals_contradictory=false
```

## What the LLM reasons and outputs

```json
{
  "signal_direction": "YES",
  "behavior_score": 8,
  "confidence": "high",
  "reasoning": "Sustained +18pp upward move over 24h, gradual shape (not a
                sudden news spike). Momentum trending UP across all horizons
                with high r² (0.88–0.99) and accelerating near close.
                All signals converge. Score 8.",
  "context_for_other_agents": "Strong, consistent upward signal. Gradual
                               move shape suggests informed accumulation
                               rather than single news event reaction."
}
```

Note: `move_shape: "gradual"` is a key signal for the Revision Agent (see Section 4).

---

---

# SECTION 4 — The False Positive Problem
### For: Revision Agent owner · Decision Agent owner · GitHub owner

## Why false positives happen

Agent B is blind. It cannot distinguish between:

- **Informed accumulation**: someone with non-public info slowly bought YES over 24h → price rose gradually
- **Public news reaction**: a major publication released Oscar predictions → price spiked in 1 hour

Both produce strong Agent B scores. But only the first represents an exploitable edge. If public information caused the move, the market already adjusted — **there is no arbitrage opportunity remaining**.

## How to detect it: the move_shape + agent comparison rule

Agent B now reports `move_shape` in `PriceJumpResult`:

| move_shape | What it suggests |
|---|---|
| `"gradual"` | Price built up over many hours — consistent with informed accumulation |
| `"sudden"` | Price jumped sharply in < 2h — consistent with public news reaction |

This alone is not enough. The definitive signal comes from **comparing both agents**:

```
Agent B behavior_score >= 7   (strong quantitative anomaly)
Agent A insider_score  <= 3   (low insider risk — info is public)
→ Label: PUBLIC_INFO_ADJUSTED
→ Recommendation: SKIP
```

The combination of high Agent B + low Agent A is the false positive signature.

## Additional patterns the Revision Agent must detect

| Pattern | Revision flag | Meaning |
|---|---|---|
| Agent B high + Agent A low | `PUBLIC_INFO_ADJUSTED` | Market already adjusted to public info — no edge |
| Agent B high + Agent A high + directions match | `NONE` | Strong convergent signal — ideal GO setup |
| Agent B SKIP + Agent A high | `PRE_SIGNAL` | Insider risk high, market hasn't moved yet |
| Agent B high + `is_sustained=false` | `REVERSION` | Price jumped but reverted — likely noise |
| Agent B signals contradictory internally | `INTERNAL_CONFLICT` | Send back to Agent B for re-examination |
| Agent B high + Agent A high + directions conflict | `DIRECTIONAL_CONFLICT` | Agents disagree on outcome — SKIP |

---

---

# SECTION 5 — Revision Agent Spec
### For: Revision Agent owner

## What you receive

Both Agent A report and Agent B report (`AgentBReport`) from the parallel fork.

## Your responsibilities

1. Run cross-checks (detect patterns from Section 4)
2. If issues found and max iterations not reached: send targeted feedback to the specific agent
3. Pass structured summary to Decision Agent

## When to send feedback back to Agent B

Only send back for **specific, actionable questions**:

**Good reasons:**
- "Momentum horizons show conflicting directions — re-examine using tighter windows"
- "Volume spike timing and price jump timing don't align — confirm correlation"

**Bad reasons (do not send back):**
- "I'm not sure about this" — not actionable
- "Run your analysis again" — produces identical output

When Agent B responds via `AgentBRevisionResponse`:
- If `finding_changed=false` → Agent B's signal was robust to scrutiny (treat as stronger, not weaker)
- If `finding_changed=true` → use updated score/direction

## For the PUBLIC_INFO_ADJUSTED case specifically

Do NOT send feedback to Agent B. Agent B's output is correct — it found a real anomaly. The issue is the interpretation, which only becomes clear when comparing with Agent A. Make the call yourself and pass it forward as a flag.

## What you pass to the Decision Agent

```json
{
  "revision_flag": "NONE | PUBLIC_INFO_ADJUSTED | PRE_SIGNAL | REVERSION | INTERNAL_CONFLICT | DIRECTIONAL_CONFLICT",
  "flag_explanation": "...",
  "agent_a_report": { ... },
  "agent_b_report": { ... },
  "revision_notes": "...",
  "recommendation_to_decision_agent": "GO_EVALUATE | SKIP | WATCH",
  "iterations_used": 0
}
```

`GO_EVALUATE` → proceed to Decision Agent weighting.
`SKIP` → Revision Agent has already made the call — Decision Agent honours it.
`WATCH` → signal detected but premature.

---

---

# SECTION 6 — Decision Agent Spec
### For: Decision Agent owner

## Step 1: Handle revision flag first

```
IF recommendation == "SKIP":  → output SKIP, use flag_explanation as reason
IF recommendation == "WATCH": → output SKIP, note "pre-signal, no market movement yet"
IF recommendation == "GO_EVALUATE": → proceed to Step 2
```

This handles `PUBLIC_INFO_ADJUSTED` automatically.

## Step 2: Dynamic weighting

| Situation | Agent A weight | Agent B weight |
|---|---|---|
| Agent B `behavior_score >= 7` + `confidence=high` | 40% | 60% |
| Agent B `behavior_score 5–6` | 50% | 50% |
| Agent B `behavior_score <= 4` or `confidence=low` | 70% | 30% |
| Agent B tools mostly skipped | 90% | 10% |

```
weighted_score = (agent_a_score × weight_a) + (agent_b_score × weight_b)
```

State weights explicitly in output — Revision Agent may review.

## Step 3: Directional alignment

```
IF agent_b.signal_direction == "YES" AND agent_a recommends YES → BET YES
IF agent_b.signal_direction == "NO"  AND agent_a recommends NO  → BET NO

IF revision_flag == "DIRECTIONAL_CONFLICT":
    → both agents are high confidence but disagree on direction
    → do NOT SKIP automatically — this is genuinely ambiguous
    → Revision Agent already labelled it GO_EVALUATE
    → use dynamic weighting to resolve: whichever agent has
      higher confidence and stronger supporting evidence gets
      higher weight — weighted score determines direction
    → if weighted_score >= threshold: take direction of higher-weighted agent
    → if weighted_score < threshold: SKIP

IF agent_b.signal_direction == "SKIP":
    → rely on Agent A direction only, reduce weight on B
```

> **Note:** DIRECTIONAL_CONFLICT → GO_EVALUATE follows the Revision Agent
> owner's specification, which prevails on this decision. When both agents
> are high-confidence in opposite directions, the Decision Agent is better
> positioned to resolve the ambiguity via dynamic weighting than to blanket SKIP
> two valid high-confidence signals.

## Step 4: GO / NO-GO

```
IF weighted_score >= threshold AND direction confirmed → GO
ELSE                                                  → SKIP
```

Threshold starting value: `6.5` (calibrate via backtesting sweep — TBD).

## Your output schema

```json
{
  "decision": "GO | SKIP",
  "bet_direction": "YES | NO | null",
  "scoring": {
    "agent_a_score": 8,
    "agent_b_score": 8,
    "weight_a": 0.40,
    "weight_b": 0.60,
    "weighted_score": 8.0,
    "weighting_rationale": "..."
  },
  "revision_flag_applied": "NONE",
  "reasoning": "...",
  "recommendation": {
    "bet": "YES",
    "risk_grade": 8,
    "agent_a_contribution": "...",
    "agent_b_contribution": "...",
    "current_price": 0.72,
    "decision": "INVEST"
  }
}
```

---

---

# SECTION 7 — Integration Guide
### For: GitHub owner

## New graph structure

Current graph (linear):
```
START → load_markets → filter_markets → classify_markets → export_results → END
```

New graph (with multi-agent layer):
```
START
  → load_markets
  → filter_markets
  → classify_markets
  → [PARALLEL FORK]
      → agent_a_node
      → agent_b_node
  → [JOIN]
  → revision_node
  → decision_node
  → export_results
END
```

Use LangGraph's `Send` or fan-out pattern for the parallel fork.

## Critical: enforce the time boundary before calling Agent B

```python
truncated_price_history = [
    pt for pt in market.price_history.data_points
    if pt.timestamp <= evaluation_date
]
```

Agent B does not enforce this itself. Passing post-T data contaminates backtesting.

## Constructing AgentBInputPackage

```python
AgentBInputPackage(
    evaluation_date  = T,
    end_date         = market.end_date,
    price_history    = truncated_price_history,
    current_price    = truncated_price_history[-1].price if truncated_price_history else 0.0,
    volume_history   = [],                    # empty until trades API is added
    volume_total_usd = market.volume,
    volume_24h_usd   = market.volume_24h,
    market_age_days  = (
        (market.end_date - market.start_date).days
        if market.start_date else None
    ),
)
```

Do NOT pass: `market.question`, `market.description`, `market.category`, `market.platform`.

## LangGraph state additions

```python
class GraphState(TypedDict):
    # ... existing fields ...
    agent_b_reports: dict[str, AgentBReport]
    agent_b_revision_responses: dict[str, AgentBRevisionResponse]
    revision_summaries: dict[str, dict]
    decision_outputs: dict[str, dict]
```

## File structure

```
src/ai_layer/agent_b/          ← Agent B implementer
    __init__.py
    params.py
    schemas.py
    tools.py
    assessment.py
    prompts.py
    agent.py

src/ai_layer/revision_agent.py ← Revision Agent owner
src/ai_layer/decision_agent.py ← Decision Agent owner
src/graph.py                   ← GitHub owner (add parallel fork + new nodes)
```

---

---

# SECTION 8 — Signal Interpretation Quick Reference
### For: All roles

| Agent A score | Agent B score | move_shape | Revision flag | Decision |
|---|---|---|---|---|
| High (7–10) | High (7–10) | gradual | NONE | GO — strong convergent signal |
| Low (1–3) | High (7–10) | sudden | PUBLIC_INFO_ADJUSTED | SKIP — market already adjusted |
| Low (1–3) | High (7–10) | gradual | PUBLIC_INFO_ADJUSTED | SKIP — but flag for review |
| High (7–10) | SKIP | any | PRE_SIGNAL | WATCH — market hasn't moved yet |
| High (7–10) | High (7–10) | any | DIRECTIONAL_CONFLICT | GO_EVALUATE — Decision Agent resolves via dynamic weighting |
| High (7–10) | High (7–10) | sudden | Review carefully | move_shape suggests news — Revision Agent must assess |
| Any | High, not sustained | any | REVERSION | SKIP — likely noise |
| Low | Low | any | NONE | SKIP — no signal |

---

---

# SECTION 9 — All Design Decisions
### For: All roles — do not override without team discussion

| Decision | Choice | Reason |
|---|---|---|
| Agent B knows market content? | No | Prevent bias, enforce quantitative-only |
| Agent B calls APIs? | No | Decoupled from data fetching |
| Tool execution | Pure Python, deterministic | Reproducible backtesting |
| LLM role | Synthesize tool outputs only | Never touches raw data |
| LLM temperature | 0 | Reproducibility |
| behavior_score type | int (1–10) | Consistent with Agent A, no false precision |
| First run mode | Deterministic node | Cheap, reproducible |
| Revision response | Single LLM call, tools re-run in Python | Adaptive but controlled |
| Thresholds | AgentBParams dataclass | Tuned later via backtest sweep |
| False positive detection | Revision Agent cross-check | Only visible when comparing both agents |
| PUBLIC_INFO_ADJUSTED handling | Revision Agent labels, Decision Agent SKIPs | No need to re-run Agent B |
| move_shape field | Reported by price_jump_detector | Gradual vs sudden helps distinguish informed vs news-driven |
| GO threshold (Decision Agent) | TBD — starting value 6.5 | Calibrate via backtesting |

---

*For Agent B implementation details: see Sections 2–3.*
*For questions about the overall workflow: see `insider_alpha_workflow.md`.*
