"""
LangGraph multi-agent pipeline for insider alpha detection.

Graph topology (multi-agent):

    START
      → load_markets
      → filter_markets
      → run_agent_a ──┐      (insider risk scoring — text-based, sees market content)
      → run_agent_b ──┤      (quantitative signals — blind, price/momentum only)
                      ↓
      → run_revision         (cross-pattern QA + feedback loop, max 2 iterations)
      → run_decision         (final GO/SKIP with Bayesian weighting)
      → export_results
    END

Design notes:
  - Agent A and B are parallel branches (fan-out from filter_markets, fan-in at
    run_revision). They share no state so execution is safe to parallelize.
  - Revision node handles the feedback loop internally (loops back to Agent B
    at most MAX_REVISION_ITERATIONS times per market before passing forward).
  - All LLM calls use temperature=0 for backtesting reproducibility.
  - Volume data is frequently absent for historical markets — Agent B degrades
    gracefully to price/momentum-only analysis.
"""

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.ai_layer.agent_a.agent import agent_a_initial, agent_a_revise
from src.ai_layer.agent_a.params import AgentAParams
from src.ai_layer.agent_a.schemas import AgentAInputPackage, AgentAReport
from src.ai_layer.agent_b.agent import agent_b_initial, agent_b_revise
from src.ai_layer.agent_b.params import AgentBParams
from src.ai_layer.agent_b.schemas import AgentBInputPackage, AgentBReport
from src.ai_layer.decision_agent.agent import decision_agent, decision_agent_deterministic
from src.ai_layer.decision_agent.params import DecisionAgentParams
from src.ai_layer.decision_agent.schemas import (
    DecisionAgentInputPackage,
    DecisionAgentOutput,
)
from src.ai_layer.revision_agent import RevisionAgentOutput, revision_agent
from src.config import EXPORTS_DIR, MIN_VOLUME_USD
from src.data_layer.models import MarketStatus, Platform, Tag, UnifiedMarket

logger = logging.getLogger(__name__)

MAX_REVISION_ITERATIONS = 2


# ── Graph State ────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    """Typed state flowing through the multi-agent classification graph."""

    # Inputs (set at invocation)
    polymarket_csv: str
    kalshi_csv: str
    volume_threshold: float
    primary_model: str          # Agent A LLM (default: gpt-4o-mini)
    agent_b_model: str          # Agent B LLM (default: gpt-4o-mini)
    revision_model: str         # Revision Agent LLM (default: gpt-4o-mini)
    decision_model: str         # Decision Agent LLM for LLM-based version (default: gpt-4o-mini)
    export_path: str
    # When True: also runs LLM-based decision agent for comparison alongside deterministic
    compare_decision_methods: bool
    max_workers: int                # parallel market concurrency (default: 20)

    # Intermediate — populated by load/filter nodes
    all_markets: list           # list[UnifiedMarket]
    filtered_markets: list      # list[UnifiedMarket]

    # Agent outputs — dicts keyed by market_id
    agent_a_reports: dict       # dict[str, dict]  (AgentAReport serialized)
    agent_b_reports: dict       # dict[str, dict]  (AgentBReport serialized)
    revision_outputs: dict      # dict[str, dict]  (RevisionAgentOutput serialized)
    decision_outputs: dict      # dict[str, dict]  (DecisionAgentOutput — deterministic, primary)
    decision_outputs_llm: dict  # dict[str, dict]  (DecisionAgentOutput — LLM-based, comparison)

    # Final
    summary: dict


# ── Node: load_markets ─────────────────────────────────────────────────────────

def load_markets(state: MultiAgentState) -> dict:
    """Load markets from CSV files and reconstruct UnifiedMarket objects."""
    dfs = []
    if state.get("polymarket_csv"):
        dfs.append(pd.read_csv(state["polymarket_csv"]))
    if state.get("kalshi_csv"):
        dfs.append(pd.read_csv(state["kalshi_csv"]))

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    markets = []
    for _, row in combined.iterrows():
        platform = (
            Platform.POLYMARKET
            if row["platform"] == "polymarket"
            else Platform.KALSHI
        )
        tags_list = []
        if pd.notna(row.get("tags", "")):
            for label in str(row["tags"]).split("|"):
                label = label.strip()
                if label:
                    tags_list.append(
                        Tag(id=0, label=label, slug=label.lower().replace(" ", "-"))
                    )

        markets.append(UnifiedMarket(
            platform=platform,
            market_id=str(row["market_id"]),
            condition_id=None,
            slug=str(row.get("slug", "")),
            question=str(row["question"]),
            description="",
            category=str(row.get("category", "")),
            volume=float(row["volume"]),
            tags=tags_list,
            status=MarketStatus.CLOSED,
        ))

    logger.info("Loaded %d markets from CSVs", len(markets))
    return {"all_markets": markets}


# ── Node: filter_markets ───────────────────────────────────────────────────────

def filter_markets(state: MultiAgentState) -> dict:
    """Filter markets by volume threshold."""
    threshold = state.get("volume_threshold", MIN_VOLUME_USD)
    filtered = [m for m in state["all_markets"] if m.volume >= threshold]
    logger.info(
        "Filtered to %d markets (volume >= $%s)",
        len(filtered), f"{threshold:,.0f}",
    )
    return {"filtered_markets": filtered}


# ── Node: run_agent_a ──────────────────────────────────────────────────────────

def run_agent_a(state: MultiAgentState) -> dict:
    """
    Agent A: insider risk scoring via text analysis.
    Sees market title, description, category, tags — blind to price data.
    """
    params = AgentAParams(
        model_name=state.get("primary_model", "gpt-4o-mini"),
        cache_enabled=True,
    )
    max_workers = state.get("max_workers", 20)
    agent_a_reports: dict[str, dict] = {}

    def _process_one_a(market):
        try:
            package = AgentAInputPackage(
                market_id=market.market_id,
                question=market.question,
                description=market.description or "",
                category=market.category or "",
                tags=[t.label for t in market.tags],
                platform=market.platform.value,
                end_date=market.end_date,
            )
            report: AgentAReport = agent_a_initial(package, params)
            return market.market_id, report.model_dump()
        except Exception as e:
            logger.error("Agent A failed for %s: %s", market.market_id, e)
            return market.market_id, {
                "market_id": market.market_id,
                "market_title": market.question,
                "platform": market.platform.value,
                "insider_risk_score": 1,
                "confidence": "low",
                "reasoning": f"Agent A error: {e}",
                "info_holders": [],
                "leak_vectors": [],
                "model_used": "error-fallback",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mid, report in executor.map(_process_one_a, state["filtered_markets"]):
            agent_a_reports[mid] = report

    logger.info("Agent A complete: %d markets", len(agent_a_reports))
    return {"agent_a_reports": agent_a_reports}


# ── Node: run_agent_b ──────────────────────────────────────────────────────────

def run_agent_b(state: MultiAgentState) -> dict:
    """
    Agent B: quantitative signal analysis — fully blind to market content.
    Works on price/momentum only. Volume absent for historical markets is normal.

    Markets loaded from CSV carry no price history — Agent B will run with
    empty price_history and return a graceful SKIP with low score.
    Price history must be attached to UnifiedMarket objects for live data.
    """
    params = AgentBParams()
    max_workers = state.get("max_workers", 20)
    agent_b_reports: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    def _process_one_b(market):
        try:
            price_history = []
            if hasattr(market, "price_history") and market.price_history:
                ph = market.price_history
                price_history = ph.data_points if hasattr(ph, "data_points") else []

            current_price = market.yes_price or 0.5
            end_date = market.end_date or now

            package = AgentBInputPackage(
                evaluation_date=now,
                end_date=end_date,
                price_history=price_history,
                current_price=current_price,
                volume_history=[],
                volume_total_usd=market.volume if market.volume > 0 else None,
                volume_24h_usd=market.volume_24h if market.volume_24h > 0 else None,
                market_age_days=(
                    (end_date - market.start_date).days
                    if market.start_date else None
                ),
            )
            report: AgentBReport = agent_b_initial(package, params)
            return market.market_id, report.model_dump()
        except Exception as e:
            logger.error("Agent B failed for %s: %s", market.market_id, e)
            return market.market_id, {
                "market_id": market.market_id,
                "signal_direction": "SKIP",
                "behavior_score": 1,
                "confidence": "low",
                "reasoning": f"Agent B error: {e}",
                "key_findings": [],
                "context_for_other_agents": "Agent B encountered an error.",
                "tools_run": [],
                "tools_skipped": ["price_jump_detector", "momentum_analyzer", "volume_spike_checker"],
                "data_quality_notes": [],
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mid, report in executor.map(_process_one_b, state["filtered_markets"]):
            agent_b_reports[mid] = report

    logger.info("Agent B complete: %d markets", len(agent_b_reports))
    return {"agent_b_reports": agent_b_reports}


# ── Node: run_revision ─────────────────────────────────────────────────────────

def run_revision(state: MultiAgentState) -> dict:
    """
    Revision Agent: cross-pattern QA with feedback loop.

    For each market:
    1. Run revision_agent(A_report, B_report)
    2. If feedback_to_send contains recipient="B":
       - Call agent_b_revise() and update B report
       - Re-run revision_agent with updated reports
    3. Repeat up to MAX_REVISION_ITERATIONS total
    """
    revision_model = state.get("revision_model", "gpt-4o-mini")
    agent_a_params = AgentAParams(
        model_name=state.get("primary_model", "gpt-4o-mini"),
        cache_enabled=False,  # Don't cache revision outputs — they differ from initial
    )
    agent_b_params = AgentBParams()
    max_workers = state.get("max_workers", 20)
    now = datetime.now(timezone.utc)

    def _process_one_revision(market):
        mid = market.market_id
        a_report = state["agent_a_reports"].get(mid, {})
        b_report = state["agent_b_reports"].get(mid, {})

        iterations = 0
        current_a_report = a_report
        current_b_report = b_report
        a_changed = False
        b_changed = False

        while iterations < MAX_REVISION_ITERATIONS:
            rev_out: RevisionAgentOutput = revision_agent(
                current_a_report, current_b_report, model=revision_model
            )
            iterations += 1
            rev_out.iterations_used = iterations

            a_feedback = [f for f in rev_out.feedback_to_send if f.recipient == "A"]
            b_feedback = [f for f in rev_out.feedback_to_send if f.recipient == "B"]

            if not (a_feedback or b_feedback) or iterations >= MAX_REVISION_ITERATIONS:
                break

            # Route feedback to Agent A
            if a_feedback:
                a_package = AgentAInputPackage(
                    market_id=market.market_id,
                    question=market.question,
                    description=market.description or "",
                    category=market.category or "",
                    tags=[t.label for t in market.tags],
                    platform=market.platform.value,
                    end_date=market.end_date,
                )
                try:
                    original_a = AgentAReport(**current_a_report)
                    a_revision = agent_a_revise(
                        original_report=original_a,
                        revision_feedback=a_feedback[0].message,
                        package=a_package,
                        params=agent_a_params,
                    )
                    current_a_report = {
                        **current_a_report,
                        "insider_risk_score": a_revision.updated_insider_risk_score,
                        "confidence": a_revision.updated_confidence,
                        "reasoning": a_revision.final_reasoning,
                        "info_holders": a_revision.updated_info_holders,
                        "leak_vectors": a_revision.updated_leak_vectors,
                    }
                    a_changed = a_changed or a_revision.finding_changed
                    logger.debug(
                        "Revision loop %d/%d for %s (A): finding_changed=%s new_score=%d",
                        iterations, MAX_REVISION_ITERATIONS, mid,
                        a_revision.finding_changed,
                        a_revision.updated_insider_risk_score,
                    )
                except Exception as e:
                    logger.error("Agent A revision failed for %s: %s", mid, e)

            # Route feedback to Agent B
            if b_feedback:
                price_history = []
                if hasattr(market, "price_history") and market.price_history:
                    ph = market.price_history
                    price_history = ph.data_points if hasattr(ph, "data_points") else []

                end_date = market.end_date or now
                b_package = AgentBInputPackage(
                    evaluation_date=now,
                    end_date=end_date,
                    price_history=price_history,
                    current_price=market.yes_price or 0.5,
                    volume_history=[],
                    volume_total_usd=market.volume if market.volume > 0 else None,
                    volume_24h_usd=market.volume_24h if market.volume_24h > 0 else None,
                    market_age_days=(
                        (end_date - market.start_date).days
                        if market.start_date else None
                    ),
                )
                try:
                    original_b = AgentBReport(**current_b_report)
                    b_revision = agent_b_revise(
                        original_report=original_b,
                        revision_feedback=b_feedback[0].message,
                        package=b_package,
                        params=agent_b_params,
                    )
                    current_b_report = {
                        **current_b_report,
                        "signal_direction": b_revision.updated_signal_direction,
                        "behavior_score": b_revision.updated_behavior_score,
                        "confidence": b_revision.updated_confidence,
                        "reasoning": b_revision.final_reasoning,
                        "context_for_other_agents": b_revision.context_for_other_agents,
                    }
                    b_changed = b_changed or b_revision.finding_changed
                    logger.debug(
                        "Revision loop %d/%d for %s (B): finding_changed=%s new_score=%d",
                        iterations, MAX_REVISION_ITERATIONS, mid,
                        b_revision.finding_changed,
                        b_revision.updated_behavior_score,
                    )
                except Exception as e:
                    logger.error("Agent B revision failed for %s: %s", mid, e)
                    break

        rev_out.agent_a_report = current_a_report
        rev_out.agent_b_report = current_b_report

        logger.info(
            "[revision] market_id=%s  flag=%s  a_changed=%s  b_changed=%s  iters=%d",
            mid, rev_out.revision_flag, a_changed, b_changed, rev_out.iterations_used,
        )
        return mid, rev_out.model_dump(), {
            "flag": rev_out.revision_flag,
            "a_changed": a_changed,
            "b_changed": b_changed,
            "iters": rev_out.iterations_used,
        }

    revision_outputs: dict[str, dict] = {}
    revision_stats: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mid, output, stat in executor.map(_process_one_revision, state["filtered_markets"]):
            revision_outputs[mid] = output
            revision_stats.append(stat)

    # Aggregate summary
    flag_counts = Counter(s["flag"] for s in revision_stats)
    n = len(revision_stats)
    avg_iters = sum(s["iters"] for s in revision_stats) / n if n else 0
    logger.info(
        "[revision] summary: %d markets | flags: %s | a_changed=%d  b_changed=%d  avg_iters=%.1f",
        n,
        "  ".join(f"{k}={v}" for k, v in flag_counts.most_common()),
        sum(s["a_changed"] for s in revision_stats),
        sum(s["b_changed"] for s in revision_stats),
        avg_iters,
    )
    return {"revision_outputs": revision_outputs}


# ── Node: run_decision ─────────────────────────────────────────────────────────

def run_decision(state: MultiAgentState) -> dict:
    """
    Decision Agent: deterministic GO/SKIP (primary) + optional LLM comparison.

    Always runs decision_agent_deterministic() — this is what backtesting sweeps.
    Also runs the LLM-based decision_agent() if compare_decision_methods=True.
    """
    det_params = DecisionAgentParams(
        model_name=state.get("decision_model", "gpt-4o-mini")
    )
    max_workers = state.get("max_workers", 20)
    now = datetime.now(timezone.utc)
    compare = state.get("compare_decision_methods", False)

    def _error_fallback(mid, rev, e):
        return {
            "market_id": mid,
            "decision": "SKIP",
            "bet_direction": "null",
            "full_reasoning": f"Decision Agent error: {e}",
            "revision_flag_applied": rev.get("revision_flag", "NONE"),
            "recommendation": {"action": "PASS"},
            "evaluation_date": now.isoformat(),
        }

    def _process_one_decision(market):
        mid = market.market_id
        rev = state["revision_outputs"].get(mid, {})
        det_result = None
        llm_result = None

        try:
            package = DecisionAgentInputPackage(
                revision_flag=rev.get("revision_flag", "NONE"),
                flag_explanation=rev.get("flag_explanation", ""),
                agent_a_report=rev.get("agent_a_report", state["agent_a_reports"].get(mid, {})),
                agent_b_report=rev.get("agent_b_report", state["agent_b_reports"].get(mid, {})),
                revision_notes=rev.get("revision_notes", ""),
                recommendation_to_decision_agent=rev.get(
                    "recommendation_to_decision_agent", "SKIP"
                ),
                current_market_price=market.yes_price or 0.5,
                evaluation_date=now,
                end_date=market.end_date or now,
                market_id=mid,
            )

            det_result = decision_agent_deterministic(package, det_params).model_dump()

            if compare:
                try:
                    llm_result = decision_agent(package, det_params).model_dump()
                except Exception as e:
                    logger.error("LLM Decision Agent failed for %s: %s", mid, e)
                    llm_result = _error_fallback(mid, rev, e)

        except Exception as e:
            logger.error("Decision Agent failed for %s: %s", mid, e)
            det_result = _error_fallback(mid, rev, e)

        return mid, det_result, llm_result

    decision_outputs: dict[str, dict] = {}
    decision_outputs_llm: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mid, det_result, llm_result in executor.map(
            _process_one_decision, state["filtered_markets"]
        ):
            decision_outputs[mid] = det_result
            if llm_result is not None:
                decision_outputs_llm[mid] = llm_result

    logger.info(
        "Decision node complete: %d markets (LLM comparison: %s)",
        len(decision_outputs), compare,
    )
    return {
        "decision_outputs": decision_outputs,
        "decision_outputs_llm": decision_outputs_llm,
    }


# ── Node: export_results ───────────────────────────────────────────────────────

def export_results(state: MultiAgentState) -> dict:
    """Export final decisions to CSV and compute summary stats."""
    llm_outputs = state.get("decision_outputs_llm", {})
    compare = bool(llm_outputs)

    rows = []
    for market in state["filtered_markets"]:
        mid = market.market_id
        a = state["agent_a_reports"].get(mid, {})
        b = state["agent_b_reports"].get(mid, {})
        rev = state["revision_outputs"].get(mid, {})
        dec = state["decision_outputs"].get(mid, {})

        row = {
            "market_id": mid,
            "market_question": market.question,
            "platform": market.platform.value,
            "yes_price": market.yes_price,
            # Agent A
            "a_insider_risk_score": a.get("insider_risk_score"),
            "a_confidence": a.get("confidence"),
            "a_info_holders": "|".join(a.get("info_holders", [])),
            # Agent B
            "b_behavior_score": b.get("behavior_score"),
            "b_signal_direction": b.get("signal_direction"),
            "b_confidence": b.get("confidence"),
            "b_tools_skipped": "|".join(b.get("tools_skipped", [])),
            # Revision
            "revision_flag": rev.get("revision_flag"),
            "revision_recommendation": rev.get("recommendation_to_decision_agent"),
            "revision_iterations": rev.get("iterations_used", 0),
            # Decision — deterministic (primary)
            "decision": dec.get("decision"),
            "bet_direction": dec.get("bet_direction"),
            "weighted_score": dec.get("analysis", {}).get("weighted_score"),
            "estimated_edge_pp": dec.get("analysis", {}).get("estimated_edge_pp"),
            "action": dec.get("recommendation", {}).get("action"),
        }

        # Decision — LLM comparison (optional)
        if compare:
            llm = llm_outputs.get(mid, {})
            row["llm_decision"] = llm.get("decision")
            row["llm_bet_direction"] = llm.get("bet_direction")
            row["llm_weighted_score"] = llm.get("analysis", {}).get("weighted_score")
            row["llm_estimated_edge_pp"] = llm.get("analysis", {}).get("estimated_edge_pp")
            row["llm_action"] = llm.get("recommendation", {}).get("action")
            # Flag disagreements
            row["decision_disagree"] = row["decision"] != row["llm_decision"]

        rows.append(row)

    export_path = state.get(
        "export_path", str(EXPORTS_DIR / "multi_agent_decisions.csv")
    )
    df = pd.DataFrame(rows)
    df.to_csv(export_path, index=False)

    decisions = [r["decision"] for r in rows if r["decision"]]
    flags = [r["revision_flag"] for r in rows if r["revision_flag"]]
    go_rows = [r for r in rows if r["decision"] == "GO"]
    top_opportunities = sorted(
        go_rows,
        key=lambda r: r.get("weighted_score") or 0,
        reverse=True,
    )[:10]

    summary = {
        "total_markets": len(rows),
        "decision_distribution": dict(Counter(decisions).most_common()),
        "revision_flag_distribution": dict(Counter(flags).most_common()),
        "go_count": decisions.count("GO"),
        "skip_count": decisions.count("SKIP"),
        "top_opportunities": [
            {
                "market_id": r["market_id"],
                "question": r["market_question"][:80],
                "bet": r["bet_direction"],
                "yes_price": r["yes_price"],
                "weighted_score": r["weighted_score"],
                "edge_pp": r["estimated_edge_pp"],
            }
            for r in top_opportunities
        ],
        "export_path": export_path,
    }

    if compare:
        disagree_count = sum(1 for r in rows if r.get("decision_disagree"))
        summary["llm_comparison"] = {
            "llm_go_count": sum(1 for r in rows if r.get("llm_decision") == "GO"),
            "llm_skip_count": sum(1 for r in rows if r.get("llm_decision") == "SKIP"),
            "disagreements": disagree_count,
            "agreement_rate_pct": round(
                100 * (1 - disagree_count / max(len(rows), 1)), 1
            ),
        }

    logger.info(
        "Export complete: %d markets → %d GO / %d SKIP — saved to %s",
        len(rows), summary["go_count"], summary["skip_count"], export_path,
    )
    return {"summary": summary}


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_multi_agent_graph():
    """Build and compile the multi-agent insider alpha detection pipeline."""
    builder = StateGraph(MultiAgentState)

    builder.add_node("load_markets", load_markets)
    builder.add_node("filter_markets", filter_markets)
    builder.add_node("run_agent_a", run_agent_a)
    builder.add_node("run_agent_b", run_agent_b)
    builder.add_node("run_revision", run_revision)
    builder.add_node("run_decision", run_decision)
    builder.add_node("export_results", export_results)

    builder.add_edge(START, "load_markets")
    builder.add_edge("load_markets", "filter_markets")
    # A and B are fully independent — run in parallel, fan-in at run_revision
    builder.add_edge("filter_markets", "run_agent_a")
    builder.add_edge("filter_markets", "run_agent_b")
    builder.add_edge("run_agent_a", "run_revision")
    builder.add_edge("run_agent_b", "run_revision")
    builder.add_edge("run_revision", "run_decision")
    builder.add_edge("run_decision", "export_results")
    builder.add_edge("export_results", END)

    return builder.compile()
