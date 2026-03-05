"""
Unit tests for the Decision Agent.

Tests integration and early-exit logic only — no LLM calls.
Covers:
  - Revision Agent flag handling (SKIP/WATCH → immediate SKIP)
  - Invalid price validation
  - Output schema integrity
  - Edge case: current_price at extremes (0.01, 0.99)
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.ai_layer.decision_agent.agent import _extract_decision, _summarize, decision_agent
from src.ai_layer.decision_agent.params import DecisionAgentParams
from src.ai_layer.decision_agent.schemas import DecisionAgentInputPackage, DecisionAgentOutput

T = datetime(2024, 3, 6, 9, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 3, 7, 9, 0, 0, tzinfo=timezone.utc)

PARAMS = DecisionAgentParams()


def make_package(
    recommendation: str = "GO_EVALUATE",
    revision_flag: str = "NONE",
    flag_explanation: str = "No issues detected.",
    current_price: float = 0.60,
    a_score: int = 8,
    b_score: int = 8,
    b_direction: str = "YES",
) -> DecisionAgentInputPackage:
    return DecisionAgentInputPackage(
        revision_flag=revision_flag,
        flag_explanation=flag_explanation,
        agent_a_report={
            "insider_risk_score": a_score,
            "confidence": "high",
            "reasoning": "Strong insider risk.",
            "info_holders": ["pharma execs"],
            "leak_vectors": ["social media"],
        },
        agent_b_report={
            "behavior_score": b_score,
            "signal_direction": b_direction,
            "confidence": "high",
            "key_findings": ["Price jumped +18pp sustained"],
            "reasoning": "Strong price anomaly.",
            "tools_skipped": [],
        },
        revision_notes="Both agents coherent.",
        recommendation_to_decision_agent=recommendation,
        current_market_price=current_price,
        evaluation_date=T,
        end_date=END,
        market_id="test-market-001",
    )


# ── Early exit logic (no LLM call needed) ─────────────────────────────────────

class TestEarlyExitLogic:
    def test_skip_recommendation_returns_skip(self):
        pkg = make_package(recommendation="SKIP", revision_flag="PUBLIC_INFO_ADJUSTED")
        result = decision_agent(pkg, PARAMS)
        assert result.decision == "SKIP"
        assert result.bet_direction == "null"
        assert result.revision_flag_applied == "PUBLIC_INFO_ADJUSTED"

    def test_watch_recommendation_returns_skip_with_watch_action(self):
        pkg = make_package(recommendation="WATCH", revision_flag="PRE_SIGNAL")
        result = decision_agent(pkg, PARAMS)
        assert result.decision == "SKIP"
        assert result.recommendation["action"] == "WATCH"

    def test_invalid_price_below_zero_returns_skip(self):
        pkg = make_package(current_price=-0.1)
        result = decision_agent(pkg, PARAMS)
        assert result.decision == "SKIP"
        assert result.revision_flag_applied == "VALIDATION_ERROR"

    def test_invalid_price_above_one_returns_skip(self):
        pkg = make_package(current_price=1.5)
        result = decision_agent(pkg, PARAMS)
        assert result.decision == "SKIP"

    def test_output_has_market_id(self):
        pkg = make_package(recommendation="SKIP")
        result = decision_agent(pkg, PARAMS)
        assert result.market_id == "test-market-001"

    def test_output_has_evaluation_date(self):
        pkg = make_package(recommendation="SKIP")
        result = decision_agent(pkg, PARAMS)
        assert result.evaluation_date == T.isoformat()


# ── _extract_decision helper ───────────────────────────────────────────────────

class TestExtractDecision:
    def test_extracts_go_from_explicit_label(self):
        text = 'Based on my analysis, decision: GO. bet_direction: YES.'
        result = _extract_decision(text, 0.60)
        assert result["decision"] == "GO"

    def test_extracts_go_from_invest(self):
        text = 'RECOMMEND INVEST on YES. The edge is meaningful.'
        result = _extract_decision(text, 0.60)
        assert result["decision"] == "GO"

    def test_defaults_to_skip_on_no_signal(self):
        text = 'The signals are weak. SKIP is appropriate here.'
        result = _extract_decision(text, 0.60)
        assert result["decision"] == "SKIP"

    def test_extracts_weight_a(self):
        text = 'Weight Agent A at 40%. Weight Agent B at 60%.'
        result = _extract_decision(text, 0.60)
        assert result["weight_a"] == 40
        assert result["weight_b"] == 60

    def test_extracts_weighted_score(self):
        text = 'Weighted score: 8.0. This is strong.'
        result = _extract_decision(text, 0.60)
        assert result["weighted_score"] == 8.0

    def test_extracts_adjusted_probability(self):
        text = 'Adjusted probability: 0.78. Edge = 18pp.'
        result = _extract_decision(text, 0.60)
        assert abs(result["adjusted_probability"] - 0.78) < 0.01

    def test_extracts_adjusted_probability_as_percentage(self):
        text = 'Adjusted probability: 78%. That is meaningful.'
        result = _extract_decision(text, 0.60)
        assert abs(result["adjusted_probability"] - 0.78) < 0.01

    def test_edge_pp_computed(self):
        text = 'Adjusted probability: 0.78.'
        result = _extract_decision(text, 0.60)
        assert abs(result["edge_pp"] - 18.0) < 1.0

    def test_edge_assessment_meaningful(self):
        text = 'The edge is meaningful. GO.'
        result = _extract_decision(text, 0.60)
        assert result["edge_assessment"] == "meaningful"

    def test_edge_assessment_not_meaningful(self):
        text = 'Edge is not meaningful. SKIP.'
        result = _extract_decision(text, 0.60)
        assert result["edge_assessment"] == "not meaningful"

    def test_bet_direction_yes(self):
        text = 'decision: GO. buy YES. Strong signal.'
        result = _extract_decision(text, 0.60)
        assert result["bet_direction"] == "YES"

    def test_bet_direction_no(self):
        text = 'decision: GO. buy NO. Price will fall.'
        result = _extract_decision(text, 0.99)
        assert result["bet_direction"] == "NO"

    def test_determinism(self):
        text = 'Weight Agent A at 40%. Weighted score: 8.0. Adjusted probability: 0.78. GO.'
        r1 = _extract_decision(text, 0.60)
        r2 = _extract_decision(text, 0.60)
        assert r1 == r2


# ── _summarize helper ──────────────────────────────────────────────────────────

class TestSummarize:
    def test_go_summary_contains_invest(self):
        parsed = {
            "decision": "GO", "bet_direction": "YES",
            "edge_pp": 16.0, "weighted_score": 8.0,
        }
        summary = _summarize(parsed)
        assert "INVEST" in summary
        assert "YES" in summary

    def test_skip_summary_contains_pass(self):
        parsed = {
            "decision": "SKIP", "bet_direction": "null",
            "edge_pp": 3.0, "weighted_score": 4.5,
        }
        summary = _summarize(parsed)
        assert "PASS" in summary


# ── LLM call path (mocked) ────────────────────────────────────────────────────

class TestDecisionAgentWithMockedLLM:
    """Smoke-test the full LLM path with a mocked response."""

    def _mock_llm_response(self, text: str):
        mock_response = MagicMock()
        mock_response.content = text
        return mock_response

    def test_go_path_with_mocked_llm(self):
        pkg = make_package(recommendation="GO_EVALUATE", a_score=8, b_score=8)
        llm_text = (
            "Step 1: GO_EVALUATE. Proceed.\n"
            "Step 2: Weight Agent A at 45%. Weight Agent B at 55%.\n"
            "Step 3: Weighted score: 8.0\n"
            "Step 4: Adjusted probability: 0.78\n"
            "Step 5: Edge = 18pp. This is meaningful.\n"
            "Step 7: decision: GO. bet_direction: YES. RECOMMEND INVEST."
        )

        with patch("src.ai_layer.decision_agent.agent.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = self._mock_llm_response(llm_text)
            mock_init.return_value = mock_llm

            result = decision_agent(pkg, PARAMS)

        assert result.decision == "GO"
        assert result.bet_direction == "YES"
        assert result.analysis["agent_a_score"] == 8
        assert result.analysis["agent_b_score"] == 8

    def test_skip_path_high_b_low_a_public_info(self):
        """PUBLIC_INFO_ADJUSTED → early exit, no LLM call."""
        pkg = make_package(
            recommendation="SKIP",
            revision_flag="PUBLIC_INFO_ADJUSTED",
            flag_explanation="Market adjusted to public news.",
            a_score=2,
            b_score=8,
        )
        result = decision_agent(pkg, PARAMS)
        assert result.decision == "SKIP"
        assert result.revision_flag_applied == "PUBLIC_INFO_ADJUSTED"

    def test_llm_failure_returns_skip(self):
        pkg = make_package(recommendation="GO_EVALUATE")
        with patch("src.ai_layer.decision_agent.agent.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("API timeout")
            mock_init.return_value = mock_llm

            result = decision_agent(pkg, PARAMS)

        assert result.decision == "SKIP"
        assert "LLM call failed" in result.full_reasoning
