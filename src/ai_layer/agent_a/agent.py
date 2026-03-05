"""
Agent A: Insider risk classifier.

Single LLM call per market. Receives market text (title, description, category, tags)
and assigns an insider_risk_score (1-10) based on information asymmetry analysis.

Design decisions (non-negotiable):
  - temperature=0 for backtesting reproducibility
  - Structured output binding via .with_structured_output()
  - Blind to price/volume data — text signals only
  - No archetype lookup — LLM reasons from first principles every time
  - Results cached by market_id for pipeline re-runs
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain.chat_models import init_chat_model

from src.config import CACHE_DIR
from src.ai_layer.agent_a.params import AgentAParams
from src.ai_layer.agent_a.schemas import (
    AgentAInputPackage,
    AgentAReport,
    AgentARevisionResponse,
    _LLMClassificationResponse,
)
from src.ai_layer.agent_a.prompts import (
    AGENT_A_SYSTEM_PROMPT,
    build_agent_a_prompt,
    build_agent_a_revision_prompt,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = CACHE_DIR / "agent_a"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_llm(params: AgentAParams, output_schema):
    return init_chat_model(
        params.model_name,
        temperature=params.temperature,
        model_provider="anthropic",
    ).with_structured_output(output_schema)


def _cache_path(market_id: str) -> Path:
    return _CACHE_DIR / f"{market_id}.json"


def _load_cached(market_id: str) -> Optional[AgentAReport]:
    path = _cache_path(market_id)
    if path.exists():
        try:
            return AgentAReport(**json.loads(path.read_text()))
        except Exception:
            return None
    return None


def _save_cached(report: AgentAReport) -> None:
    _cache_path(report.market_id).write_text(report.model_dump_json(indent=2))


def agent_a_initial(
    package: AgentAInputPackage,
    params: AgentAParams,
    llm=None,
) -> AgentAReport:
    """
    Run Agent A on a single market.

    Returns AgentAReport with insider_risk_score (1-10), confidence,
    reasoning, info_holders, and leak_vectors.
    """
    if params.cache_enabled:
        cached = _load_cached(package.market_id)
        if cached:
            logger.debug("Agent A cache hit: %s", package.market_id)
            return cached

    if llm is None:
        llm = _get_llm(params, _LLMClassificationResponse)

    user_prompt = build_agent_a_prompt(package)

    parsed: _LLMClassificationResponse = llm.invoke([
        {"role": "system", "content": AGENT_A_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    report = AgentAReport(
        market_id=package.market_id,
        market_title=package.question,
        platform=package.platform,
        insider_risk_score=parsed.insider_risk_score,
        confidence=parsed.confidence,
        reasoning=parsed.reasoning,
        info_holders=parsed.info_holders,
        leak_vectors=parsed.leak_vectors,
        model_used=params.model_name,
    )

    if params.cache_enabled:
        _save_cached(report)

    logger.debug(
        "Agent A: %s → score=%d confidence=%s",
        package.market_id, report.insider_risk_score, report.confidence,
    )
    return report


def agent_a_revise(
    original_report: AgentAReport,
    revision_feedback: str,
    package: AgentAInputPackage,
    params: AgentAParams,
    llm=None,
) -> AgentARevisionResponse:
    """
    Revise an Agent A report in response to Revision Agent feedback.

    Returns AgentARevisionResponse with updated score, reasoning,
    and a delta explanation of what changed (or why original stands).
    """
    if llm is None:
        llm = _get_llm(params, AgentARevisionResponse)

    user_prompt = build_agent_a_revision_prompt(original_report, revision_feedback, package)

    result: AgentARevisionResponse = llm.invoke([
        {"role": "system", "content": AGENT_A_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    logger.debug(
        "Agent A revision: %s → score=%d finding_changed=%s",
        package.market_id, result.updated_insider_risk_score, result.finding_changed,
    )
    return result
