"""Minimal config for Decision Agent — the LLM handles all decision logic."""

from dataclasses import dataclass


@dataclass
class DecisionAgentParams:
    model_name: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0  # Mandatory for reproducibility
