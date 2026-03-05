from dataclasses import dataclass


@dataclass
class AgentAParams:
    model_name: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    cache_enabled: bool = True
