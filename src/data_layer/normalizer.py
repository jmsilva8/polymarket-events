"""Price normalization across platforms.

Both Polymarket and Kalshi already express prices in the 0–1 range,
but the raw formats differ:
  - Polymarket: float 0.0–1.0  (already implied probability)
  - Kalshi:     dollar string like "0.6500"  (divide by nothing, just parse)
"""


def polymarket_to_probability(price: float) -> float:
    """Polymarket prices are already 0.0–1.0. Clamp for safety."""
    return max(0.0, min(1.0, float(price)))


def kalshi_to_probability(dollar_str: str | float) -> float:
    """
    Kalshi prices are dollar amounts as fixed-point decimals (e.g. "0.6500").
    Already in the 0–1 range. Parse and clamp.
    """
    return max(0.0, min(1.0, float(dollar_str)))
