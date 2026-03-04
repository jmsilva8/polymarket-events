"""Centralized configuration loaded from .env and constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPORTS_DIR = DATA_DIR / "exports"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARCHETYPES_PATH = DATA_DIR / "archetypes.json"

# ── Load .env ──────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

# ── LLM Keys ──────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Kalshi ─────────────────────────────────────────────────────────
KALSHI_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

# ── Polymarket ─────────────────────────────────────────────────────
POLYMARKET_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_BASE_URL = "https://clob.polymarket.com"

# Rate limits: Polymarket ~1000 req/hr, Kalshi 20 req/sec (basic tier)
POLYMARKET_RATE = 0.25  # requests per second (conservative)
POLYMARKET_BURST = 5
KALSHI_RATE = 10.0  # requests per second (conservative vs 20 limit)
KALSHI_BURST = 10

# ── Platform toggles ──────────────────────────────────────────────
POLYMARKET_ENABLED = os.getenv("POLYMARKET_ENABLED", "true").lower() == "true"
KALSHI_ENABLED = os.getenv("KALSHI_ENABLED", "true").lower() == "true"

# ── Filters ────────────────────────────────────────────────────────
MIN_VOLUME_USD = 20_000  # Only classify markets above this volume

# Polymarket event-level tag labels used to identify entertainment/pop-culture
# markets.  These are matched against the tag "label" field (case-insensitive).
# Discovered empirically from the Gamma API /events endpoint.
ENTERTAINMENT_TAG_LABELS = [
    "culture", "movies", "music", "entertainment", "awards",
    "tv", "oscar", "grammy", "emmy", "golden globe",
    "celebrity", "celebrities", "box office", "hollywood",
    "netflix", "streaming", "album", "song", "film",
    "super bowl", "halftime", "reality", "eurovision",
    "spotify", "tiktok", "pop culture", "poker",
]

# ── Ensure directories exist ──────────────────────────────────────
for _dir in [
    CACHE_DIR / "events",
    CACHE_DIR / "markets",
    CACHE_DIR / "price_history",
    CACHE_DIR / "tags",
    EXPORTS_DIR,
    REPORTS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
