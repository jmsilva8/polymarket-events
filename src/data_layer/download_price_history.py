"""
Download price history for filtered Polymarket markets into SQLite.

Filters applied to the existing CSV export:
  - End date in 2024-2025 (complete closed years)
  - Volume >= $50,000
  - Excludes: tweet counting, daily up/down, crypto, weather, YouTube views, sports

Downloads Yes-token price history only (No = 1 - Yes).
Resumable: skips markets already in SQLite + per-token file cache.

Usage:
    python -m src.data_layer.download_price_history
"""

import csv
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    DATA_DIR,
    EXPORTS_DIR,
    POLYMARKET_CLOB_RATE,
    POLYMARKET_CLOB_BURST,
)
from src.data_layer.polymarket_client import PolymarketClient
from src.data_layer.rate_limiter import TokenBucketRateLimiter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Filter settings ─────────────────────────────────────────────────
MIN_VOLUME = 50_000
VALID_END_YEARS = ("2024", "2025")

# ── Content exclusion patterns ──────────────────────────────────────
_EXCLUDE_TAGS = {"Tweet Markets", "Mentions"}
_EXCLUDE_PATTERNS = [
    re.compile(r"(up or down|opens up or down)\s+on", re.I),
    re.compile(r"close (above|below)\s+\$?\d", re.I),
    re.compile(
        r"\b(bitcoin|btc|eth\b|ethereum|solana|sol\b|doge|crypto|token|memecoin)\b",
        re.I,
    ),
    re.compile(r"(temperature increase|temperature decrease|weather)", re.I),
    re.compile(r"\d+[mk]?\s*(million\s+)?(views|subscribers)", re.I),
    re.compile(r"\b(nba|nfl|mlb|nhl|premier league|super bowl|ufc|mma)\b", re.I),
]


def should_exclude(question: str, tags: str) -> bool:
    """Return True if market should be excluded from price history download."""
    for tag in _EXCLUDE_TAGS:
        if tag in tags:
            return True
    for pat in _EXCLUDE_PATTERNS:
        if pat.search(question):
            return True
    return False


def load_and_filter_markets(csv_path: Path) -> list[dict]:
    """Load markets from CSV and apply all filters."""
    markets = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Year filter on end_date
            end_date = row.get("end_date", "")
            if end_date[:4] not in VALID_END_YEARS:
                continue
            # Volume filter
            volume = float(row.get("volume", 0) or 0)
            if volume < MIN_VOLUME:
                continue
            # Content filter
            question = row.get("question", "")
            tags = row.get("tags", "")
            if should_exclude(question, tags):
                continue
            # Must have CLOB token ID
            clob_ids = row.get("clob_token_ids", "")
            if not clob_ids:
                continue
            markets.append(row)
    return markets


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with markets and price_history tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id TEXT PRIMARY KEY,
            event_id TEXT,
            question TEXT,
            event_title TEXT,
            volume REAL,
            start_date TEXT,
            end_date TEXT,
            tags TEXT,
            clob_token_id_yes TEXT,
            resolved_yes INTEGER,
            price_points INTEGER DEFAULT 0,
            downloaded_at TEXT
        );

        CREATE TABLE IF NOT EXISTS price_history (
            market_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            price REAL NOT NULL,
            PRIMARY KEY (market_id, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_ph_market
            ON price_history(market_id);
    """)
    conn.commit()
    return conn


def get_downloaded_ids(conn: sqlite3.Connection) -> set[str]:
    """Get set of market_ids that already have price history downloaded."""
    cursor = conn.execute(
        "SELECT market_id FROM markets WHERE price_points > 0"
    )
    return {row[0] for row in cursor.fetchall()}


def download_all(
    markets: list[dict],
    conn: sqlite3.Connection,
    client: PolymarketClient,
):
    """Download price history for each market and store in SQLite."""
    already_done = get_downloaded_ids(conn)
    to_download = [m for m in markets if m["market_id"] not in already_done]

    logger.info(
        "Markets: %d total filtered, %d already downloaded, %d remaining",
        len(markets), len(already_done), len(to_download),
    )
    if not to_download:
        logger.info("All markets already downloaded!")
        return

    est_seconds = len(to_download) / POLYMARKET_CLOB_RATE
    logger.info(
        "Estimated time: %.0f min (%.1f hrs) at %.1f req/s",
        est_seconds / 60, est_seconds / 3600, POLYMARKET_CLOB_RATE,
    )

    start_time = time.time()
    success = 0
    errors = 0
    total_points = 0

    for i, market in enumerate(to_download):
        market_id = market["market_id"]
        clob_ids = market["clob_token_ids"].split("|")
        yes_token_id = clob_ids[0]  # First token = Yes

        try:
            ph = client.get_price_history(
                market_id=market_id,
                token_id=yes_token_id,
                interval="max",
                fidelity=720,  # 12-hour resolution (best available for closed markets)
            )

            # Resolve boolean
            resolved = market.get("resolved_yes", "")
            resolved_int = None
            if resolved == "True":
                resolved_int = 1
            elif resolved == "False":
                resolved_int = 0

            # Insert market metadata
            conn.execute(
                """INSERT OR REPLACE INTO markets
                   (market_id, event_id, question, event_title, volume,
                    start_date, end_date, tags, clob_token_id_yes,
                    resolved_yes, price_points, downloaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (
                    market_id,
                    market.get("event_id", ""),
                    market.get("question", ""),
                    market.get("event_title", ""),
                    float(market.get("volume", 0)),
                    market.get("start_date", ""),
                    market.get("end_date", ""),
                    market.get("tags", ""),
                    yes_token_id,
                    resolved_int,
                    len(ph.data_points),
                ),
            )

            # Insert price points
            if ph.data_points:
                conn.executemany(
                    """INSERT OR IGNORE INTO price_history
                       (market_id, timestamp, price)
                       VALUES (?, ?, ?)""",
                    [
                        (market_id, int(dp.timestamp.timestamp()), dp.price)
                        for dp in ph.data_points
                    ],
                )
                total_points += len(ph.data_points)

            success += 1

            # Commit every 50 markets
            if success % 50 == 0:
                conn.commit()

        except Exception as e:
            errors += 1
            logger.warning(
                "Error fetching market %s: %s", market_id, str(e)[:120]
            )

        # Progress logging every 100 markets or at the end
        done = i + 1
        if done % 100 == 0 or done == len(to_download):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining_s = (len(to_download) - done) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | %d OK, %d err | "
                "%d pts | %.2f req/s | ETA: %.0f min",
                done, len(to_download), done / len(to_download) * 100,
                success, errors, total_points, rate, remaining_s / 60,
            )

    # Final commit
    conn.commit()
    elapsed = time.time() - start_time
    logger.info(
        "Done! %d markets (%d errors) in %.1f min. %d price points stored.",
        success, errors, elapsed / 60, total_points,
    )


def main():
    csv_path = EXPORTS_DIR / "polymarket_tagged_sample.csv"
    db_path = DATA_DIR / "price_history.db"

    if not csv_path.exists():
        logger.error(
            "CSV not found: %s -- run download_sample.py first.", csv_path
        )
        sys.exit(1)

    # Load and filter markets
    logger.info("Loading markets from %s", csv_path)
    markets = load_and_filter_markets(csv_path)
    logger.info(
        "Filtered to %d markets (end_date 2024-2025, >= $%s, content-filtered)",
        len(markets), f"{MIN_VOLUME:,.0f}",
    )

    # Init SQLite
    conn = init_db(db_path)
    logger.info("SQLite database: %s", db_path)

    # Init client with CLOB rate limit (faster than default Gamma rate)
    client = PolymarketClient()
    client.rate_limiter = TokenBucketRateLimiter(
        rate=POLYMARKET_CLOB_RATE, burst=POLYMARKET_CLOB_BURST
    )

    try:
        download_all(markets, conn, client)
    except KeyboardInterrupt:
        conn.commit()
        logger.info("Interrupted! Progress saved. Re-run to resume.")
    finally:
        conn.close()
        client.close()

    # Quick summary
    conn = sqlite3.connect(str(db_path))
    mkt_count = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    pt_count = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
    avg_pts = conn.execute(
        "SELECT AVG(price_points) FROM markets WHERE price_points > 0"
    ).fetchone()[0] or 0
    conn.close()

    logger.info(
        "Database: %d markets, %d price points (avg %.0f pts/market) in %s",
        mkt_count, pt_count, avg_pts, db_path,
    )


if __name__ == "__main__":
    main()
