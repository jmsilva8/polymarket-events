"""Download Polymarket price history for classified markets (score >= 3).

Downloads hourly price data for the YES token of each market.
Caches each response as JSON in data/cache/price_history/.
Skips markets that already have cached data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)

from src.data_layer.polymarket_client import PolymarketClient

MIN_SCORE = 3  # Download for score >= 3 (covers all sweep ranges + control)

# Load classifications
clf = pd.read_csv("data/exports/classifications_gpt4omini.csv")
poly = clf[(clf["platform"] == "polymarket") & (clf["insider_risk_score"] >= MIN_SCORE)]
print(f"Polymarket markets with score >= {MIN_SCORE}: {len(poly)}")

# Load cached events to get clob_token_ids
events_data = json.loads(
    Path("data/cache/events/polymarket_entertainment_closed.json").read_text()
)
market_tokens = {}
for event in events_data:
    for m in event.get("markets", []):
        mid = str(m.get("id", ""))
        tokens_raw = m.get("clobTokenIds", "[]")
        if isinstance(tokens_raw, str):
            tokens = json.loads(tokens_raw)
        else:
            tokens = tokens_raw or []
        market_tokens[mid] = tokens

# Check which already have cached price history
cache_dir = Path("data/cache/price_history")
already_cached = 0
to_download = []
for _, row in poly.iterrows():
    mid = str(row["market_id"])
    tokens = market_tokens.get(mid, [])
    if not tokens:
        continue
    # Use first token (YES outcome)
    token_id = tokens[0]
    cache_path = cache_dir / f"{token_id}_max_60.json"
    if cache_path.exists():
        already_cached += 1
    else:
        to_download.append((mid, token_id, row["market_title"]))

print(f"Already cached: {already_cached}")
print(f"Need to download: {len(to_download)}")

if not to_download:
    print("All price histories already cached!")
    sys.exit(0)

# Estimate time (rate = 0.25 req/sec = 4 sec per request)
est_minutes = len(to_download) * 4 / 60
print(f"Estimated time: ~{est_minutes:.0f} minutes")
print()

# Download
client = PolymarketClient(cache_enabled=True)
successes = 0
failures = 0

try:
    for mid, token_id, title in tqdm(to_download, desc="Downloading price history"):
        try:
            ph = client.get_price_history(
                market_id=mid,
                token_id=token_id,
                interval="max",
                fidelity=60,  # hourly
            )
            successes += 1
            if len(ph.data_points) == 0:
                tqdm.write(f"  [empty] {title[:60]}")
        except Exception as e:
            failures += 1
            tqdm.write(f"  [FAIL] {title[:60]}: {e}")
finally:
    client.close()

print(f"\nDone! Successes: {successes}, Failures: {failures}")
print(f"Total cached: {already_cached + successes}")
