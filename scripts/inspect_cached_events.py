"""Inspect cached events JSON to find clob_token_ids for price history download."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

# Load cached events
data = json.loads(
    Path("data/cache/events/polymarket_all_closed.json").read_text()
)
print(f"Cached events: {len(data)}")

# Build market_id -> clob_token_ids mapping
market_tokens = {}
market_end_dates = {}
for event in data:
    for m in event.get("markets", []):
        mid = str(m.get("id", ""))
        tokens_raw = m.get("clobTokenIds", "[]")
        if isinstance(tokens_raw, str):
            tokens = json.loads(tokens_raw)
        else:
            tokens = tokens_raw or []
        market_tokens[mid] = tokens
        market_end_dates[mid] = m.get("endDate", "")

print(f"Markets with clob_token_ids: {sum(1 for v in market_tokens.values() if v)}")
print(f"Markets without: {sum(1 for v in market_tokens.values() if not v)}")

# Load classifications
clf = pd.read_csv("data/exports/classifications_gpt4omini.csv")
poly = clf[clf["platform"] == "polymarket"]
poly_high = poly[poly["insider_risk_score"] >= 5]

print(f"\nPolymarket markets score >= 5: {len(poly_high)}")

# Check how many have clob_token_ids
has_tokens = 0
missing_tokens = 0
for _, row in poly_high.iterrows():
    mid = str(row["market_id"])
    tokens = market_tokens.get(mid, [])
    if tokens:
        has_tokens += 1
    else:
        missing_tokens += 1

print(f"  With clob_token_ids: {has_tokens}")
print(f"  Missing clob_token_ids: {missing_tokens}")

# Also check score >= 3 (for control group comparison)
poly_mid = poly[poly["insider_risk_score"] >= 3]
has_mid = sum(1 for _, r in poly_mid.iterrows() if market_tokens.get(str(r["market_id"]), []))
print(f"\nPolymarket markets score >= 3: {len(poly_mid)}, with tokens: {has_mid}")
