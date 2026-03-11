"""
build_demo_dataset.py

Creates a self-contained demo dataset with 100 markets so the notebook
can run end-to-end without querying Polymarket.

Outputs (relative to project root):
  demo/data/exports/polymarket_tagged_sample.parquet
  demo/data/price_history.db
  demo/data/backtest/agent_a.jsonl
  demo/data/backtest/agent_b.jsonl
  demo/data/backtest/revision.jsonl
"""

import json
import random
import sqlite3
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "backtest"
DEMO = ROOT / "demo" / "data"

# ---------------------------------------------------------------------------
# 1. Load all cached LLM results
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records[str(d["market_id"])] = d
    return records

print("Loading jsonl files...")
a_all = load_jsonl(DATA / "agent_a.jsonl")
b_all = load_jsonl(DATA / "agent_b.jsonl")
r_all = load_jsonl(DATA / "revision.jsonl")

all_ids = sorted(set(a_all) & set(b_all) & set(r_all))
print(f"  {len(all_ids)} markets in all three files")

# ---------------------------------------------------------------------------
# 2. Select 100 diverse markets
#    - Always include market 523151 (the one hardcoded in the notebook)
#    - Stratify by insider_risk_score (low/mid/high) for diversity
# ---------------------------------------------------------------------------

PINNED = "523151"

scored = []
for mid in all_ids:
    score = a_all[mid].get("insider_risk_score", 5)
    signal = b_all[mid].get("signal_direction", "SKIP")
    behavior = b_all[mid].get("behavior_score", 1)
    flag = r_all[mid].get("revision_flag", "NONE")
    scored.append((mid, score, signal, behavior, flag))

# Sort by score for stratified sampling
scored.sort(key=lambda x: x[1])

low   = [x for x in scored if x[1] <= 3]
mid   = [x for x in scored if 4 <= x[1] <= 6]
high  = [x for x in scored if x[1] >= 7]

def pick(group, n):
    step = max(1, len(group) // n)
    return [group[i][0] for i in range(0, min(len(group), n * step), step)][:n]

selected_ids = set()
selected_ids.update(pick(low,  33))
selected_ids.update(pick(mid,  34))
selected_ids.update(pick(high, 33))
selected_ids.add(PINNED)
selected_ids = list(selected_ids)[:100]

# Make sure PINNED is always included
if PINNED not in selected_ids:
    selected_ids[0] = PINNED

print(f"  Selected {len(selected_ids)} markets (including pinned {PINNED})")

# ---------------------------------------------------------------------------
# 3. Build parquet — synthesize the columns the notebook needs
# ---------------------------------------------------------------------------

CATEGORY_HINTS = {
    "oscar":     "Entertainment",
    "academy":   "Entertainment",
    "grammy":    "Entertainment",
    "emmy":      "Entertainment",
    "billboard": "Entertainment",
    "box office": "Entertainment",
    "film":      "Entertainment",
    "movie":     "Entertainment",
    "album":     "Entertainment",
    "artist":    "Entertainment",
    "song":      "Entertainment",
    "fed":       "Finance",
    "rate":      "Finance",
    "inflation": "Finance",
    "gdp":       "Finance",
    "recession": "Finance",
    "stock":     "Finance",
    "market":    "Finance",
    "s&p":       "Finance",
    "tech":      "Technology",
    "ai":        "Technology",
    "openai":    "Technology",
    "apple":     "Technology",
    "google":    "Technology",
    "microsoft": "Technology",
    "elon":      "Technology",
    "twitter":   "Technology",
    "world cup": "Sports",
    "super bowl": "Sports",
    "nba":       "Sports",
    "nfl":       "Sports",
    "mlb":       "Sports",
    "bitcoin":   "Crypto",
    "ethereum":  "Crypto",
    "crypto":    "Crypto",
    "election":  "Elections",
    "president": "Politics",
    "senate":    "Politics",
    "congress":  "Politics",
    "ukraine":   "Geopolitics",
    "russia":    "Geopolitics",
    "china":     "Geopolitics",
    "taiwan":    "Geopolitics",
    "nato":      "Geopolitics",
    "climate":   "Environment",
    "covid":     "Health",
    "vaccine":   "Health",
    "fda":       "Health",
}

def infer_category(title: str) -> str:
    t = title.lower()
    for keyword, cat in CATEGORY_HINTS.items():
        if keyword in t:
            return cat
    return "Other"


rows = []
for mid in selected_ids:
    rec_a = a_all[mid]
    rec_b = b_all[mid]

    title = rec_a.get("market_title", f"Market {mid}")
    category = infer_category(title)

    # Derive end_date from agent_b evaluation_date (+24h)
    eval_iso = rec_b.get("evaluation_date", "2024-06-01T00:00:00+00:00")
    eval_dt  = datetime.fromisoformat(eval_iso)
    end_dt   = eval_dt + timedelta(hours=24)
    start_dt = end_dt - timedelta(days=random.randint(30, 180))

    # Synthetic but plausible volume
    risk_score = rec_a.get("insider_risk_score", 5)
    behavior   = rec_b.get("behavior_score", 1)
    volume = random.randint(25_000, 2_000_000)

    # yes_price: last price in backtest window (or synthetic)
    yes_price = round(random.uniform(0.15, 0.90), 3)

    tags = "|".join(rec_a.get("info_holders", [])[:3])

    rows.append({
        "market_id":  mid,
        "question":   title,
        "category":   category,
        "status":     "closed",
        "volume":     float(volume),
        "yes_price":  yes_price,
        "end_date":   end_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "start_date": start_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "tags":       tags,
        "platform":   "polymarket",
    })

df = pd.DataFrame(rows)

# Save parquet
exports_dir = DEMO / "exports"
exports_dir.mkdir(parents=True, exist_ok=True)
parquet_path = exports_dir / "polymarket_tagged_sample.parquet"
df.to_parquet(parquet_path, index=False)
print(f"  Wrote parquet: {parquet_path} ({len(df)} rows)")

# ---------------------------------------------------------------------------
# 4. Build price_history.db — synthetic realistic price series
# ---------------------------------------------------------------------------

def generate_price_series(end_dt: datetime, n_points: int = 60,
                          start_price: float = 0.5,
                          outcome: float = None) -> list[tuple]:
    """
    Generate a realistic binary prediction market price series.
    Prices evolve as a bounded random walk and converge toward outcome near end.
    Returns list of (unix_timestamp, price).
    """
    window_hours = 120 + 24  # 5 days before end + 24h buffer
    start_ts = end_dt - timedelta(hours=window_hours)

    if outcome is None:
        outcome = float(random.choice([0.05, 0.95]))  # markets resolve to YES or NO

    prices = []
    price = start_price
    for i in range(n_points):
        frac = i / n_points
        # Drift toward outcome as market approaches resolution
        drift = (outcome - price) * (frac ** 2) * 0.4
        noise = np.random.normal(0, 0.015) * (1 - frac * 0.5)
        price = float(np.clip(price + drift + noise, 0.01, 0.99))
        ts = start_ts + timedelta(hours=window_hours * i / n_points)
        prices.append((int(ts.timestamp()), round(price, 4)))

    return prices


db_path = DEMO / "price_history.db"
db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS price_history (
        market_id TEXT,
        timestamp INTEGER,
        price     REAL
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_market ON price_history(market_id)")
conn.execute("DELETE FROM price_history")

price_rows = []
for row in rows:
    mid     = str(row["market_id"])
    end_dt  = datetime.fromisoformat(row["end_date"])
    s_price = row["yes_price"]
    outcome = random.choice([0.04, 0.96])
    series  = generate_price_series(end_dt, n_points=60,
                                    start_price=s_price, outcome=outcome)
    for ts, price in series:
        price_rows.append((mid, ts, price))

conn.executemany("INSERT INTO price_history VALUES (?, ?, ?)", price_rows)
conn.commit()
conn.close()

print(f"  Wrote price_history.db: {db_path} ({len(price_rows)} price points)")

# ---------------------------------------------------------------------------
# 5. Write trimmed jsonl files
# ---------------------------------------------------------------------------

backtest_dir = DEMO / "backtest"
backtest_dir.mkdir(parents=True, exist_ok=True)

for name, source in [("agent_a", a_all), ("agent_b", b_all), ("revision", r_all)]:
    out_path = backtest_dir / f"{name}.jsonl"
    with open(out_path, "w") as f:
        for mid in selected_ids:
            f.write(json.dumps(source[mid]) + "\n")
    print(f"  Wrote {out_path} ({len(selected_ids)} records)")

# ---------------------------------------------------------------------------
# 6. Copy archetypes.json
# ---------------------------------------------------------------------------
import shutil
archetypes_src = ROOT / "data" / "archetypes.json"
archetypes_dst = DEMO / "archetypes.json"
shutil.copy(archetypes_src, archetypes_dst)
print(f"  Copied archetypes.json")

# ---------------------------------------------------------------------------
# 7. Copy backtest results CSVs (full — they're small and tell the story)
# ---------------------------------------------------------------------------
results_src = ROOT / "data" / "backtest" / "results"
results_dst = DEMO / "backtest" / "results"
results_dst.mkdir(parents=True, exist_ok=True)
for csv_file in results_src.glob("*.csv"):
    shutil.copy(csv_file, results_dst / csv_file.name)
print(f"  Copied {len(list(results_src.glob('*.csv')))} result CSVs")

print("\n✅ Demo dataset ready in demo/data/")
print(f"   Markets: {len(selected_ids)}")
print(f"   Parquet: {parquet_path.stat().st_size // 1024} KB")
print(f"   DB:      {db_path.stat().st_size // 1024} KB")
