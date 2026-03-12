"""
build_demo_dataset.py

Creates a self-contained demo dataset with 100 **real** markets so the
notebook can run end-to-end without querying Polymarket or making LLM calls.

All data is sourced from the full cached pipeline outputs:
  - data/exports/polymarket_tagged_sample.parquet  (real market metadata)
  - data/price_history.db                          (real CLOB price history)
  - data/backtest/agent_a.jsonl                    (real LLM classification)
  - data/backtest/agent_b.jsonl                    (real LLM analysis)
  - data/backtest/revision.jsonl                   (real revision results)

Selection strategy:
  - 100 markets chosen for maximum signal diversity (A-high/B-high,
    A-high/B-low, A-low/B-high, both-medium, both-low, etc.)
  - Preference for recognizable, fun, or memeable markets
  - No duplication of nearly-identical markets (e.g. only one Iran strike
    market, one Fed meeting, etc.)
  - Always includes market 523151 (GPT-4.5 release — notebook default)

Outputs (relative to project root):
  demo/data/exports/polymarket_tagged_sample.parquet
  demo/data/price_history.db
  demo/data/backtest/agent_a.jsonl
  demo/data/backtest/agent_b.jsonl
  demo/data/backtest/revision.jsonl
"""

import json
import sqlite3
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEMO = ROOT / "demo" / "data"

# ---------------------------------------------------------------------------
# 1. Load all cached data
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records[str(d["market_id"])] = d
    return records

print("Loading source data...")
a_all = load_jsonl(DATA / "backtest" / "agent_a.jsonl")
b_all = load_jsonl(DATA / "backtest" / "agent_b.jsonl")
r_all = load_jsonl(DATA / "backtest" / "revision.jsonl")

# Real market metadata from parquet
table = pq.read_table(DATA / "exports" / "polymarket_tagged_sample.parquet")
parquet_by_id = {}
for i in range(len(table)):
    mid = str(table.column("market_id")[i].as_py())
    parquet_by_id[mid] = {col: table.column(col)[i].as_py() for col in table.column_names}

# Markets present in ALL sources
all_agent_ids = set(a_all) & set(b_all) & set(r_all)
eligible = all_agent_ids & set(parquet_by_id)

# Real price history — check which markets have enough data points
conn_src = sqlite3.connect(DATA / "price_history.db")
cur = conn_src.cursor()
cur.execute("SELECT market_id, COUNT(*) FROM price_history GROUP BY market_id")
price_counts = {str(r[0]): r[1] for r in cur.fetchall()}

# Require at least 10 price points for meaningful demo
eligible = {mid for mid in eligible if price_counts.get(mid, 0) >= 10}
print(f"  {len(eligible)} markets eligible (all sources + >=10 price points)")

# ---------------------------------------------------------------------------
# 2. Build a scored index for smart selection
# ---------------------------------------------------------------------------

market_index = []
for mid in eligible:
    a = a_all[mid]
    b = b_all[mid]
    r = r_all[mid]
    p = parquet_by_id[mid]

    a_score = a.get("insider_risk_score", 0)
    b_score = b.get("behavior_score", 0)
    b_dir = b.get("signal_direction", "SKIP")
    rev_flag = r.get("revision_flag", "NONE")
    rev_rec = r.get("recommendation_to_decision_agent", "GO_EVALUATE")
    title = a.get("market_title", "")
    volume = float(p.get("volume", 0) or 0)

    # Classify signal profile
    a_level = "high" if a_score >= 7 else ("med" if a_score >= 4 else "low")
    b_level = "high" if b_score >= 7 else ("med" if b_score >= 4 else "low")
    profile = f"{a_level}_{b_level}"

    market_index.append({
        "mid": mid, "title": title, "volume": volume,
        "a_score": a_score, "b_score": b_score, "b_dir": b_dir,
        "rev_flag": rev_flag, "rev_rec": rev_rec, "profile": profile,
        "price_points": price_counts.get(mid, 0),
    })

# ---------------------------------------------------------------------------
# 3. Hand-curated selection for maximum diversity and interest
#
# Target allocation across signal profiles:
#   a_high_b_high  (DIRECTIONAL_CONFLICT):  ~12  - the "hot" markets
#   a_high_b_low   (PRE_SIGNAL / WATCH):    ~12  - insider risk but no move yet
#   a_low_b_high   (PUBLIC_INFO_ADJUSTED):   ~12  - price moved but public info
#   a_med_b_high:                            ~14  - moderate insider + strong signal
#   a_high_b_med:                            ~6   - rare profile
#   a_med_b_med:                             ~10  - balanced / uncertain
#   a_low_b_low:                             ~12  - quiet / no signal
#   a_med_b_low:                             ~14  - typical markets
#   a_low_b_med:                             ~8   - low risk but some movement
#
# Within each bucket we prefer:
#   1. Higher volume (more recognizable)
#   2. More price data points
#   3. Diverse topics (avoid 10 Iran markets)
# ---------------------------------------------------------------------------

PINNED = "523151"  # GPT-4.5 release — always included

# Target counts per profile
TARGETS = {
    "high_high": 12,
    "high_low":  12,
    "low_high":  12,
    "med_high":  14,
    "high_med":  6,
    "med_med":   10,
    "low_low":   12,
    "med_low":   14,
    "low_med":   8,
}

# Group markets by profile
by_profile = {}
for m in market_index:
    by_profile.setdefault(m["profile"], []).append(m)

# Sort each bucket: prefer higher volume, then more price points
for profile in by_profile:
    by_profile[profile].sort(key=lambda x: (-x["volume"], -x["price_points"]))


def deduplicate_topics(markets, max_per_topic=2):
    """
    Avoid picking too many near-identical markets (e.g. 5 Iran strike markets).
    Uses simple keyword clustering.
    """
    topic_keys = [
        "iran", "israel", "fed rate", "fed interest", "fed decrease", "fed increase",
        "fed emergency", "elon tweet", "trump tweet", "tiktok",
        "oscar", "grammy", "emmy", "eurovision", "box office",
        "olympics", "hurricane", "recession", "bitcoin", "ethereum",
        "openai", "gpt-5", "gpt-4", "taylor swift", "north korea",
        "ukraine", "russia", "china", "apple", "nvidia", "tesla",
        "spacex", "netflix", "drone", "bird flu", "pandemic",
    ]
    topic_counts = {}
    result = []
    for m in markets:
        title_lower = m["title"].lower()
        # Find which topic cluster this market belongs to
        matched_topic = None
        for tk in topic_keys:
            if tk in title_lower:
                matched_topic = tk
                break
        if matched_topic:
            count = topic_counts.get(matched_topic, 0)
            if count >= max_per_topic:
                continue
            topic_counts[matched_topic] = count + 1
        result.append(m)
    return result


selected_ids = set()

# Always include pinned market
if PINNED in eligible:
    selected_ids.add(PINNED)

# Fill each profile bucket
for profile, target in TARGETS.items():
    pool = by_profile.get(profile, [])
    pool = deduplicate_topics(pool)
    # Skip already-selected
    pool = [m for m in pool if m["mid"] not in selected_ids]
    for m in pool[:target]:
        selected_ids.add(m["mid"])

# If we're short of 100, fill from the largest remaining markets
if len(selected_ids) < 100:
    remaining = [m for m in market_index if m["mid"] not in selected_ids]
    remaining = deduplicate_topics(remaining)
    remaining.sort(key=lambda x: (-x["volume"], -x["price_points"]))
    for m in remaining:
        if len(selected_ids) >= 100:
            break
        selected_ids.add(m["mid"])

# If we have more than 100 (unlikely), trim
selected_ids = list(selected_ids)[:100]

# Verify pinned is still there
assert PINNED in selected_ids, f"Pinned market {PINNED} was lost during selection"

print(f"  Selected {len(selected_ids)} markets (including pinned {PINNED})")

# Print distribution summary
profile_counts = {}
for mid in selected_ids:
    a_score = a_all[mid].get("insider_risk_score", 0)
    b_score = b_all[mid].get("behavior_score", 0)
    a_level = "high" if a_score >= 7 else ("med" if a_score >= 4 else "low")
    b_level = "high" if b_score >= 7 else ("med" if b_score >= 4 else "low")
    profile = f"A-{a_level}_B-{b_level}"
    profile_counts[profile] = profile_counts.get(profile, 0) + 1

print("\n  Signal profile distribution:")
for p in sorted(profile_counts):
    print(f"    {p}: {profile_counts[p]}")

# Revision flag distribution
flag_counts = {}
for mid in selected_ids:
    flag = r_all[mid].get("revision_flag", "NONE")
    flag_counts[flag] = flag_counts.get(flag, 0) + 1
print("\n  Revision flag distribution:")
for f in sorted(flag_counts):
    print(f"    {f}: {flag_counts[f]}")

# ---------------------------------------------------------------------------
# 4. Build parquet from REAL market metadata
# ---------------------------------------------------------------------------

print("\nBuilding demo parquet...")
columns = table.column_names
demo_rows = {col: [] for col in columns}

for mid in selected_ids:
    row = parquet_by_id[mid]
    for col in columns:
        demo_rows[col].append(row[col])

demo_table = pa.table(demo_rows, schema=table.schema)

exports_dir = DEMO / "exports"
exports_dir.mkdir(parents=True, exist_ok=True)
parquet_path = exports_dir / "polymarket_tagged_sample.parquet"
pq.write_table(demo_table, parquet_path)
print(f"  Wrote parquet: {parquet_path} ({len(selected_ids)} rows)")

# ---------------------------------------------------------------------------
# 5. Build price_history.db from REAL price data
# ---------------------------------------------------------------------------

print("Building demo price_history.db...")
db_path = DEMO / "price_history.db"
db_path.parent.mkdir(parents=True, exist_ok=True)

conn_dst = sqlite3.connect(db_path)
conn_dst.execute("DROP TABLE IF EXISTS price_history")
conn_dst.execute("""
    CREATE TABLE price_history (
        market_id TEXT,
        timestamp INTEGER,
        price     REAL
    )
""")

total_price_rows = 0
for mid in selected_ids:
    cur.execute(
        "SELECT market_id, timestamp, price FROM price_history WHERE market_id = ?",
        (mid,)
    )
    rows = cur.fetchall()
    if not rows:
        # Try numeric form
        cur.execute(
            "SELECT market_id, timestamp, price FROM price_history WHERE market_id = ?",
            (int(mid),)
        )
        rows = cur.fetchall()
    conn_dst.executemany("INSERT INTO price_history VALUES (?, ?, ?)", rows)
    total_price_rows += len(rows)

conn_dst.execute("CREATE INDEX idx_market ON price_history(market_id)")
conn_dst.commit()
conn_dst.close()
conn_src.close()

print(f"  Wrote price_history.db: {db_path} ({total_price_rows} price points)")

# ---------------------------------------------------------------------------
# 6. Write trimmed JSONL files (real LLM results)
# ---------------------------------------------------------------------------

print("Writing demo JSONL files...")
backtest_dir = DEMO / "backtest"
backtest_dir.mkdir(parents=True, exist_ok=True)

for name, source in [("agent_a", a_all), ("agent_b", b_all), ("revision", r_all)]:
    out_path = backtest_dir / f"{name}.jsonl"
    with open(out_path, "w") as f:
        for mid in selected_ids:
            f.write(json.dumps(source[mid]) + "\n")
    print(f"  Wrote {out_path} ({len(selected_ids)} records)")

# ---------------------------------------------------------------------------
# 7. Copy archetypes.json
# ---------------------------------------------------------------------------

archetypes_src = DATA / "archetypes.json"
archetypes_dst = DEMO / "archetypes.json"
if archetypes_src.exists():
    shutil.copy(archetypes_src, archetypes_dst)
    print("  Copied archetypes.json")

# ---------------------------------------------------------------------------
# 8. Copy backtest results CSVs
# ---------------------------------------------------------------------------

results_src = DATA / "backtest" / "results"
results_dst = DEMO / "backtest" / "results"
results_dst.mkdir(parents=True, exist_ok=True)
csv_count = 0
for csv_file in results_src.glob("*.csv"):
    shutil.copy(csv_file, results_dst / csv_file.name)
    csv_count += 1
print(f"  Copied {csv_count} result CSVs")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Demo dataset ready in demo/data/")
print(f"  Markets:      {len(selected_ids)}")
print(f"  Parquet:      {parquet_path.stat().st_size // 1024} KB")
print(f"  Price DB:     {db_path.stat().st_size // 1024} KB ({total_price_rows} rows)")
print(f"  Agent A JSONL: real LLM results")
print(f"  Agent B JSONL: real LLM results")
print(f"  Revision JSONL: real results")
print(f"{'='*60}")

# Print a few sample markets for verification
print("\nSample markets:")
for mid in list(selected_ids)[:10]:
    a = a_all[mid]
    b = b_all[mid]
    r = r_all[mid]
    title = a.get("market_title", "")[:70]
    print(f"  [{mid}] A={a.get('insider_risk_score',0)} "
          f"B={b.get('behavior_score',0)} "
          f"dir={b.get('signal_direction','?')} "
          f"rev={r.get('revision_flag','?')} "
          f"| {title}")
