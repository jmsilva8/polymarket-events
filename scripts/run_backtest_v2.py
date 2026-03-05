"""
Backtesting pipeline v2 — full A → B → Revision → Decision chain.

Stages:
  1. Load markets from parquet
  2. Filter to markets with sufficient price data in SQLite [end_date-120h, end_date-24h]
  3. Run Agent A (LLM, cached in data/backtest/agent_a.jsonl)
  4. Run Agent B (LLM, cached in data/backtest/agent_b.jsonl)
  5. Run Revision (LLM, cached in data/backtest/revision.jsonl)
  6. Run Decision configs (deterministic, no LLM)
  7. Aggregate results + export to data/backtest/results/

Usage:
    # Full run
    python scripts/run_backtest_v2.py

    # Smoke test (first 50 markets, skip LLM if not cached)
    python scripts/run_backtest_v2.py --limit 50 --skip-llm

    # Re-run decisions only (no LLM stages)
    python scripts/run_backtest_v2.py --decisions-only

    # Add LLM decision variant for GO_EVALUATE markets
    python scripts/run_backtest_v2.py --llm-decision
"""

import argparse
import csv
import json
import logging
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from tqdm import tqdm

from src.config import DATA_DIR, EXPORTS_DIR
from src.data_layer.models import PricePoint
from src.ai_layer.agent_a.agent import agent_a_initial
from src.ai_layer.agent_a.params import AgentAParams
from src.ai_layer.agent_a.schemas import AgentAInputPackage, _LLMClassificationResponse
from src.ai_layer.agent_b.agent import agent_b_initial
from src.ai_layer.agent_b.params import AgentBParams
from src.ai_layer.agent_b.schemas import AgentBInputPackage, AgentBReport, _LLMAgentBResponse
from src.ai_layer.revision_agent import revision_agent_deterministic, RevisionAgentOutput
from src.ai_layer.decision_agent.agent import decision_agent_deterministic, decision_agent
from src.ai_layer.decision_agent.params import DecisionAgentParams
from src.ai_layer.decision_agent.schemas import DecisionAgentInputPackage

import httpx

# ── Paths ─────────────────────────────────────────────────────────────────────

BACKTEST_DIR = DATA_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

AGENT_A_JSONL    = BACKTEST_DIR / "agent_a.jsonl"
AGENT_B_JSONL    = BACKTEST_DIR / "agent_b.jsonl"
REVISION_JSONL   = BACKTEST_DIR / "revision.jsonl"
REV_LOOPS_CSV    = BACKTEST_DIR / "revision_loops.csv"
RESULTS_DIR      = BACKTEST_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH      = DATA_DIR / "price_history.db"
CSV_PATH = EXPORTS_DIR / "polymarket_tagged_sample.csv"

MAX_REVISION_ITERATIONS = 2  # match graph.py

# ── Decision configs ──────────────────────────────────────────────────────────
# Add new configs here — each runs independently on the fixed A/B/Revision outputs.

DECISION_CONFIGS = [
    {"name": "default",          "params": DecisionAgentParams()},
    {"name": "aggressive",       "params": DecisionAgentParams(
        min_a_score=3, min_b_score=3, go_score_threshold=5.0, min_edge_pp=3.0)},
    {"name": "conservative",     "params": DecisionAgentParams(
        min_a_score=6, min_b_score=6, go_score_threshold=7.0)},
    {"name": "a_heavy",          "params": DecisionAgentParams(weight_a=0.6, weight_b=0.4)},
    {"name": "b_heavy",          "params": DecisionAgentParams(weight_a=0.3, weight_b=0.7)},
    {"name": "no_conf_penalty",  "params": DecisionAgentParams(
        confidence_medium=1.0, confidence_low=1.0)},
]

# ── Thread-safe JSONL locks ───────────────────────────────────────────────────

_a_lock        = threading.Lock()
_b_lock        = threading.Lock()
_rev_lock      = threading.Lock()
_rev_loop_lock = threading.Lock()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Cost tracking ─────────────────────────────────────────────────────────────

class CostTracker:
    """Thread-safe cumulative token/cost tracker for OpenAI LLM calls."""

    # gpt-4o-mini pricing ($/1M tokens)
    INPUT_COST_PER_M = 0.15
    OUTPUT_COST_PER_M = 0.60

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._stage_snapshots: dict[str, tuple[int, int, int]] = {}
        self._stage_costs: list[tuple[str, int, float]] = []  # (name, calls, cost)

    def record(self, prompt_tokens: int, completion_tokens: int):
        """Record token usage from one LLM call (thread-safe)."""
        with self._lock:
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            self.calls += 1

    @property
    def cost(self) -> float:
        return (
            self.input_tokens * self.INPUT_COST_PER_M
            + self.output_tokens * self.OUTPUT_COST_PER_M
        ) / 1_000_000

    def snapshot(self, stage: str):
        """Save current counters before a stage begins."""
        self._stage_snapshots[stage] = (self.input_tokens, self.output_tokens, self.calls)

    def log_stage(self, stage: str):
        """Log cost delta for one stage + cumulative total."""
        prev_in, prev_out, prev_calls = self._stage_snapshots.get(stage, (0, 0, 0))
        d_in = self.input_tokens - prev_in
        d_out = self.output_tokens - prev_out
        d_calls = self.calls - prev_calls
        d_cost = (d_in * self.INPUT_COST_PER_M + d_out * self.OUTPUT_COST_PER_M) / 1_000_000
        self._stage_costs.append((stage, d_calls, d_cost))
        logger.info(
            "Cost [%s]: %d calls | %dk in + %dk out tokens | $%.4f stage | $%.4f cumulative",
            stage, d_calls, d_in // 1000, d_out // 1000, d_cost, self.cost,
        )

    def merge_from(self, other: "CostTracker", stage_name: str):
        """Merge another tracker's totals into this one and log the stage."""
        with self._lock:
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens
            self.calls += other.calls
        self._stage_costs.append((stage_name, other.calls, other.cost))

    def print_summary(self):
        """Print a final cost summary table."""
        SEP = "=" * 60
        print(f"\n{SEP}")
        print("LLM COST SUMMARY")
        print("-" * 60)
        for name, calls, cost in self._stage_costs:
            print(f"  {name:<16} {calls:>5} calls | ${cost:.4f}")
        print("-" * 60)
        print(f"  {'TOTAL':<16} {self.calls:>5} calls | ${self.cost:.4f}")
        print(f"{SEP}")


# ── Httpx-based LLM (bypasses OpenAI SDK — hangs on Python 3.14) ─────────────

class HttpxStructuredLLM:
    """
    Lightweight OpenAI chat completion client using httpx directly.

    The OpenAI Python SDK v2.24.0 hangs indefinitely on Python 3.14.
    Raw httpx works fine, so this class bypasses the SDK entirely.

    Provides .invoke(messages) → Pydantic model, matching the interface
    that agents expect from LangChain's .with_structured_output().
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        schema: type,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        timeout: float = 30.0,
        cost_tracker: CostTracker | None = None,
    ):
        import os
        self._schema = schema
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._cost_tracker = cost_tracker
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set")
        # Shared httpx client (thread-safe)
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        # Use json_object mode (avoids strict schema compatibility issues)
        self._response_format = {"type": "json_object"}
        # Build human-readable schema instruction from Pydantic model
        self._schema_instruction = self._build_schema_instruction(schema)

    @staticmethod
    def _build_schema_instruction(schema: type) -> str:
        """Generate readable field descriptions from a Pydantic model."""
        s = schema.model_json_schema()
        lines = ["Respond with a JSON object containing exactly these fields:"]
        for name, prop in s.get("properties", {}).items():
            typ = prop.get("type", "")
            enum = prop.get("enum")
            if enum:
                lines.append(f'- {name}: one of {", ".join(repr(e) for e in enum)}')
            elif typ == "array":
                item_type = prop.get("items", {}).get("type", "string")
                lines.append(f"- {name}: list of {item_type}s")
            elif typ == "integer":
                lines.append(f"- {name}: integer")
            elif typ == "object":
                lines.append(f"- {name}: object (dict)")
            else:
                lines.append(f"- {name}: {typ}")
            # Add description if available
            desc = prop.get("description")
            if desc:
                lines[-1] += f" — {desc}"
        # Handle nested $defs (e.g. FeedbackMessage in RevisionAgentOutput)
        defs = s.get("$defs", {})
        for def_name, def_schema in defs.items():
            lines.append(f"\n{def_name} object fields:")
            for pname, pprop in def_schema.get("properties", {}).items():
                penum = pprop.get("enum")
                if penum:
                    lines.append(f'  - {pname}: one of {", ".join(repr(e) for e in penum)}')
                else:
                    lines.append(f"  - {pname}: {pprop.get('type', 'string')}")
        return "\n".join(lines)

    def invoke(self, messages: list[dict]) -> object:
        """Call OpenAI API and return parsed Pydantic model."""
        # Inject readable schema instruction so the LLM knows the
        # exact JSON fields to produce (json_object mode only forces
        # JSON output, it doesn't enforce a schema).
        patched = list(messages)
        if patched and patched[0].get("role") == "system":
            patched[0] = {
                **patched[0],
                "content": patched[0]["content"] + "\n\n" + self._schema_instruction,
            }
        else:
            patched.insert(0, {"role": "system", "content": self._schema_instruction})

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": patched,
            "response_format": self._response_format,
        }
        # Retry with exponential backoff for rate limits (429) and server errors (5xx)
        last_exc = None
        for attempt in range(5):
            try:
                resp = self._client.post(self.API_URL, json=payload)
                if resp.status_code in (429, 500, 502, 503):
                    wait = min(2 ** attempt * 1.0, 30.0)
                    logger.debug("OpenAI %d — retry %d in %.1fs", resp.status_code, attempt + 1, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                last_exc = e
                wait = min(2 ** attempt * 1.0, 30.0)
                logger.debug("Timeout — retry %d in %.1fs", attempt + 1, wait)
                time.sleep(wait)
        else:
            raise last_exc or RuntimeError("OpenAI API failed after 5 retries")
        data = resp.json()

        # Track cost
        usage = data.get("usage", {})
        if self._cost_tracker and usage:
            self._cost_tracker.record(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        # Parse structured output
        content = data["choices"][0]["message"]["content"]
        return self._schema.model_validate_json(content)


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> dict:
    """Load JSONL file into {market_id: row_dict}. Empty dict if file missing."""
    cache: dict = {}
    if not path.exists():
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                mid = row.get("market_id")
                if mid:
                    cache[mid] = row
            except json.JSONDecodeError:
                pass
    return cache


def append_jsonl(path: Path, market_id: str, data: dict, lock: threading.Lock) -> None:
    """Thread-safely append one record to a JSONL file."""
    row = {"market_id": market_id, **data}
    with lock:
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


# ── Revision loop CSV ─────────────────────────────────────────────────────────

_REV_LOOP_FIELDNAMES = [
    "market_id", "question", "revision_flag", "recommendation",
    "had_feedback", "feedback_to_a", "feedback_to_b",
]
_rev_loop_header_written = False


def log_revision_loop(market_id: str, question: str, rev_out: dict) -> None:
    """Append to revision_loops.csv if feedback was sent."""
    global _rev_loop_header_written

    feedback = rev_out.get("feedback_to_send", [])
    had_feedback = len(feedback) > 0
    feedback_to_a = " | ".join(
        f["message"] for f in feedback if f.get("recipient") == "A"
    )
    feedback_to_b = " | ".join(
        f["message"] for f in feedback if f.get("recipient") == "B"
    )

    row = {
        "market_id": market_id,
        "question": question,
        "revision_flag": rev_out.get("revision_flag", ""),
        "recommendation": rev_out.get("recommendation_to_decision_agent", ""),
        "had_feedback": had_feedback,
        "feedback_to_a": feedback_to_a,
        "feedback_to_b": feedback_to_b,
    }

    with _rev_loop_lock:
        write_header = not REV_LOOPS_CSV.exists() or not _rev_loop_header_written
        with open(REV_LOOPS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_REV_LOOP_FIELDNAMES)
            if write_header:
                writer.writeheader()
                _rev_loop_header_written = True
            writer.writerow(row)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_markets(market_path: Path, limit: int | None = None) -> list[dict]:
    """Load markets from CSV, filtering to closed polymarket markets with outcomes."""
    with open(market_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("platform") == "polymarket"]
    rows = [r for r in rows if r.get("resolved_yes") not in (None, "", "nan")]
    if limit:
        rows = rows[:limit]
    logger.info("Loaded %d resolved polymarket markets from CSV", len(rows))
    return rows


def _parse_end_date(end_str: str) -> datetime | None:
    if not end_str or str(end_str) in ("nan", "None", ""):
        return None
    try:
        dt = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _load_price_points(
    conn: sqlite3.Connection,
    market_id: str,
    start_ts: int,
    end_ts: int,
) -> list[PricePoint]:
    """Fetch price points in [start_ts, end_ts] (inclusive, unix seconds)."""
    rows = conn.execute(
        "SELECT timestamp, price FROM price_history "
        "WHERE market_id = ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (market_id, start_ts, end_ts),
    ).fetchall()
    return [
        PricePoint(
            timestamp=datetime.fromtimestamp(r[0], tz=timezone.utc),
            price=float(r[1]),
            raw_price=float(r[1]),
        )
        for r in rows
    ]


def _get_price_at(
    conn: sqlite3.Connection,
    market_id: str,
    target_ts: int,
    tolerance_sec: int = 7_200,  # ±2 h
) -> float | None:
    """Return price closest to target_ts within tolerance."""
    row = conn.execute(
        "SELECT price FROM price_history "
        "WHERE market_id = ? AND timestamp BETWEEN ? AND ? "
        "ORDER BY ABS(timestamp - ?) LIMIT 1",
        (market_id, target_ts - tolerance_sec, target_ts + tolerance_sec, target_ts),
    ).fetchone()
    return float(row[0]) if row else None


def prepare_markets(markets: list[dict], db_path: Path = DB_PATH) -> list[dict]:
    """
    For each market, load price history from SQLite filtered to
    [end_date − 120h, end_date − 24h] and attach to the dict.

    Markets with < 3 price points in that window are skipped.
    """
    if not db_path.exists():
        logger.error("SQLite DB not found: %s", db_path)
        return []
    conn = sqlite3.connect(str(db_path))
    skipped = {"no_end_date": 0, "no_price_data": 0, "insufficient_points": 0}
    ready: list[dict] = []

    try:
        for m in markets:
            market_id = str(m["market_id"])
            end_date = _parse_end_date(str(m.get("end_date", "")))
            if end_date is None:
                skipped["no_end_date"] += 1
                continue

            evaluation_date = end_date - timedelta(hours=24)
            window_start    = end_date - timedelta(hours=120)

            start_ts = int(window_start.timestamp())
            end_ts   = int(evaluation_date.timestamp())

            price_points = _load_price_points(conn, market_id, start_ts, end_ts)
            if not price_points:
                skipped["no_price_data"] += 1
                continue
            if len(price_points) < 3:
                skipped["insufficient_points"] += 1
                continue

            # Price at the decision moment (24 h before close)
            current_price = _get_price_at(
                conn, market_id, int(evaluation_date.timestamp())
            )
            if current_price is None:
                current_price = price_points[-1].price  # fallback: last visible point

            # Parse resolved_yes to bool
            rv = m.get("resolved_yes")
            if rv in (True, 1, "True", "true", "1"):
                resolved_yes = True
            elif rv in (False, 0, "False", "false", "0"):
                resolved_yes = False
            else:
                resolved_yes = None  # unknown — excluded from P&L

            # Parse start_date for market_age_days
            market_age_days = None
            start_str = str(m.get("start_date", ""))
            start_dt = _parse_end_date(start_str)
            if start_dt:
                market_age_days = (end_date - start_dt).total_seconds() / 86_400

            tags_list = [
                t.strip() for t in str(m.get("tags", "")).split("|") if t.strip()
            ]

            ready.append({
                **m,
                "market_id":       market_id,
                "end_date":        end_date,
                "evaluation_date": evaluation_date,
                "price_history":   price_points,
                "current_price":   current_price,
                "resolved_yes":    resolved_yes,
                "tags_list":       tags_list,
                "market_age_days": market_age_days,
            })
    finally:
        conn.close()

    logger.info(
        "Markets ready: %d  |  skipped → no_end_date=%d  no_price=%d  insufficient=%d",
        len(ready),
        skipped["no_end_date"],
        skipped["no_price_data"],
        skipped["insufficient_points"],
    )
    return ready


# ── Stage 3: Agent A ──────────────────────────────────────────────────────────

def run_stage_agent_a(
    markets: list[dict],
    cache: dict,
    max_workers: int = 20,
    skip_llm: bool = False,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """
    Run Agent A for all markets.

    Uses agent_a_initial() with cache_enabled=True (reuses data/cache/agent_a/*.json
    from previous pipeline runs). Saves results to data/backtest/agent_a.jsonl.

    Returns {market_id: report_dict}.
    """
    results = dict(cache)
    to_run  = [m for m in markets if m["market_id"] not in cache]

    if not to_run:
        logger.info("Agent A: all %d markets already cached", len(markets))
        return results

    if skip_llm:
        logger.info(
            "Agent A: --skip-llm set — skipping %d uncached markets", len(to_run)
        )
        return results

    logger.info(
        "Agent A: %d markets to run  (%d already cached)", len(to_run), len(cache)
    )
    params = AgentAParams(cache_enabled=True)

    # Httpx-based LLM (bypasses OpenAI SDK hang on Python 3.14)
    structured_llm = HttpxStructuredLLM(
        _LLMClassificationResponse, timeout=30.0, cost_tracker=cost_tracker,
    )

    def _run_one(m: dict) -> tuple[str, dict | None]:
        mid = m["market_id"]
        pkg = AgentAInputPackage(
            market_id=mid,
            question=str(m.get("question", "")),
            description=str(m.get("event_title", "")),
            category=str(m.get("category", "")),
            tags=m["tags_list"],
            platform="polymarket",
            end_date=m["end_date"],
        )
        try:
            t0 = time.time()
            report = agent_a_initial(pkg, params, llm=structured_llm)
            logger.info("Agent A [%s] done in %.1fs", mid[:12], time.time() - t0)
            return mid, report.model_dump()
        except Exception as e:
            logger.warning("Agent A failed for %s (%.1fs): %s", mid, time.time() - t0, e)
            return mid, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, m): m["market_id"] for m in to_run}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Agent A"):
            mid, report = fut.result()
            if report:
                results[mid] = report
                append_jsonl(AGENT_A_JSONL, mid, report, _a_lock)

    logger.info("Agent A: %d results total", len(results))
    return results


# ── Stage 4: Agent B ──────────────────────────────────────────────────────────

def run_stage_agent_b(
    markets: list[dict],
    cache: dict,
    max_workers: int = 20,
    skip_llm: bool = False,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """
    Run Agent B for all markets.

    Price history is pre-filtered to [end_date-120h, end_date-24h].
    evaluation_date = end_date − 24h (decision made at 24 h before close).
    volume_total_usd = final volume from CSV (used as proxy signal).

    Returns {market_id: report_dict}.
    """
    results = dict(cache)
    to_run  = [m for m in markets if m["market_id"] not in cache]

    if not to_run:
        logger.info("Agent B: all %d markets already cached", len(markets))
        return results

    if skip_llm:
        logger.info(
            "Agent B: --skip-llm set — skipping %d uncached markets", len(to_run)
        )
        return results

    logger.info(
        "Agent B: %d markets to run  (%d already cached)", len(to_run), len(cache)
    )
    b_params = AgentBParams()

    # Httpx-based LLM (bypasses OpenAI SDK hang on Python 3.14)
    structured_llm = HttpxStructuredLLM(
        _LLMAgentBResponse, timeout=30.0, cost_tracker=cost_tracker,
    )

    def _run_one(m: dict) -> tuple[str, dict | None]:
        mid = m["market_id"]
        t0 = time.time()
        try:
            volume_raw = m.get("volume", None)
            volume_total = float(volume_raw) if volume_raw not in (None, "", "nan") else None

            pkg = AgentBInputPackage(
                evaluation_date  = m["evaluation_date"],
                end_date         = m["end_date"],
                price_history    = m["price_history"],
                current_price    = m["current_price"],
                volume_total_usd = volume_total,  # final volume as proxy
                market_age_days  = m["market_age_days"],
            )
            report = agent_b_initial(pkg, b_params, llm=structured_llm)
            logger.info("Agent B [%s] done in %.1fs", mid[:12], time.time() - t0)
            return mid, report.model_dump()
        except Exception as e:
            logger.warning("Agent B failed for %s (%.1fs): %s", mid, time.time() - t0, e)
            return mid, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, m): m["market_id"] for m in to_run}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Agent B"):
            mid, report = fut.result()
            if report:
                results[mid] = report
                append_jsonl(AGENT_B_JSONL, mid, report, _b_lock)

    logger.info("Agent B: %d results total", len(results))
    return results


# ── Stage 5: Revision ─────────────────────────────────────────────────────────

def run_stage_revision(
    markets: list[dict],
    a_results: dict,
    b_results: dict,
    cache: dict,
) -> dict:
    """
    Run deterministic Revision Agent for markets where both A and B exist.

    No LLM — applies cross-pattern rules in Python (microseconds per market).
    Logs markets where feedback would have been sent to revision_loops.csv.

    Returns {market_id: revision_dict}.
    """
    results = dict(cache)

    eligible = [
        m for m in markets
        if m["market_id"] in a_results
        and m["market_id"] in b_results
        and m["market_id"] not in cache
    ]

    if not eligible:
        logger.info("Revision: all %d markets already cached", len(results))
        return results

    logger.info(
        "Revision: %d markets to run  (%d already cached)",
        len(eligible), len(cache)
    )

    for m in tqdm(eligible, desc="Revision"):
        mid = m["market_id"]
        out = revision_agent_deterministic(
            agent_a_report=a_results[mid],
            agent_b_report=b_results[mid],
        )
        d = out.model_dump()
        d.pop("agent_a_report", None)
        d.pop("agent_b_report", None)
        results[mid] = d
        append_jsonl(REVISION_JSONL, mid, d, _rev_lock)
        if d.get("feedback_to_send"):
            log_revision_loop(mid, str(m.get("question", "")), d)

    feedback_count = sum(
        1 for r in results.values() if r.get("feedback_to_send")
    )
    logger.info(
        "Revision: %d results total  |  %d markets had feedback (logged to %s)",
        len(results), feedback_count, REV_LOOPS_CSV.name,
    )
    return results


# ── Stage 6 + 7: Decision + Results ──────────────────────────────────────────

def _pnl(decision: str, bet_direction: str, entry_price: float, resolved_yes: bool | None) -> float | None:
    """
    Binary prediction market P&L per unit stake.

    YES token: costs entry_price, pays 1.0 if resolved YES.
    NO  token: costs (1 - entry_price), pays 1.0 if resolved NO.

    Returns None if resolved_yes is unknown.
    """
    if decision != "GO" or resolved_yes is None:
        return None
    if bet_direction == "YES":
        return (1.0 - entry_price) if resolved_yes else -entry_price
    if bet_direction == "NO":
        return entry_price if not resolved_yes else -(1.0 - entry_price)
    return None


def run_stage_decisions(
    markets: list[dict],
    a_results: dict,
    b_results: dict,
    rev_results: dict,
    run_llm_decision: bool = False,
) -> dict[str, list[dict]]:
    """
    Run all decision configs against the fixed A/B/Revision outputs.

    Returns {config_name: list_of_trade_dicts}.
    """
    # Market lookup for price / dates
    market_lookup = {m["market_id"]: m for m in markets}

    configs = list(DECISION_CONFIGS)
    if run_llm_decision:
        configs.append({"name": "llm", "params": DecisionAgentParams(), "llm": True})

    all_trades: dict[str, list[dict]] = {}

    for cfg in configs:
        name        = cfg["name"]
        params      = cfg["params"]
        use_llm     = cfg.get("llm", False)
        trade_rows: list[dict] = []

        eligible_ids = set(a_results) & set(b_results) & set(rev_results)

        for mid in tqdm(eligible_ids, desc=f"Decision [{name}]"):
            m = market_lookup.get(mid)
            if m is None:
                continue

            a_rep  = a_results[mid]
            b_rep  = b_results[mid]
            rev    = rev_results[mid]

            pkg = DecisionAgentInputPackage(
                revision_flag                  = rev.get("revision_flag", "NONE"),
                flag_explanation               = rev.get("flag_explanation", ""),
                agent_a_report                 = a_rep,
                agent_b_report                 = b_rep,
                revision_notes                 = rev.get("revision_notes", ""),
                recommendation_to_decision_agent = rev.get(
                    "recommendation_to_decision_agent", "GO_EVALUATE"
                ),
                current_market_price           = m["current_price"],
                evaluation_date                = m["evaluation_date"],
                end_date                       = m["end_date"],
                market_id                      = mid,
            )

            try:
                if use_llm:
                    out = decision_agent(pkg, params)
                else:
                    out = decision_agent_deterministic(pkg, params)
            except Exception as e:
                logger.warning("Decision [%s] failed for %s: %s", name, mid, e)
                continue

            entry_price  = m["current_price"]
            resolved_yes = m["resolved_yes"]
            pnl          = _pnl(out.decision, out.bet_direction, entry_price, resolved_yes)

            trade_rows.append({
                "market_id":      mid,
                "question":       str(m.get("question", ""))[:120],
                "a_score":        a_rep.get("insider_risk_score"),
                "b_score":        b_rep.get("behavior_score"),
                "a_confidence":   a_rep.get("confidence"),
                "b_confidence":   b_rep.get("confidence"),
                "revision_flag":  rev.get("revision_flag"),
                "recommendation": rev.get("recommendation_to_decision_agent"),
                "decision":       out.decision,
                "bet_direction":  out.bet_direction,
                "entry_price":    round(entry_price, 4),
                "resolved_yes":   resolved_yes,
                "pnl":            round(pnl, 4) if pnl is not None else None,
                "weighted_score": out.analysis.get("weighted_score"),
                "estimated_edge": out.analysis.get("estimated_edge_pp"),
                "evaluation_date": m["evaluation_date"].isoformat(),
                "end_date":        m["end_date"].isoformat(),
            })

        all_trades[name] = trade_rows
        go_count = sum(1 for r in trade_rows if r["decision"] == "GO")
        logger.info(
            "Decision [%s]: %d total  |  %d GO  |  %d SKIP",
            name, len(trade_rows), go_count, len(trade_rows) - go_count,
        )

    return all_trades


def compute_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics for one config's trade list."""
    go_trades = [r for r in trades if r["decision"] == "GO"]
    if not go_trades:
        return {
            "total_markets": len(trades),
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "avg_pnl": None,
            "roi_pct": None,
            "sharpe": None,
            "max_drawdown": 0.0,
            "avg_entry_price": None,
        }

    pnl_known = [r["pnl"] for r in go_trades if r["pnl"] is not None]
    wins   = sum(1 for p in pnl_known if p > 0)
    losses = sum(1 for p in pnl_known if p <= 0)

    total_pnl = sum(pnl_known)

    # Stake per trade: YES bet costs entry_price, NO bet costs (1 - entry_price)
    go_with_pnl = [r for r in go_trades if r["pnl"] is not None]
    effective_stakes = [
        r["entry_price"] if r["bet_direction"] == "YES" else (1.0 - r["entry_price"])
        for r in go_with_pnl
    ]
    total_stakes = sum(effective_stakes)
    roi_pct = (total_pnl / total_stakes * 100) if total_stakes > 0 else None

    sharpe = None
    n = len(pnl_known)
    if n >= 2:
        mean_pnl = total_pnl / n
        variance = sum((p - mean_pnl) ** 2 for p in pnl_known) / (n - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = round(mean_pnl / std, 3)

    # Max drawdown on cumulative P&L
    max_dd = 0.0
    cum = 0.0
    peak = 0.0
    for p in pnl_known:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    entry_prices = [r["entry_price"] for r in go_trades]

    return {
        "total_markets":  len(trades),
        "total_signals":  len(go_trades),
        "wins":           wins,
        "losses":         losses,
        "win_rate":       round(wins / n, 4) if n > 0 else None,
        "total_pnl":      round(total_pnl, 4),
        "avg_pnl":        round(total_pnl / n, 4) if n > 0 else None,
        "roi_pct":        round(roi_pct, 2) if roi_pct is not None else None,
        "sharpe":         sharpe,
        "max_drawdown":   round(max_dd, 4),
        "avg_entry_price": round(sum(entry_prices) / len(entry_prices), 4),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_results(all_trades: dict[str, list[dict]], markets: list[dict]) -> None:
    """Save per-config trade CSVs and a master summary CSV."""
    summary_rows = []

    for cfg_name, trades in all_trades.items():
        # Save individual trades
        trade_path = RESULTS_DIR / f"{cfg_name}_trades.csv"
        _write_csv(trade_path, trades)
        logger.info("  Saved %d rows → %s", len(trades), trade_path)

        # Compute metrics
        m = compute_metrics(trades)
        summary_rows.append({"config": cfg_name, **m})

    # Save summary
    summary_path = RESULTS_DIR / "all_configs_summary.csv"
    _write_csv(summary_path, summary_rows)
    logger.info("Saved summary → %s", summary_path)

    # Print leaderboard
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("DECISION CONFIG LEADERBOARD")
    print("-" * 90)
    go_rows = [r for r in summary_rows if r["total_signals"] > 0]
    if not go_rows:
        print("  No configs produced GO signals.")
    else:
        go_rows.sort(key=lambda r: r["roi_pct"] if r["roi_pct"] is not None else float("-inf"), reverse=True)
        print(
            f"  {'Config':<22} {'Signals':>8} {'WinRate':>8} {'ROI%':>8} "
            f"{'Sharpe':>8} {'MaxDD':>8}"
        )
        print("-" * 90)
        for row in go_rows:
            wr  = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
            roi = f"{row['roi_pct']:.1f}%" if row["roi_pct"] is not None else "—"
            sh  = f"{row['sharpe']:.2f}"   if row["sharpe"]  is not None else "—"
            dd  = f"{row['max_drawdown']:.3f}"
            print(
                f"  {row['config']:<22} {int(row['total_signals']):>8} {wr:>8} "
                f"{roi:>9} {sh:>8} {dd:>8}"
            )

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"{SEP}")


# ── Funnel logging ────────────────────────────────────────────────────────────

def print_funnel(
    total_markets: int,
    markets_with_price: int,
    a_count: int,
    b_count: int,
    rev_count: int,
    all_trades: dict[str, list[dict]],
) -> None:
    print("\n" + "=" * 60)
    print("SIGNAL FUNNEL")
    print("-" * 60)
    print(f"  Total markets in CSV:            {total_markets:>6}")
    print(f"  With price data (>=3 points):    {markets_with_price:>6}")
    print(f"  Agent A completed:               {a_count:>6}")
    print(f"  Agent B completed:               {b_count:>6}")
    print(f"  Revision completed:              {rev_count:>6}")
    for name, trades in all_trades.items():
        go = sum(1 for r in trades if r["decision"] == "GO")
        wins = sum(1 for r in trades if r["decision"] == "GO" and r.get("pnl") is not None and r["pnl"] > 0)
        print(
            f"  [{name:<20}]  GO={go:>4}  Wins={wins:>4}"
        )
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest pipeline v2")
    p.add_argument("--limit",          type=int, default=None,
                   help="Limit number of markets (smoke test)")
    p.add_argument("--skip-llm",       action="store_true",
                   help="Skip LLM stages; only use cached results")
    p.add_argument("--decisions-only", action="store_true",
                   help="Skip all LLM stages, run decisions on existing cache")
    p.add_argument("--llm-decision",   action="store_true",
                   help="Also run LLM-based decision for GO_EVALUATE markets")
    p.add_argument("--workers",        type=int, default=4,
                   help="ThreadPoolExecutor workers per LLM stage (default 4)")
    p.add_argument("--csv",            type=Path, default=CSV_PATH,
                   help=f"Markets CSV path (default: {CSV_PATH})")
    p.add_argument("--db",             type=Path, default=DB_PATH,
                   help=f"SQLite price history path (default: {DB_PATH})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    skip_llm = args.skip_llm or args.decisions_only
    tracker = CostTracker()

    # ── Stage 1: Load markets ─────────────────────────────────────────────────
    logger.info("Stage 1: Loading markets")
    all_markets = load_markets(args.csv, limit=args.limit)
    total_in_csv = len(all_markets)

    # ── Stage 2: Filter to markets with price data ────────────────────────────
    logger.info("Stage 2: Filtering markets with price data")
    markets = prepare_markets(all_markets, db_path=args.db)

    if not markets:
        logger.error("No markets with price data found. Check %s exists.", args.db)
        sys.exit(1)

    # ── Load existing JSONL caches ────────────────────────────────────────────
    logger.info("Loading existing caches from %s", BACKTEST_DIR)
    a_cache   = load_jsonl(AGENT_A_JSONL)
    b_cache   = load_jsonl(AGENT_B_JSONL)
    rev_cache = load_jsonl(REVISION_JSONL)
    logger.info(
        "  Cached: A=%d  B=%d  Rev=%d", len(a_cache), len(b_cache), len(rev_cache)
    )

    # ── Stage 3+4: Agent A & B (concurrent) ────────────────────────────────────
    # Each stage gets its own tracker to avoid cross-contamination during
    # concurrent execution, then results are merged into the main tracker.
    logger.info("Stage 3+4: Agent A & B (running concurrently)")
    a_tracker = CostTracker()
    b_tracker = CostTracker()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stage") as stage_pool:
        fut_a = stage_pool.submit(
            run_stage_agent_a,
            markets, a_cache, max_workers=args.workers, skip_llm=skip_llm,
            cost_tracker=a_tracker,
        )
        fut_b = stage_pool.submit(
            run_stage_agent_b,
            markets, b_cache, max_workers=args.workers, skip_llm=skip_llm,
            cost_tracker=b_tracker,
        )
        a_results = fut_a.result()
        b_results = fut_b.result()

    # Merge per-stage trackers into main tracker
    for name, st in [("Agent A", a_tracker), ("Agent B", b_tracker)]:
        tracker.merge_from(st, name)
        logger.info(
            "Cost [%s]: %d calls | %dk in + %dk out tokens | $%.4f stage | $%.4f cumulative",
            name, st.calls, st.input_tokens // 1000, st.output_tokens // 1000,
            st.cost, tracker.cost,
        )

    # ── Stage 5: Revision (deterministic — no LLM) ─────────────────────────────
    logger.info("Stage 5: Revision (deterministic)")
    rev_results = run_stage_revision(
        markets, a_results, b_results, rev_cache,
    )

    # ── Stage 6: Decisions ────────────────────────────────────────────────────
    logger.info("Stage 6: Running %d decision configs", len(DECISION_CONFIGS))
    all_trades = run_stage_decisions(
        markets, a_results, b_results, rev_results,
        run_llm_decision=args.llm_decision,
    )

    # ── Stage 7: Export ───────────────────────────────────────────────────────
    logger.info("Stage 7: Exporting results")
    export_results(all_trades, markets)

    print_funnel(
        total_in_csv,
        len(markets),
        len(a_results),
        len(b_results),
        len(rev_results),
        all_trades,
    )

    tracker.print_summary()
    logger.info("Done. Run `python scripts/plot_backtest_v2.py` to generate charts.")


if __name__ == "__main__":
    main()
