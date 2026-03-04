"""Backtest runner with parameter sweep grid search."""

import json
import logging
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import CACHE_DIR, EXPORTS_DIR
from src.data_layer.models import Platform, PricePoint, PriceHistory
from src.backtest_engine.strategy import InsiderAlphaStrategy, StrategyParams, TradeSignal
from src.backtest_engine.metrics import BacktestMetrics, compute_metrics

logger = logging.getLogger(__name__)


# Default sweep ranges
DEFAULT_PRICE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DEFAULT_HOURS_BEFORE_CLOSE = [6, 12, 24, 48, 72]
DEFAULT_MIN_LEAK_SCORES = [5, 6, 7, 8, 9]
# None = no jump requirement (pure threshold); values add the abnormal-movement filter
DEFAULT_MIN_PRICE_JUMPS: list[Optional[float]] = [None, 0.05, 0.10, 0.15]


class BacktestRunner:
    """
    Runs the insider alpha strategy against historical market data
    with configurable parameter sweeps.

    Data flow:
    1. Load classifications (insider risk scores)
    2. Load Polymarket market data from ALL JSON files in events_cache_dir
    3. Load cached price histories (sorted by timestamp)
    4. For each parameter combination, run the strategy and compute metrics
    """

    def __init__(
        self,
        classifications_path: Optional[Path] = None,
        events_cache_dir: Optional[Path] = None,
        price_cache_dir: Optional[Path] = None,
    ):
        self.classifications_path = classifications_path or EXPORTS_DIR / "classifications_gpt4omini.csv"
        self.events_cache_dir = events_cache_dir or CACHE_DIR / "events"
        self.price_cache_dir = price_cache_dir or CACHE_DIR / "price_history"

        # Loaded data
        self._classifications: Optional[pd.DataFrame] = None
        self._market_data: dict = {}   # market_id -> {end_date, resolved_yes, volume, clob_token_ids}
        self._price_histories: dict = {}  # market_id -> PriceHistory

    def load_data(self) -> None:
        """Load all required data for backtesting."""
        self._load_classifications()
        self._load_market_data()
        self._load_price_histories()

    def _load_classifications(self) -> None:
        """Load classification results."""
        df = pd.read_csv(self.classifications_path)
        # Only Polymarket markets (Kalshi has no price history)
        self._classifications = df[df["platform"] == "polymarket"].copy()
        logger.info("Loaded %d Polymarket classifications", len(self._classifications))

    def _load_market_data(self) -> None:
        """Load market metadata from ALL cached events JSON files."""
        cache_files = sorted(self.events_cache_dir.glob("*.json"))
        if not cache_files:
            logger.warning("No event cache files found in %s", self.events_cache_dir)
            return

        logger.info("Loading market metadata from %d cache file(s)", len(cache_files))

        for cache_file in cache_files:
            try:
                events_data = json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read %s: %s", cache_file.name, exc)
                continue

            for event in events_data:
                for m in event.get("markets", []):
                    mid = str(m.get("id", ""))
                    tokens_raw = m.get("clobTokenIds", "[]")
                    if isinstance(tokens_raw, str):
                        try:
                            tokens = json.loads(tokens_raw)
                        except json.JSONDecodeError:
                            tokens = []
                    else:
                        tokens = tokens_raw or []

                    # Parse outcome prices
                    outcome_prices_raw = m.get("outcomePrices", "[]")
                    if isinstance(outcome_prices_raw, str):
                        try:
                            outcome_prices = [float(x) for x in json.loads(outcome_prices_raw)]
                        except (json.JSONDecodeError, ValueError):
                            outcome_prices = []
                    else:
                        outcome_prices = [float(x) for x in (outcome_prices_raw or [])]

                    # Determine the YES outcome index from the outcomes labels.
                    # The API returns outcomes as a JSON-encoded list like
                    # '["Yes","No"]' or '["No","Yes"]' — we must not assume
                    # index 0 is always YES.
                    outcomes_raw = m.get("outcomes", '["Yes","No"]')
                    if isinstance(outcomes_raw, str):
                        try:
                            outcomes = json.loads(outcomes_raw)
                        except json.JSONDecodeError:
                            outcomes = ["Yes", "No"]
                    else:
                        outcomes = list(outcomes_raw) if outcomes_raw else ["Yes", "No"]

                    yes_index = 0  # safe default
                    for i, label in enumerate(outcomes):
                        if str(label).strip().lower() in ("yes", "true"):
                            yes_index = i
                            break

                    # Determine resolved_yes using the correct index
                    resolved_yes = None
                    if m.get("closed") and outcome_prices:
                        if yes_index < len(outcome_prices):
                            resolved_yes = outcome_prices[yes_index] > 0.5
                        else:
                            logger.debug(
                                "Market %s: yes_index %d out of range "
                                "(outcome_prices len=%d)",
                                mid, yes_index, len(outcome_prices),
                            )

                    # Parse end date
                    end_date = None
                    end_str = m.get("endDate") or m.get("closedTime")
                    if end_str:
                        try:
                            end_date = datetime.fromisoformat(
                                end_str.replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            logger.debug(
                                "Market %s: could not parse end_date %r",
                                mid, end_str,
                            )

                    volume_raw = m.get("volumeNum", m.get("volume", 0))
                    try:
                        volume = float(volume_raw) if volume_raw else 0.0
                    except (ValueError, TypeError):
                        volume = 0.0

                    self._market_data[mid] = {
                        "end_date": end_date,
                        "resolved_yes": resolved_yes,
                        "volume": volume,
                        "clob_token_ids": tokens,
                    }

        missing_end_date = sum(1 for v in self._market_data.values() if v["end_date"] is None)
        unresolved = sum(1 for v in self._market_data.values() if v["resolved_yes"] is None)
        logger.info(
            "Market data: %d total, %d missing end_date, %d unresolved (no outcome yet)",
            len(self._market_data), missing_end_date, unresolved,
        )

    def _load_price_histories(self) -> None:
        """Load and sort cached price history JSON files."""
        loaded = 0
        no_clob = 0
        no_cache = 0
        empty = 0

        for clf_row in self._classifications.itertuples():
            mid = str(clf_row.market_id)
            mdata = self._market_data.get(mid)
            if not mdata or not mdata["clob_token_ids"]:
                no_clob += 1
                continue

            token_id = mdata["clob_token_ids"][0]  # YES token
            cache_path = self.price_cache_dir / f"{token_id}_max_60.json"

            if not cache_path.exists():
                no_cache += 1
                logger.debug("Market %s: no price cache for token %s", mid, token_id)
                continue

            raw = json.loads(cache_path.read_text())
            data_points: list[PricePoint] = []
            for pt in raw.get("history", []):
                ts = datetime.fromtimestamp(pt["t"], tz=timezone.utc)
                price = float(pt["p"])
                data_points.append(PricePoint(timestamp=ts, price=price, raw_price=price))

            if not data_points:
                empty += 1
                logger.debug("Market %s: price cache exists but has no data points", mid)
                continue

            # Sort chronologically so strategy loops can rely on order
            data_points.sort(key=lambda dp: dp.timestamp)

            self._price_histories[mid] = PriceHistory(
                market_id=mid,
                platform=Platform.POLYMARKET,
                token_id=token_id,
                question="",
                outcome_label="Yes",
                data_points=data_points,
            )
            loaded += 1

        logger.info(
            "Price histories: %d loaded, %d no CLOB IDs, %d no cache file, %d empty",
            loaded, no_clob, no_cache, empty,
        )

    # ── Core run methods ───────────────────────────────────────────

    def run_single(self, params: StrategyParams) -> tuple[list[TradeSignal], BacktestMetrics]:
        """Run strategy with a single parameter set. Returns (signals, metrics)."""
        strategy = InsiderAlphaStrategy(params)
        signals: list[TradeSignal] = []

        for clf_row in self._classifications.itertuples():
            mid = str(clf_row.market_id)
            score = int(clf_row.insider_risk_score)

            # Quick filter before looking up data
            if score < params.min_leak_score:
                continue

            mdata = self._market_data.get(mid)
            if not mdata:
                continue

            ph = self._price_histories.get(mid)
            if not ph:
                continue

            signal = strategy.evaluate(
                price_history=ph,
                market_id=mid,
                market_title=str(clf_row.market_title),
                platform="polymarket",
                insider_risk_score=score,
                end_date=mdata["end_date"],
                volume=mdata["volume"],
                resolved_yes=mdata["resolved_yes"],
            )

            if signal:
                signals.append(signal)

        metrics = compute_metrics(
            signals,
            price_threshold=params.price_threshold,
            hours_before_close=params.hours_before_close,
            min_leak_score=params.min_leak_score,
        )
        return signals, metrics

    def run_sweep(
        self,
        price_thresholds: Optional[list[float]] = None,
        hours_before_close: Optional[list[float]] = None,
        min_leak_scores: Optional[list[int]] = None,
        min_price_jumps: Optional[list[Optional[float]]] = None,
        ignore_window: bool = False,
        strategy_type_label: str = "insider_alpha",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations.

        Args:
            price_thresholds: List of price threshold values to sweep.
            hours_before_close: List of window sizes (hours) to sweep.
                Ignored when ignore_window=True.
            min_leak_scores: List of minimum insider risk scores to sweep.
            min_price_jumps: List of minimum price-jump values to sweep.
                None entries disable the jump filter for that combo.
            ignore_window: If True, search entire price history (baseline B mode).
            strategy_type_label: Label written to the ``strategy_type`` column.
            verbose: Show tqdm progress bar.

        Returns:
            DataFrame with one row per parameter combo, sorted by ROI descending.
        """
        pts = price_thresholds or DEFAULT_PRICE_THRESHOLDS
        hbc = hours_before_close or DEFAULT_HOURS_BEFORE_CLOSE
        mls = min_leak_scores or DEFAULT_MIN_LEAK_SCORES
        mpjs = min_price_jumps if min_price_jumps is not None else DEFAULT_MIN_PRICE_JUMPS

        combos = list(product(pts, hbc, mls, mpjs))
        if verbose:
            print(f"Running {len(combos)} parameter combinations [{strategy_type_label}]...")

        all_metrics: list[dict] = []

        iterator = tqdm(combos, desc=strategy_type_label) if verbose else combos
        for pt, h, ms, mpj in iterator:
            params = StrategyParams(
                price_threshold=pt,
                hours_before_close=h,
                min_leak_score=ms,
                min_price_jump=mpj,
                ignore_window=ignore_window,
            )
            signals, metrics = self.run_single(params)
            all_metrics.append({
                "strategy_type": strategy_type_label,
                "price_threshold": pt,
                "hours_before_close": h if not ignore_window else None,
                "min_leak_score": ms,
                "min_price_jump": mpj,
                "total_signals": metrics.total_signals,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "win_rate": round(metrics.win_rate, 4),
                "total_pnl": round(metrics.total_pnl, 4),
                "avg_pnl": round(metrics.avg_pnl, 4),
                "roi_pct": round(metrics.roi, 2),
                "sharpe": round(metrics.sharpe, 4) if metrics.sharpe is not None else None,
                "max_drawdown": round(metrics.max_drawdown, 4),
                "avg_entry_price": round(metrics.avg_entry_price, 4),
                "avg_hours_before": round(metrics.avg_hours_before_close, 1),
            })

        df = pd.DataFrame(all_metrics)
        df = df.sort_values(["roi_pct", "win_rate"], ascending=[False, False])
        df = df.reset_index(drop=True)
        return df

    def get_data_summary(self) -> dict:
        """Return summary of loaded data."""
        return {
            "classifications": len(self._classifications) if self._classifications is not None else 0,
            "market_metadata": len(self._market_data),
            "price_histories": len(self._price_histories),
            "score_distribution": (
                self._classifications.groupby("insider_risk_score").size().to_dict()
                if self._classifications is not None else {}
            ),
        }
