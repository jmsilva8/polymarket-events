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


# Default sweep ranges from the plan
DEFAULT_PRICE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DEFAULT_HOURS_BEFORE_CLOSE = [6, 12, 24, 48, 72]
DEFAULT_MIN_LEAK_SCORES = [5, 6, 7, 8, 9]


class BacktestRunner:
    """
    Runs the insider alpha strategy against historical market data
    with configurable parameter sweeps.

    Data flow:
    1. Load classifications (insider risk scores)
    2. Load Polymarket market data (end_date, resolved_yes, clob_token_ids)
    3. Load cached price histories
    4. For each parameter combination, run the strategy and compute metrics
    """

    def __init__(
        self,
        classifications_path: Optional[Path] = None,
        markets_csv_path: Optional[Path] = None,
        events_cache_path: Optional[Path] = None,
        price_cache_dir: Optional[Path] = None,
    ):
        self.classifications_path = classifications_path or EXPORTS_DIR / "classifications_gpt4omini.csv"
        self.markets_csv_path = markets_csv_path or EXPORTS_DIR / "polymarket_entertainment_sample.csv"
        self.events_cache_path = events_cache_path or CACHE_DIR / "events" / "polymarket_entertainment_closed.json"
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
        """Load market metadata from cached events JSON."""
        events_data = json.loads(self.events_cache_path.read_text())
        for event in events_data:
            for m in event.get("markets", []):
                mid = str(m.get("id", ""))
                tokens_raw = m.get("clobTokenIds", "[]")
                if isinstance(tokens_raw, str):
                    tokens = json.loads(tokens_raw)
                else:
                    tokens = tokens_raw or []

                # Parse outcome prices to determine resolution
                outcome_prices_raw = m.get("outcomePrices", "[]")
                if isinstance(outcome_prices_raw, str):
                    try:
                        outcome_prices = [float(x) for x in json.loads(outcome_prices_raw)]
                    except (json.JSONDecodeError, ValueError):
                        outcome_prices = []
                else:
                    outcome_prices = [float(x) for x in (outcome_prices_raw or [])]

                # Determine resolved_yes
                resolved_yes = None
                if m.get("closed"):
                    if outcome_prices:
                        resolved_yes = outcome_prices[0] > 0.5

                # Parse end date
                end_date = None
                end_str = m.get("endDate") or m.get("closedTime")
                if end_str:
                    try:
                        end_date = datetime.fromisoformat(
                            end_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

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

        logger.info("Loaded metadata for %d markets", len(self._market_data))

    def _load_price_histories(self) -> None:
        """Load cached price history JSON files."""
        loaded = 0
        for clf_row in self._classifications.itertuples():
            mid = str(clf_row.market_id)
            mdata = self._market_data.get(mid)
            if not mdata or not mdata["clob_token_ids"]:
                continue

            token_id = mdata["clob_token_ids"][0]  # YES token
            cache_path = self.price_cache_dir / f"{token_id}_max_60.json"

            if not cache_path.exists():
                continue

            raw = json.loads(cache_path.read_text())
            data_points = []
            for pt in raw.get("history", []):
                ts = datetime.fromtimestamp(pt["t"], tz=timezone.utc)
                price = float(pt["p"])
                data_points.append(PricePoint(timestamp=ts, price=price, raw_price=price))

            if data_points:
                self._price_histories[mid] = PriceHistory(
                    market_id=mid,
                    platform=Platform.POLYMARKET,
                    token_id=token_id,
                    question="",
                    outcome_label="Yes",
                    data_points=data_points,
                )
                loaded += 1

        logger.info("Loaded %d price histories", loaded)

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
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations.

        Returns DataFrame with one row per parameter combo, sorted by ROI.
        """
        pts = price_thresholds or DEFAULT_PRICE_THRESHOLDS
        hbc = hours_before_close or DEFAULT_HOURS_BEFORE_CLOSE
        mls = min_leak_scores or DEFAULT_MIN_LEAK_SCORES

        combos = list(product(pts, hbc, mls))
        if verbose:
            print(f"Running {len(combos)} parameter combinations...")

        all_metrics: list[dict] = []

        iterator = tqdm(combos, desc="Sweep") if verbose else combos
        for pt, h, ms in iterator:
            params = StrategyParams(
                price_threshold=pt,
                hours_before_close=h,
                min_leak_score=ms,
            )
            signals, metrics = self.run_single(params)
            all_metrics.append({
                "price_threshold": pt,
                "hours_before_close": h,
                "min_leak_score": ms,
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
        # Sort by ROI descending, then by win rate
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
