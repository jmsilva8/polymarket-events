"""Run backtest grid search across strategies and display results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd

from src.backtest_engine.backtester import BacktestRunner
from src.backtest_engine.strategy import StrategyParams

# ── Load data ──────────────────────────────────────────────────────
print("Loading data...")
runner = BacktestRunner()
runner.load_data()

summary = runner.get_data_summary()
print(f"  Classifications:  {summary['classifications']}")
print(f"  Market metadata:  {summary['market_metadata']}")
print(f"  Price histories:  {summary['price_histories']}")
print(f"  Score distribution: {summary['score_distribution']}")
print()

# ── Strategy sweeps ────────────────────────────────────────────────

# Insider alpha: full sweep with score filter + optional jump filter
results_alpha = runner.run_sweep(strategy_type_label="insider_alpha")

# Baseline A: same window/threshold sweep but NO score filter.
# Shows whether the AI insider score adds value beyond the price signal.
results_baseline_a = runner.run_sweep(
    min_leak_scores=[0],
    min_price_jumps=[None],        # no jump requirement for clean comparison
    strategy_type_label="baseline_no_score",
)

# Baseline B: bet YES on the favourite regardless of time-to-close.
# No score filter, no time window — just "price > threshold, enter immediately".
results_baseline_b = runner.run_sweep(
    price_thresholds=[0.60, 0.65, 0.70, 0.75],
    min_leak_scores=[0],
    min_price_jumps=[None],
    ignore_window=True,
    strategy_type_label="baseline_favourite",
)

# Combine all results
all_results = pd.concat(
    [results_alpha, results_baseline_a, results_baseline_b],
    ignore_index=True,
)

# ── Display ────────────────────────────────────────────────────────
SEP = "=" * 110

def print_table(df: pd.DataFrame, title: str, n: int = 20) -> None:
    cols = [
        "strategy_type", "price_threshold", "hours_before_close",
        "min_leak_score", "min_price_jump",
        "total_signals", "wins", "win_rate", "roi_pct", "sharpe",
    ]
    present = [c for c in cols if c in df.columns]
    print(f"\n{SEP}")
    print(title)
    print("-" * 110)
    print(df.head(n)[present].to_string(index=False))


print_table(all_results, f"ALL STRATEGIES — TOP 20 BY ROI ({len(all_results)} total combos)")

significant = all_results[all_results["total_signals"] >= 10].copy()
if not significant.empty:
    print_table(significant, "TOP 20 BY ROI (min 10 signals)", n=20)

    by_wr = significant.sort_values("win_rate", ascending=False)
    print_table(by_wr, "TOP 10 BY WIN RATE (min 10 signals)", n=10)

# Per-strategy best row
print(f"\n\n{SEP}")
print("BEST RESULT PER STRATEGY TYPE")
print("-" * 110)
src = significant if not significant.empty else all_results
for stype, grp in src.groupby("strategy_type"):
    best = grp.iloc[0]
    print(
        f"  [{stype}]  ROI={best['roi_pct']:.1f}%  "
        f"win_rate={best['win_rate']:.1%}  "
        f"signals={int(best['total_signals'])}  "
        f"threshold={best['price_threshold']}  "
        f"hours={best['hours_before_close']}  "
        f"score>={best['min_leak_score']}  "
        f"jump>={best['min_price_jump']}"
    )

# Overall summary
print(f"\n\n{SEP}")
print("OVERALL SUMMARY")
print(f"{SEP}")
print(f"Total combos tested: {len(all_results)}")
print(f"Combos with positive ROI:   {len(all_results[all_results['roi_pct'] > 0])}")
print(f"Combos with win_rate > 60%: {len(all_results[all_results['win_rate'] > 0.6])}")
print(f"Combos with win_rate > 70%: {len(all_results[all_results['win_rate'] > 0.7])}")
if not significant.empty:
    print(f"\nWith >= 10 signals:")
    print(f"  Best win rate: {significant['win_rate'].max():.1%}")
    print(f"  Best ROI:      {significant['roi_pct'].max():.1f}%")

# ── Export results ─────────────────────────────────────────────────
export_dir = Path("data/exports")
export_dir.mkdir(parents=True, exist_ok=True)

sweep_path = export_dir / "backtest_sweep_results.csv"
all_results.to_csv(sweep_path, index=False)
print(f"\nExported {len(all_results)} rows to {sweep_path}")

# Detailed trades for the best insider_alpha combo (min 10 signals)
alpha_sig = results_alpha[results_alpha["total_signals"] >= 10]
if not alpha_sig.empty:
    best = alpha_sig.iloc[0]
    params = StrategyParams(
        price_threshold=best["price_threshold"],
        hours_before_close=best["hours_before_close"],
        min_leak_score=int(best["min_leak_score"]),
        min_price_jump=best["min_price_jump"] if pd.notna(best["min_price_jump"]) else None,
    )
    signals, _ = runner.run_single(params)
    if signals:
        trade_rows = []
        for s in sorted(signals, key=lambda x: -(x.pnl or 0)):
            trade_rows.append({
                "market_id": s.market_id,
                "market_title": s.market_title,
                "insider_risk_score": s.insider_risk_score,
                "entry_price": round(s.entry_price, 4),
                "signal_time": s.signal_time.isoformat(),
                "end_date": s.end_date.isoformat(),
                "hours_before_close": round(s.hours_before_close_actual, 1),
                "resolved_yes": s.resolved_yes,
                "pnl": round(s.pnl, 4) if s.pnl is not None else 0,
                "strategy_type": "insider_alpha",
            })
        trades_df = pd.DataFrame(trade_rows)
        trades_path = export_dir / "backtest_best_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Exported {len(trade_rows)} trades from best insider_alpha combo to {trades_path}")

        print(
            f"\nBEST INSIDER ALPHA TRADES "
            f"(threshold={best['price_threshold']}, "
            f"hours={best['hours_before_close']}, "
            f"score>={int(best['min_leak_score'])}, "
            f"jump>={best['min_price_jump']}):"
        )
        print("-" * 110)
        for s in sorted(signals, key=lambda x: -(x.pnl or 0)):
            result = "WIN " if s.resolved_yes else "LOSS"
            print(
                f"  [{result}] PnL={s.pnl:+.3f} | Entry={s.entry_price:.3f} | "
                f"Score={s.insider_risk_score} | {s.hours_before_close_actual:.0f}h before | "
                f"{s.market_title[:55]}"
            )

print("\nDone. Run `python scripts/plot_results.py` to generate charts.")
