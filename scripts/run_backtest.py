"""Run backtest grid search and display results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

from src.backtest_engine.backtester import BacktestRunner

# Load all data
print("Loading data...")
runner = BacktestRunner()
runner.load_data()

summary = runner.get_data_summary()
print(f"  Classifications: {summary['classifications']}")
print(f"  Market metadata: {summary['market_metadata']}")
print(f"  Price histories: {summary['price_histories']}")
print(f"  Score distribution: {summary['score_distribution']}")
print()

# Run parameter sweep
results = runner.run_sweep()

print(f"\n{'='*100}")
print(f"BACKTEST RESULTS — {len(results)} parameter combinations")
print(f"{'='*100}\n")

# Top 20 by ROI
print("TOP 20 BY ROI:")
print("-" * 100)
cols = ["price_threshold", "hours_before_close", "min_leak_score",
        "total_signals", "wins", "losses", "win_rate", "total_pnl",
        "roi_pct", "sharpe", "avg_entry_price"]
top20 = results.head(20)
print(top20[cols].to_string(index=False))

# Filter: at least 10 signals for statistical significance
significant = results[results["total_signals"] >= 10].copy()
if not significant.empty:
    print(f"\n\nTOP 20 BY ROI (min 10 signals):")
    print("-" * 100)
    top20_sig = significant.head(20)
    print(top20_sig[cols].to_string(index=False))

# Best win rate combos (min 10 signals)
if not significant.empty:
    by_wr = significant.sort_values("win_rate", ascending=False)
    print(f"\n\nTOP 10 BY WIN RATE (min 10 signals):")
    print("-" * 100)
    print(by_wr.head(10)[cols].to_string(index=False))

# Summary stats
print(f"\n\n{'='*100}")
print("OVERALL SUMMARY")
print(f"{'='*100}")
print(f"Total parameter combos tested: {len(results)}")
print(f"Combos with positive ROI: {len(results[results['roi_pct'] > 0])}")
print(f"Combos with win rate > 60%: {len(results[results['win_rate'] > 0.6])}")
print(f"Combos with win rate > 70%: {len(results[results['win_rate'] > 0.7])}")
if not significant.empty:
    print(f"\nWith >= 10 signals:")
    print(f"  Combos with positive ROI: {len(significant[significant['roi_pct'] > 0])}")
    print(f"  Best win rate: {significant['win_rate'].max():.1%}")
    print(f"  Best ROI: {significant['roi_pct'].max():.1f}%")
    best = significant.iloc[0]
    print(f"  Best combo: threshold={best['price_threshold']}, "
          f"hours={best['hours_before_close']}, "
          f"min_score={best['min_leak_score']}")

# Export results
export_path = Path("data/exports/backtest_sweep_results.csv")
results.to_csv(export_path, index=False)
print(f"\nExported full results to {export_path}")

# Also export the detailed trades for the best combo (min 10 signals)
if not significant.empty:
    best = significant.iloc[0]
    from src.backtest_engine.strategy import StrategyParams
    params = StrategyParams(
        price_threshold=best["price_threshold"],
        hours_before_close=best["hours_before_close"],
        min_leak_score=int(best["min_leak_score"]),
    )
    signals, metrics = runner.run_single(params)
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
                "pnl": round(s.pnl, 4) if s.pnl else 0,
            })
        import pandas as pd
        trades_df = pd.DataFrame(trade_rows)
        trades_path = Path("data/exports/backtest_best_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Exported {len(trade_rows)} trades from best combo to {trades_path}")

        # Show the trades
        print(f"\nBEST COMBO TRADES (threshold={best['price_threshold']}, "
              f"hours={best['hours_before_close']}, min_score={int(best['min_leak_score'])}):")
        print("-" * 100)
        for s in sorted(signals, key=lambda x: -(x.pnl or 0)):
            result = "WIN " if s.resolved_yes else "LOSS"
            print(f"  [{result}] PnL={s.pnl:+.3f} | Entry={s.entry_price:.3f} | "
                  f"Score={s.insider_risk_score} | {s.hours_before_close_actual:.0f}h before | "
                  f"{s.market_title[:55]}")
