"""Backtest performance metrics."""

from dataclasses import dataclass
from typing import Optional

from src.backtest_engine.strategy import TradeSignal


@dataclass
class BacktestMetrics:
    """Performance metrics for a parameter combination."""
    # Strategy params that produced these metrics
    price_threshold: float
    hours_before_close: float
    min_leak_score: int

    # Counts
    total_signals: int = 0
    wins: int = 0
    losses: int = 0

    # Performance
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    roi: float = 0.0           # Total PnL / total capital deployed
    sharpe: Optional[float] = None
    max_drawdown: float = 0.0

    # Detail
    avg_entry_price: float = 0.0
    avg_hours_before_close: float = 0.0


def compute_metrics(
    signals: list[TradeSignal],
    price_threshold: float,
    hours_before_close: float,
    min_leak_score: int,
) -> BacktestMetrics:
    """Compute performance metrics from a list of trade signals."""
    metrics = BacktestMetrics(
        price_threshold=price_threshold,
        hours_before_close=hours_before_close,
        min_leak_score=min_leak_score,
    )

    if not signals:
        return metrics

    metrics.total_signals = len(signals)

    pnls = [s.pnl for s in signals if s.pnl is not None]
    if not pnls:
        return metrics

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    metrics.wins = len(wins)
    metrics.losses = len(losses)
    metrics.win_rate = metrics.wins / metrics.total_signals if metrics.total_signals > 0 else 0.0
    metrics.total_pnl = sum(pnls)
    metrics.avg_pnl = metrics.total_pnl / len(pnls)

    if wins:
        metrics.avg_win = sum(wins) / len(wins)
        metrics.max_win = max(wins)
    if losses:
        metrics.avg_loss = sum(losses) / len(losses)
        metrics.max_loss = min(losses)

    # ROI: total PnL / total capital deployed (sum of entry prices)
    total_capital = sum(s.entry_price for s in signals if s.entry_price > 0)
    metrics.roi = (metrics.total_pnl / total_capital * 100) if total_capital > 0 else 0.0

    # Sharpe ratio (simplified: mean / std of per-trade returns)
    if len(pnls) >= 2:
        import math
        mean_pnl = metrics.avg_pnl
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0
        metrics.sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else None

    # Max drawdown (cumulative P&L curve)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    metrics.max_drawdown = max_dd

    # Averages
    entry_prices = [s.entry_price for s in signals]
    metrics.avg_entry_price = sum(entry_prices) / len(entry_prices)
    hours = [s.hours_before_close_actual for s in signals]
    metrics.avg_hours_before_close = sum(hours) / len(hours)

    return metrics
