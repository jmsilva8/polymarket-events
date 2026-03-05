"""
Generate charts from backtesting v2 results for presentation.

Reads:
    data/backtest/results/all_configs_summary.csv
    data/backtest/results/{config_name}_trades.csv

Writes (all to data/backtest/charts/):
    1. leaderboard.png       — ROI and win rate side-by-side bar chart for all configs
    2. cumulative_pnl.png    — Cumulative P&L curves, one line per config
    3. signal_funnel.png     — Markets → price data → A+B → GO → WIN
    4. score_scatter.png     — a_score vs b_score, colored by WIN/LOSS (GO trades only)

Usage:
    python scripts/plot_backtest_v2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.config import DATA_DIR

RESULTS_DIR = DATA_DIR / "backtest" / "results"
CHARTS_DIR  = DATA_DIR / "backtest" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]

WIN_COLOR  = "#4CAF50"
LOSS_COLOR = "#F44336"
SKIP_COLOR = "#9E9E9E"


def _savefig(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_summary() -> pd.DataFrame | None:
    path = RESULTS_DIR / "all_configs_summary.csv"
    if not path.exists():
        print(f"  [skip] {path} not found — run run_backtest_v2.py first.")
        return None
    return pd.read_csv(path)


def load_trades(config_name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{config_name}_trades.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


# ── Chart 1: Leaderboard — ROI + Win Rate per config ─────────────────────────

def plot_leaderboard(summary: pd.DataFrame) -> None:
    """Side-by-side bar chart: ROI% and win rate for all configs with GO signals."""
    df = summary[summary["total_signals"] > 0].copy()
    if df.empty:
        print("  [skip] leaderboard — no configs with GO signals")
        return

    df = df.sort_values("roi_pct", ascending=False).reset_index(drop=True)
    n   = len(df)
    x   = np.arange(n)
    w   = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, n * 1.5), 5))
    ax2 = ax1.twinx()

    # ROI bars
    bars_roi = ax1.bar(
        x - w / 2, df["roi_pct"].fillna(0), w,
        color=[WIN_COLOR if v >= 0 else LOSS_COLOR for v in df["roi_pct"].fillna(0)],
        alpha=0.85, label="ROI %",
    )
    # Win rate bars
    wr_vals = (df["win_rate"].fillna(0) * 100).tolist()
    bars_wr = ax2.bar(
        x + w / 2, wr_vals, w,
        color=PALETTE[0], alpha=0.65, label="Win Rate %",
    )

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("ROI (%)", fontsize=12)
    ax2.set_ylabel("Win Rate (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["config"].tolist(), rotation=20, ha="right", fontsize=10)
    ax1.set_title("Decision Config Leaderboard — ROI vs Win Rate", fontsize=14, pad=14)

    # Annotate signals count
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.text(
            i - w / 2, (df["roi_pct"].fillna(0).iloc[i] or 0) + 0.3,
            f"n={int(row['total_signals'])}", ha="center", fontsize=8, color="black",
        )

    handles = [
        mpatches.Patch(color=WIN_COLOR, label="ROI (positive)"),
        mpatches.Patch(color=LOSS_COLOR, label="ROI (negative)"),
        mpatches.Patch(color=PALETTE[0], alpha=0.65, label="Win rate"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    _savefig(fig, "leaderboard.png")


# ── Chart 2: Cumulative P&L curves ───────────────────────────────────────────

def plot_cumulative_pnl(summary: pd.DataFrame) -> None:
    """One cumulative P&L curve per config (GO trades only, sorted by evaluation_date)."""
    configs_with_signals = summary[summary["total_signals"] > 0]["config"].tolist()
    if not configs_with_signals:
        print("  [skip] cumulative_pnl — no configs with GO signals")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = 0

    for i, name in enumerate(configs_with_signals):
        df = load_trades(name)
        if df is None:
            continue
        go = df[df["decision"] == "GO"].dropna(subset=["pnl"]).copy()
        if go.empty:
            continue
        if "evaluation_date" in go.columns:
            try:
                go["evaluation_date"] = pd.to_datetime(go["evaluation_date"])
                go = go.sort_values("evaluation_date")
            except Exception:
                pass
        cum_pnl = go["pnl"].cumsum().reset_index(drop=True)
        color   = PALETTE[i % len(PALETTE)]
        ax.plot(cum_pnl.index, cum_pnl.values, label=name, color=color, linewidth=2)
        # Mark final value
        ax.scatter(
            [len(cum_pnl) - 1], [cum_pnl.iloc[-1]],
            color=color, s=60, zorder=5
        )
        plotted += 1

    if plotted == 0:
        print("  [skip] cumulative_pnl — no trades with P&L found")
        plt.close(fig)
        return

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Trade #", fontsize=12)
    ax.set_ylabel("Cumulative P&L (unit stakes)", fontsize=12)
    ax.set_title("Cumulative P&L by Decision Config", fontsize=14, pad=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "cumulative_pnl.png")


# ── Chart 3: Signal funnel ────────────────────────────────────────────────────

def plot_signal_funnel(summary: pd.DataFrame) -> None:
    """
    Horizontal bar chart showing the signal funnel for the default config
    (or the config with the most signals if default isn't present).
    """
    config_to_use = "default"
    if config_to_use not in summary["config"].values:
        config_to_use = summary.sort_values("total_signals", ascending=False)["config"].iloc[0]

    row = summary[summary["config"] == config_to_use].iloc[0]
    df  = load_trades(config_to_use)

    total_markets = int(row.get("total_markets", 0))
    total_signals = int(row.get("total_signals", 0))
    wins          = int(row.get("wins", 0))
    losses        = int(row.get("losses", 0))

    # Try to get additional funnel stage counts from the trades file
    n_revision = total_markets  # fallback — ideally we'd track this
    if df is not None and len(df) > 0:
        n_revision = len(df)  # all markets that reached decision stage

    labels = [
        "Markets w/ price data",
        "A + B completed",
        "Reached decision",
        "GO signal",
        "Wins",
    ]
    values = [total_markets, total_markets, n_revision, total_signals, wins]

    colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], WIN_COLOR]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.55)

    # Annotate counts and percentages
    for bar, val in zip(bars, values):
        pct = f"  {val:,}"
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            pct, va="center", fontsize=10,
        )

    ax.set_xlabel("# Markets", fontsize=11)
    ax.set_title(
        f"Signal Funnel — config: {config_to_use}  (losses={losses})",
        fontsize=13, pad=12,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "signal_funnel.png")


# ── Chart 4: Score scatter (a_score vs b_score, GO trades only) ───────────────

def plot_score_scatter(summary: pd.DataFrame) -> None:
    """
    Scatter plot of a_score vs b_score for GO trades across all configs.
    Color = WIN (green) / LOSS (red). Useful for threshold selection.
    """
    dfs = []
    for name in summary["config"].tolist():
        df = load_trades(name)
        if df is None:
            continue
        go = df[df["decision"] == "GO"].dropna(subset=["pnl", "a_score", "b_score"]).copy()
        if go.empty:
            continue
        go["config"] = name
        dfs.append(go)

    if not dfs:
        print("  [skip] score_scatter — no GO trades found")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined["outcome"] = combined["pnl"].apply(lambda p: "WIN" if p > 0 else "LOSS")
    combined["color"]   = combined["outcome"].map({"WIN": WIN_COLOR, "LOSS": LOSS_COLOR})

    fig, ax = plt.subplots(figsize=(8, 6))

    for outcome, color in [("WIN", WIN_COLOR), ("LOSS", LOSS_COLOR)]:
        sub = combined[combined["outcome"] == outcome]
        ax.scatter(
            sub["a_score"] + np.random.uniform(-0.15, 0.15, len(sub)),
            sub["b_score"] + np.random.uniform(-0.15, 0.15, len(sub)),
            c=color, alpha=0.5, s=30, label=outcome,
        )

    ax.set_xlabel("Agent A — Insider Risk Score", fontsize=12)
    ax.set_ylabel("Agent B — Behavior Score", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_title("GO Trades: Score Distribution (all configs combined)", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "score_scatter.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    summary = load_summary()
    if summary is None:
        return

    print(f"\nGenerating charts → {CHARTS_DIR}")
    plot_leaderboard(summary)
    plot_cumulative_pnl(summary)
    plot_signal_funnel(summary)
    plot_score_scatter(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
