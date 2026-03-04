"""
Generate charts from backtest results.

Usage:
    python scripts/plot_results.py

Reads:
    data/exports/backtest_sweep_results.csv
    data/exports/backtest_best_trades.csv   (optional)

Writes:
    data/exports/charts/roi_heatmap_<strategy>.png
    data/exports/charts/strategy_comparison.png
    data/exports/charts/cumulative_pnl.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CHARTS_DIR = Path("data/exports/charts")
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ── Helpers ────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 1: ROI heatmap (price_threshold × hours_before_close) ───

def plot_roi_heatmap(df: pd.DataFrame, strategy_type: str) -> None:
    """Heatmap of best ROI per (price_threshold, hours_before_close) cell."""
    sub = df[df["strategy_type"] == strategy_type].copy()
    sub = sub.dropna(subset=["hours_before_close"])
    if sub.empty:
        print(f"  Skipping heatmap for '{strategy_type}' — no windowed data.")
        return

    # Best ROI for each grid cell (across all min_leak_score / min_price_jump combos)
    pivot = (
        sub.groupby(["hours_before_close", "price_threshold"])["roi_pct"]
        .max()
        .unstack("price_threshold")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = max(abs(pivot.values[np.isfinite(pivot.values)].max()), 1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.0f}h" for y in pivot.index], fontsize=9)
    ax.set_xlabel("Price Threshold")
    ax.set_ylabel("Hours Before Close")
    ax.set_title(f"ROI % Heatmap — {strategy_type}\n(best across all score/jump combos)")

    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="ROI %", fraction=0.03)
    _savefig(fig, CHARTS_DIR / f"roi_heatmap_{strategy_type}.png")


# ── Chart 2: Strategy comparison bar chart ─────────────────────────

def plot_strategy_comparison(df: pd.DataFrame) -> None:
    """Side-by-side ROI and win-rate bars, best combo per strategy."""
    min_signals = 5
    src = df[df["total_signals"] >= min_signals] if (df["total_signals"] >= min_signals).any() else df
    # For each strategy type, pick the row with the highest ROI
    best_rows = src.loc[src.groupby("strategy_type")["roi_pct"].idxmax()].reset_index(drop=True)

    strategies = best_rows["strategy_type"].tolist()
    x = np.arange(len(strategies))
    colors = PALETTE[: len(strategies)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Strategy Comparison — Best Parameter Combo per Type", fontsize=12)

    # ROI
    bars1 = ax1.bar(x, best_rows["roi_pct"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("ROI %")
    ax1.set_title("Best ROI")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars1, best_rows["roi_pct"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3 * (1 if val >= 0 else -1),
                 f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=8)

    # Win rate
    bars2 = ax2.bar(x, best_rows["win_rate"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Win Rate %")
    ax2.set_title("Best Win Rate")
    ax2.axhline(50, color="red", linewidth=0.8, linestyle="--", label="50% (coin-flip)")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, best_rows["win_rate"] * 100):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # Annotate signal counts
    for ax, df_row in [(ax1, best_rows), (ax2, best_rows)]:
        for i, (_, row) in enumerate(df_row.iterrows()):
            ax.text(i, ax.get_ylim()[0] * 0.95,
                    f"n={int(row['total_signals'])}", ha="center",
                    va="bottom", fontsize=7, color="grey")

    plt.tight_layout()
    _savefig(fig, CHARTS_DIR / "strategy_comparison.png")


# ── Chart 3: Cumulative P&L curve ─────────────────────────────────

def plot_cumulative_pnl(trades_df: pd.DataFrame) -> None:
    """Chronological cumulative P&L with drawdown shading."""
    if trades_df.empty:
        print("  Skipping cumulative P&L — no trade data.")
        return

    df = trades_df.copy()
    if "signal_time" in df.columns:
        df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
        df = df.sort_values("signal_time")
    df = df.reset_index(drop=True)

    cumulative = df["pnl"].cumsum().values
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Best Insider Alpha Strategy — Cumulative P&L", fontsize=12)

    xs = np.arange(len(cumulative))

    # P&L curve
    ax1.plot(xs, cumulative, color=PALETTE[0], linewidth=2, label="Cumulative P&L")
    ax1.fill_between(xs, cumulative, alpha=0.15, color=PALETTE[0])
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Win/Loss markers
    wins = df[df["pnl"] > 0].index.tolist()
    losses = df[df["pnl"] <= 0].index.tolist()
    if wins:
        ax1.scatter(wins, cumulative[wins], color="#4CAF50", s=20, zorder=5, label="Win")
    if losses:
        ax1.scatter(losses, cumulative[losses], color="#F44336", s=20, zorder=5, label="Loss")

    # Drawdown
    ax2.fill_between(xs, -drawdown, 0, color="#F44336", alpha=0.4, label="Drawdown")
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Trade #")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    _savefig(fig, CHARTS_DIR / "cumulative_pnl.png")


# ── Chart 4: ROI distribution by min_price_jump ───────────────────

def plot_jump_comparison(df: pd.DataFrame) -> None:
    """Box plot of ROI distribution for each min_price_jump value."""
    alpha = df[df["strategy_type"] == "insider_alpha"].copy()
    if alpha.empty or "min_price_jump" not in alpha.columns:
        return

    alpha["jump_label"] = alpha["min_price_jump"].apply(
        lambda v: "None (threshold only)" if pd.isna(v) or v is None else f"+{v:.2f}"
    )
    groups = [grp["roi_pct"].values for _, grp in alpha.groupby("jump_label", sort=False)]
    labels = [label for label, _ in alpha.groupby("jump_label", sort=False)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("min_price_jump setting")
    ax.set_ylabel("ROI % across all other parameter combos")
    ax.set_title("Insider Alpha — ROI Distribution by Price-Jump Filter")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _savefig(fig, CHARTS_DIR / "jump_comparison.png")


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    sweep_path = Path("data/exports/backtest_sweep_results.csv")
    trades_path = Path("data/exports/backtest_best_trades.csv")

    if not sweep_path.exists():
        print(f"ERROR: {sweep_path} not found. Run scripts/run_backtest.py first.")
        sys.exit(1)

    df = pd.read_csv(sweep_path)
    print(f"Loaded {len(df)} rows from {sweep_path}")

    if "strategy_type" not in df.columns:
        df["strategy_type"] = "insider_alpha"

    strategy_types = df["strategy_type"].unique().tolist()
    print(f"Strategy types: {strategy_types}")
    print(f"\nGenerating charts into {CHARTS_DIR}/...")

    # Heatmaps per strategy (only windowed strategies have hours_before_close)
    for stype in strategy_types:
        plot_roi_heatmap(df, stype)

    # Strategy comparison
    plot_strategy_comparison(df)

    # Price-jump comparison (insider_alpha only)
    plot_jump_comparison(df)

    # Cumulative P&L
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)
        plot_cumulative_pnl(trades_df)
    else:
        print(f"  {trades_path} not found — skipping cumulative P&L chart.")

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
