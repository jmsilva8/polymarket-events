# Insider Alpha on Prediction Markets

An experimental multi-agent system to detect whether insider-driven price signals in prediction markets are identifiable by an outside observer — and whether those signals can be traded on profitably.

---

## Motivation

Prediction markets like [Polymarket](https://polymarket.com) and [Kalshi](https://kalshi.com) aggregate beliefs about future events. Because some participants may hold non-public information (insiders), their activity could leave detectable traces in price and volume data before market resolution.

The central question this project explores is: **can a neutral outside observer identify markets where insiders are likely trading, and extract a reliable edge from those signals?**

To investigate this, we built a multi-agent AI pipeline that combines textual analysis of market descriptions with quantitative analysis of historical price behavior. The system was backtested on Polymarket data from 2024–2025.

**Finding**: The best configurations produce ~4–5% ROI before transaction costs. This is not sufficient for live trading at present, but represents an interesting starting point. The main limitations were data granularity (historical price data available only at ~12-hour intervals) and the absence of volume history for backtesting — both of which constrained the quantitative signal detection. With higher-frequency data and further parameter tuning, there may be something to build on here.

---

## System Architecture

### Data Layer

- **Polymarket**: Data collected via the Gamma API (market metadata, tags, prices) and the CLOB API (price timeseries). Historical markets from 2024–2025.
- **Kalshi**: Market metadata collected via the public trade API. Kalshi markets were included in classification but could not be backtested (no historical price API available).
- All data is normalized to a unified market schema (`UnifiedMarket`, `UnifiedEvent`) and cached locally.

### Multi-Agent Pipeline

The classification system is built with [LangGraph](https://github.com/langchain-ai/langgraph) and consists of four specialized agents:

```
         ┌─────────────┐
         │ load_markets│
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │filter_markets│
         └──────┬───────┘
                │
      ┌─────────┴──────────┐
      ▼                    ▼
┌───────────┐        ┌───────────┐
│  Agent A  │        │  Agent B  │   (run in parallel)
└───────────┘        └───────────┘
      └─────────┬──────────┘
                ▼
        ┌───────────────┐
        │Revision Agent │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │Decision Agent │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ export results│
        └───────────────┘
```

**Agent A — Insider Risk Scorer**

Analyzes market text (title, description, category, tags) to assess the likelihood of information asymmetry. It identifies who might hold advance knowledge (`info_holders`), how that information could leak (`leak_vectors`), and produces an insider risk score (1–10). Agent A is deliberately blind to price and volume data.

**Agent B — Quantitative Signal Detector**

Analyzes price timeseries to detect unusual market behavior. It runs a set of deterministic Python tools:

- **Price jump detector**: flags abnormal price spikes (absolute and relative)
- **Momentum analyzer**: linear regression over multiple time windows (6/12/24/48h) to determine if a move is part of a sustained trend or an isolated jump
- **Volume spike checker**: detects anomalous volume surges (3× baseline)
- **Consistency checker**: validates that price and volume signals are coherent

Agent B was designed to detect the combination of abnormal price movement, abnormal volume/liquidity changes, and momentum direction. In practice, backtesting was limited to price-only signals because hourly volume history is not available via the Polymarket API — this significantly constrained Agent B's capabilities.

Agent B is blind to market text.

**Revision Agent**

Cross-validates the outputs of Agents A and B. It checks for internal coherence within each report and detects cross-agent conflicts (e.g., high insider risk score but no corresponding price movement). It can send feedback to either agent for up to two revision iterations before passing a recommendation to the Decision Agent. Key flags include `DIRECTIONAL_CONFLICT` (A and B disagree), `PUBLIC_INFO_ADJUSTED` (signal explained by public information), and `PRE_SIGNAL` (signal exists but market has not yet resolved).

**Decision Agent**

Synthesizes all signals into a final `GO` or `SKIP` decision with an estimated edge. Operates in two modes:

- **Deterministic** (primary): weighted scoring formula — `0.4 × A_score + 0.3 × B_score + 0.3 × revision_boost` — with configurable thresholds
- **LLM-based** (optional): the model reasons through all inputs directly

All LLM calls use `temperature=0` for reproducibility. Outputs are structured via Pydantic schemas.

### Backtesting Engine

The backtesting framework evaluates historical Polymarket markets against a parameter grid. For each combination, a `BUY YES` signal is generated when:
- The market's insider risk score meets a minimum threshold
- The price crosses a configured threshold within a defined window before resolution
- Optional: an abnormal price move is also detected

Six decision configurations were evaluated, each re-run against the same fixed Agent A / B / Revision outputs:

| Config | Description |
|---|---|
| `default` | Balanced defaults (A: 0.4, B: 0.3, revision boost: 0.3) |
| `aggressive` | Lower GO thresholds to generate more signals |
| `conservative` | Higher thresholds — only acts on strong signals |
| `a_heavy` | More weight on Agent A (insider risk text score) |
| `b_heavy` | More weight on Agent B (quantitative price signals) |
| `no_conf_penalty` | Ignores confidence levels when scoring |

An optional seventh config (`llm`) uses LLM reasoning for the final decision instead of the deterministic formula, enabled via `--llm-decision`.

---

## Results

Best configurations achieved **~4–5% ROI before transaction costs** across the backtested dataset. This is below the break-even threshold once Polymarket fees are accounted for, so live deployment is not warranted at this stage.

Key limiting factors:

- **Price granularity**: Historical price data is only available at ~12-hour intervals via the CLOB API, making it impossible to detect short-window signals reliably. Live deployment would have access to hourly (or finer) data.
- **No volume history**: There is no API for historical volume timeseries, so Agent B's volume spike and liquidity analysis could not contribute meaningfully in backtesting.
- **Kalshi exclusion**: Kalshi markets could not be included in the backtest due to the absence of a historical price API.
- **LLM parameter sensitivity**: Agent A and the Revision Agent are sensitive to prompt and model configuration; there is room for further tuning.

The 4–5% ROI figure is nonetheless an interesting starting point. With higher-frequency pricing data, volume history, and better-calibrated agent parameters, the signal may become more actionable.

---

## How to Run the Demo

### Prerequisites

- Python 3.10+
- An OpenAI API key (for Agent A, B, Revision, Decision)
- Optionally: an Anthropic API key (fallback LLM)

### Setup

```bash
git clone <repo-url>
cd polymarket-events
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # optional
```

### Run the Notebook (Recommended)

The easiest way to explore the system end-to-end is via the Jupyter notebook. It uses a **pre-built demo dataset** (100 markets with pre-computed LLM outputs) so no API calls or credits are required:

```bash
jupyter notebook INSIDER_ALPHA_FINAL.ipynb
```

The notebook walks through:
1. Loading the demo dataset
2. Running the full multi-agent pipeline
3. Inspecting agent outputs and revision flags
4. Viewing backtest results and ranked opportunities

### Run the Classification Pipeline

To classify markets using live data:

```bash
python scripts/run_classification.py
```

This downloads markets from Polymarket and Kalshi, runs all agents, and exports results to `data/exports/`.

### Run the Backtesting Pipeline

```bash
# Full backtest (runs LLM stages, then sweeps parameters)
python scripts/run_backtest_v2.py

# Skip LLM stages if already cached
python scripts/run_backtest_v2.py --decisions-only

# Include LLM-based decision variant
python scripts/run_backtest_v2.py --llm-decision
```

Results are exported to `data/backtest/results/`.

---

## Project Structure

```
polymarket-events/
├── src/
│   ├── ai_layer/
│   │   ├── agent_a/           # Insider risk scorer (text analysis)
│   │   ├── agent_b/           # Quantitative signal detector (price/volume)
│   │   ├── decision_agent/    # GO/SKIP decision with edge estimation
│   │   └── revision_agent.py  # Cross-validation and feedback loop
│   ├── data_layer/            # API clients, caching, data models
│   ├── graph.py               # LangGraph multi-agent orchestration
│   └── config.py
├── scripts/
│   ├── run_classification.py  # Live classification pipeline
│   ├── run_backtest_v2.py     # Backtesting pipeline
│   └── build_demo_dataset.py  # Builds self-contained demo dataset
├── demo/
│   └── data/
│       ├── exports/           # polymarket_tagged_sample.parquet
│       ├── backtest/          # Agent JSONL outputs + results CSVs
│       ├── price_history.db   # SQLite price snapshots
│       └── archetypes.json    # Market archetype definitions
├── data/                      # Local cache, exports, backtest results (gitignored)
├── tests/
└── INSIDER_ALPHA_FINAL.ipynb  # Interactive demo notebook
```

---

## Authors

Claude Code (Anthropic), Jose Manuel Silva, Santiago Bambach, Ricardo Ramos, Jose Romero
