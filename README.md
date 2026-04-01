# ZigChain Transaction Reasoning Engine

A CLI-based **transaction analysis and explanation system** for [ZigChain](https://zigchain.com), a Cosmos SDK blockchain. Feed it a transaction hash, and it will deterministically analyze, classify, and explain the transaction in plain English — then let you ask follow-up questions in a guarded chat session.

> **Target integration:** ZigScan frontend (API mode). CLI is the development/testing interface.

## How It Works

The engine processes transactions through a strict **4-layer pipeline** with an intelligent query gate:

```
TX Hash
  │
  ▼
┌───────────────────────┐
│  Layer 1 — Fetcher    │  Pulls raw data from ZigChain Cosmos RPC
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Layer 2 — Normalizer │  Decodes events, transfers, WASM actions,
│                       │  contract messages into structured JSON
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Layer 3 — Interpreter│  Deterministic rules engine — classifies
│                       │  tx type, generates warnings, scores complexity
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Layer 4 — LLM        │  Translates structured data into a
│  (Tiered Routing)     │  human-readable explanation (no invention)
│                       │  simple→⚡fast | moderate→🧠std | complex→🔬powerful
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Query Intelligence   │  TX-aware reasoning gate for follow-up Q&A
│  Layer                │  (replaces naive keyword filtering)
└───────────────────────┘
```

> **Design philosophy:** The LLM is a *translator*, not a *thinker*. All factual analysis is done deterministically in Layers 2–3. The LLM reformats structured truth into natural language under strict system constraints.

## Key Features

### Single Model
The engine uses one selected explanation model for all requests:
- **Default model:** `glm-4.7-flash:latest`
- Transaction complexity is still computed deterministically for analysis quality and stats
- Startup warmup only loads the selected production model

### Model Warmup
On startup, only the selected production model is preloaded so the first user query avoids a cold start without paying the cost of loading multiple models.

### Query Intelligence Layer
Follow-up questions are validated through a structured reasoning pipeline:
1. **Feature extraction** — maps question to entities (gas, signer, swap...) + intent (causal, explain, quantitative...)
2. **TX-data validation** — checks if the question can be answered from *this specific transaction*
3. **Dynamic prompt conditioning** — shapes LLM response quality based on question strength
4. **Rejection logging** — blocked queries are logged to `query_rejections.jsonl` for future analysis

## Supported Transaction Types

| Category | Types |
|---|---|
| **DeFi** | DEX Swap, Liquidity Provision / Withdrawal |
| **Banking** | Token Send, Multi-Send |
| **Staking** | Delegate, Undelegate, Redelegate, Reward Claim |
| **Governance** | Vote |
| **IBC** | IBC Transfer, IBC Relay |
| **Smart Contracts** | CosmWasm Execute, Instantiate, Store Code |
| **Tokens** | Mint, Burn, Transfer |

## Quick Start

### Prerequisites

- **Python 3.11+**
- Access to a **ZigChain RPC endpoint**
- Access to an **Ollama-compatible LLM API** (self-hosted or remote)

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd ZIG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt


### Usage

```bash
# Analyze a transaction directly
python main.py <TX_HASH>

# Or start the interactive shell and load a tx later
python main.py
```

### CLI Commands

| Command | Description |
|---|---|
| `/tx <hash>` | Load and analyze a new transaction |
| `/raw` | Print the normalized JSON for the current tx |
| `/interpret` | Print the deterministic interpretation |
| `/stats` | Show session stats (type, complexity, model, message count) |
| `/help` | Show help |
| `/quit` | Exit |
| *free text* | Ask a follow-up question about the current tx |

## Project Structure

```
ZIG/
├── main.py                 # CLI entry point, REPL loop, model warmup
├── requirements.txt        # Python dependencies
├── .env                    # Environment configuration (not committed)
├── query_rejections.jsonl  # Auto-generated rejection log (training data)
└── src/
    ├── __init__.py
    ├── config.py           # Loads .env, exposes settings
    ├── fetcher.py          # Layer 1 — RPC data fetch
    ├── normalizer.py       # Layer 2 — Raw → structured JSON
    ├── interpreter.py      # Layer 3 — Deterministic rules engine
    ├── llm.py              # Layer 4 — Single-model LLM translation layer
    ├── query_engine.py     # Query Intelligence Layer (tx-aware intent gate)
    └── chat.py             # Per-TX chat session manager
```

## Dependencies

| Package | Purpose |
|---|---|
| `requests` | HTTP calls to ZigChain RPC and LLM API |
| `python-dotenv` | `.env` file loading |
| `colorama` | Terminal colors for the CLI |

## License

This project is not yet licensed. Contact the author for usage terms.
