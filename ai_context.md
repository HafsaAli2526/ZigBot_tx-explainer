# AI Context — ZigChain TX Reasoning Engine

> This file provides structured context for AI assistants working on this codebase.

## Project Identity

- **Name:** ZigChain Transaction Reasoning Engine
- **Type:** CLI application — Python (target: API for ZigScan frontend)
- **Domain:** Blockchain transaction analysis (Cosmos SDK / ZigChain)
- **Python version:** 3.11+
- **Package manager:** pip (requirements.txt)
- **No framework** — pure Python with a `src/` package

## Architecture Overview

The system follows a strict **4-layer pipeline** with a **Query Intelligence Layer** and **tiered model routing**:

```
fetch_tx() → normalize_tx() → interpret() → call_llm(complexity=...) → ChatSession + QueryEngine
```

### Layer Responsibilities

| Layer | Module | Role | Uses LLM? |
|---|---|---|---|
| 1 — Fetch | `src/fetcher.py` | Pulls raw JSON from ZigChain Cosmos RPC `/tx?hash=` endpoint | No |
| 2 — Normalize | `src/normalizer.py` | Parses raw RPC events into structured dict: fee, gas, signer, transfers, WASM actions, contract messages. Decodes base64 tx body to extract message types and embedded JSON. | No |
| 3 — Interpret | `src/interpreter.py` | Deterministic rules engine. Classifies tx type (20+ categories), builds summary, generates warnings (failure, gas), annotates transfers/swaps, scores complexity. | No |
| 4 — Explain | `src/llm.py` | Sends normalized data + interpretation to an Ollama-compatible LLM. **Tiered model routing**: simple→fast, moderate→standard, complex→powerful. Supports model warmup and dynamic prompt directives. | **Yes** |
| Query Engine | `src/query_engine.py` | TX-aware intent reasoning gate. Extracts structured question features, validates against tx data, generates context notes + prompt directives, logs rejections. | No |
| Chat | `src/chat.py` | Per-tx session manager. Tracks conversation history with token-budget-aware trimming. Stores `QueryContext` for the loaded tx. | No |
| Config | `src/config.py` | Loads `.env`, exposes all settings as module-level constants. | No |
| CLI | `main.py` | REPL loop. Model warmup on boot. Handles `/tx`, `/raw`, `/interpret`, `/stats`, `/help`, `/quit` commands and free-text questions. | No |

### Critical Design Principles

1. **The LLM does NOT think. It translates.** All factual analysis is deterministic (Layers 2–3).
2. **Tiered model routing.** Complexity score drives model selection — don't burn tokens on simple txs.
3. **Context notes, not rejections.** The query engine never rejects based on data absence — it passes context notes to help the LLM answer accurately.
4. **Dynamic prompt conditioning.** Question quality shapes how the LLM responds (conservative for vague, thorough for specific).

## Data Flow

```
1. STARTUP: warmup_models() → preloads all model tiers into GPU memory
2. User provides TX hash
3. fetcher.py → GET {RPC}/tx?hash=0x{HASH} → raw JSON
4. normalizer.py → Extracts from raw:
   - tx_hash, height, status, code, failure_reason
   - gas (wanted, used, fee, fee_payer, efficiency %)
   - signer, sequence
   - messages (action, sender, module)
   - transfers (from, to, amount, denom, finalized?)
   - wasm_actions (type, contract, parsed swap details)
   - contract_executions
   - tx_body (decoded base64 → msg_types + contract_msgs)
5. interpreter.py → Produces:
   - tx_type (dex_swap | bank_send | staking_delegate | 20+ types)
   - summary (human-readable one-liner)
   - warnings (critical/warning/info with messages)
   - annotations (labeled transfers + swaps)
   - complexity (simple | moderate | complex)
6. QueryContext created from normalized + interpretation (pre-computes tx flags)
7. llm.py → Selects model tier based on complexity → sends prompt → receives explanation
8. ChatSession created → user can ask follow-up questions
9. For each follow-up:
   a. query_engine.check_question() → (allowed, note, directive)
   b. If allowed: call_llm() with complexity + prompt_directive
   c. If blocked: show rejection, log to query_rejections.jsonl
```

## Normalized Data Schema

Output of `normalize_tx()` — the canonical data structure used by all downstream layers:

```python
{
    "tx_hash": str,
    "height": str,
    "status": "success" | "failed",
    "code": int,                    # 0 = success
    "failure_reason": str | None,
    "gas": {
        "wanted": int,
        "used": int,
        "fee": str | None,         # e.g. "5000uzig"
        "fee_payer": str | None,
        "efficiency": float,       # percentage
    },
    "signer": str | None,
    "sequence": int | None,
    "messages": [{"action": str, "sender": str, "module": str, "msg_index": str}],
    "tx_body": {
        "msg_types": [str],        # e.g. ["/cosmwasm.wasm.v1.MsgExecuteContract"]
        "contract_msgs": [dict],   # decoded JSON from contract call payloads
    } | None,
    "transfers": [{
        "from": str, "to": str,
        "amount": int, "denom": str, "raw_amount": str,
        "status": "finalized" | "emitted_but_tx_failed",
        "msg_index": str,
    }],
    "wasm_actions": [{
        "type": str,               # "swap", "provide_liquidity", etc.
        "contract": str,
        "status": "finalized" | "emitted_but_tx_failed",
        "msg_index": str,
        "details": dict,
        "parsed": dict | None,     # only for swap: offer/return/spread/commission/reserves
    }],
    "contract_executions": [{"contract": str, "msg_index": str}],
    "raw_event_count": int,
}
```

## Tiered Model Routing

Defined in `src/llm.py` → `MODEL_TIERS`:

| Complexity Score | Model | Use Case |
|---|---|---|
| `simple` | `LLM_FAST_MODEL` (glm-4.7-flash) | Basic sends, votes, simple staking |
| `moderate` | `LLM_MODEL_NAME` (qwen3:32b) | Swaps, contract calls, standard DeFi |
| `complex` | `LLM_POWERFUL_MODEL` (qwen3-coder-next) | Multi-msg txs, failed contracts, deep analysis |

Complexity is scored deterministically in `interpreter.py` → `_score_complexity()`:
- Points for: multiple messages, WASM actions (×2), many transfers, multiple contract executions (×2), failure (+2)
- Score ≤3 = simple, 4–8 = moderate, >8 = complex

## Query Intelligence Layer

The query engine (`src/query_engine.py`) implements a structured reasoning pipeline:

### Data Structures
- **`QuestionFeatures`** — entities (set), intent (str), scope (str)
- **`QueryContext`** — pre-computed tx flags (has_transfers, has_swap, is_failed, tx_type, complexity)

### Pipeline
1. **Hard domain boundary** — keyword blocklist (price, market, bitcoin, jokes, predictions)
2. **Feature extraction** — maps question text to:
   - Entities: `{gas, signer, transfer, failure, amount, address, sequence, contract, swap, status, events, staking, governance, ibc}`
   - Intents: `{causal, actor, info, explain, quantitative, unknown}`
3. **TX-data validation** — validates features against QueryContext. **Never rejects on absence** — returns context notes instead
4. **Prompt conditioning** — generates directives based on question quality:
   - Vague → "answer conservatively"
   - Failure on success → "clearly state it succeeded"
   - Causal + failed → "provide root-cause analysis"
   - Specific multi-entity → "thorough structured explanation"
   - Complex tx + explain → "step-by-step breakdown"

### Return Signature
```python
check_question(question, ctx) -> tuple[bool, str | None, str | None]
# (is_allowed, note_or_rejection, prompt_directive)
```

## Transaction Type Classification

The interpreter classifies transactions using a priority chain:

1. **WASM action types** → `dex_swap`, `liquidity_provision`, `liquidity_withdrawal`
2. **Contract message keys** → `swap`, `provide_liquidity`, `transfer`, `send`, `mint`, `burn`, `stake`
3. **Cosmos message type patterns** → `MsgSend`, `MsgDelegate`, `MsgVote`, `MsgTransfer`, etc.
4. **Module fallback** → `wasm` → contract_execution, `bank` → bank_send, `staking` → staking

## External Dependencies

| Dependency | Endpoint | Auth |
|---|---|---|
| ZigChain Cosmos RPC | `ZIGCHAIN_RPC_URL` (default: `https://zigchain-rpc.degenter.io`) | None |
| Ollama-style LLM API | `LLM_API_URL` → POST `/api/generate` | Optional HTTP Basic Auth |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ZIGCHAIN_RPC_URL` | `https://zigchain-rpc.degenter.io` | ZigChain RPC endpoint |
| `LLM_API_URL` | *required* | Ollama-compatible API URL |
| `LLM_MODEL_NAME` | `qwen3:32b` | Standard model (moderate tier) |
| `LLM_FAST_MODEL` | `glm-4.7-flash:latest` | Fast model (simple tier) |
| `LLM_POWERFUL_MODEL` | `qwen3-coder-next:latest` | Powerful model (complex tier) |
| `LLM_API_USER` | — | HTTP Basic Auth user |
| `LLM_API_PASSWORD` | — | HTTP Basic Auth password |
| `LLM_TEMPERATURE` | `0.1` | Low temperature for factual output |
| `LLM_TOP_P` | `0.9` | Nucleus sampling |
| `LLM_MAX_TOKENS` | `2048` | Max response tokens |
| `LLM_TIMEOUT` | `120` | Request timeout in seconds |

## File-by-File Summary

| File | Lines | Purpose |
|---|---|---|
| `main.py` | ~200 | CLI REPL — entry point, model warmup, command routing, colored output |
| `src/config.py` | 23 | `.env` loader, module-level config constants |
| `src/fetcher.py` | 43 | HTTP GET to RPC, hash normalization, error handling |
| `src/normalizer.py` | 273 | Event parsing, base64 decoding, transfer/WASM/contract extraction |
| `src/interpreter.py` | 250 | Rules engine — classification, summary, warnings, annotations, complexity |
| `src/llm.py` | ~170 | Tiered LLM routing, warmup, prompt construction, Ollama API call |
| `src/query_engine.py` | ~260 | Query Intelligence Layer — feature extraction, tx-aware validation, prompt conditioning, rejection logging |
| `src/chat.py` | ~55 | Chat session with QueryContext and token-budget-aware history |

## Generated Files

| File | Purpose |
|---|---|
| `query_rejections.jsonl` | Auto-generated log of rejected queries (timestamp, question, reason, tx_type). Training data for improving the query engine. |

## Known Gaps / Future Work

- No tests exist — correctness is critical for a tx analysis tool.
- No async support — all RPC and LLM calls are synchronous blocking.
- No caching of fetched transaction data (txs are immutable, should cache).
- The base64 tx body decoding uses raw string matching rather than Protobuf deserialization (pragmatic but lossy for edge cases).
- No API layer yet — CLI only. Target: FastAPI service for ZigScan frontend.
- No multi-turn suggested follow-ups (e.g., "ask about gas / failure / transfers").

## How to Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Configure .env
python main.py <TX_HASH>
```
