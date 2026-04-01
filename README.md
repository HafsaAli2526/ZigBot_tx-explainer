# ZigChain Transaction Reasoning Engine

A deterministic, transaction-scoped intelligence system for ZigChain / Cosmos transactions.

This engine does not behave like a generic AI chatbot.  
It analyzes a single transaction using structured logic, then uses an LLM only as a controlled explanation layer.

---

## 🚀 Why This Exists

Most blockchain “AI explainers”:

- hallucinate
- lose context
- mix unrelated data
- give generic answers

This system is built differently:

- **Facts first** — deterministic extraction and interpretation
- **Strict scope** — answers only about the loaded transaction
- **Controlled intelligence** — LLM explains, never decides
- **Context-aware** — adds protocol meaning without overriding truth

---

## 🧠 Design Philosophy

- The **transaction is the source of truth**
- Deterministic layers extract and interpret facts
- The **LLM is a translator, not a thinker**
- Chat is **retrieval + explanation**, not re-analysis
- Context (docs) is **support**, never authority
- If context conflicts with facts → **facts win**

---

## ⚙️ Core Capabilities

- Deterministic transaction parsing and classification
- Fact-index-based direct answers (no LLM latency)
- Structured explanation templates
- Context-aware explanations via:
  - Manifest (zero-latency module knowledge)
  - Optional vector-doc fallback
- Strict query scoping and rejection system
- Session-based caching and repeat-query optimization
- Multi-interface support (Web, CLI, Telegram)

---

## 🏗️ Architecture Overview

```text
TX Hash
  → Fetcher
  → Normalizer
  → Interpreter
  → Fact Index
  → Context Keys
  → Context Router
  → Response Engine
      ├── Deterministic (facts)
      ├── Template
      ├── Manifest Context
      ├── Vector Context (optional)
      └── LLM (controlled)

🧠 Routing Logic (Critical)
Every user query is processed through a decision system:

Deterministic

Direct fact lookup

No LLM call

Example: "Who paid gas?"

Template

Structured explanation using known patterns

Example: failure explanation

Manifest Context

Module-level explanation (bank, wasm, staking)

Zero latency, deterministic

Vector Context (Optional)

Only triggered for deeper explanation questions

Uses approved documentation only

LLM

Used only when necessary

Receives facts + context

Cannot override transaction truth

Refusal

Out-of-scope queries are rejected strictly

🧪 Example Output
Transaction failed due to insufficient funds.

Fee payer: zig1x...

Gas used: 122,045

Attempted transfer: 10,000 uzig

Explanation:

The transaction attempted to execute a contract operation, but the account balance was lower than required.
As a result, execution failed before any state changes were finalized.

🧩 Context-Aware Reasoning (RAG Done Right)
This system does NOT use naive global RAG.

Instead:

Tier 1 — Manifest Context (Deterministic)
Local, trusted module explanations

Zero latency

Always aligned with protocol behavior

Tier 2 — Vector Context (Optional)
Triggered only for explanatory queries

Limited to approved documentation

Never overrides transaction facts

Key Rule:
Context explains the transaction — it never rewrites it.

🖥️ Interfaces

Interface	Entry Point
Web UI / API	web_ui.py
CLI REPL	main.py
Telegram Bot	tg_bot.py

⚡ Quick Start (Local)

1. Setup
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt

2. Configure
Create .env from .env.example:

ZIGCHAIN_RPC_URL=https://zigchain-rpc.degenter.io

LLM_API_URL=http://your-llm-host:11434/api/generate
LLM_MODEL_NAME=qwen3:32b

WEB_UI_HOST=0.0.0.0
WEB_UI_PORT=8787

3. Run
Web UI:

python web_ui.py
CLI:

python main.py
Telegram Bot:

python tg_bot.py

---

🧪 Quick Demo
Open:

http://localhost:8787
Paste a transaction hash

Try:

Why did this fail?
Who paid gas?
Explain this transaction
What does this module do?

---

🐳 Docker Deployment

Run Web UI
docker compose up --build
Optional
# Telegram bot
docker compose --profile telegram up

# CLI
docker compose --profile cli run --rm cli

---

📁 Project Structure
.
├── main.py
├── web_ui.py
├── tg_bot.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── src/
    ├── fetcher.py
    ├── normalizer.py
    ├── interpreter.py
    ├── facts.py
    ├── fact_formatter.py
    ├── query_engine.py
    ├── context_router.py
    ├── context_manifest.py
    ├── context_keys.py
    ├── context_vector.py
    ├── llm.py
    ├── chat.py
    ├── tx_digest.py
    ├── context_manifest_store/
    └── context_vector_corpus/

---

⚠️ Limitations
Explanations are limited to available transaction data

Unknown contracts may have limited context

Vector retrieval is optional and scoped

Not designed for multi-transaction analytics

---

🚀 Future Direction
This system is designed to evolve into:

ZigChain Intelligence API Layer

Explorer integrations (e.g., ZigScan)

Wallet-level transaction explainability

AI-powered analytics and automation

---

🔐 Security Notes
Never commit .env

Keep API keys and tokens private

Use .env.example for shared configuration

---

📌 Final Note
This is not just a tool —
it is a transaction intelligence layer designed for real-world integration.

---

License
Not yet licensed. Contact the author for usage terms.

---

# 🧠 SYSTEM EVOLUTION — WHAT CHANGED (IMPORTANT)

---

# 🔴 OLD SYSTEM (Before)
TX → Fetch → Normalize → Interpret → LLM → Chat


Problems:
- LLM doing too much ❌
- repeated computation ❌
- slow ❌
- weak control ❌
- no context intelligence ❌

---

# 🟢 NEW SYSTEM (Now)
TX → Deterministic Core → Fact Index → Routing → Controlled LLM


---

# 🧱 LAYER BREAKDOWN (FINAL)

## 1. Fetcher
👉 Gets raw transaction from RPC

---

## 2. Normalizer
👉 Converts raw logs → structured JSON

---

## 3. Interpreter
👉 Classifies:
- tx type
- warnings
- complexity
- annotations

---

## 4. Fact Index (🔥 BIG UPGRADE)

👉 Converts everything into **directly answerable truth**

Example:
status = failed
fee_payer = zig1...
gas_used = 122045
failure_reason = insufficient funds


👉 This removes need for LLM in 70% cases

---

## 5. Context Layer (🔥 NEW)

### Context Keys
👉 Extract:
- module
- message type
- tx type

---

### Manifest (Tier 1)
👉 Deterministic explanations
MsgExecuteContract → explanation

---

### Vector (Tier 2)
👉 Only for deep explanation
👉 Optional fallback

---

## 6. Query Engine (UPGRADED)

👉 Not just filtering anymore

It:
- understands intent
- checks answerability
- routes query

---

## 7. Context Router (🔥 CORE INTELLIGENCE)

👉 Decides:
fact?
template?
context?
llm?
reject?
```
