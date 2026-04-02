"""
Query Intelligence Layer — Transaction-Aware Reasoning Gate

Core question: "Can this question be answered using THIS transaction's data?"

Not a keyword filter. A structured reasoning system that:
1. Extracts question features (entities + intent)
2. Validates against actual tx data + interpretation
3. Returns context notes for the LLM instead of hard rejections
4. Provides dynamic prompt directives based on question quality
5. Logs rejected queries for future analysis
"""

from html import entities
import json
import os
from datetime import datetime, timezone

from src.context_router import ContextRouter
from src.fact_formatter import format_direct_fact_entities

# ══════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════

class QuestionFeatures:
    """Structured representation of what a user question is asking about."""

    def __init__(self):
        self.entities: set[str] = set()    # gas, signer, transfer, swap, etc.
        self.intent: str = "unknown"       # causal, actor, info, explain, quantitative, unknown
        self.scope: str = "unknown"        # tx_specific, off_topic, unknown

    def __repr__(self):
        return f"QuestionFeatures(entities={self.entities}, intent={self.intent}, scope={self.scope})"


class QueryContext:
    """Transaction context for query evaluation.
    Pre-computes tx characteristics so validation is fast lookups, not repeated scans.
    """

    def __init__(self, normalized_data: dict, interpretation: dict):
        self.tx = normalized_data
        self.interpretation = interpretation

        # Pre-computed flags
        self.tx_type: str = interpretation.get("tx_type", "unknown")
        self.complexity: str = interpretation.get("complexity", "simple")
        self.has_transfers: bool = bool(normalized_data.get("transfers"))
        self.has_wasm: bool = bool(normalized_data.get("wasm_actions"))
        self.has_contracts: bool = bool(normalized_data.get("contract_executions"))
        self.is_failed: bool = normalized_data.get("status") == "failed"
        self.has_swap: bool = any(
            a.get("type") == "swap" for a in normalized_data.get("wasm_actions", [])
        )


# ══════════════════════════════════════════════
# Constants — Controlled Vocabulary
# ══════════════════════════════════════════════

REJECTION_MESSAGE = "This assistant only answers questions about this specific transaction."

# Hard domain boundary — topics that are NEVER about a specific tx
HARD_REJECT_CONCEPTS = [
    "price", "market", "buy", "sell", "investment",
    "bitcoin", "ethereum", "solana",
    "joke", "who are you", "news", "predict",
    "forecast", "portfolio", "trading volume",
    "market cap", "should i", "weather", "opinion",
]

# Controlled entity mapping — structured, not keyword spam
ENTITY_MAP = {
    "gas":           ["gas", "cost", "charged", "paid", "deducted", "expensive"],
    "fee":           ["fee", "fee amount", "tx fee", "transaction fee", "fee payer", "gas fee"],
    "signer":        ["signer", "sender", "payer", "who sent", "who signed", "who paid", "who initiated", "who submitted"],
    "transfer":      ["transfer", "send", "receive", "sent", "received", "move", "went", "moved"],
    "failure":       ["fail", "error", "wrong", "broke", "break", "revert"],
    "amount":        ["amount", "how much", "how many", "value", "total"],
    "address":       ["address", "wallet", "account", "recipient", "destination"],
    "sequence":      ["sequence", "nonce"],
    "contract":      ["contract", "execute", "wasm", "smart contract", "invoke"],
    "swap":          ["swap", "trade", "exchange", "slippage", "spread", "commission", "liquidity"],
    "status":        ["status", "success", "succeed", "work", "go through", "confirm"],
    "events":        ["event", "log", "emit"],
    "staking":       ["stake", "delegate", "undelegate", "redelegate", "bond", "unbond", "validator"],
    "governance":    ["vote", "governance", "proposal"],
    "ibc":           ["ibc", "cross-chain", "relay", "packet"],
    "memo":          ["memo", "note", "message text", "annotation", "label"],
    "fee_recipient": ["fee recipient", "who got the fee", "fee receiver", "fee collector", "who received the fee"],
}

# Intent detection based on question starters / patterns
INTENT_PATTERNS = {
    "causal":       ["why", "how come", "how did", "what caused", "reason"],
    "actor":        ["who"],
    "info":         ["what", "which", "where"],
    "explain":      ["explain", "tell me", "describe", "break down", "walk me through",
                     "summarize", "summary", "detail", "what happen", "what occur",
                     "what is this", "what did"],
    "quantitative": ["how much", "how many", "how long"],
}

DIRECT_FACT_ENTITIES = {"signer", "gas", "fee", "status", "failure", "memo"}


# ══════════════════════════════════════════════
# Step 1 — Structured Feature Extraction
# ══════════════════════════════════════════════

def extract_features(question: str) -> QuestionFeatures:
    """Extract structured features from a natural-language question.

    Maps raw text into controlled entities + intent + scope.
    """
    q = question.lower().strip()
    features = QuestionFeatures()

    # Intent detection — check patterns
    for intent, patterns in INTENT_PATTERNS.items():
        if any(q.startswith(p) or f" {p}" in q for p in patterns):
            features.intent = intent
            break

    # Entity detection — controlled mapping
    for entity, signals in ENTITY_MAP.items():
        if any(signal in q for signal in signals):
            features.entities.add(entity)

    # Scope detection
    if any(concept in q for concept in HARD_REJECT_CONCEPTS):
        features.scope = "off_topic"
    elif features.entities or features.intent in ("causal", "explain", "actor", "info", "quantitative"):
        features.scope = "tx_specific"
    else:
        features.scope = "unknown"

    return features


# ══════════════════════════════════════════════
# Step 2 — TX-Aware Validation
# ══════════════════════════════════════════════

def validate_against_tx(features: QuestionFeatures, ctx: QueryContext) -> tuple[bool, str | None]:
    """Validate question against actual transaction data.

    DESIGN PRINCIPLE: Never reject based on data absence.
    Instead, return context notes that help the LLM answer accurately.

    Returns (is_valid, context_note_or_None).
    """
    notes = []

    # General explanation — always valid
    if features.intent == "explain":
        return True, None

    # Status — always answerable
    if "status" in features.entities:
        return True, None

    # Gas/fee — always answerable (every tx has gas)
    if "gas" in features.entities:
        return True, None

    # Fee ? always answerable
    if "fee" in features.entities:
        return True, None

    # Signer — always answerable
    if "signer" in features.entities:
        return True, None

    # Address — always answerable
    if "address" in features.entities:
        return True, None

    # Amount — always answerable
    if "amount" in features.entities:
        return True, None

    # Sequence — always answerable
    if "sequence" in features.entities:
        return True, None

    # Events — always answerable
    if "events" in features.entities:
        return True, None

    # Failure on a successful tx — VALID, answer is "it didn't fail"
    if "failure" in features.entities and not ctx.is_failed:
        return True, "Note: this transaction did not fail — it completed successfully."

    if "failure" in features.entities and ctx.is_failed:
        return True, None

    # Transfers — valid, but note absence
    if "transfer" in features.entities and not ctx.has_transfers:
        notes.append("Note: this transaction has no transfer events.")

    # Contracts — valid, but note absence
    if "contract" in features.entities and not ctx.has_contracts and not ctx.has_wasm:
        notes.append("Note: this transaction does not involve any contract execution.")

    # Swaps — valid, but note absence
    if "swap" in features.entities and not ctx.has_swap:
        notes.append("Note: this transaction does not contain a swap action.")

    # Staking — valid, but note if not relevant
    if "staking" in features.entities and ctx.tx_type not in (
        "staking_delegate", "staking_undelegate", "staking_redelegate", "staking", "reward_claim"
    ):
        notes.append("Note: this transaction is not a staking operation.")

    # Governance — valid, but note if not relevant
    if "governance" in features.entities and ctx.tx_type != "governance_vote":
        notes.append("Note: this transaction is not a governance action.")

    # IBC — valid, but note if not relevant
    if "ibc" in features.entities and ctx.tx_type not in ("ibc_transfer", "ibc_relay"):
        notes.append("Note: this transaction is not an IBC operation.")

    note = " ".join(notes) if notes else None
    return True, note


# ══════════════════════════════════════════════
# Step 3 — Dynamic Prompt Conditioning
# ══════════════════════════════════════════════

def get_prompt_directive(features: QuestionFeatures, ctx: QueryContext) -> str | None:
    """Generate a prompt directive to guide LLM response quality.

    This shapes HOW the LLM answers based on question strength and context.
    """
    # Vague/unrecognized question → conservative answer
    if not features.entities and features.intent == "unknown":
        return (
            "The user's question is vague. Answer conservatively using only "
            "explicit data from the transaction. Do not speculate."
        )

    # Asking about failure on a successful tx → redirect
    if "failure" in features.entities and not ctx.is_failed:
        return (
            "The user is asking about failure but this transaction succeeded. "
            "Clearly state it succeeded and explain what it actually did."
        )

    # Causal / "why" question on a failed tx → deep dive
    if features.intent == "causal" and ctx.is_failed:
        return (
            "The user wants to understand WHY this transaction failed. "
            "Provide a detailed root-cause analysis using the failure_reason, "
            "gas usage, and any relevant event data."
        )

    # Strong, specific question with multiple entities → thorough answer
    if len(features.entities) >= 2 or features.intent in ("causal", "quantitative"):
        return "Provide a thorough, structured explanation addressing all aspects of the question."

    # Complex tx + explain intent → structured breakdown
    if ctx.complexity == "complex" and features.intent == "explain":
        return (
            "This is a complex transaction. Provide a step-by-step breakdown "
            "of each action, clearly separating what happened in each stage."
        )

    return None


# ══════════════════════════════════════════════
# Step 4 — Rejection Logging
# ══════════════════════════════════════════════

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "query_rejections.jsonl")


def _log_rejection(question: str, reason: str, tx_type: str = "unknown"):
    """Log rejected queries for future training data / analysis."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "reason": reason,
            "tx_type": tx_type,
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging should never break the app


# ══════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════

def check_question(question: str, ctx: QueryContext) -> tuple[bool, str | None, str | None]:
    """Transaction-aware query reasoning gate.

    Returns (is_allowed, note_or_rejection, prompt_directive):
      - (True, None, directive)        → allowed, optional prompt directive
      - (True, "Note: ...", directive) → allowed with context note for the LLM
      - (False, rejection_msg, None)   → blocked
    """
    q = question.strip()
    q_lower = q.lower()

    # Empty or very short — treat as "tell me more"
    if len(q) < 3:
        return True, None, None

    # Step 1: Hard domain boundary
    if any(concept in q_lower for concept in HARD_REJECT_CONCEPTS):
        _log_rejection(q, "off_topic_concept", ctx.tx_type)
        return False, REJECTION_MESSAGE, None

    # Step 2: Extract structured features
    features = extract_features(q)

    # Step 3: Handle unrecognized questions
    if not features.entities and features.intent == "unknown":
        # Short follow-ups are likely contextual ("and?", "more", "go on")
        if len(q.split()) <= 5:
            directive = get_prompt_directive(features, ctx)
            return True, None, directive
        # Longer unrecognized questions are probably off-topic
        _log_rejection(q, "no_recognized_intent", ctx.tx_type)
        return False, REJECTION_MESSAGE, None

    # Step 4: Validate against actual tx data
    is_valid, note = validate_against_tx(features, ctx)
    if not is_valid:
        _log_rejection(q, "not_answerable", ctx.tx_type)
        return False, REJECTION_MESSAGE, None

    # Step 5: Generate prompt directive
    directive = get_prompt_directive(features, ctx)

    return True, note, directive


def _short_addr(addr: str | None) -> str:
    if not addr:
        return "?"
    if len(addr) > 24:
        return f"{addr[:10]}...{addr[-6:]}"
    return addr


def _format_transfer_line(transfer: dict) -> str:
    return (
        f"- {_short_addr(transfer.get('from'))} -> {_short_addr(transfer.get('to'))}, "
        f"{transfer.get('raw_amount') or '?'} ({transfer.get('status') or 'unknown'})"
    )


def _aggregate_transfer_amounts(transfers: list[dict]) -> str:
    totals: dict[str, int] = {}
    for transfer in transfers:
        denom = transfer.get("denom") or ""
        amount = transfer.get("amount")
        if denom and isinstance(amount, int):
            totals[denom] = totals.get(denom, 0) + amount
    if not totals:
        return "unknown"
    return ", ".join(f"{amount}{denom}" for denom, amount in totals.items())


def _is_direct_fact_query(features: QuestionFeatures) -> bool:
    """True when question is a direct factual ask that should bypass the LLM."""
    if not features.entities:
        return False
    if not features.entities.issubset(DIRECT_FACT_ENTITIES):
        return False
    # Keep explanation-style requests on template/LLM paths.
    if features.intent == "explain":
        return False
    return True


def _direct_fact_answer(features: QuestionFeatures, fact_index: dict) -> str | None:
    """Direct answers for signer/gas/fee/status/failure from fact index."""
    return format_direct_fact_entities(features.entities, fact_index)


def _deterministic_answer(features: QuestionFeatures, fact_index: dict, ctx: QueryContext) -> str | None:
    """Return direct factual answer when question maps to indexed facts."""
    entities = features.entities
    status = fact_index.get("status")
    failure_reason = fact_index.get("failure_reason")
    signer = fact_index.get("signer")
    sequence = fact_index.get("sequence")
    fee_payer = fact_index.get("fee_payer")
    fee_amount = fact_index.get("fee_amount")
    gas = fact_index.get("gas", {})
    transfers = fact_index.get("transfers", [])
    addresses = fact_index.get("addresses_involved", [])
    raw_event_count = ctx.tx.get("raw_event_count", 0)
    fee_recipient = fact_index.get("fee_recipient")
    memo = fact_index.get("memo", "")

    lines = []

    if "status" in entities:
        lines.append(f"Status: {status or 'unknown'}.")
    if "failure" in entities:
        if status == "failed":
            lines.append(f"Failure reason: {failure_reason or 'unknown failure reason'}.")
        else:
            lines.append("This transaction did not fail; status is success.")
    if "signer" in entities or (features.intent == "actor" and not entities):
        lines.append(f"Signer: {signer or 'unknown'}.")
    if "sequence" in entities:
        lines.append(f"Sequence: {sequence if sequence is not None else 'unknown'}.")
    if "gas" in entities:
        lines.append(
            "Gas: "
            f"used={gas.get('used', 'unknown')}, "
            f"wanted={gas.get('wanted', 'unknown')}, "
            f"fee={fee_amount or 'unknown'}, "
            f"fee payer={fee_payer or 'unknown'}."
        )
    if "events" in entities:
        lines.append(f"Raw event count: {raw_event_count}.")
    if "amount" in entities:
        if not transfers:
            lines.append("No transfer amount is recorded for this transaction.")
        elif len(transfers) == 1:
            lines.append(f"Transfer amount: {transfers[0].get('raw_amount') or 'unknown'}.")
        else:
            lines.append(
                f"Transfer amounts across {len(transfers)} transfer events total: "
                f"{_aggregate_transfer_amounts(transfers)}."
            )
    if "address" in entities:
        if not addresses:
            lines.append("No involved addresses were extracted.")
        elif len(addresses) == 1:
            lines.append(f"Involved address: {addresses[0]}.")
        else:
            preview = ", ".join(_short_addr(addr) for addr in addresses[:8])
            extra = len(addresses) - 8
            if extra > 0:
                lines.append(f"Involved addresses: {preview}, and {extra} more.")
            else:
                lines.append(f"Involved addresses: {preview}.")
    if "memo" in entities:
        if memo:
            lines.append(f"Memo: {memo}")
        else:
            lines.append("No memo was attached to this transaction.")

    if "fee_recipient" in entities:
        if fee_recipient:
            lines.append(f"Fee recipient: {fee_recipient}")
        else:
            lines.append("Fee recipient could not be determined from this transaction's events.")

    if lines:
        return "\n".join(lines)
    return None


def _template_answer(features: QuestionFeatures, fact_index: dict, ctx: QueryContext) -> str | None:
    """Return structured template answers for common explanation-style requests."""
    entities = features.entities
    transfers = fact_index.get("transfers", [])
    contract_data = fact_index.get("contracts", {})
    contract_actions = contract_data.get("actions", [])

    if "transfer" in entities:
        if not transfers:
            return "Transfer summary:\n- This transaction contains no transfer events."
        lines = [f"Transfer summary:\n- Transfer events: {len(transfers)}"]
        for transfer in transfers[:5]:
            lines.append(_format_transfer_line(transfer))
        if len(transfers) > 5:
            lines.append(f"- Additional transfer events not shown: {len(transfers) - 5}")
        return "\n".join(lines)

    if "contract" in entities or "swap" in entities:
        if not contract_actions:
            return "Contract action summary:\n- This transaction has no contract actions."
        lines = [f"Contract action summary:\n- Actions: {len(contract_actions)}"]
        for action in contract_actions[:6]:
            lines.append(
                f"- contract={_short_addr(action.get('contract'))}, "
                f"type={action.get('action_type') or 'unknown'}, "
                f"status={action.get('status') or 'unknown'}"
            )
        if len(contract_actions) > 6:
            lines.append(f"- Additional contract actions not shown: {len(contract_actions) - 6}")
        return "\n".join(lines)

#    if features.intent == "explain" and ctx.complexity in ("simple", "moderate"):        return (
#            "Transaction summary:\n"
#            f"- Status: {fact_index.get('status') or 'unknown'}\n"
#            f"- Type: {fact_index.get('tx_type') or 'unknown'}\n"
#            f"- Summary: {fact_index.get('summary') or 'n/a'}\n"
#            f"- Signer: {fact_index.get('signer') or 'unknown'}\n"
#            f"- Gas used/wanted: {fact_index.get('gas', {}).get('used', 'unknown')}/"
#            f"{fact_index.get('gas', {}).get('wanted', 'unknown')}\n"
#            f"- Transfers: {len(transfers)}\n"
#            f"- Contract actions: {len(contract_actions)}"
#        )

    return None


def route_question(
    question: str,
    ctx: QueryContext,
    fact_index: dict,
    context_artifacts: dict | None = None,
) -> dict:
    """Route a question to deterministic, template, context, or LLM path.

    Response shape:
      {
        "allowed": bool,
        "mode": "deterministic" | "template" | "manifest_context" | "vector_fallback" | "llm_context" | "llm" | "rejected",
        "response": str | None,
        "note": str | None,
        "directive": str | None,
        "source_refs": list[str] | None,   # optional: compact context source references
        "context_update": dict | None,     # optional: session context updates from retrieval
      }
    """
    allowed, note, directive = check_question(question, ctx)
    if not allowed:
        return {
            "allowed": False,
            "mode": "rejected",
            "response": note,
            "note": None,
            "directive": None,
        }

    features = extract_features(question)

    # Priority route: these direct fact questions should bypass the LLM.
    if _is_direct_fact_query(features):
        direct_core = _direct_fact_answer(features, fact_index)
        if direct_core:
            if note:
                direct_core = f"{direct_core}\n{note}"
            return {
                "allowed": True,
                "mode": "deterministic",
                "response": direct_core,
                "note": None,
                "directive": None,
            }

    direct = _deterministic_answer(features, fact_index, ctx)
    if direct:
        if note:
            direct = f"{direct}\n{note}"
        return {
            "allowed": True,
            "mode": "deterministic",
            "response": direct,
            "note": None,
            "directive": None,
        }

    templated = _template_answer(features, fact_index, ctx)
    if templated:
        if note:
            templated = f"{templated}\n{note}"
        return {
            "allowed": True,
            "mode": "template",
            "response": templated,
            "note": None,
            "directive": None,
        }

    context_route = ContextRouter(
        question=question,
        features=features,
        query_ctx=ctx,
        fact_index=fact_index,
        context_artifacts=context_artifacts,
        note=note,
        directive=directive,
    ).route()
    if context_route is not None:
        return context_route

    return {
        "allowed": True,
        "mode": "llm",
        "response": None,
        "note": note,
        "directive": directive,
    }
