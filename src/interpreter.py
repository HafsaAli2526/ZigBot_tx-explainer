"""
Layer 3 — Deterministic Interpretation (Rules Engine)
NO LLM. Pure logic. Classifies and annotates normalized tx data.
"""

from src.tokens import registry as token_registry


def interpret(normalized: dict) -> dict:
    """Apply deterministic rules to normalized tx data."""
    messages = normalized.get("messages", [])
    wasm_actions = normalized.get("wasm_actions", [])
    tx_body = normalized.get("tx_body") or {}
    contract_msgs = tx_body.get("contract_msgs", [])
    msg_types = tx_body.get("msg_types", [])

    tx_type = _classify_tx_type(messages, wasm_actions, msg_types, contract_msgs)
    summary = _build_summary(normalized, tx_type)
    warnings = _generate_warnings(normalized)
    annotations = _annotate_actions(normalized)
    complexity = _score_complexity(normalized)

    return {
        "tx_type": tx_type,
        "summary": summary,
        "warnings": warnings,
        "annotations": annotations,
        "complexity": complexity,
    }


# ──────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────

def _classify_tx_type(messages, wasm_actions, msg_types, contract_msgs) -> str:
    action_types = {a["type"] for a in wasm_actions}

    if "swap" in action_types:
        return "dex_swap"
    if "provide_liquidity" in action_types:
        return "liquidity_provision"
    if "withdraw_liquidity" in action_types:
        return "liquidity_withdrawal"

    for msg in contract_msgs:
        if "swap" in msg:
            return "dex_swap"
        if "provide_liquidity" in msg:
            return "liquidity_provision"
        if "withdraw_liquidity" in msg:
            return "liquidity_withdrawal"
        if "transfer" in msg or "send" in msg:
            return "token_transfer"
        if "mint" in msg:
            return "token_mint"
        if "burn" in msg:
            return "token_burn"
        if "stake" in msg or "bond" in msg:
            return "staking"

    type_map = {
        "MsgSend": "bank_send",
        "MsgMultiSend": "bank_multi_send",
        "MsgDelegate": "staking_delegate",
        "MsgUndelegate": "staking_undelegate",
        "MsgBeginRedelegate": "staking_redelegate",
        "MsgWithdrawDelegatorReward": "reward_claim",
        "MsgVote": "governance_vote",
        "MsgTransfer": "ibc_transfer",
        "MsgUpdateClient": "ibc_relay",
        "MsgRecvPacket": "ibc_relay",
        "MsgAcknowledgement": "ibc_relay",
        "MsgExecuteContract": "contract_execution",
        "MsgInstantiateContract": "contract_instantiation",
        "MsgStoreCode": "contract_upload",
    }
    for mt in msg_types:
        for suffix, tx_type in type_map.items():
            if suffix in mt:
                return tx_type

    modules = {m.get("module") for m in messages if m.get("module")}
    if "wasm" in modules:
        return "contract_execution"
    if "bank" in modules:
        return "bank_send"
    if "staking" in modules:
        return "staking"

    return "unknown"


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

def _build_summary(normalized: dict, tx_type: str) -> str:
    status = normalized["status"]
    signer = _short_addr(normalized.get("signer"))
    fee = normalized["gas"]["fee"] or "unknown fee"

    if status == "failed":
        reason = normalized.get("failure_reason", "unknown reason")
        return f"FAILED transaction by {signer}: {_tx_type_label(tx_type)} — {reason}"

    wasm_actions = normalized.get("wasm_actions", [])
    if tx_type == "dex_swap" and wasm_actions:
        for action in wasm_actions:
            if action["type"] == "swap" and "parsed" in action:
                p = action["parsed"]
                offer = _format_amount(p.get("offer_amount", 0), p.get("offer_asset", "?"))
                ret = _format_amount(p.get("return_amount", 0), p.get("ask_asset", "?"))
                return f"{signer} swapped {offer} -> {ret} (fee: {fee})"

    transfers = normalized.get("transfers", [])
    if tx_type in ("bank_send", "token_transfer") and transfers:
        t = transfers[0]
        return f"{signer} sent {t['raw_amount']} to {_short_addr(t['to'])} (fee: {fee})"

    return f"{_tx_type_label(tx_type)} by {signer} (fee: {fee})"


# ──────────────────────────────────────────────
# Warnings
# ──────────────────────────────────────────────

def _generate_warnings(normalized: dict) -> list:
    warnings = []
    status = normalized["status"]
    gas = normalized["gas"]

    if status == "failed":
        warnings.append({"level": "critical", "message": "Transaction FAILED — no state changes were finalized"})
        if normalized.get("transfers"):
            warnings.append({"level": "warning", "message": "Transfer events emitted but NOT finalized due to tx failure"})
        if normalized.get("wasm_actions"):
            warnings.append({"level": "warning", "message": "Contract events emitted but NOT finalized due to tx failure"})

    if gas["efficiency"] > 95:
        warnings.append({"level": "warning", "message": f"Gas usage very high ({gas['efficiency']}%) — tx nearly ran out of gas"})

    if status == "failed" and gas["used"] > 0:
        warnings.append({"level": "info", "message": f"Gas fee of {gas['fee']} was still charged despite failure"})

    return warnings


# ──────────────────────────────────────────────
# Annotations
# ──────────────────────────────────────────────

def _annotate_actions(normalized: dict) -> list:
    annotations = []

    for t in normalized.get("transfers", []):
        # Check if this is a fee payment (first transfer to fee collector)
        label = "Fee payment" if t["to"] and "17xpfv" in t["to"] else "Token transfer"
        annotations.append({
            "type": "transfer",
            "label": label,
            "from": _short_addr(t["from"]),
            "to": _short_addr(t["to"]),
            "amount": t["raw_amount"],
            "finalized": t["status"] == "finalized",
        })

    for action in normalized.get("wasm_actions", []):
        if action["type"] == "swap" and "parsed" in action:
            p = action["parsed"]
            annotations.append({
                "type": "swap",
                "label": "DEX Swap",
                "offer": _format_amount(p.get("offer_amount", 0), p.get("offer_asset", "?")),
                "received": _format_amount(p.get("return_amount", 0), p.get("ask_asset", "?")),
                "commission": p.get("commission_amount", 0),
                "spread": p.get("spread_amount", 0),
                "finalized": action["status"] == "finalized",
            })
        else:
            annotations.append({
                "type": "contract_action",
                "label": action["type"],
                "contract": _short_addr(action.get("contract")),
                "finalized": action["status"] == "finalized",
            })

    return annotations


# ──────────────────────────────────────────────
# Complexity scoring
# ──────────────────────────────────────────────

def _score_complexity(normalized: dict) -> str:
    score = 0
    msg_count = len(normalized.get("messages", []))
    if msg_count > 1:
        score += msg_count
    score += len(normalized.get("wasm_actions", [])) * 2
    transfer_count = len(normalized.get("transfers", []))
    if transfer_count > 3:
        score += transfer_count
    exec_count = len(normalized.get("contract_executions", []))
    if exec_count > 1:
        score += exec_count * 2
    if normalized["status"] == "failed":
        score += 2

    if score <= 3:
        return "simple"
    elif score <= 8:
        return "moderate"
    return "complex"


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────

TX_TYPE_LABELS = {
    "dex_swap": "DEX Swap", "liquidity_provision": "Liquidity Provision",
    "liquidity_withdrawal": "Liquidity Withdrawal", "bank_send": "Token Send",
    "bank_multi_send": "Multi-Send", "token_transfer": "Token Transfer",
    "token_mint": "Token Mint", "token_burn": "Token Burn",
    "staking_delegate": "Staking Delegation", "staking_undelegate": "Staking Undelegation",
    "staking_redelegate": "Staking Redelegation", "reward_claim": "Reward Claim",
    "governance_vote": "Governance Vote", "ibc_transfer": "IBC Transfer",
    "ibc_relay": "IBC Relay", "contract_execution": "Contract Execution",
    "contract_instantiation": "Contract Instantiation", "contract_upload": "Contract Upload",
    "staking": "Staking", "governance": "Governance",
}


def _tx_type_label(tx_type: str) -> str:
    return TX_TYPE_LABELS.get(tx_type, tx_type.replace("_", " ").title())


def _short_addr(addr: str | None) -> str:
    if not addr:
        return "?"
    if len(addr) > 20:
        return f"{addr[:10]}...{addr[-6:]}"
    return addr


def _format_amount(amount: int, denom: str) -> str:
    """Format amount using token registry for exponent-aware display."""
    return token_registry.format_amount(amount, denom)
