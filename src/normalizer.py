"""
Layer 2 — Normalization Engine (CRITICAL)
Converts raw Cosmos RPC transaction data into structured, deterministic JSON.
No guessing. No LLM. Pure data transformation.
"""

import base64
import json


def normalize_tx(raw: dict) -> dict:
    """Convert raw RPC tx result into normalized structured data."""
    tx_result = raw.get("tx_result", {})
    tx_hash = raw.get("hash", "")
    height = raw.get("height", "")

    code = tx_result.get("code", 0)
    is_success = code == 0

    gas_wanted = int(tx_result.get("gas_wanted", "0"))
    gas_used = int(tx_result.get("gas_used", "0"))

    events = tx_result.get("events", [])

    fee = _extract_fee(events)
    fee_payer = _extract_fee_payer(events)
    signer_seq = _extract_signer_sequence(events)
    messages = _extract_messages(events)
    transfers = _extract_transfers(events, is_success)
    wasm_actions = _extract_wasm_actions(events, is_success)
    contract_executions = _extract_contract_executions(events)

    tx_body = _decode_tx_body(raw.get("tx", ""))

    failure_reason = None
    if not is_success:
        failure_reason = _extract_failure_reason(tx_result)

    return {
        "tx_hash": tx_hash,
        "height": height,
        "status": "success" if is_success else "failed",
        "code": code,
        "failure_reason": failure_reason,
        "gas": {
            "wanted": gas_wanted,
            "used": gas_used,
            "fee": fee,
            "fee_payer": fee_payer,
            "efficiency": round(gas_used / gas_wanted * 100, 1) if gas_wanted > 0 else 0,
        },
        "signer": signer_seq.get("signer"),
        "sequence": signer_seq.get("sequence"),
        "messages": messages,
        "tx_body": tx_body,
        "transfers": transfers,
        "wasm_actions": wasm_actions,
        "contract_executions": contract_executions,
        "raw_event_count": len(events),
    }


# ──────────────────────────────────────────────
# Event extraction helpers
# ──────────────────────────────────────────────

def _extract_fee(events: list) -> str | None:
    for event in events:
        if event.get("type") == "tx":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            if "fee" in attrs:
                return attrs["fee"]
    return None


def _extract_fee_payer(events: list) -> str | None:
    for event in events:
        if event.get("type") == "tx":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            if "fee_payer" in attrs:
                return attrs["fee_payer"]
    return None


def _extract_signer_sequence(events: list) -> dict:
    for event in events:
        if event.get("type") == "tx":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            if "acc_seq" in attrs:
                parts = attrs["acc_seq"].rsplit("/", 1)
                return {
                    "signer": parts[0] if len(parts) > 0 else None,
                    "sequence": int(parts[1]) if len(parts) > 1 else None,
                }
    return {"signer": None, "sequence": None}


def _extract_messages(events: list) -> list:
    messages = []
    for event in events:
        if event.get("type") == "message":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            if "action" in attrs:
                messages.append({
                    "action": attrs["action"],
                    "sender": attrs.get("sender"),
                    "module": attrs.get("module"),
                    "msg_index": attrs.get("msg_index"),
                })
    return messages


def _extract_transfers(events: list, is_success: bool) -> list:
    transfers = []
    for event in events:
        if event.get("type") == "transfer":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            amount_str = attrs.get("amount", "")
            amount, denom = _parse_amount(amount_str)
            transfers.append({
                "from": attrs.get("sender"),
                "to": attrs.get("recipient"),
                "amount": amount,
                "denom": denom,
                "raw_amount": amount_str,
                "status": "finalized" if is_success else "emitted_but_tx_failed",
                "msg_index": attrs.get("msg_index"),
            })
    return transfers


def _extract_wasm_actions(events: list, is_success: bool) -> list:
    actions = []
    for event in events:
        if event.get("type") == "wasm":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            contract = attrs.pop("_contract_address", None)
            action_type = attrs.pop("action", "unknown")
            msg_index = attrs.pop("msg_index", None)

            action = {
                "type": action_type,
                "contract": contract,
                "status": "finalized" if is_success else "emitted_but_tx_failed",
                "msg_index": msg_index,
                "details": attrs,
            }

            if action_type == "swap":
                action["parsed"] = _parse_swap(attrs)

            actions.append(action)
    return actions


def _extract_contract_executions(events: list) -> list:
    executions = []
    for event in events:
        if event.get("type") == "execute":
            attrs = {a["key"]: a["value"] for a in event.get("attributes", [])}
            executions.append({
                "contract": attrs.get("_contract_address"),
                "msg_index": attrs.get("msg_index"),
            })
    return executions


# ──────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────

def _parse_swap(attrs: dict) -> dict:
    parsed = {}
    for key in ["offer_asset", "ask_asset"]:
        if key in attrs:
            parsed[key] = attrs[key]
    for key in ["offer_amount", "return_amount", "spread_amount", "commission_amount", "maker_fee_amount"]:
        if key in attrs:
            parsed[key] = int(attrs[key])
    if "reserves" in attrs:
        reserves = {}
        for pair in attrs["reserves"].split(","):
            parts = pair.rsplit(":", 1)
            if len(parts) == 2:
                reserves[parts[0]] = int(parts[1])
        parsed["reserves"] = reserves
    return parsed


def _parse_amount(amount_str: str) -> tuple:
    if not amount_str:
        return (0, "")
    i = 0
    while i < len(amount_str) and amount_str[i].isdigit():
        i += 1
    if i == 0:
        return (0, amount_str)
    return (int(amount_str[:i]), amount_str[i:])


def _extract_failure_reason(tx_result: dict) -> str:
    raw_log = tx_result.get("log", "")
    if raw_log:
        return raw_log
    codespace = tx_result.get("codespace", "")
    code = tx_result.get("code", 0)
    if codespace:
        return f"{codespace}/{code}"
    return f"Transaction failed with code {code}"


def _decode_tx_body(tx_base64: str) -> dict | None:
    if not tx_base64:
        return None
    try:
        raw_bytes = base64.b64decode(tx_base64)
        decoded_text = raw_bytes.decode("utf-8", errors="ignore")
        msg_types = []
        patterns = [
            "/cosmwasm.wasm.v1.MsgExecuteContract",
            "/cosmwasm.wasm.v1.MsgInstantiateContract",
            "/cosmwasm.wasm.v1.MsgStoreCode",
            "/cosmos.bank.v1beta1.MsgSend",
            "/cosmos.bank.v1beta1.MsgMultiSend",
            "/cosmos.staking.v1beta1.MsgDelegate",
            "/cosmos.staking.v1beta1.MsgUndelegate",
            "/cosmos.staking.v1beta1.MsgBeginRedelegate",
            "/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward",
            "/cosmos.gov.v1beta1.MsgVote",
            "/cosmos.gov.v1.MsgVote",
            "/ibc.core.client.v1.MsgUpdateClient",
            "/ibc.core.channel.v1.MsgRecvPacket",
            "/ibc.core.channel.v1.MsgAcknowledgement",
            "/ibc.applications.transfer.v1.MsgTransfer",
        ]
        for pattern in patterns:
            if pattern in decoded_text:
                msg_types.append(pattern)

        contract_msgs = []
        idx = 0
        while idx < len(decoded_text):
            start = decoded_text.find('{"', idx)
            if start == -1:
                break
            depth = 0
            end = start
            for j in range(start, len(decoded_text)):
                if decoded_text[j] == '{':
                    depth += 1
                elif decoded_text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end > start:
                try:
                    msg = json.loads(decoded_text[start:end])
                    contract_msgs.append(msg)
                except (json.JSONDecodeError, ValueError):
                    pass
            idx = end if end > start else start + 1

        result = {}
        if msg_types:
            result["msg_types"] = msg_types
        if contract_msgs:
            result["contract_msgs"] = contract_msgs
        return result if result else None

    except Exception:
        return None
