"""
Transaction Digest Builder

Builds compact LLM-ready transaction digests from processed transaction data.
"""

from __future__ import annotations


def _clip(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    return items[:limit]


def _short_addr(addr: str | None) -> str:
    if not addr:
        return "?"
    if len(addr) > 24:
        return f"{addr[:10]}...{addr[-6:]}"
    return addr


def _build_key_actions(normalized_data: dict, max_actions: int) -> list[dict]:
    actions: list[dict] = []

    for msg in normalized_data.get("messages", []):
        if len(actions) >= max_actions:
            break
        actions.append(
            {
                "kind": "message",
                "action": msg.get("action"),
                "module": msg.get("module"),
                "sender": msg.get("sender"),
                "msg_index": msg.get("msg_index"),
            }
        )

    for transfer in normalized_data.get("transfers", []):
        if len(actions) >= max_actions:
            break
        actions.append(
            {
                "kind": "transfer",
                "from": _short_addr(transfer.get("from")),
                "to": _short_addr(transfer.get("to")),
                "amount": transfer.get("raw_amount"),
                "status": transfer.get("status"),
                "msg_index": transfer.get("msg_index"),
            }
        )

    for action in normalized_data.get("wasm_actions", []):
        if len(actions) >= max_actions:
            break
        entry = {
            "kind": "wasm_action",
            "type": action.get("type"),
            "contract": _short_addr(action.get("contract")),
            "contract_full": action.get("contract"),
            "status": action.get("status"),
            "msg_index": action.get("msg_index"),
        }
        parsed = action.get("parsed")
        if parsed:
            entry["parsed"] = parsed
        actions.append(entry)

    return actions


def _build_contracts(normalized_data: dict, max_items: int) -> dict:
    addresses: list[str] = []
    seen: set[str] = set()

    for action in normalized_data.get("wasm_actions", []):
        addr = action.get("contract")
        if addr and addr not in seen:
            seen.add(addr)
            addresses.append(addr)

    for execution in normalized_data.get("contract_executions", []):
        addr = execution.get("contract")
        if addr and addr not in seen:
            seen.add(addr)
            addresses.append(addr)

    action_types = []
    for action in normalized_data.get("wasm_actions", []):
        action_types.append(action.get("type") or "unknown")

    return {
        "count": len(addresses),
        "addresses": _clip(addresses, max_items),
        "action_types": _clip(action_types, max_items),
    }


def build_tx_digest(
    normalized_data: dict,
    interpretation: dict,
    *,
    max_actions: int = 12,
    max_contracts: int = 10,
) -> dict:
    """Build compact digest for LLM input.

    Required fields:
    - status
    - failure_reason
    - signer
    - fee
    - gas_used
    - key_actions
    - contracts
    """
    gas = normalized_data.get("gas", {})
    return {
        "tx_hash": normalized_data.get("tx_hash"),
        "status": normalized_data.get("status"),
        "failure_reason": normalized_data.get("failure_reason"),
        "signer": normalized_data.get("signer"),
        "fee": {
            "amount": gas.get("fee"),
            "payer": gas.get("fee_payer"),
        },
        "gas_used": gas.get("used"),
        "gas_wanted": gas.get("wanted"),
        "tx_type": interpretation.get("tx_type"),
        "summary": interpretation.get("summary"),
        "complexity": interpretation.get("complexity"),
        "key_actions": _build_key_actions(normalized_data, max_actions=max_actions),
        "contracts": _build_contracts(normalized_data, max_items=max_contracts),
        "contract_params": (normalized_data.get("tx_body") or {}).get("contract_msgs", []),
    }

