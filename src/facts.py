"""
Fact Index Layer

Builds a structured dictionary of directly answerable facts from:
- normalized transaction data (explicit facts)
- deterministic interpretation output (derived facts)
"""

from __future__ import annotations


def _append_unique(items: list[str], seen: set[str], value: str | None) -> None:
    if not value:
        return
    cleaned = str(value).strip()
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    items.append(cleaned)


def build_fact_index(normalized_data: dict, interpretation: dict) -> dict:
    """Build direct-answer fact index from normalized + interpreted transaction data."""
    gas = normalized_data.get("gas", {})
    transfers = normalized_data.get("transfers", [])
    wasm_actions = normalized_data.get("wasm_actions", [])
    contract_executions = normalized_data.get("contract_executions", [])
    messages = normalized_data.get("messages", [])

    transfer_facts = []
    for transfer in transfers:
        transfer_facts.append(
            {
                "from": transfer.get("from"),
                "to": transfer.get("to"),
                "amount": transfer.get("amount"),
                "denom": transfer.get("denom"),
                "raw_amount": transfer.get("raw_amount"),
                "status": transfer.get("status"),
                "msg_index": transfer.get("msg_index"),
            }
        )

    contract_actions = []
    contract_addresses: list[str] = []
    contract_seen: set[str] = set()

    for action in wasm_actions:
        contract = action.get("contract")
        _append_unique(contract_addresses, contract_seen, contract)
        contract_actions.append(
            {
                "contract": contract,
                "action_type": action.get("type"),
                "status": action.get("status"),
                "msg_index": action.get("msg_index"),
            }
        )

    for execution in contract_executions:
        contract = execution.get("contract")
        _append_unique(contract_addresses, contract_seen, contract)
        contract_actions.append(
            {
                "contract": contract,
                "action_type": "execute",
                "status": "unknown",
                "msg_index": execution.get("msg_index"),
            }
        )

    addresses_involved: list[str] = []
    address_seen: set[str] = set()

    _append_unique(addresses_involved, address_seen, normalized_data.get("signer"))
    _append_unique(addresses_involved, address_seen, gas.get("fee_payer"))

    for transfer in transfers:
        _append_unique(addresses_involved, address_seen, transfer.get("from"))
        _append_unique(addresses_involved, address_seen, transfer.get("to"))

    for message in messages:
        _append_unique(addresses_involved, address_seen, message.get("sender"))

    for contract in contract_addresses:
        _append_unique(addresses_involved, address_seen, contract)

    return {
        "status": normalized_data.get("status"),
        "failure_reason": normalized_data.get("failure_reason"),
        "signer": normalized_data.get("signer"),
        "fee_payer": gas.get("fee_payer"),
        "sequence": normalized_data.get("sequence"),
        "gas": {
            "used": gas.get("used"),
            "wanted": gas.get("wanted"),
            "efficiency_pct": gas.get("efficiency"),
        },
        "fee_amount": gas.get("fee"),
        "transfers": transfer_facts,
        "contracts": {
            "addresses": contract_addresses,
            "actions": contract_actions,
        },
        "addresses_involved": addresses_involved,
        # Derived deterministic facts (already produced by interpreter)
        "tx_type": interpretation.get("tx_type"),
        "complexity": interpretation.get("complexity"),
        "summary": interpretation.get("summary"),
    }

