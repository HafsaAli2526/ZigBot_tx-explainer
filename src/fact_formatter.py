"""
Deterministic Fact Response Formatter

Formats direct fact-index answers into concise natural language without using the LLM.
"""

from __future__ import annotations


def _format_int(value) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return "unknown"


def _split_amount_denom(amount_text: str) -> tuple[int | None, str]:
    if not amount_text:
        return None, ""

    i = 0
    while i < len(amount_text) and amount_text[i].isdigit():
        i += 1

    if i == 0:
        return None, amount_text

    return int(amount_text[:i]), amount_text[i:]


def _format_fee_amount(fee_amount: str | None) -> str:
    if not fee_amount:
        return "unknown fee"

    parts = [part.strip() for part in str(fee_amount).split(",") if part.strip()]
    if not parts:
        return "unknown fee"

    formatted = []
    for part in parts:
        amount, denom = _split_amount_denom(part)
        if amount is None:
            formatted.append(part)
        elif denom:
            formatted.append(f"{amount:,} {denom}")
        else:
            formatted.append(f"{amount:,}")

    return ", ".join(formatted)


def format_status(status: str | None) -> str:
    if status == "failed":
        return "This transaction failed."
    if status == "success":
        return "This transaction succeeded."
    return "The transaction status is unknown."


def format_failure(status: str | None, failure_reason: str | None) -> str:
    if status != "failed":
        return "It did not fail."
    if not failure_reason:
        return "It failed, but the failure reason is not available."

    reason_lower = failure_reason.lower()
    if "insufficient funds" in reason_lower:
        return "It failed because the account did not have enough funds."
    if "out of gas" in reason_lower:
        return "It failed because it ran out of gas."
    return f"It failed because: {failure_reason}."


def format_signer(signer: str | None) -> str:
    if signer:
        return f"The transaction was signed by {signer}."
    return "The signer is not available in this transaction data."


def format_fee(fee_amount: str | None, fee_payer: str | None) -> str:
    fee_text = _format_fee_amount(fee_amount)
    if fee_payer:
        return f"This transaction used a fee of {fee_text}. The fee payer was {fee_payer}."
    return f"This transaction used a fee of {fee_text}."


def format_gas(gas: dict | None) -> str:
    gas = gas or {}
    used = gas.get("used")
    wanted = gas.get("wanted")
    efficiency = gas.get("efficiency_pct")

    if isinstance(used, int) and isinstance(wanted, int):
        base = f"It used {_format_int(used)} gas out of {_format_int(wanted)} gas wanted."
    elif isinstance(used, int):
        base = f"It used {_format_int(used)} gas."
    elif isinstance(wanted, int):
        base = f"The gas wanted was {_format_int(wanted)}."
    else:
        return "Gas usage is not available."

    if isinstance(efficiency, (int, float)):
        return f"{base} Efficiency was {efficiency}%."
    return base


def format_direct_fact_entities(entities: set[str], fact_index: dict) -> str | None:
    """Compose concise natural-language direct answers for fact-only routes."""
    if not entities:
        return None

    status = fact_index.get("status")
    failure_reason = fact_index.get("failure_reason")
    signer = fact_index.get("signer")
    fee_amount = fact_index.get("fee_amount")
    fee_payer = fact_index.get("fee_payer")
    gas = fact_index.get("gas", {})

    lines = []
    if "status" in entities:
        lines.append(format_status(status))
    if "failure" in entities:
        lines.append(format_failure(status, failure_reason))
    if "signer" in entities:
        lines.append(format_signer(signer))
    if "fee" in entities:
        lines.append(format_fee(fee_amount, fee_payer))
    if "gas" in entities:
        lines.append(format_gas(gas))

    if not lines:
        return None
    return "\n".join(lines)

