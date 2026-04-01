"""
Layer 1 — Data Fetch
Fetches raw transaction data from ZigChain Cosmos RPC.
"""

import requests
from src.config import ZIGCHAIN_RPC_URL


def _raise_non_json_rpc_error(status_code: int) -> None:
    """Map non-JSON upstream responses to stable transport-style errors."""
    # Common edge/proxy timeout codes.
    if status_code in (504, 522, 524):
        raise TimeoutError(
            f"ZigChain RPC upstream timed out (HTTP {status_code}). "
            "Please retry in a moment."
        )

    # Common gateway/unavailable responses from RPC edge layers.
    if status_code in (502, 503, 520, 521, 523):
        raise ConnectionError(
            f"ZigChain RPC is temporarily unavailable (HTTP {status_code}). "
            "Please retry shortly."
        )

    raise RuntimeError(f"RPC returned HTTP {status_code} (non-JSON response)")


def fetch_tx(tx_hash: str) -> dict:
    """Fetch a transaction by hash from ZigChain RPC.

    Returns the raw RPC result dict or raises on failure.
    """
    tx_hash = tx_hash.strip()
    if tx_hash.lower().startswith("0x"):
        tx_hash = tx_hash[2:]
    tx_hash = tx_hash.upper()

    # Validate hash format before hitting the RPC
    if len(tx_hash) != 64:
        raise ValueError(
            f"Invalid tx hash: expected 64 hex characters, got {len(tx_hash)}. "
            f"Check for typos."
        )
    if not all(c in "0123456789ABCDEF" for c in tx_hash):
        raise ValueError("Invalid tx hash: contains non-hex characters.")

    url = f"{ZIGCHAIN_RPC_URL}/tx?hash=0x{tx_hash}"

    try:
        resp = requests.get(url, timeout=30)
    except requests.ConnectionError:
        raise ConnectionError(f"Cannot reach ZigChain RPC at {ZIGCHAIN_RPC_URL}")
    except requests.Timeout:
        raise TimeoutError("ZigChain RPC request timed out")

    # Cosmos RPC returns errors as HTTP 500 + JSON body.
    # Always try to parse JSON first for a clean error message.
    try:
        data = resp.json()
    except Exception:
        _raise_non_json_rpc_error(resp.status_code)

    if "error" in data:
        error_data = data["error"].get("data", "")
        error_msg = data["error"].get("message", "Unknown error")

        # Provide clean, human-readable errors
        if "not found" in str(error_data).lower():
            raise ValueError(f"Transaction not found on-chain. Hash: {tx_hash[:16]}...")
        else:
            raise ValueError(f"RPC error: {error_msg} — {error_data}")

    result = data.get("result", {})
    if not result:
        raise ValueError("Empty result from RPC")

    return result
