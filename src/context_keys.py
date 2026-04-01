"""
Transaction Context Key Inference

Builds compact, stable transaction-scoped context keys from existing artifacts:
- normalized transaction data
- deterministic interpretation
- fact index
- tx digest
"""

from __future__ import annotations

from src.tx_digest import build_tx_digest


# Prefix-level message namespace -> module key
_MSG_PREFIX_TO_MODULE = {
    "/cosmos.bank.": "bank",
    "/cosmos.staking.": "staking",
    "/cosmos.distribution.": "staking",
    "/cosmos.gov.": "gov",
    "/ibc.": "ibc",
    "/cosmwasm.wasm.": "wasm",
}


# Deterministic tx_type -> module hints (fallback when module fields are sparse)
_TX_TYPE_TO_MODULE = {
    "bank_send": "bank",
    "bank_multi_send": "bank",
    "token_transfer": "bank",
    "staking": "staking",
    "staking_delegate": "staking",
    "staking_undelegate": "staking",
    "staking_redelegate": "staking",
    "reward_claim": "staking",
    "governance": "gov",
    "governance_vote": "gov",
    "ibc_transfer": "ibc",
    "ibc_relay": "ibc",
    "contract_execution": "wasm",
    "contract_instantiation": "wasm",
    "contract_upload": "wasm",
    "dex_swap": "wasm",
    "liquidity_provision": "wasm",
    "liquidity_withdrawal": "wasm",
}


# Optional aliases for manifest compatibility.
_TX_TYPE_ALIASES = {
    "dex_swap": "swap",
}


def infer_context_keys(
    normalized_data: dict,
    interpretation: dict,
    *,
    fact_index: dict | None = None,
    tx_digest: dict | None = None,
) -> list[str]:
    """Infer stable context keys from existing transaction artifacts."""
    fact_index = fact_index or {}
    if tx_digest is None:
        tx_digest = build_tx_digest(normalized_data, interpretation)

    keys: list[str] = []
    seen: set[str] = set()

    def add_key(namespace: str, value: str | None) -> None:
        if not value:
            return
        cleaned = _normalize_key_value(value, namespace=namespace)
        if not cleaned:
            return
        key = f"{namespace}:{cleaned}"
        if key in seen:
            return
        seen.add(key)
        keys.append(key)

    for module_name in _collect_modules(normalized_data, interpretation, tx_digest):
        add_key("module", module_name)

    for message_type in _collect_message_types(normalized_data):
        add_key("message_type", message_type)

    for tx_type in _collect_tx_types(interpretation, tx_digest):
        add_key("tx_type", tx_type)

    failure_category = _infer_failure_category(normalized_data, fact_index, tx_digest)
    if failure_category:
        add_key("failure_category", failure_category)

    return keys


def _collect_modules(normalized_data: dict, interpretation: dict, tx_digest: dict) -> list[str]:
    modules: list[str] = []
    seen: set[str] = set()

    def push(module_name: str | None) -> None:
        if not module_name:
            return
        normalized = _normalize_key_value(module_name, namespace="module")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        modules.append(normalized)

    for msg in normalized_data.get("messages", []):
        push(msg.get("module"))

    for action in tx_digest.get("key_actions", []):
        if action.get("kind") == "message":
            push(action.get("module"))

    tx_body = normalized_data.get("tx_body") or {}
    for raw in tx_body.get("msg_types") or []:
        inferred_from_full = _module_from_full_msg_type(raw)
        if inferred_from_full:
            push(inferred_from_full)

    for msg_type in _collect_message_types(normalized_data):
        inferred = _module_from_msg_type(msg_type)
        if inferred:
            push(inferred)

    tx_type = interpretation.get("tx_type") or tx_digest.get("tx_type")
    if isinstance(tx_type, str):
        push(_TX_TYPE_TO_MODULE.get(tx_type))

    return modules


def _module_from_full_msg_type(full_type: object) -> str | None:
    if not isinstance(full_type, str):
        return None
    raw = full_type.strip().lower()
    if not raw:
        return None

    if not raw.startswith("/"):
        raw = "/" + raw

    for prefix, module_name in _MSG_PREFIX_TO_MODULE.items():
        if raw.startswith(prefix):
            return module_name
    return None


def _collect_message_types(normalized_data: dict) -> list[str]:
    tx_body = normalized_data.get("tx_body") or {}
    raw_types = tx_body.get("msg_types") or []

    message_types: list[str] = []
    seen: set[str] = set()
    for raw in raw_types:
        if not isinstance(raw, str):
            continue
        msg_type = _extract_msg_type_name(raw)
        if not msg_type:
            continue
        if msg_type in seen:
            continue
        seen.add(msg_type)
        message_types.append(msg_type)

    return message_types


def _collect_tx_types(interpretation: dict, tx_digest: dict) -> list[str]:
    tx_types: list[str] = []
    seen: set[str] = set()

    def push(value: str | None) -> None:
        if not value:
            return
        normalized = _normalize_key_value(value, namespace="tx_type")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tx_types.append(normalized)

    tx_type = interpretation.get("tx_type") or tx_digest.get("tx_type")
    if isinstance(tx_type, str):
        push(tx_type)
        push(_TX_TYPE_ALIASES.get(tx_type))

    return tx_types


def _module_from_msg_type(msg_type: str) -> str | None:
    # When only short names are available, infer using known groups.
    if msg_type in {"MsgSend", "MsgMultiSend"}:
        return "bank"
    if msg_type in {"MsgDelegate", "MsgUndelegate", "MsgBeginRedelegate", "MsgWithdrawDelegatorReward"}:
        return "staking"
    if msg_type == "MsgVote":
        return "gov"
    if msg_type in {"MsgTransfer", "MsgUpdateClient", "MsgRecvPacket", "MsgAcknowledgement"}:
        return "ibc"
    if msg_type in {"MsgExecuteContract", "MsgInstantiateContract", "MsgStoreCode"}:
        return "wasm"
    return None


def _extract_msg_type_name(full_type: str) -> str | None:
    cleaned = full_type.strip()
    if not cleaned:
        return None

    # Input examples:
    #   /cosmos.bank.v1beta1.MsgSend
    #   /cosmwasm.wasm.v1.MsgExecuteContract
    #   MsgSend
    leaf = cleaned.split("/")[-1]
    if "." in leaf:
        leaf = leaf.split(".")[-1]
    leaf = leaf.strip()
    if not leaf:
        return None
    return leaf


def _infer_failure_category(normalized_data: dict, fact_index: dict, tx_digest: dict) -> str | None:
    status = normalized_data.get("status") or fact_index.get("status") or tx_digest.get("status")
    if status != "failed":
        return None

    reason = (
        normalized_data.get("failure_reason")
        or fact_index.get("failure_reason")
        or tx_digest.get("failure_reason")
        or ""
    )
    text = str(reason).strip().lower()
    tokens = set(_tokenize_words(text))

    if _matches_phrase_or_tokens(text, tokens, ("insufficient funds",), (("insufficient", "funds"),)):
        return "insufficient_funds"
    if _matches_phrase_or_tokens(text, tokens, ("out of gas",), (("out", "gas"),)):
        return "out_of_gas"
    if _matches_phrase_or_tokens(
        text,
        tokens,
        ("not authorized", "unauthorized"),
        (("unauthorized",), ("not", "authorized")),
    ):
        return "unauthorized"
    if _matches_phrase_or_tokens(
        text,
        tokens,
        ("account sequence mismatch", "invalid sequence"),
        (("account", "sequence", "mismatch"), ("invalid", "sequence")),
    ):
        return "invalid_sequence"
    if _matches_phrase_or_tokens(
        text,
        tokens,
        ("max spread assertion", "slippage"),
        (("max", "spread", "assertion"), ("slippage",)),
    ):
        return "slippage"
    if _matches_phrase_or_tokens(text, tokens, ("not found",), (("not", "found"),)):
        return "not_found"
    if _matches_phrase_or_tokens(text, tokens, ("already exists",), (("already", "exists"),)):
        return "already_exists"

    if "/" in text:
        maybe_code = text.rsplit("/", 1)[-1].strip()
        if maybe_code.isdigit():
            return "chain_error"

    return "unknown"


def _matches_phrase_or_tokens(
    text: str,
    tokens: set[str],
    phrases: tuple[str, ...],
    token_groups: tuple[tuple[str, ...], ...],
) -> bool:
    for phrase in phrases:
        if phrase and phrase in text:
            return True
    for group in token_groups:
        if all(token in tokens for token in group):
            return True
    return False


def _tokenize_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalnum():
            current.append(ch)
            continue
        if current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _normalize_key_value(value: str, *, namespace: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""

    if namespace == "message_type":
        # Keep canonical message type case (e.g. MsgSend) for lookup readability.
        out = [ch for ch in raw if ch.isalnum() or ch == "_"]
        return "".join(out).strip("_")

    lowered = raw.lower()
    out: list[str] = []
    prev_underscore = False
    for ch in lowered:
        if ch.isalnum() or ch == "_":
            out.append(ch)
            prev_underscore = ch == "_"
            continue
        if ch in (" ", "-", ".", "/", ":"):
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
            continue

    normalized = "".join(out).strip("_")
    return normalized
