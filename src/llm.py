"""
Layer 4 - LLM Wrapper

The LLM does NOT think. It translates structured truth into human explanation.
Uses a single Ollama-style /api/generate model for all requests.
"""

import json

import requests

from src.config import (
    LLM_API_PASSWORD,
    LLM_API_URL,
    LLM_API_USER,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from src.tx_digest import build_tx_digest


def _json_compact(data: dict) -> str:
    """Serialize JSON with minimal whitespace to reduce prompt size."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _clip_list(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    return items[:limit]


def _clip_text(text: str | None, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return f"{value[:max_chars - 3]}..."


def _clip_str_list(values: list, limit: int, max_chars: int) -> list[str]:
    result: list[str] = []
    for value in values[:limit]:
        if not isinstance(value, str):
            continue
        cleaned = _clip_text(value, max_chars)
        if cleaned:
            result.append(cleaned)
    return result


def _build_secondary_context(context_artifacts: dict | None) -> dict | None:
    """Build compact optional explanatory context layers for LLM support."""
    if not isinstance(context_artifacts, dict):
        return None

    secondary: dict = {
        "authority": (
            "secondary_explanatory_context_only; "
            "explicit transaction facts and deterministic interpretation are authoritative"
        )
    }

    manifest = context_artifacts.get("manifest_context")
    if isinstance(manifest, dict):
        manifest_out: dict = {}

        version = manifest.get("manifest_version")
        if isinstance(version, str) and version.strip():
            manifest_out["manifest_version"] = _clip_text(version, 32)

        matched = manifest.get("matched_keys", [])
        if isinstance(matched, list):
            manifest_out["matched_keys"] = _clip_str_list(matched, limit=8, max_chars=80)

        blocks = manifest.get("blocks", [])
        if isinstance(blocks, list):
            compact_blocks = []
            for block in blocks[:4]:
                if not isinstance(block, dict):
                    continue
                compact = {}
                key = block.get("key")
                title = block.get("title")
                summary = block.get("summary")
                notes = block.get("notes", [])

                if isinstance(key, str) and key.strip():
                    compact["key"] = _clip_text(key, 80)
                if isinstance(title, str) and title.strip():
                    compact["title"] = _clip_text(title, 80)
                if isinstance(summary, str) and summary.strip():
                    compact["summary"] = _clip_text(summary, 260)
                if isinstance(notes, list):
                    compact_notes = _clip_str_list(notes, limit=2, max_chars=180)
                    if compact_notes:
                        compact["notes"] = compact_notes
                if compact:
                    compact_blocks.append(compact)
            if compact_blocks:
                manifest_out["blocks"] = compact_blocks

        if manifest_out:
            secondary["manifest_context"] = manifest_out

    retrieved = context_artifacts.get("retrieved_context", [])
    if isinstance(retrieved, list):
        compact_retrieved = []
        for item in retrieved[:3]:
            if isinstance(item, str):
                text = _clip_text(item, 260)
                if text:
                    compact_retrieved.append({"summary": text})
                continue
            if not isinstance(item, dict):
                continue

            compact_item = {}
            key = item.get("key")
            title = item.get("title")
            summary = item.get("summary")
            notes = item.get("notes", [])

            if isinstance(key, str) and key.strip():
                compact_item["key"] = _clip_text(key, 80)
            if isinstance(title, str) and title.strip():
                compact_item["title"] = _clip_text(title, 80)
            if isinstance(summary, str) and summary.strip():
                compact_item["summary"] = _clip_text(summary, 260)
            if isinstance(notes, list):
                compact_notes = _clip_str_list(notes, limit=1, max_chars=160)
                if compact_notes:
                    compact_item["notes"] = compact_notes

            if compact_item:
                compact_retrieved.append(compact_item)

        if compact_retrieved:
            secondary["retrieved_context"] = compact_retrieved

    sources = context_artifacts.get("context_sources", [])
    if isinstance(sources, list):
        compact_sources = _clip_str_list(sources, limit=4, max_chars=80)
        if compact_sources:
            secondary["context_sources"] = compact_sources

    if len(secondary) == 1:
        return None
    return secondary


def _build_fact_digest(normalized_data: dict, interpretation: dict) -> dict:
    """Build explicit + derived fact layers for evidence-first prompting."""
    gas = normalized_data.get("gas", {})
    explicit_facts = {
        "tx_hash": normalized_data.get("tx_hash"),
        "height": normalized_data.get("height"),
        "status": normalized_data.get("status"),
        "code": normalized_data.get("code"),
        "failure_reason": normalized_data.get("failure_reason"),
        "signer": normalized_data.get("signer"),
        "sequence": normalized_data.get("sequence"),
        "gas": {
            "wanted": gas.get("wanted"),
            "used": gas.get("used"),
            "fee": gas.get("fee"),
            "fee_payer": gas.get("fee_payer"),
            "efficiency_pct": gas.get("efficiency"),
        },
        "counts": {
            "messages": len(normalized_data.get("messages", [])),
            "transfers": len(normalized_data.get("transfers", [])),
            "wasm_actions": len(normalized_data.get("wasm_actions", [])),
            "contract_executions": len(normalized_data.get("contract_executions", [])),
            "raw_event_count": normalized_data.get("raw_event_count", 0),
        },
    }

    derived_facts: dict = {}
    for key in ("tx_type", "summary", "complexity"):
        value = interpretation.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        derived_facts[key] = value

    warnings = [
        {
            "level": warning.get("level", "info"),
            "message": warning.get("message", ""),
        }
        for warning in interpretation.get("warnings", [])
        if isinstance(warning, dict)
    ]
    if warnings:
        derived_facts["warnings"] = warnings

    result = {
        "explicit_facts": explicit_facts,
        "forbidden_assumptions": [
            "Do not infer user intent, strategy, profitability, motivation, or emotion.",
            "Do not claim facts not present in explicit or derived evidence.",
        ],
    }
    if derived_facts:
        result["derived_facts"] = derived_facts
    return result


def _should_use_expanded_context(
    normalized_data: dict,
    complexity: str,
    user_question: str,
    prompt_directive: str | None,
) -> bool:
    """Route to larger context only when likely needed."""
    if normalized_data.get("status") == "failed":
        return True
    if complexity == "complex":
        return True

    question_len = len((user_question or "").strip())
    if question_len > 180:
        return True

    directive = (prompt_directive or "").lower()
    for marker in ("detailed", "step-by-step", "root-cause", "thorough", "all aspects"):
        if marker in directive:
            return True

    return False


def _build_compact_context(normalized_data: dict, interpretation: dict) -> dict:
    """Build compact context with enough detail for most follow-up questions."""
    tx_digest = build_tx_digest(normalized_data, interpretation, max_actions=10, max_contracts=8)
    return {
        "tx_digest": tx_digest,
        "sequence": normalized_data.get("sequence"),
        "raw_event_count": normalized_data.get("raw_event_count", 0),
        "warnings": interpretation.get("warnings", []),
        "annotations": _clip_list(interpretation.get("annotations", []), 20),
    }


def _build_expanded_digest_context(normalized_data: dict, interpretation: dict) -> dict:
    """Build richer digest-only context for harder questions (no full raw JSON)."""
    tx_digest = build_tx_digest(normalized_data, interpretation, max_actions=20, max_contracts=16)
    return {
        "tx_digest": tx_digest,
        "sequence": normalized_data.get("sequence"),
        "raw_event_count": normalized_data.get("raw_event_count", 0),
        "warnings": interpretation.get("warnings", []),
        "annotations": _clip_list(interpretation.get("annotations", []), 40),
        "transfers_sample": _clip_list(normalized_data.get("transfers", []), 12),
        "wasm_actions_sample": _clip_list(normalized_data.get("wasm_actions", []), 12),
    }


def _build_llm_context_payload(
    normalized_data: dict,
    interpretation: dict,
    complexity: str,
    user_question: str,
    prompt_directive: str | None,
    context_artifacts: dict | None = None,
) -> dict:
    """Build context payload with routing between compact and expanded modes."""
    mode = (
        "expanded"
        if _should_use_expanded_context(normalized_data, complexity, user_question, prompt_directive)
        else "compact"
    )
    tx_digest = build_tx_digest(normalized_data, interpretation, max_actions=12, max_contracts=10)
    payload = {
        "context_mode": mode,
        "tx_digest": tx_digest,
        "fact_digest": _build_fact_digest(normalized_data, interpretation),
    }

    if mode == "expanded":
        payload["supporting_context"] = _build_expanded_digest_context(normalized_data, interpretation)
    else:
        payload["supporting_context"] = _build_compact_context(normalized_data, interpretation)

    secondary_context = _build_secondary_context(context_artifacts)
    if secondary_context:
        payload["secondary_context"] = secondary_context

    return payload


def _get_auth() -> tuple | None:
    """Return HTTP Basic Auth tuple or None."""
    if LLM_API_USER and LLM_API_PASSWORD:
        return (LLM_API_USER, LLM_API_PASSWORD)
    return None


def warmup_models(verbose: bool = True) -> dict[str, bool]:
    """Warm only the selected production model."""
    if not LLM_API_URL:
        if verbose:
            print(f"  failed {LLM_MODEL_NAME} (LLM_API_URL not configured)")
        return {LLM_MODEL_NAME: False}

    try:
        payload = {
            "model": LLM_MODEL_NAME,
            "prompt": "hello",
            "system": "respond with ok",
            "stream": False,
            "options": {"num_predict": 1},
        }
        resp = requests.post(
            LLM_API_URL,
            json=payload,
            auth=_get_auth(),
            timeout=30,
        )
        resp.raise_for_status()
        if verbose:
            print(f"  loaded {LLM_MODEL_NAME}")
        return {LLM_MODEL_NAME: True}
    except Exception as exc:
        if verbose:
            print(f"  failed {LLM_MODEL_NAME} ({type(exc).__name__})")
        return {LLM_MODEL_NAME: False}


SYSTEM_PROMPT = """You are the ZigChain transaction explanation layer.

You are operating on ONE already-loaded transaction. The transaction context is fixed for this chat.
Use only the structured evidence provided in the prompt payload (for example: tx_digest, explicit_facts, derived_facts, fact_digest, supporting_context, prior answers).

PRIMARY CONTRACT:
- Stay strictly transaction-scoped.
- Do not hallucinate.
- Do not invent missing data.
- Do not recompute chain logic.
- Do not expand beyond provided evidence.

EVIDENCE HIERARCHY:
1) Explicit facts: treat as highest-confidence facts.
2) Derived facts: use as interpretation; phrase with appropriate certainty.
3) Secondary context (manifest/retrieved blocks): explanatory support only; never override explicit or derived facts.
4) Unknowns: if not supported by evidence, say it cannot be determined from this transaction.

REQUIRED BEHAVIOR:
1. Answer only with transaction-grounded evidence.
2. Keep answers concise by default.
3. Do not restate the entire transaction unless asked.
4. Keep repeated answers consistent with prior answers unless corrected by newer evidence in context.
5. If transaction status is failed, clearly state failure and use provided failure evidence.
6. Distinguish attempted/emitted actions from finalized state changes when relevant.
7. Never infer user intent, strategy, profitability, motivation, or emotion.
8. If provided context conflicts with transaction facts, transaction facts are authoritative.

REFUSAL RULE:
If a question is outside this transaction scope, reply with exactly:
"This assistant only answers questions about this specific transaction."

OUTPUT RULES:
- Plain text only (no markdown).
- If user asks for a full breakdown, use:
  TITLE:
  KEY POINTS:
  DETAILED EXPLANATION:
  ADDITIONAL CONTEXT:
- Otherwise provide a direct concise answer."""


def call_llm(
    normalized_data: dict,
    interpretation: dict,
    user_question: str = "Explain this transaction",
    chat_history: list = None,
    complexity: str = "moderate",
    prompt_directive: str = None,
    context_artifacts: dict | None = None,
) -> str:
    """Call the selected LLM with structured tx data and get an explanation."""
    context_payload = _build_llm_context_payload(
        normalized_data=normalized_data,
        interpretation=interpretation,
        complexity=complexity,
        user_question=user_question,
        prompt_directive=prompt_directive,
        context_artifacts=context_artifacts,
    )

    prompt_parts = [
        "Transaction Evidence Payload:",
        _json_compact(context_payload),
    ]

    if chat_history:
        prompt_parts.append("\nPrevious conversation:")
        for msg in chat_history:
            prompt_parts.append(f"  {msg['role']}: {msg['content']}")

    if prompt_directive:
        prompt_parts.append(f"\n[System directive: {prompt_directive}]")

    prompt_parts.append(f"\nUser Question:\n{user_question}")
    full_prompt = "\n".join(prompt_parts)

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": full_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "num_predict": LLM_MAX_TOKENS,
        },
    }

    try:
        if not LLM_API_URL:
            return "[ERROR] LLM_API_URL not configured. Set LLM_API_URL environment variable."

        resp = requests.post(
            LLM_API_URL,
            json=payload,
            auth=_get_auth(),
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "").strip()
    except requests.ConnectionError:
        return "[ERROR] Cannot reach LLM API. Check your connection and LLM_API_URL."
    except requests.Timeout:
        return "[ERROR] LLM request timed out. The model may be overloaded."
    except requests.HTTPError:
        return f"[ERROR] LLM API returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return f"[ERROR] LLM call failed: {str(exc)}"
