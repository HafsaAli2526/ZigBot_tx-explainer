"""
Approved Documentation Vector Retrieval

Narrow, deterministic vector retrieval over an approved local documentation corpus.
Used only as explanatory fallback for transaction-scoped questions.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache


_CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_vector_corpus")
_CORPUS_FILE = os.path.join(_CORPUS_DIR, "approved_docs_v1.json")
_CORPUS_SOURCE = "approved_docs:approved_docs_v1.json"


def retrieve_approved_docs_context(
    question: str,
    context_keys: list[str] | tuple[str, ...] | None,
    *,
    top_k: int = 2,
    min_score: float = 0.14,
) -> list[dict]:
    """Retrieve compact high-relevance snippets from approved local docs only."""
    q = (question or "").strip()
    if not q:
        return []

    index = _load_vector_index()
    docs = index.get("docs", [])
    if not docs:
        return []

    query_vector = _build_query_vector(q, index.get("idf", {}))
    if not query_vector:
        return []

    context_key_set = {key.strip().lower() for key in (context_keys or []) if isinstance(key, str) and key.strip()}
    scored: list[tuple[float, dict]] = []

    for doc in docs:
        cosine = _cosine_sparse(query_vector, doc.get("vector", {}), doc.get("norm", 1.0))
        if cosine < min_score:
            continue

        tags = doc.get("tags", set())
        overlap = len(context_key_set.intersection(tags)) if context_key_set else 0

        # Require stronger text similarity when no context-key overlap is present.
        if context_key_set and overlap == 0 and cosine < (min_score + 0.08):
            continue

        final_score = cosine + min(0.12, overlap * 0.04)
        scored.append((final_score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[dict] = []
    for final_score, doc in scored[:top_k]:
        results.append(
            {
                "key": f"doc:{doc.get('id', 'unknown')}",
                "title": doc.get("title", "Approved documentation snippet"),
                "summary": _clip_text(doc.get("summary", ""), 280),
                "notes": _build_doc_notes(doc, final_score),
                "source": doc.get("source", _CORPUS_SOURCE),
                "score": round(final_score, 3),
            }
        )
    return results


@lru_cache(maxsize=1)
def _load_vector_index() -> dict:
    """Load and vectorize approved docs once per process."""
    raw_docs = _load_docs_file()
    if not raw_docs:
        return {"idf": {}, "docs": []}

    tokenized_docs: list[dict] = []
    doc_freq: dict[str, int] = {}

    for raw in raw_docs:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue

        tokens = _tokenize(text)
        if not tokens:
            continue

        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        unique_tokens = set(token_counts.keys())
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

        tags = {
            tag.strip().lower()
            for tag in raw.get("tags", [])
            if isinstance(tag, str) and tag.strip()
        }

        tokenized_docs.append(
            {
                "id": str(raw.get("id", "unknown")).strip() or "unknown",
                "title": str(raw.get("title", "Approved doc")).strip() or "Approved doc",
                "summary": str(raw.get("summary", "")).strip() or text,
                "source": str(raw.get("source", _CORPUS_SOURCE)).strip() or _CORPUS_SOURCE,
                "counts": token_counts,
                "tags": tags,
            }
        )

    if not tokenized_docs:
        return {"idf": {}, "docs": []}

    idf: dict[str, float] = {}
    total_docs = len(tokenized_docs)
    for token, freq in doc_freq.items():
        idf[token] = math.log((total_docs + 1) / (freq + 1)) + 1.0

    docs: list[dict] = []
    for doc in tokenized_docs:
        counts = doc["counts"]
        vector: dict[str, float] = {}
        norm_sq = 0.0
        max_tf = max(counts.values()) if counts else 1
        for token, count in counts.items():
            tf = count / max_tf
            weight = tf * idf.get(token, 1.0)
            vector[token] = weight
            norm_sq += weight * weight

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        docs.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "summary": doc["summary"],
                "source": doc["source"],
                "tags": doc["tags"],
                "vector": vector,
                "norm": norm,
            }
        )

    return {"idf": idf, "docs": docs}


def _load_docs_file() -> list[dict]:
    try:
        with open(_CORPUS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    docs = data.get("docs", [])
    if not isinstance(docs, list):
        return []

    approved: list[dict] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        approved.append(item)
    return approved


def _build_query_vector(question: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = _tokenize(question)
    if not tokens:
        return {}

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    max_tf = max(counts.values()) if counts else 1
    vector: dict[str, float] = {}
    norm_sq = 0.0
    for token, count in counts.items():
        tf = count / max_tf
        weight = tf * idf.get(token, 0.0)
        if weight <= 0:
            continue
        vector[token] = weight
        norm_sq += weight * weight

    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    if norm <= 0:
        return {}

    # Store normalized query vector so cosine uses one norm value.
    for token in list(vector.keys()):
        vector[token] = vector[token] / norm
    return vector


def _cosine_sparse(query_vec: dict[str, float], doc_vec: dict[str, float], doc_norm: float) -> float:
    if not query_vec or not doc_vec or doc_norm <= 0:
        return 0.0
    dot = 0.0
    for token, q_weight in query_vec.items():
        d_weight = doc_vec.get(token)
        if d_weight is None:
            continue
        dot += q_weight * d_weight
    return dot / doc_norm


def _tokenize(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            current.append(ch)
            continue
        if current:
            word = "".join(current)
            if len(word) >= 3:
                words.append(word)
            current = []
    if current:
        word = "".join(current)
        if len(word) >= 3:
            words.append(word)
    return words


def _build_doc_notes(doc: dict, score: float) -> list[str]:
    notes = []
    source = str(doc.get("source", _CORPUS_SOURCE)).strip()
    if source:
        notes.append(f"Source: {source}")
    notes.append(f"Relevance score: {round(score, 3)}")
    return notes


def _clip_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return f"{value[:max_chars - 3]}..."
