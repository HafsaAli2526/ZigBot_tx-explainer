"""
Deterministic Context Manifest Lookup

Loads local explanatory context snippets and resolves them by inferred context keys.
Manifest context is explanatory only; transaction facts remain primary truth.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache


_MANIFEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_manifest_store")
_MANIFEST_FILE = os.path.join(_MANIFEST_DIR, "manifest_v1.json")
_MANIFEST_SOURCE = "local_manifest:manifest_v1.json"


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    """Load local manifest once per process for deterministic low-latency lookups."""
    try:
        with open(_MANIFEST_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"version": "unknown", "entries": {}}

    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}

    version = str(data.get("version", "unknown"))
    return {
        "version": version,
        "entries": entries,
    }


def lookup_manifest_context(context_keys: list[str] | tuple[str, ...] | None, *, max_blocks: int = 6) -> dict:
    """Resolve concise explanatory blocks for matched context keys.

    Returns:
      {
        "manifest_context": dict,
        "retrieved_context": list[dict],
        "context_sources": list[str],
      }
    """
    normalized_keys = _normalize_keys(context_keys)
    if not normalized_keys:
        return {"manifest_context": {}, "retrieved_context": [], "context_sources": []}

    manifest = _load_manifest()
    entries = manifest.get("entries", {})

    matched_keys: list[str] = []
    blocks: list[dict] = []

    for key in normalized_keys:
        entry = entries.get(key)
        if not isinstance(entry, dict):
            continue
        matched_keys.append(key)
        blocks.append(_build_block(key, entry))
        if len(blocks) >= max_blocks:
            break

    if not blocks:
        return {"manifest_context": {}, "retrieved_context": [], "context_sources": []}

    manifest_context = {
        "manifest_version": manifest.get("version", "unknown"),
        "matched_keys": matched_keys,
        "blocks": blocks,
        "policy": (
            "Manifest blocks are explanatory context only. "
            "Transaction facts and deterministic interpretation remain authoritative."
        ),
    }

    return {
        "manifest_context": manifest_context,
        "retrieved_context": list(blocks),
        "context_sources": [_MANIFEST_SOURCE],
    }


def _normalize_keys(context_keys: list[str] | tuple[str, ...] | None) -> list[str]:
    if not context_keys:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for raw in context_keys:
        if not isinstance(raw, str):
            continue
        key = raw.strip()
        if not key:
            continue
        lowered = key.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(key)
    return result


def _build_block(key: str, entry: dict) -> dict:
    title = str(entry.get("title", "")).strip()
    summary = str(entry.get("summary", "")).strip()
    notes = entry.get("notes", [])

    concise_notes: list[str] = []
    if isinstance(notes, list):
        for item in notes:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                concise_notes.append(cleaned)
            if len(concise_notes) >= 4:
                break

    return {
        "key": key,
        "title": title,
        "summary": summary,
        "notes": concise_notes,
    }
