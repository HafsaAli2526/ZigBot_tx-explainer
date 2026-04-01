"""Simple in-memory cache used by the web UI.

This provides a minimal `get(key)` / `put(key, normalized, interpretation)`
API expected by `web_ui.py`.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

_lock = threading.Lock()
_store: Dict[str, Tuple[dict, dict]] = {}


def get(key: str) -> Optional[Tuple[dict, dict]]:
    """Return cached (normalized, interpretation) for `key`, or None."""
    with _lock:
        return _store.get(key)


def put(key: str, normalized: dict, interpretation: dict) -> None:
    """Store `(normalized, interpretation)` under `key`.

    This is intentionally simple — an in-memory dict protected by a lock.
    """
    with _lock:
        _store[key] = (normalized, interpretation)


def clear() -> None:
    """Clear the cache."""
    with _lock:
        _store.clear()
