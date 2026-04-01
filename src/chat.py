"""
Chat Session Manager
Manages per-TX chat context with token-budget-aware history.
TX data is the real context, not the chat.
"""

from collections import OrderedDict
import copy
import threading

from src.context_keys import infer_context_keys
from src.context_manifest import lookup_manifest_context
from src.facts import build_fact_index
from src.query_engine import QueryContext
from src.tx_digest import build_tx_digest


class ChatSession:
    """Manages a single transaction chat session."""

    # Token budget for chat history (tx data is always resent separately)
    MAX_HISTORY_CHARS = 6000  # ~1500 tokens at ~4 chars/token
    # Keep only a bounded number of repeated-question answers per tx session.
    MAX_ANSWER_CACHE = 40
    # Keep a bounded in-process cache of processed transactions.
    MAX_PROCESSED_TX_CACHE = 256
    _processed_lock = threading.Lock()
    _processed_tx_cache: OrderedDict[str, tuple[dict, dict]] = OrderedDict()

    def __init__(
        self,
        tx_hash: str,
        normalized_data: dict,
        interpretation: dict,
        load_explanation: str | None = None,
        context_keys: list[str] | tuple[str, ...] | None = None,
        manifest_context: dict | None = None,
        retrieved_context: list | tuple | None = None,
        context_sources: list | tuple | None = None,
    ):
        self.tx_hash = tx_hash
        self.normalized_data = normalized_data
        self.interpretation = interpretation
        self.fact_index = build_fact_index(normalized_data, interpretation)
        self.tx_digest = build_tx_digest(normalized_data, interpretation)
        self.query_ctx = QueryContext(normalized_data, interpretation)
        self.load_explanation = (load_explanation or "").strip()
        self.history = []

        self._lock = threading.Lock()
        self._answer_cache: OrderedDict[str, tuple[str, bool]] = OrderedDict()
        self.cache_hits = 0
        # Optional transaction-scoped context artifacts.
        # These are additive-only state fields for future context-aware routing.
        if context_keys is None:
            inferred = infer_context_keys(
                normalized_data,
                interpretation,
                fact_index=self.fact_index,
                tx_digest=self.tx_digest,
            )
            self.context_keys = self._normalize_context_keys(inferred)
        else:
            self.context_keys = self._normalize_context_keys(context_keys)

        manifest_lookup = None
        if manifest_context is None or retrieved_context is None or context_sources is None:
            manifest_lookup = lookup_manifest_context(self.context_keys)

        if manifest_context is None:
            default_manifest = manifest_lookup.get("manifest_context", {}) if manifest_lookup else {}
            self.manifest_context = self._clone_mapping(default_manifest)
        else:
            self.manifest_context = self._clone_mapping(manifest_context)

        if retrieved_context is None:
            default_retrieved = manifest_lookup.get("retrieved_context", []) if manifest_lookup else []
            self.retrieved_context = self._clone_sequence(default_retrieved)
        else:
            self.retrieved_context = self._clone_sequence(retrieved_context)

        if context_sources is None:
            default_sources = manifest_lookup.get("context_sources", []) if manifest_lookup else []
            self.context_sources = self._clone_sequence(default_sources)
        else:
            self.context_sources = self._clone_sequence(context_sources)

    @property
    def complexity(self) -> str:
        return self.query_ctx.complexity

    @staticmethod
    def normalize_tx_hash(tx_hash: str) -> str:
        value = (tx_hash or "").strip()
        if value.lower().startswith("0x"):
            value = value[2:]
        return value.upper()

    @classmethod
    def get_or_load_processed(cls, tx_hash: str) -> tuple[str, dict, dict, bool]:
        """Return normalized+interpreted tx data, loading once per process.

        Returns: (clean_hash, normalized, interpretation, cache_hit)
        """
        clean_hash = cls.normalize_tx_hash(tx_hash)

        with cls._processed_lock:
            cached = cls._processed_tx_cache.get(clean_hash)
            if cached is not None:
                cls._processed_tx_cache.move_to_end(clean_hash)
                normalized, interpretation = cached
                return clean_hash, normalized, interpretation, True

        # Import lazily so chat/session stays light until tx load is needed.
        from src.fetcher import fetch_tx
        from src.interpreter import interpret
        from src.normalizer import normalize_tx

        raw = fetch_tx(clean_hash)
        normalized = normalize_tx(raw)
        interpretation = interpret(normalized)

        with cls._processed_lock:
            cls._processed_tx_cache[clean_hash] = (normalized, interpretation)
            cls._processed_tx_cache.move_to_end(clean_hash)
            while len(cls._processed_tx_cache) > cls.MAX_PROCESSED_TX_CACHE:
                cls._processed_tx_cache.popitem(last=False)

        return clean_hash, normalized, interpretation, False

    @classmethod
    def processed_cache_size(cls) -> int:
        with cls._processed_lock:
            return len(cls._processed_tx_cache)

    def add_user_message(self, message: str):
        with self._lock:
            self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        with self._lock:
            self.history.append({"role": "assistant", "content": message})

    def get_fact_index(self) -> dict:
        """Return transaction fact index for deterministic answer routing/use."""
        return self.fact_index

    def set_context_artifacts(
        self,
        *,
        context_keys: list[str] | tuple[str, ...] | None = None,
        manifest_context: dict | None = None,
        retrieved_context: list | tuple | None = None,
        context_sources: list | tuple | None = None,
    ) -> None:
        """Attach or update optional transaction-scoped context artifacts."""
        with self._lock:
            if context_keys is not None:
                self.context_keys = self._normalize_context_keys(context_keys)
            if manifest_context is not None:
                self.manifest_context = self._clone_mapping(manifest_context)
            if retrieved_context is not None:
                self.retrieved_context = self._clone_sequence(retrieved_context)
            if context_sources is not None:
                self.context_sources = self._clone_sequence(context_sources)

    def get_context_artifacts(self) -> dict:
        """Return detached copies of optional transaction-scoped context artifacts."""
        with self._lock:
            return {
                "context_keys": list(self.context_keys),
                "manifest_context": copy.deepcopy(self.manifest_context),
                "retrieved_context": copy.deepcopy(self.retrieved_context),
                "context_sources": copy.deepcopy(self.context_sources),
            }

    def get_load_explanation(self, note: str | None = None) -> str | None:
        """Return precomputed load-time explanation for chat reuse."""
        if not self.load_explanation:
            return None
        if not note:
            return self.load_explanation
        return f"{self.load_explanation}\n\n{note}"

    def get_context_history(self) -> list:
        """Return recent history trimmed to token budget.

        Walks backwards from newest, keeps what fits.
        TX data is always resent by the caller - this is just chat history.
        """
        with self._lock:
            if not self.history:
                return []

            result = []
            total_chars = 0

            for msg in reversed(self.history):
                msg_chars = len(msg["content"])
                if total_chars + msg_chars > self.MAX_HISTORY_CHARS:
                    break
                result.insert(0, msg)
                total_chars += msg_chars

            return result

    def get_cached_response(self, question: str) -> tuple[str, bool] | None:
        """Return cached (response, allowed) for near-identical questions."""
        fingerprint = self._question_fingerprint(question)
        if not fingerprint:
            return None

        with self._lock:
            cached = self._answer_cache.get(fingerprint)
            if cached is None:
                return None

            # LRU behavior for repeated use.
            self._answer_cache.move_to_end(fingerprint)
            self.cache_hits += 1
            return cached

    def cache_response(self, question: str, response: str, allowed: bool) -> None:
        """Cache response for the current transaction session only."""
        fingerprint = self._question_fingerprint(question)
        if not fingerprint:
            return

        with self._lock:
            self._answer_cache[fingerprint] = (response, allowed)
            self._answer_cache.move_to_end(fingerprint)
            while len(self._answer_cache) > self.MAX_ANSWER_CACHE:
                self._answer_cache.popitem(last=False)

    @staticmethod
    def _question_fingerprint(question: str) -> str:
        """Create a lightweight normalized fingerprint for near-repeat detection."""
        cleaned = (question or "").strip().lower()
        if not cleaned:
            return ""

        # Normalize common punctuation and whitespace without regex-heavy parsing.
        cleaned = cleaned.translate(str.maketrans({
            "?": " ",
            "!": " ",
            ".": " ",
            ",": " ",
            ";": " ",
            ":": " ",
            "\t": " ",
            "\n": " ",
            "\r": " ",
        }))

        tokens = [token for token in cleaned.split(" ") if token]
        if not tokens:
            return ""

        return " ".join(tokens)

    @staticmethod
    def _normalize_context_keys(context_keys: list[str] | tuple[str, ...] | None) -> list[str]:
        if not context_keys:
            return []

        result: list[str] = []
        seen: set[str] = set()
        for raw in context_keys:
            if not isinstance(raw, str):
                continue
            value = raw.strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            result.append(value)
        return result

    @staticmethod
    def _clone_mapping(value: dict | None) -> dict:
        if not value:
            return {}
        return copy.deepcopy(value)

    @staticmethod
    def _clone_sequence(value: list | tuple | None) -> list:
        if not value:
            return []
        return list(copy.deepcopy(value))

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "tx_hash": self.tx_hash,
                "messages": len(self.history),
                "tx_type": self.interpretation.get("tx_type", "unknown"),
                "complexity": self.interpretation.get("complexity", "unknown"),
                "cached_answers": len(self._answer_cache),
                "cache_hits": self.cache_hits,
                "fact_fields": len(self.fact_index),
                "processed_tx_cache_size": self.processed_cache_size(),
            }
