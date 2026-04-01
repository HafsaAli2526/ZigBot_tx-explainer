"""
Context Router

Deterministically routes explanatory transaction-scoped questions to:
- manifest context answer
- vector fallback answer
- LLM with attached context note/directive
"""

from __future__ import annotations

from src.context_vector import retrieve_approved_docs_context


class ContextRouter:
    """Context-aware routing stage that runs after deterministic/template routes."""

    def __init__(
        self,
        *,
        question: str,
        features,
        query_ctx,
        fact_index: dict,
        context_artifacts: dict | None,
        note: str | None,
        directive: str | None,
    ):
        self.question = (question or "").strip()
        self.features = features
        self.query_ctx = query_ctx
        self.fact_index = fact_index or {}
        self.context_artifacts = context_artifacts or {}
        self.note = note
        self.directive = directive

    def route(self) -> dict | None:
        """Return context route dict, or None to continue normal LLM fallback."""
        if not self._should_use_context():
            return None

        manifest_blocks_raw = self._manifest_blocks()
        manifest_blocks, manifest_drop = self._sanitize_context_blocks(manifest_blocks_raw, source_label="manifest")
        if manifest_blocks:
            source_refs = self._manifest_source_refs()
            if self._can_answer_directly_from_manifest():
                response = self._build_manifest_response(
                    manifest_blocks,
                    source_refs=source_refs,
                    dropped=manifest_drop,
                )
                return {
                    "allowed": True,
                    "mode": "manifest_context",
                    "response": response,
                    "note": None,
                    "directive": None,
                    "source_refs": source_refs,
                }

            return {
                "allowed": True,
                "mode": "llm_context",
                "response": None,
                "note": self._build_context_note(
                    manifest_blocks,
                    source_label="manifest",
                    dropped=manifest_drop,
                ),
                "directive": self._build_context_directive(source_label="manifest"),
                "source_refs": source_refs,
            }

        vector_blocks_raw = self._vector_blocks()
        vector_blocks, vector_drop = self._sanitize_context_blocks(vector_blocks_raw, source_label="vector")
        if vector_blocks:
            source_refs = self._vector_source_refs(vector_blocks)
            if self._can_answer_directly_from_vector():
                response = self._build_vector_response(
                    vector_blocks,
                    source_refs=source_refs,
                    dropped=vector_drop,
                )
                return {
                    "allowed": True,
                    "mode": "vector_fallback",
                    "response": response,
                    "note": None,
                    "directive": None,
                    "context_update": self._vector_context_update(vector_blocks),
                    "source_refs": source_refs,
                }

            return {
                "allowed": True,
                "mode": "llm_context",
                "response": None,
                "note": self._build_context_note(
                    vector_blocks,
                    source_label="vector",
                    dropped=vector_drop,
                ),
                "directive": self._build_context_directive(source_label="vector"),
                "context_update": self._vector_context_update(vector_blocks),
                "source_refs": source_refs,
            }

        return None

    def _should_use_context(self) -> bool:
        # Context routing is explanatory-only and must remain tx-scoped.
        if self.features.scope != "tx_specific":
            return False

        if self.features.intent in ("explain", "causal"):
            return True

        # "info" questions can still be explanatory when asking meaning/semantics.
        if self.features.intent == "info" and self._is_semantic_info_question():
            return True

        return False

    def _is_semantic_info_question(self) -> bool:
        q = self.question.lower()
        semantic_markers = (
            "what is",
            "what does",
            "what means",
            "meaning",
            "module",
            "message type",
            "msg",
        )
        return any(marker in q for marker in semantic_markers)

    def _manifest_blocks(self) -> list[dict]:
        manifest_context = self.context_artifacts.get("manifest_context")
        if not isinstance(manifest_context, dict):
            return []
        blocks = manifest_context.get("blocks", [])
        if not isinstance(blocks, list):
            return []

        result: list[dict] = []
        for block in blocks:
            if isinstance(block, dict):
                result.append(block)
        return result

    def _vector_blocks(self) -> list[dict]:
        manifest_blocks = self._manifest_blocks()
        if manifest_blocks:
            return []

        persisted = self._persisted_vector_blocks()
        if persisted and self._is_short_followup_question():
            return persisted

        if not self._should_attempt_vector_retrieval():
            return []

        context_keys = self.context_artifacts.get("context_keys", [])
        if not isinstance(context_keys, list):
            context_keys = []

        return retrieve_approved_docs_context(
            self.question,
            context_keys,
            top_k=2,
            min_score=0.14,
        )

    def _is_short_followup_question(self) -> bool:
        q = self.question.strip()
        return len(q) <= 48 or len(q.split()) <= 8

    def _persisted_vector_blocks(self) -> list[dict]:
        retrieved = self.context_artifacts.get("retrieved_context", [])
        if not isinstance(retrieved, list):
            return []

        context_sources = self.context_artifacts.get("context_sources", [])
        if isinstance(context_sources, list):
            has_vector_source = any(
                isinstance(source, str) and source.strip().lower().startswith("approved_docs:")
                for source in context_sources
            )
            if not has_vector_source:
                return []

        blocks: list[dict] = []
        for item in retrieved:
            if isinstance(item, dict):
                blocks.append(item)
                continue
            if isinstance(item, str):
                text = item.strip()
                if text:
                    blocks.append({"title": "Retrieved context", "summary": text})
        return blocks

    def _should_attempt_vector_retrieval(self) -> bool:
        # Keep retrieval narrow: explanatory transaction questions only.
        if not self._should_use_context():
            return False

        q = self.question.strip()
        if len(q) < 18 and len(q.split()) < 4:
            return False

        # Skip vector retrieval for extremely generic short follow-ups.
        low_signal = {"more", "and", "why", "how", "what"}
        words = {token.lower() for token in q.split()}
        if words and words.issubset(low_signal):
            return False

        return True

    def _can_answer_directly_from_manifest(self) -> bool:
        # Prefer direct concise context answers for semantic asks.
        return self._is_semantic_info_question() or self.features.intent == "explain"

    def _can_answer_directly_from_vector(self) -> bool:
        # Keep vector fallback conservative and concise.
        return self.features.intent in ("explain", "info")

    def _build_manifest_response(
        self,
        blocks: list[dict],
        *,
        source_refs: list[str] | None = None,
        dropped: dict | None = None,
    ) -> str:
        lines = [
            "Context explanation for this transaction:",
            f"- Transaction status: {self.fact_index.get('status') or 'unknown'}",
            f"- Transaction type: {self.fact_index.get('tx_type') or 'unknown'}",
        ]
        lines.extend(self._format_blocks(blocks, limit=3))
        lines.append(
            "These notes explain module/message semantics only; transaction facts above remain authoritative."
        )
        drop_line = self._dropped_context_line(dropped)
        if drop_line:
            lines.append(drop_line)
        compact_sources = self._compact_source_refs(source_refs)
        if compact_sources:
            lines.append(f"Sources: {', '.join(compact_sources)}")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def _build_vector_response(
        self,
        blocks: list[dict],
        *,
        source_refs: list[str] | None = None,
        dropped: dict | None = None,
    ) -> str:
        lines = [
            "Retrieved context for this transaction:",
            f"- Transaction status: {self.fact_index.get('status') or 'unknown'}",
            f"- Transaction type: {self.fact_index.get('tx_type') or 'unknown'}",
        ]
        lines.extend(self._format_blocks(blocks, limit=3))
        lines.append(
            "This is supporting context only; explicit transaction facts remain the source of truth."
        )
        drop_line = self._dropped_context_line(dropped)
        if drop_line:
            lines.append(drop_line)
        compact_sources = self._compact_source_refs(source_refs)
        if compact_sources:
            lines.append(f"Sources: {', '.join(compact_sources)}")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def _build_context_note(self, blocks: list[dict], *, source_label: str, dropped: dict | None = None) -> str:
        base = self.note.strip() if isinstance(self.note, str) and self.note.strip() else ""
        lines = []
        if base:
            lines.append(base)
        lines.append(f"Context ({source_label}) for this transaction:")
        lines.extend(self._format_blocks(blocks, limit=4))
        drop_line = self._dropped_context_line(dropped)
        if drop_line:
            lines.append(drop_line)
        lines.append(
            "Use these blocks as explanatory support only; do not override explicit transaction facts."
        )
        return "\n".join(lines)

    def _build_context_directive(self, *, source_label: str) -> str:
        addition = (
            f"Use provided {source_label} context blocks only as supplemental explanation. "
            "Prioritize explicit transaction facts and deterministic interpretation."
        )
        if self.directive:
            return f"{self.directive} {addition}"
        return addition

    def _format_blocks(self, blocks: list[dict], *, limit: int) -> list[str]:
        lines: list[str] = []
        for block in blocks[:limit]:
            title = str(block.get("title", "")).strip() or str(block.get("key", "Context")).strip()
            summary = str(block.get("summary", "")).strip()
            if summary:
                lines.append(f"- {title}: {summary}")
            else:
                lines.append(f"- {title}")

            notes = block.get("notes", [])
            if isinstance(notes, list):
                for note in notes[:2]:
                    if isinstance(note, str) and note.strip():
                        lines.append(f"  - {note.strip()}")
        return lines

    def _vector_source_label(self, blocks: list[dict]) -> str | None:
        for block in blocks:
            source = block.get("source")
            if isinstance(source, str) and source.strip():
                return source.strip()
        return None

    def _manifest_source_refs(self) -> list[str]:
        sources = self.context_artifacts.get("context_sources", [])
        refs: list[str] = []
        seen: set[str] = set()

        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, str):
                    continue
                cleaned = source.strip()
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if "manifest" not in lowered:
                    continue
                if lowered in seen:
                    continue
                seen.add(lowered)
                refs.append(cleaned)

        manifest_context = self.context_artifacts.get("manifest_context")
        if isinstance(manifest_context, dict):
            version = manifest_context.get("manifest_version")
            if isinstance(version, str) and version.strip():
                tag = f"manifest_version:{version.strip()}"
                lowered = tag.lower()
                if lowered not in seen:
                    seen.add(lowered)
                    refs.append(tag)

        return self._compact_source_refs(refs)

    def _vector_source_refs(self, blocks: list[dict]) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()

        for block in blocks:
            source = block.get("source")
            if not isinstance(source, str):
                continue
            cleaned = source.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            refs.append(cleaned)

        if not refs:
            fallback = self._vector_source_label(blocks)
            if fallback:
                refs.append(fallback)

        return self._compact_source_refs(refs)

    def _compact_source_refs(self, refs: list[str] | None) -> list[str]:
        if not refs:
            return []

        result: list[str] = []
        seen: set[str] = set()
        for source in refs:
            if not isinstance(source, str):
                continue
            cleaned = source.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            result.append(cleaned if len(cleaned) <= 80 else f"{cleaned[:77]}...")
            if len(result) >= 2:
                break
        return result

    def _vector_context_update(self, blocks: list[dict]) -> dict:
        sources: list[str] = []
        seen: set[str] = set()
        for block in blocks:
            source = block.get("source")
            if not isinstance(source, str):
                continue
            cleaned = source.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            sources.append(cleaned)

        return {
            "retrieved_context": blocks,
            "context_sources": sources,
        }

    def _sanitize_context_blocks(self, blocks: list[dict], *, source_label: str) -> tuple[list[dict], dict]:
        kept: list[dict] = []
        dropped_conflict = 0
        dropped_low_value = 0

        for block in blocks:
            normalized = self._normalize_block(block)
            if not normalized:
                dropped_low_value += 1
                continue

            if self._conflicts_with_tx_truth(normalized):
                dropped_conflict += 1
                continue

            if self._is_low_value_block(normalized, source_label=source_label):
                dropped_low_value += 1
                continue

            kept.append(normalized)

        return kept, {
            "conflict": dropped_conflict,
            "low_value": dropped_low_value,
        }

    def _normalize_block(self, block: dict) -> dict | None:
        if not isinstance(block, dict):
            return None

        key = block.get("key")
        title = block.get("title")
        summary = block.get("summary")
        notes = block.get("notes", [])

        normalized: dict = {}
        if isinstance(key, str) and key.strip():
            normalized["key"] = key.strip()
        if isinstance(title, str) and title.strip():
            normalized["title"] = title.strip()
        if isinstance(summary, str) and summary.strip():
            normalized["summary"] = summary.strip()

        compact_notes: list[str] = []
        if isinstance(notes, list):
            for note in notes:
                if not isinstance(note, str):
                    continue
                cleaned = note.strip()
                if cleaned:
                    compact_notes.append(cleaned)
                if len(compact_notes) >= 3:
                    break
        if compact_notes:
            normalized["notes"] = compact_notes

        source = block.get("source")
        if isinstance(source, str) and source.strip():
            normalized["source"] = source.strip()

        score = block.get("score")
        if isinstance(score, (int, float)):
            normalized["score"] = float(score)

        if not normalized.get("summary") and not normalized.get("notes"):
            return None
        return normalized

    def _is_low_value_block(self, block: dict, *, source_label: str) -> bool:
        summary = str(block.get("summary", "")).strip()
        notes = block.get("notes", [])

        if source_label == "vector":
            score = block.get("score")
            if isinstance(score, (int, float)) and float(score) < 0.18:
                return True

        if not summary and not notes:
            return True

        if summary and len(summary) < 24 and not notes:
            return True

        if summary and self._is_generic_summary(summary) and not notes:
            return True

        return False

    def _is_generic_summary(self, summary: str) -> bool:
        text = summary.strip().lower()
        if not text:
            return True

        generic_phrases = (
            "provides functionality",
            "handles operations",
            "is used for",
            "general purpose",
            "various actions",
        )
        if any(phrase in text for phrase in generic_phrases):
            return True

        tokens = self._tokenize_words(text)
        if not tokens:
            return True

        stop = {
            "the", "and", "for", "with", "this", "that", "from", "into", "over", "under",
            "transaction", "module", "message", "type", "used", "use",
        }
        informative = {token for token in tokens if token not in stop}
        return len(informative) < 4

    def _conflicts_with_tx_truth(self, block: dict) -> bool:
        status = str(self.fact_index.get("status") or "").strip().lower()
        tx_type = str(self.fact_index.get("tx_type") or "").strip().lower()
        key = str(block.get("key") or "").strip().lower()

        text_parts = []
        summary = block.get("summary")
        if isinstance(summary, str):
            text_parts.append(summary.lower())
        notes = block.get("notes", [])
        if isinstance(notes, list):
            for note in notes:
                if isinstance(note, str):
                    text_parts.append(note.lower())
        text = " ".join(text_parts)

        if status == "failed":
            success_claims = (
                "this transaction succeeded",
                "this tx succeeded",
                "transaction completed successfully",
                "status is success",
                "status: success",
            )
            if any(claim in text for claim in success_claims):
                return True
        elif status == "success":
            failure_claims = (
                "this transaction failed",
                "this tx failed",
                "status is failed",
                "status: failed",
            )
            if any(claim in text for claim in failure_claims):
                return True

        if status == "success" and key.startswith("failure_category:"):
            return True

        if tx_type and tx_type != "unknown" and key.startswith("tx_type:"):
            block_type = key.split(":", 1)[1].strip()
            if block_type and block_type not in self._tx_type_equivalents(tx_type):
                return True

        return False

    def _tx_type_equivalents(self, tx_type: str) -> set[str]:
        pairs = {
            "dex_swap": {"dex_swap", "swap"},
            "swap": {"swap", "dex_swap"},
        }
        base = tx_type.strip().lower()
        return pairs.get(base, {base})

    def _tokenize_words(self, text: str) -> list[str]:
        words: list[str] = []
        current: list[str] = []
        for ch in text:
            if ch.isalnum() or ch == "_":
                current.append(ch)
                continue
            if current:
                words.append("".join(current))
                current = []
        if current:
            words.append("".join(current))
        return words

    def _dropped_context_line(self, dropped: dict | None) -> str | None:
        if not isinstance(dropped, dict):
            return None

        conflict = int(dropped.get("conflict") or 0)
        low_value = int(dropped.get("low_value") or 0)
        if conflict <= 0 and low_value <= 0:
            return None

        parts = []
        if conflict > 0:
            parts.append(f"{conflict} conflicting")
        if low_value > 0:
            parts.append(f"{low_value} low-value")
        return f"Skipped {' and '.join(parts)} context snippet(s) to preserve transaction-grounded accuracy."
