"""
ZigChain TX Reasoning Engine - Telegram Bot

Telegram frontend for transaction explanations and follow-up Q&A.
"""

import asyncio
import html
import io
import json
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.chat import ChatSession
from src.config import CHAT_LLM_FALLBACK, TELEGRAM_BOT_TOKEN
from src.llm import call_llm, warmup_models
from src.query_engine import route_question
from src.tokens import registry as token_registry


logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("zig-bot")

sessions: dict[int, ChatSession] = {}
MAX_MSG_LEN = 4096


def _escape(text: str) -> str:
    return html.escape(str(text))


def _chat_meta(update: Update) -> dict:
    user = update.effective_user
    chat = update.effective_chat
    return {
        "chat_id": chat.id if chat else None,
        "user_id": user.id if user else None,
        "username": user.username if user else None,
    }


def _split_message(text: str) -> list[str]:
    """Split long text into valid Telegram-sized non-empty chunks."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= MAX_MSG_LEN:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= MAX_MSG_LEN:
            chunks.append(remaining.strip())
            break

        split_at = remaining.rfind("\n", 0, MAX_MSG_LEN)
        if split_at == -1 or split_at < MAX_MSG_LEN // 2:
            split_at = remaining.rfind(" ", 0, MAX_MSG_LEN)
        if split_at == -1 or split_at < MAX_MSG_LEN // 2:
            split_at = MAX_MSG_LEN

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    return [chunk for chunk in chunks if chunk]


def _merge_source_refs(existing: list | None, incoming: list | None, *, max_items: int = 4) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for group in (existing or [], incoming or []):
        if not isinstance(group, list):
            continue
        for source in group:
            if not isinstance(source, str):
                continue
            cleaned = source.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(cleaned if len(cleaned) <= 80 else f"{cleaned[:77]}...")
            if len(merged) >= max_items:
                return merged

    return merged


def _append_source_refs(text: str, source_refs: list | None) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    if not isinstance(source_refs, list):
        return text

    compact = _merge_source_refs([], source_refs, max_items=2)
    if not compact:
        return text
    if text.strip().startswith("[ERROR]"):
        return text
    return f"{text}\n\nSources: {', '.join(compact)}"


def _fallback_tx_message(normalized: dict, interpretation: dict) -> str:
    """Fallback when the LLM returns an empty response."""
    lines = [
        f"Transaction status: {normalized['status']}.",
        f"Type: {interpretation['tx_type']}.",
        f"Summary: {interpretation['summary']}.",
    ]
    warnings = interpretation.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning.get('message', '')}")
    return "\n".join(lines)


async def _send_text_chunks(
    send_func,
    text: str,
    *,
    fallback_text: str | None = None,
    parse_mode: str | None = None,
):
    """Send text safely and never attempt to send an empty Telegram message."""
    chunks = _split_message(text)
    if not chunks and fallback_text:
        chunks = _split_message(fallback_text)
    if not chunks:
        logger.warning("Skipped empty outgoing Telegram message")
        return

    for chunk in chunks:
        kwargs = {"parse_mode": parse_mode} if parse_mode else {}
        await send_func(chunk, **kwargs)


def _analyze_tx_sync(tx_hash: str) -> tuple[dict, dict, str, ChatSession]:
    clean_hash = ChatSession.normalize_tx_hash(tx_hash)
    clean_hash, normalized, interpretation, cache_hit = ChatSession.get_or_load_processed(clean_hash)
    if cache_hit:
        logger.info("TX cache hit for %s", clean_hash[:16])

    complexity = interpretation["complexity"]
    explanation = call_llm(normalized, interpretation, complexity=complexity)

    if not (explanation or "").strip():
        explanation = _fallback_tx_message(normalized, interpretation)

    session = ChatSession(clean_hash, normalized, interpretation, load_explanation=explanation)
    session.add_assistant_message(explanation)
    return normalized, interpretation, explanation, session


def _ask_question_sync(session: ChatSession, question: str) -> tuple[str, bool]:
    cached = session.get_cached_response(question)
    if cached is not None:
        response, allowed = cached
        session.add_user_message(question)
        session.add_assistant_message(response)
        return response, allowed

    route = route_question(
        question,
        session.query_ctx,
        session.get_fact_index(),
        context_artifacts=session.get_context_artifacts(),
    )
    if not route["allowed"]:
        response = route["response"] or "This assistant only answers questions about this specific transaction."
        session.cache_response(question, response, False)
        return response, False

    context_update = route.get("context_update")
    if isinstance(context_update, dict):
        existing_sources = session.get_context_artifacts().get("context_sources", [])
        merged_sources = _merge_source_refs(existing_sources, context_update.get("context_sources"))
        session.set_context_artifacts(
            retrieved_context=context_update.get("retrieved_context"),
            context_sources=merged_sources,
        )

    if route["mode"] in ("deterministic", "template", "manifest_context", "vector_fallback"):
        response = route["response"] or "I could not resolve that from the indexed transaction facts."
        session.add_user_message(question)
        session.add_assistant_message(response)
        session.cache_response(question, response, True)
        return response, True

    note = route["note"]
    directive = route["directive"]
    if route["mode"] == "llm":
        reused = session.get_load_explanation(note=note)
        if reused:
            session.add_user_message(question)
            session.add_assistant_message(reused)
            session.cache_response(question, reused, True)
            return reused, True

    if not CHAT_LLM_FALLBACK:
        response = "I could not answer from cached transaction analysis in this session."
        session.add_user_message(question)
        session.add_assistant_message(response)
        session.cache_response(question, response, True)
        return response, True

    session.add_user_message(question)
    history = session.get_context_history()

    effective_question = question
    if note:
        effective_question = f"{question}\n\n[System note: {note}]"

    response = call_llm(
        session.normalized_data,
        session.interpretation,
        user_question=effective_question,
        chat_history=history[:-1],
        complexity=session.complexity,
        prompt_directive=directive,
        context_artifacts=session.get_context_artifacts(),
    )
    if route["mode"] == "llm_context":
        response = _append_source_refs(response, route.get("source_refs"))

    if not (response or "").strip():
        response = "I could not generate an explanation for that question from the transaction data."

    session.cache_response(question, response, True)
    session.add_assistant_message(response)
    return response, True


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "<b>ZigChain Transaction Explainer</b>\n\n"
        "Send a 64-character transaction hash or use /tx <code>&lt;hash&gt;</code>.\n"
        "After a transaction is loaded, ask follow-up questions in plain text."
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    meta = _chat_meta(update)

    if not context.args:
        await update.message.reply_text("Usage: /tx <code>&lt;hash&gt;</code>", parse_mode=ParseMode.HTML)
        return

    tx_hash = context.args[0].strip()
    logger.info("TX request chat_id=%s user_id=%s username=%s tx_hash=%s",
                meta["chat_id"], meta["user_id"], meta["username"], tx_hash)

    await update.effective_chat.send_action(ChatAction.TYPING)
    status_msg = await update.message.reply_text("Fetching transaction and generating explanation...")

    try:
        normalized, interpretation, explanation, session = await asyncio.to_thread(_analyze_tx_sync, tx_hash)
    except ValueError as exc:
        logger.warning("TX request failed chat_id=%s tx_hash=%s error=%s", meta["chat_id"], tx_hash, exc)
        await status_msg.edit_text(f"Error: {_escape(str(exc))}", parse_mode=ParseMode.HTML)
        return
    except (ConnectionError, TimeoutError) as exc:
        logger.warning("TX request unavailable chat_id=%s tx_hash=%s error=%s", meta["chat_id"], tx_hash, exc)
        await status_msg.edit_text(f"Error: {_escape(str(exc))}", parse_mode=ParseMode.HTML)
        return
    except Exception as exc:
        logger.error("TX request crashed chat_id=%s tx_hash=%s error=%s",
                     meta["chat_id"], tx_hash, exc, exc_info=True)
        await status_msg.edit_text(f"Error: {_escape(str(exc))}", parse_mode=ParseMode.HTML)
        return

    sessions[chat_id] = session

    logger.info(
        "TX response chat_id=%s tx_hash=%s status=%s type=%s complexity=%s explanation_len=%s",
        meta["chat_id"],
        normalized["tx_hash"],
        normalized["status"],
        interpretation["tx_type"],
        interpretation["complexity"],
        len(explanation),
    )

    await status_msg.edit_text("Explanation ready. You can ask follow-up questions below.")
    await _send_text_chunks(
        update.effective_chat.send_message,
        explanation,
        fallback_text=_fallback_tx_message(normalized, interpretation),
    )


async def cmd_raw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = sessions.get(chat_id)

    if not session:
        await update.message.reply_text(
            "No transaction loaded. Use /tx <code>&lt;hash&gt;</code> first.",
            parse_mode=ParseMode.HTML,
        )
        return

    raw_json = json.dumps(session.normalized_data, indent=2)
    if len(raw_json) <= MAX_MSG_LEN - 20:
        await update.message.reply_text(f"<pre>{_escape(raw_json)}</pre>", parse_mode=ParseMode.HTML)
        return

    file = io.BytesIO(raw_json.encode("utf-8"))
    file.name = f"tx_{session.tx_hash[:12]}.json"
    await update.message.reply_document(document=file, caption="Normalized transaction data")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = sessions.get(chat_id)

    if not session:
        await update.message.reply_text(
            "No transaction loaded. Use /tx <code>&lt;hash&gt;</code> first.",
            parse_mode=ParseMode.HTML,
        )
        return

    stats = session.get_stats()
    text = (
        "<b>Session Stats</b>\n\n"
        f"<b>TX:</b> <code>{_escape(stats['tx_hash'][:20])}...</code>\n"
        f"<b>Type:</b> {_escape(stats['tx_type'])}\n"
        f"<b>Complexity:</b> {_escape(stats['complexity'])}\n"
        f"<b>Messages:</b> {stats['messages']}\n"
        f"<b>Cached answers:</b> {stats.get('cached_answers', 0)}\n"
        f"<b>Cache hits:</b> {stats.get('cache_hits', 0)}\n"
        f"<b>Fact fields:</b> {stats.get('fact_fields', 0)}\n"
        f"<b>Tokens loaded:</b> {token_registry.token_count}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    meta = _chat_meta(update)

    if not text:
        return

    logger.info("Incoming message chat_id=%s user_id=%s username=%s text=%r",
                meta["chat_id"], meta["user_id"], meta["username"], text)

    clean = text.lower().replace("0x", "")
    if len(clean) == 64 and all(char in "0123456789abcdef" for char in clean):
        context.args = [text]
        await cmd_tx(update, context)
        return

    session = sessions.get(chat_id)
    if not session:
        await update.message.reply_text(
            "No transaction loaded.\n\nSend a 64-character tx hash or use /tx <code>&lt;hash&gt;</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.effective_chat.send_action(ChatAction.TYPING)

    try:
        response, allowed = await asyncio.to_thread(_ask_question_sync, session, text)
    except Exception as exc:
        logger.error("Question failed chat_id=%s tx_hash=%s question=%r error=%s",
                     meta["chat_id"], session.tx_hash, text, exc, exc_info=True)
        await update.message.reply_text(f"Error: {_escape(str(exc))}", parse_mode=ParseMode.HTML)
        return

    logger.info(
        "Question handled chat_id=%s tx_hash=%s allowed=%s question=%r response_len=%s",
        meta["chat_id"],
        session.tx_hash,
        allowed,
        text,
        len(response or ""),
    )

    if not allowed:
        await update.message.reply_text(response)
        return

    await _send_text_chunks(update.message.reply_text, response)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled Telegram error: %s", context.error, exc_info=context.error)


async def post_init(application: Application):
    logger.info("Loading token registry...")
    await asyncio.to_thread(token_registry.load, True)
    logger.info("Token registry loaded: %s tokens", token_registry.token_count)

    logger.info("Warming up LLM models...")
    results = await asyncio.to_thread(warmup_models, True)
    loaded = sum(1 for ready in results.values() if ready)
    logger.info("Model warmup ready: %s/%s", loaded, len(results))


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("Get a token from @BotFather on Telegram and add it to your .env file.")
        return

    logger.info("Starting ZigChain TX Bot...")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("tx", cmd_tx))
    app.add_handler(CommandHandler("raw", cmd_raw))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
