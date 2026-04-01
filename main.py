"""
ZigChain TX Reasoning Engine - CLI Interface

TX -> Normalized State -> Deterministic Interpretation -> LLM Explanation -> Guarded Chat
"""

import json
import sys
import threading

from colorama import Fore, Style, init

from src.chat import ChatSession
from src.config import CHAT_LLM_FALLBACK, LLM_MODEL_NAME
from src.llm import call_llm, warmup_models
from src.query_engine import route_question
from src.tokens import registry as token_registry

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}============================================================
          ZigChain Transaction Reasoning Engine

  Deterministic analysis + LLM explanation for ZigChain
============================================================{Style.RESET_ALL}
"""

HELP_TEXT = f"""
{Fore.YELLOW}Commands:{Style.RESET_ALL}
  {Fore.GREEN}/tx <hash>{Style.RESET_ALL}      - Analyze a new transaction
  {Fore.GREEN}/raw{Style.RESET_ALL}            - Show raw normalized JSON for current tx
  {Fore.GREEN}/interpret{Style.RESET_ALL}      - Show deterministic interpretation
  {Fore.GREEN}/stats{Style.RESET_ALL}          - Show current session stats
  {Fore.GREEN}/help{Style.RESET_ALL}           - Show this help message
  {Fore.GREEN}/quit{Style.RESET_ALL}           - Exit

  Or just type a question about the current transaction.
"""

_init_done = threading.Event()
_init_results = {}


def _background_init():
    """Run model warmup + token loading in background."""
    global _init_results

    token_count = token_registry.load(verbose=True)
    model_results = warmup_models(verbose=True)
    loaded = sum(1 for ok in model_results.values() if ok)

    _init_results = {
        "tokens": token_count,
        "models_loaded": loaded,
        "models_total": len(model_results),
    }
    _init_done.set()


def print_error(msg: str):
    print(f"\n{Fore.RED}! {msg}{Style.RESET_ALL}")


def print_info(msg: str):
    print(f"\n{Fore.YELLOW}> {msg}{Style.RESET_ALL}")


def print_llm_response(text: str):
    print(f"\n{Fore.WHITE}{text}{Style.RESET_ALL}\n")


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


def print_warnings(warnings: list):
    for warning in warnings:
        level = warning.get("level", "info")
        msg = warning.get("message", "")
        if level == "critical":
            print(f"  {Fore.RED}[!] {msg}{Style.RESET_ALL}")
        elif level == "warning":
            print(f"  {Fore.YELLOW}[!] {msg}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.CYAN}[i] {msg}{Style.RESET_ALL}")


def analyze_tx(tx_hash: str) -> ChatSession | None:
    """Full pipeline: Fetch -> Normalize -> Interpret -> LLM Explain."""
    clean_hash = ChatSession.normalize_tx_hash(tx_hash)
    print_info(f"Fetching tx {clean_hash[:16]}...")

    try:
        clean_hash, normalized, interpretation, cache_hit = ChatSession.get_or_load_processed(clean_hash)
    except Exception as exc:
        print_error(str(exc))
        return None

    if cache_hit:
        print_info("Using cached normalized + interpreted transaction data...")
    else:
        print_info("Normalizing transaction data...")
        print_info("Running deterministic analysis...")

    complexity = interpretation["complexity"]

    print(f"\n{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Type:{Style.RESET_ALL}       {interpretation['tx_type']}")
    print(f"  {Fore.WHITE}Status:{Style.RESET_ALL}     {normalized['status'].upper()}")
    print(f"  {Fore.WHITE}Complexity:{Style.RESET_ALL} {complexity}")
    print(f"  {Fore.WHITE}Model:{Style.RESET_ALL}      {LLM_MODEL_NAME}")
    print(f"  {Fore.WHITE}Summary:{Style.RESET_ALL}    {interpretation['summary']}")

    if interpretation["warnings"]:
        print(f"\n  {Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        print_warnings(interpretation["warnings"])
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    print_info(f"Generating explanation ({LLM_MODEL_NAME})...")
    explanation = call_llm(normalized, interpretation, complexity=complexity)
    print_llm_response(explanation)

    session = ChatSession(clean_hash, normalized, interpretation, load_explanation=explanation)
    session.add_assistant_message(explanation)
    return session


def handle_question(session: ChatSession, question: str):
    """Handle a follow-up question through the query intelligence layer."""
    cached = session.get_cached_response(question)
    if cached is not None:
        response, allowed = cached
        session.add_user_message(question)
        session.add_assistant_message(response)
        if not allowed:
            print(f"\n{Fore.YELLOW}{response}{Style.RESET_ALL}\n")
            return
        print_info("Using cached answer from this transaction session.")
        print_llm_response(response)
        return

    route = route_question(
        question,
        session.query_ctx,
        session.get_fact_index(),
        context_artifacts=session.get_context_artifacts(),
    )
    if not route["allowed"]:
        response = route["response"] or "This assistant only answers questions about this specific transaction."
        session.cache_response(question, response, False)
        print(f"\n{Fore.YELLOW}{response}{Style.RESET_ALL}\n")
        return

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
        print_info(f"Answered via {route['mode']} route.")
        print_llm_response(response)
        return

    note = route["note"]
    directive = route["directive"]
    if route["mode"] == "llm":
        reused = session.get_load_explanation(note=note)
        if reused:
            session.add_user_message(question)
            session.add_assistant_message(reused)
            session.cache_response(question, reused, True)
            print_info("Using load-time transaction explanation (no extra LLM call).")
            print_llm_response(reused)
            return

    if not CHAT_LLM_FALLBACK:
        response = "I could not answer from cached transaction analysis in this session."
        session.add_user_message(question)
        session.add_assistant_message(response)
        session.cache_response(question, response, True)
        print_llm_response(response)
        return

    session.add_user_message(question)
    history = session.get_context_history()

    effective_question = question
    if note:
        effective_question = f"{question}\n\n[System note: {note}]"

    if route["mode"] == "llm_context":
        print_info(f"Thinking with routed context ({LLM_MODEL_NAME})...")
    else:
        print_info(f"Thinking ({LLM_MODEL_NAME})...")
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

    session.cache_response(question, response, True)
    session.add_assistant_message(response)
    print_llm_response(response)


def main():
    print(BANNER)

    print(f"{Fore.YELLOW}> Initializing (tokens + model loading in background)...{Style.RESET_ALL}")
    init_thread = threading.Thread(target=_background_init, daemon=True)
    init_thread.start()

    print(HELP_TEXT)

    session = None

    if len(sys.argv) > 1:
        if not _init_done.wait(timeout=60):
            print_info("Background init still running, proceeding anyway...")
        session = analyze_tx(sys.argv[1])

    while True:
        try:
            prompt_color = Fore.GREEN if session else Fore.YELLOW
            prompt_label = f"tx:{session.tx_hash[:8]}..." if session else "no-tx"
            user_input = input(f"{prompt_color}[{prompt_label}]>{Style.RESET_ALL} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.CYAN}Goodbye.{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                print(f"\n{Fore.CYAN}Goodbye.{Style.RESET_ALL}")
                break
            if cmd in ("/help", "/h"):
                print(HELP_TEXT)
                continue
            if cmd == "/tx":
                if len(parts) < 2:
                    print_error("Usage: /tx <hash>")
                else:
                    session = analyze_tx(parts[1])
                continue
            if cmd == "/raw":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    print(f"\n{json.dumps(session.normalized_data, indent=2)}\n")
                continue
            if cmd == "/interpret":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    print(f"\n{json.dumps(session.interpretation, indent=2)}\n")
                continue
            if cmd == "/stats":
                if not session:
                    print_error("No transaction loaded. Use /tx <hash> first.")
                else:
                    stats = session.get_stats()
                    print(f"\n  TX: {stats['tx_hash']}")
                    print(f"  Type: {stats['tx_type']}")
                    print(f"  Complexity: {stats['complexity']}")
                    print(f"  Model: {LLM_MODEL_NAME}")
                    print(f"  Messages: {stats['messages']}")
                    print(f"  Cached answers: {stats.get('cached_answers', 0)}")
                    print(f"  Cache hits: {stats.get('cache_hits', 0)}")
                    print(f"  Fact fields: {stats.get('fact_fields', 0)}")
                    print(f"  Tokens loaded: {token_registry.token_count}\n")
                continue

            print_error(f"Unknown command: {cmd}. Type /help for available commands.")
            continue

        if not session:
            print_error("No transaction loaded. Use /tx <hash> first.")
            continue

        handle_question(session, user_input)


if __name__ == "__main__":
    main()
