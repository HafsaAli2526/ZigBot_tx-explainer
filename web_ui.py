"""
ZigChain TX Reasoning Engine - Local Web UI

Run:
  python web_ui.py

Then open:
  http://127.0.0.1:8787
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from src.chat import ChatSession
from src.config import CHAT_LLM_FALLBACK
from src.llm import call_llm, warmup_models
from src.query_engine import route_question
from src.tokens import registry as token_registry


logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("zig-web")

HOST = os.getenv("WEB_UI_HOST", "127.0.0.1")
PORT = int(os.getenv("WEB_UI_PORT", "8787"))
SESSION_COOKIE = "zig_session"

sessions: dict[str, ChatSession] = {}
sessions_lock = threading.Lock()

init_state = {
    "tokens_loaded": False,
    "models_ready": 0,
    "models_total": 0,
    "ready": False,
}


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>ZigChain TX Explainer</title>
  <style>
    :root {
      --bg-1: #f6f2ea;
      --bg-2: #e3edf2;
      --card: #ffffff;
      --ink: #11232d;
      --muted: #49606b;
      --line: #cad7dd;
      --accent: #0f8a62;
      --accent-ink: #ffffff;
      --warning: #8a5b0f;
      --error: #8f1d1d;
      --radius: 14px;
      --shadow: 0 8px 24px rgba(17, 35, 45, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at 15% 20%, #fef4dc 0%, transparent 35%),
        radial-gradient(circle at 90% 0%, #dff2f5 0%, transparent 30%),
        linear-gradient(135deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
    }

    .page {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }

    .hero {
      background: linear-gradient(145deg, #0d3d5e, #0f5f65);
      color: #f2fbff;
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 20px;
      margin-bottom: 16px;
    }

    .hero h1 {
      margin: 0 0 6px;
      font-size: 1.45rem;
      letter-spacing: 0.2px;
    }

    .hero p {
      margin: 0;
      color: #cdeef7;
      font-size: 0.95rem;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 14px;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }

    .input, .button, .textarea {
      border-radius: 10px;
      border: 1px solid var(--line);
      font: inherit;
    }

    .input, .textarea {
      width: 100%;
      padding: 11px 12px;
      color: var(--ink);
      background: #fff;
    }

    .textarea {
      min-height: 90px;
      resize: vertical;
    }

    .button {
      padding: 11px 14px;
      background: var(--accent);
      color: var(--accent-ink);
      border: none;
      cursor: pointer;
      font-weight: 600;
      min-width: 110px;
      transition: transform .12s ease, opacity .12s ease;
    }

    .button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .button:active { transform: translateY(1px); }

    .meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }

    .pill {
      border: 1px solid var(--line);
      background: #f9fcfd;
      border-radius: 11px;
      padding: 8px 10px;
      font-size: 0.92rem;
    }

    .k {
      color: var(--muted);
      margin-right: 6px;
      font-size: 0.83rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .status {
      margin-top: 8px;
      font-size: 0.9rem;
      color: var(--muted);
    }

    .status.error { color: var(--error); }
    .status.warning { color: var(--warning); }

    .block {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fbfdfe;
      padding: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .chat {
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }

    .msg {
      padding: 10px 12px;
      border-radius: 11px;
      max-width: 90%;
      white-space: pre-wrap;
      line-height: 1.4;
      border: 1px solid var(--line);
    }

    .msg.user {
      margin-left: auto;
      background: #e8fff6;
      border-color: #b7e8d4;
    }

    .msg.assistant {
      margin-right: auto;
      background: #ffffff;
    }

    .small {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.82rem;
    }

    @media (max-width: 760px) {
      .meta { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .button { width: 100%; }
      .msg { max-width: 100%; }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>ZigChain Transaction Explainer</h1>
      <p>Analyze one transaction and ask follow-up questions in the same context.</p>
    </section>

    <section class="card">
      <div class="row">
        <input id="txHash" class="input" placeholder="Paste 64-char tx hash (with or without 0x)" />
        <button id="analyzeBtn" class="button">Analyze</button>
      </div>
      <div id="appStatus" class="status">Loading engine status...</div>
      <div class="meta">
        <div class="pill"><span class="k">Status</span><span id="txStatus">-</span></div>
        <div class="pill"><span class="k">Type</span><span id="txType">-</span></div>
        <div class="pill"><span class="k">Complexity</span><span id="txComplexity">-</span></div>
      </div>
      <div class="small" id="txSummary"></div>
    </section>

    <section class="card">
      <h3 style="margin-top:0">Explanation</h3>
      <div id="explanation" class="block">No transaction loaded yet.</div>
    </section>

    <section class="card">
      <h3 style="margin-top:0">Follow-up</h3>
      <div id="chatLog" class="chat"></div>
      <div class="row">
        <textarea id="question" class="textarea" placeholder="Ask a question about the loaded transaction..."></textarea>
        <button id="askBtn" class="button" disabled>Ask</button>
      </div>
      <div class="small">Only questions about the currently loaded transaction are allowed.</div>
    </section>
  </main>

  <script>
    const txHashEl = document.getElementById("txHash");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const askBtn = document.getElementById("askBtn");
    const appStatusEl = document.getElementById("appStatus");
    const txStatusEl = document.getElementById("txStatus");
    const txTypeEl = document.getElementById("txType");
    const txComplexityEl = document.getElementById("txComplexity");
    const txSummaryEl = document.getElementById("txSummary");
    const explanationEl = document.getElementById("explanation");
    const questionEl = document.getElementById("question");
    const chatLogEl = document.getElementById("chatLog");

    let txLoaded = false;

    function setStatus(text, kind = "") {
      appStatusEl.textContent = text;
      appStatusEl.className = "status" + (kind ? " " + kind : "");
    }

    function appendMessage(role, text) {
      const msg = document.createElement("div");
      msg.className = "msg " + role;
      msg.textContent = text;
      chatLogEl.appendChild(msg);
      chatLogEl.scrollTop = chatLogEl.scrollHeight;
    }

    async function api(path, body) {
      const resp = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json().catch(() => ({ ok: false, error: "Invalid server response" }));
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || ("Request failed (" + resp.status + ")"));
      }
      return data;
    }

    async function loadHealth() {
      try {
        const resp = await fetch("/api/health");
        const data = await resp.json();
        if (!data.ok) {
          setStatus("Engine status unavailable", "warning");
          return;
        }
        if (data.ready) {
          setStatus("Engine ready");
        } else {
          setStatus("Engine warming up: models " + data.models_ready + "/" + data.models_total, "warning");
        }
      } catch {
        setStatus("Failed to load engine status", "error");
      }
    }

    async function analyzeTx() {
      const txHash = txHashEl.value.trim();
      if (!txHash) {
        setStatus("Please enter a transaction hash.", "warning");
        return;
      }

      analyzeBtn.disabled = true;
      askBtn.disabled = true;
      setStatus("Fetching transaction and generating explanation...");
      explanationEl.textContent = "Working...";
      chatLogEl.innerHTML = "";

      try {
        const data = await api("/api/tx", { tx_hash: txHash });
        txLoaded = true;
        askBtn.disabled = false;

        txStatusEl.textContent = data.status || "-";
        txTypeEl.textContent = data.tx_type || "-";
        txComplexityEl.textContent = data.complexity || "-";
        txSummaryEl.textContent = data.summary || "";
        explanationEl.textContent = data.explanation || "(No explanation returned)";
        appendMessage("assistant", data.explanation || "(No explanation returned)");
        setStatus("Explanation ready.");
      } catch (err) {
        txLoaded = false;
        explanationEl.textContent = "No transaction loaded yet.";
        setStatus(err.message || "Failed to analyze transaction.", "error");
      } finally {
        analyzeBtn.disabled = false;
      }
    }

    async function askQuestion() {
      const question = questionEl.value.trim();
      if (!txLoaded) {
        setStatus("Load a transaction first.", "warning");
        return;
      }
      if (!question) {
        setStatus("Type a question first.", "warning");
        return;
      }

      askBtn.disabled = true;
      appendMessage("user", question);
      questionEl.value = "";
      setStatus("Thinking...");

      try {
        const data = await api("/api/ask", { question });
        appendMessage("assistant", data.response || "(No response)");
        if (data.allowed === false) {
          setStatus("Question blocked by transaction scope check.", "warning");
        } else {
          setStatus("Answer ready.");
        }
      } catch (err) {
        appendMessage("assistant", "Error: " + (err.message || "Failed to answer question."));
        setStatus("Failed to answer question.", "error");
      } finally {
        askBtn.disabled = false;
      }
    }

    analyzeBtn.addEventListener("click", analyzeTx);
    askBtn.addEventListener("click", askQuestion);
    txHashEl.addEventListener("keydown", (e) => { if (e.key === "Enter") analyzeTx(); });
    questionEl.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") askQuestion();
    });

    loadHealth();
    setInterval(loadHealth, 10000);
  </script>
</body>
</html>
"""


def _split_cookie(cookie_header: str) -> dict[str, str]:
    parts = [p.strip() for p in cookie_header.split(";") if p.strip()]
    result: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _normalize_hash(tx_hash: str) -> str:
    return ChatSession.normalize_tx_hash(tx_hash)


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


def _analyze_tx_sync(tx_hash: str) -> tuple[dict, dict, str, ChatSession]:
    clean_hash = _normalize_hash(tx_hash)
    clean_hash, normalized, interpretation, _ = ChatSession.get_or_load_processed(clean_hash)

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


def _background_init():
    logger.info("Loading token registry...")
    token_registry.load(verbose=True)
    init_state["tokens_loaded"] = True
    logger.info("Token registry loaded: %s tokens", token_registry.token_count)

    logger.info("Warming up LLM models...")
    results = warmup_models(verbose=True)
    loaded = sum(1 for ready in results.values() if ready)
    init_state["models_ready"] = loaded
    init_state["models_total"] = len(results)
    init_state["ready"] = loaded > 0
    logger.info("Model warmup ready: %s/%s", loaded, len(results))


class ZigWebHandler(BaseHTTPRequestHandler):
    server_version = "ZigWebUI/1.0"

    def log_message(self, fmt: str, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _get_session_id(self) -> str | None:
        cookies = _split_cookie(self.headers.get("Cookie", ""))
        session_id = cookies.get(SESSION_COOKIE)
        if session_id:
            return session_id
        return None

    def _get_or_create_session_id(self) -> tuple[str, bool]:
        existing = self._get_session_id()
        if existing:
            return existing, False
        return uuid.uuid4().hex, True

    def _send_json(self, code: int, payload: dict, set_session_id: str | None = None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if set_session_id:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={set_session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length_raw = self.headers.get("Content-Length")
        if not length_raw:
            return {}
        try:
            length = int(length_raw)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Request body must be valid JSON") from exc

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return

        if parsed.path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "ready": init_state["ready"],
                    "tokens_loaded": init_state["tokens_loaded"],
                    "tokens_count": token_registry.token_count,
                    "models_ready": init_state["models_ready"],
                    "models_total": init_state["models_total"],
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)

        try:
            data = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/tx":
            self._handle_tx(data)
            return

        if parsed.path == "/api/ask":
            self._handle_ask(data)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def _handle_tx(self, data: dict):
        tx_hash = str(data.get("tx_hash", "")).strip()
        if not tx_hash:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "tx_hash is required"})
            return

        try:
            normalized, interpretation, explanation, session = _analyze_tx_sync(tx_hash)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        except (ConnectionError, TimeoutError) as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            logger.exception("Analyze tx failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return

        session_id, created = self._get_or_create_session_id()
        with sessions_lock:
            sessions[session_id] = session

        payload = {
            "ok": True,
            "tx_hash": normalized.get("tx_hash"),
            "status": normalized.get("status"),
            "tx_type": interpretation.get("tx_type"),
            "complexity": interpretation.get("complexity"),
            "summary": interpretation.get("summary"),
            "warnings": interpretation.get("warnings", []),
            "explanation": explanation,
        }
        self._send_json(HTTPStatus.OK, payload, set_session_id=session_id if created else None)

    def _handle_ask(self, data: dict):
        question = str(data.get("question", "")).strip()
        if not question:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "question is required"})
            return

        session_id = self._get_session_id()
        if not session_id:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"ok": False, "error": "No active transaction session. Analyze a tx first."},
            )
            return

        with sessions_lock:
            session = sessions.get(session_id)

        if not session:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"ok": False, "error": "Session expired. Analyze a transaction again."},
            )
            return

        try:
            response, allowed = _ask_question_sync(session, question)
        except Exception as exc:
            logger.exception("Ask question failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "allowed": allowed,
                "response": response,
            },
        )


def main():
    logger.info("Starting ZigChain Web UI...")
    init_thread = threading.Thread(target=_background_init, daemon=True)
    init_thread.start()

    server = ThreadingHTTPServer((HOST, PORT), ZigWebHandler)
    logger.info("Web UI running at http://%s:%s", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
