import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ZigChain RPC
ZIGCHAIN_RPC_URL = os.getenv("ZIGCHAIN_RPC_URL", "https://kore-archive.wickhub.cc/")

# LLM API (Ollama-style)
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:3b")
LLM_API_USER = os.getenv("LLM_API_USER")
LLM_API_PASSWORD = os.getenv("LLM_API_PASSWORD")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
# Optional fallback for additional chat-time LLM calls if load-time explanation is missing.
CHAT_LLM_FALLBACK = os.getenv("CHAT_LLM_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
