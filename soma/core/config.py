"""acsis/core/config.py — All configuration in one place."""
from dataclasses import dataclass, field
import os


@dataclass
class AcsisConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    base_model: str       = "Qwen/Qwen3-7B"          # swap to phi-3 for smaller runs
    device: str           = "auto"                    # "cuda", "cpu", "auto"
    load_in_4bit: bool    = True                      # quantize for T4 VRAM
    lora_rank: int        = 8
    lora_alpha: int       = 16

    # ── Research tools ───────────────────────────────────────────────────────
    tavily_api_key: str   = field(default_factory=lambda: os.getenv("TAVILY_API_KEY",""))
    serper_api_key: str   = field(default_factory=lambda: os.getenv("SERPER_API_KEY",""))
    max_sources_per_query: int = 6
    max_search_depth: int = 2                         # how many follow-up searches
    arxiv_max_results: int = 5

    # ── Memory ───────────────────────────────────────────────────────────────
    chroma_path: str      = "./acsis_chroma"
    neo4j_uri: str        = field(default_factory=lambda: os.getenv("NEO4J_URI","bolt://localhost:7687"))
    neo4j_user: str       = "neo4j"
    neo4j_password: str   = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD","acsis2026"))
    embedding_model: str  = "all-MiniLM-L6-v2"       # sentence-transformers, free

    # ── Reasoning ────────────────────────────────────────────────────────────
    confidence_threshold: float = 0.70               # below this → SOMA growth flagged
    max_reasoning_steps: int = 10
    code_execution_timeout: int = 30                  # seconds per code run

    # ── Notifications ────────────────────────────────────────────────────────
    notifications_enabled: bool = False               # set True with Telegram token
    telegram_token: str   = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN",""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID",""))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir: str          = "./acsis_logs"
    verbose: bool         = False

    # ── SOMA integration ─────────────────────────────────────────────────────
    soma_enabled: bool    = False                     # Phase 3: enable SOMA growth
    soma_model_dir: str   = "./soma_adapters"
    max_adapters: int     = 20
