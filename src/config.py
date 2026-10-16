"""Central configuration — environment variables and defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
LOGS_DIR = PROJECT_ROOT / "logs"
LLM_CACHE_DIR = DATA_DIR / "llm_cache"

# ── openFDA ────────────────────────────────────────────────────────────
OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_REQUEST_DELAY = 0.5  # seconds between requests

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# ── ChromaDB ───────────────────────────────────────────────────────────
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = "drug_labels"

# ── LLM ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
LLM_MAX_RETRIES = 3

# ── Retrieval ──────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 5
RRF_K = 60  # Reciprocal Rank Fusion constant

# ── Confidence thresholds ──────────────────────────────────────────────
CONFIDENCE_HIGH = 0.45
CONFIDENCE_MEDIUM = 0.25  # below this → abstain

# ── Logging ────────────────────────────────────────────────────────────
APP_LOG_FILE = LOGS_DIR / "app.jsonl"
REVIEW_QUEUE_FILE = LOGS_DIR / "human_review_queue.jsonl"
