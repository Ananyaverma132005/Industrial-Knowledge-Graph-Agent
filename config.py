"""Central configuration. Values come from environment variables or a local .env file."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line) so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

# --- Storage layer (Neo4j) ---------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
# "auto" = use Neo4j when reachable, else the in-memory fallback. "false" forces memory.
USE_NEO4J = os.getenv("USE_NEO4J", "auto").lower()

# --- Cognitive engine (Ollama / Mistral) ---------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
USE_LLM = os.getenv("USE_LLM", "auto").lower()

# --- Paths ---------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
MANUALS_DIR = DATA_DIR / "manuals"
ASSET_REGISTRY = DATA_DIR / "asset_registry.json"
TICKETS_FILE = DATA_DIR / "tickets.json"
SNAPSHOT_FILE = DATA_DIR / "graph_snapshot.json"
MODEL_DIR = BASE_DIR / "models"
ML_MODEL_FILE = MODEL_DIR / "failure_model.joblib"
STATIC_DIR = BASE_DIR / "static"

# --- NLP / vectors -------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dimensional
EMBEDDING_DIM = 384
VECTOR_INDEX = "sop_embeddings"
VECTOR_TOP_K = 2

# --- Runtime -------------------------------------------------------------------
TELEMETRY_INTERVAL = float(os.getenv("TELEMETRY_INTERVAL", "3"))
ALERT_TEMP_C = 90.0
AT_RISK_MIN_TICKETS = 2
