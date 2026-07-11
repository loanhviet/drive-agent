import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Application
APP_DB_PATH = os.getenv("APP_DB_PATH", str(BASE_DIR / ".data" / "drive_agent.db"))
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or (
    "gemini-2.5-flash" if LLM_PROVIDER == "gemini" else "claude-sonnet-4-20250514"
)
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "8"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Embeddings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Qdrant
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / ".data" / "qdrant"))
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "agent_memory")

# RAG
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MEMORY_SCORE_THRESHOLD = float(os.getenv("MEMORY_SCORE_THRESHOLD", "0.3"))

# Google Drive
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
