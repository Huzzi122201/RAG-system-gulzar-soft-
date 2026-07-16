"""
Configuration module for the RAG Customer Support System.

Centralizes all configurable parameters: model names, retrieval weights,
confidence thresholds, and API settings. Uses environment variables for
sensitive values (API keys).
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Prevent HuggingFace read timeouts on slow connections
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# Gemini LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash" 

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
EMBEDDING_DIMENSION = 384

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "cloudsync_docs"

# Retrieval Configuration
# Hybrid retrieval weights (must sum to 1.0)
SEMANTIC_WEIGHT = 0.6   # Weight for embedding-based semantic search
KEYWORD_WEIGHT = 0.4    # Weight for TF-IDF keyword search

# Number of results to retrieve at each stage
INITIAL_RETRIEVAL_K = 15    # Candidates from each retrieval method
FINAL_TOP_K = 5             # Final results after re-ranking

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.55    # Score above this = HIGH confidence
MEDIUM_CONFIDENCE_THRESHOLD = 0.30  # Score above this = MEDIUM confidence
NO_RESULT_THRESHOLD = 0.20         # Score below this = no relevant results

# Re-Ranking Configuration
RECENCY_DECAY_DAYS = 365  # Documents older than this get recency penalty
TICKET_BOOST = 1.1        # Boost for tickets on troubleshooting queries
DOC_BOOST = 1.1           # Boost for docs on factual/feature queries

# Data File Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", )
PRODUCT_DOCS_PATH = os.path.join(DATA_DIR, "product_docs.json")
SUPPORT_TICKETS_PATH = os.path.join(DATA_DIR, "support_tickets.json")
TEST_QUERIES_PATH = os.path.join(DATA_DIR, "test_queries.json")

# Synonym Map for Query Expansion
SYNONYM_MAP = {
    "slow": ["performance", "speed", "throttling", "bandwidth", "lag", "latency"],
    "login": ["sign in", "log in", "authenticate", "authentication", "credentials"],
    "crash": ["crashing", "crashes", "freezing", "not responding", "force close", "stops working"],
    "sync": ["synchronize", "synchronization", "syncing", "synchronizing", "update across devices"],
    "share": ["sharing", "shared", "collaborate", "collaboration", "invite"],
    "price": ["pricing", "cost", "charge", "billing", "payment", "subscription", "plan"],
    "delete": ["remove", "erase", "uninstall"],
    "password": ["credentials", "login", "sign in", "authentication"],
    "upload": ["uploading", "send", "transfer"],
    "download": ["downloading", "get", "fetch", "retrieve"],
    "error": ["issue", "problem", "bug", "fail", "failure", "broken"],
    "setup": ["install", "installation", "configure", "configuration", "getting started", "onboarding"],
    "cancel": ["cancellation", "unsubscribe", "downgrade", "stop subscription"],
    "security": ["secure", "encryption", "privacy", "protection", "safe", "safety"],
    "mobile": ["phone", "ios", "android", "app", "smartphone", "tablet"],
    "api": ["integration", "developer", "sdk", "endpoint", "rest", "webhook"],
    "version": ["update", "upgrade", "v2.0", "v2.1", "release"],
    "storage": ["space", "capacity", "quota", "disk", "gb"],
    "photo": ["photos", "image", "images", "picture", "pictures", "camera"],
    "folder": ["directory", "folders", "directories"],
}

# Negation Patterns
NEGATION_PATTERNS = [
    "can't", "cannot", "can not", "couldn't", "could not",
    "won't", "will not", "wouldn't", "would not",
    "not working", "not syncing", "not visible", "not showing",
    "unable to", "failed to", "failing", "doesn't", "does not",
    "isn't", "is not", "aren't", "are not",
    "stopped", "broken", "missing", "lost",
]

# Query Type Definitions
QUERY_TYPE_KEYWORDS = {
    "troubleshooting": ["not working", "issue", "problem", "fix", "error", "broken", "can't", "unable",
                        "help", "trouble", "stuck", "crash", "fail", "wrong", "slow", "aren't syncing", "not syncing"],
    "billing_issue": ["charge", "charged", "billing", "refund", "payment", "invoice", "price",
                      "subscription", "cost", "money", "pay", "paid"],
    "feature_usage": ["how do i", "how to", "how can i", "what is", "where is", "guide",
                      "tutorial", "steps", "instructions", "use", "access", "enable"],
    "comparison": ["difference", "differences", "compare", "comparison", "vs", "versus",
                   "between free and pro", "free vs pro", "plans"],
    "security": ["security", "secure", "encryption", "privacy", "2fa", "two-factor",
                 "password", "protection", "safe"],
    "developer": ["api", "sdk", "integration", "integrate", "developer", "endpoint", "webhook",
                  "rest", "oauth", "code", "programming", "application"],
    "cancellation": ["cancel", "cancellation", "unsubscribe", "downgrade", "stop",
                     "end subscription", "quit"],
    "known_issue": ["known issue", "bug", "pending", "reported", "acknowledged"],
    "performance": ["slow", "speed", "performance", "fast", "bandwidth", "throttle",
                    "lag", "latency", "optimize"],
    "factual": ["what", "how", "when", "where", "who", "which", "tell me about", "explain"],
}

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
