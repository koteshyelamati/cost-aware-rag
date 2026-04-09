import os
import pytest

# Ensure all required env vars exist before any app module is imported
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("DB_NAME", "rag_db")
os.environ.setdefault("COLLECTION_NAME", "documents")
os.environ.setdefault("COSTS_COLLECTION", "query_costs")
os.environ.setdefault("SIMPLE_MODEL", "gemini-flash")
os.environ.setdefault("COMPLEX_MODEL", "gemini-pro")
os.environ.setdefault("COMPLEXITY_THRESHOLD", "0.4")
os.environ.setdefault("CACHE_TTL", "3600")
os.environ.setdefault("CACHE_SIMILARITY_THRESHOLD", "0.92")
os.environ.setdefault("LOG_LEVEL", "INFO")
