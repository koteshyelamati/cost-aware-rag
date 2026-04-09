import pytest
from unittest.mock import patch

from app.services.classifier import classify


@pytest.fixture(autouse=True)
def _patch_cfg(monkeypatch):
    monkeypatch.setenv("COMPLEXITY_THRESHOLD", "0.4")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("DB_NAME", "rag_db")
    monkeypatch.setenv("COLLECTION_NAME", "documents")
    monkeypatch.setenv("COSTS_COLLECTION", "query_costs")
    monkeypatch.setenv("SIMPLE_MODEL", "gemini-flash")
    monkeypatch.setenv("COMPLEX_MODEL", "gemini-pro")
    monkeypatch.setenv("CACHE_TTL", "3600")
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.92")


def test_simple_short_query():
    result = classify("what is Redis?")
    assert result.tier == "simple"
    assert result.score < 0.4


def test_complex_keyword_triggers_complex():
    result = classify("compare cosine similarity and dot product similarity")
    assert result.tier == "complex"
    assert any("keywords" in s for s in result.signals)


def test_complex_architecture_keyword():
    result = classify("explain the architecture of a vector database")
    assert result.tier == "complex"


def test_long_query_bumps_score():
    long_query = " ".join(["word"] * 90)
    result = classify(long_query)
    assert any("token_count" in s for s in result.signals)


def test_multi_part_question():
    result = classify("what is Redis and how does it handle persistence?")
    assert any("multi_part" in s for s in result.signals)


def test_score_capped_at_one():
    query = "compare and analyze the trade-off and architecture design " + " ".join(["detail"] * 90) + "?"
    result = classify(query)
    assert result.score <= 1.0


def test_returns_complexity_result_fields():
    result = classify("what is vector search?")
    assert hasattr(result, "score")
    assert hasattr(result, "tier")
    assert hasattr(result, "signals")
    assert result.tier in ("simple", "complex")
