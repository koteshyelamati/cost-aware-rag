import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import CostMetadata, RetrievedChunk


_COST_META = CostMetadata(
    model_used="gemini-flash",
    tokens_in=100,
    tokens_out=40,
    estimated_cost_usd=0.0,
    cache_hit=False,
    latency_ms=500.0,
)

_CHUNK = RetrievedChunk(
    doc_id="doc-1",
    chunk_index=0,
    score=0.85,
    text="MongoDB Atlas Vector Search uses HNSW indexing.",
    source_file="test.md",
)


def _make_app_state(redis_ok: bool = True):
    """Patch app.state dependencies so TestClient works without live services."""
    db_mock = MagicMock()
    db_mock.__getitem__ = MagicMock(return_value=MagicMock())

    embedder_mock = AsyncMock()
    embedder_mock.embed = AsyncMock(return_value=[[0.1] * 768])

    redis_mock = AsyncMock() if redis_ok else None

    app.state.db = db_mock
    app.state.embedder = embedder_mock
    app.state.redis = redis_mock


@pytest.fixture()
def client():
    _make_app_state()
    return TestClient(app, raise_server_exceptions=False)


# ── Validation ────────────────────────────────────────────────────────────────

def test_empty_query_returns_422(client):
    resp = client.post(
        "/api/v1/query",
        json={"query": ""},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 422


def test_missing_api_key_returns_401(client):
    resp = client.post("/api/v1/query", json={"query": "what is Redis?"})
    assert resp.status_code == 422  # missing header → validation error


def test_wrong_api_key_returns_401(client):
    resp = client.post(
        "/api/v1/query",
        json={"query": "what is Redis?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ── Ingest validation ─────────────────────────────────────────────────────────

def test_ingest_rejects_exe(client):
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 400
    assert ".exe" in resp.json()["detail"]


def test_ingest_accepts_md(client):
    with patch("app.routers.ingest.get_embedder", return_value=lambda r: AsyncMock()), \
         patch("app.routers.ingest.get_db", return_value=lambda r: MagicMock()):
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("doc.md", b"# Hello world", "text/markdown")},
            headers={"X-API-Key": "test-key"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "processing"
    assert "doc_id" in body


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    app.state.db = MagicMock()
    app.state.db.client.admin.command = AsyncMock(return_value={"ok": 1})
    app.state.redis = AsyncMock()
    app.state.redis.ping = AsyncMock(return_value=True)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("ok", "degraded")


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_returns_expected_keys(client):
    mock_collection = MagicMock()
    mock_cursor = AsyncMock()
    mock_cursor.to_list = AsyncMock(return_value=[{
        "totals": [{"total_queries": 5, "cache_hit_rate": 0.6,
                    "avg_cost_usd": 0.0001, "total_cost_usd": 0.0005}],
        "simple_count": [{"n": 4}],
        "complex_count": [{"n": 1}],
    }])
    mock_collection.aggregate = MagicMock(return_value=mock_cursor)
    app.state.db.__getitem__ = MagicMock(return_value=mock_collection)

    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("total_queries", "cache_hit_rate", "avg_cost_usd",
                 "total_cost_usd", "simple_query_count", "complex_query_count"):
        assert key in body
