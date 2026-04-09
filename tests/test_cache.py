import json
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import CostMetadata
from app.services.cache import SemanticCache, _cosine


def _make_cost_meta(**kwargs) -> CostMetadata:
    defaults = dict(
        model_used="gemini-flash",
        tokens_in=100,
        tokens_out=50,
        estimated_cost_usd=0.0,
        cache_hit=False,
        latency_ms=500.0,
    )
    return CostMetadata(**{**defaults, **kwargs})


def _unit_vec(size: int = 768, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(size).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ── _cosine ──────────────────────────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = _unit_vec()
    assert abs(_cosine(v, v) - 1.0) < 1e-5


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── SemanticCache.get ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_get_returns_none_when_empty():
    redis = AsyncMock()
    redis.keys = AsyncMock(return_value=[])
    cache = SemanticCache(redis)
    assert await cache.get(_unit_vec()) is None


@pytest.mark.asyncio
async def test_cache_get_hit_above_threshold():
    vec = _unit_vec(seed=1)
    cost_meta = _make_cost_meta()

    redis = AsyncMock()
    redis.keys = AsyncMock(return_value=[b"rag:cache:abc"])
    redis.hget = AsyncMock(return_value=json.dumps(vec).encode())
    redis.hgetall = AsyncMock(return_value={
        b"answer": b"cached answer",
        b"cost_metadata": cost_meta.model_dump_json().encode(),
    })

    cache = SemanticCache(redis)
    # Use the same vector — cosine similarity == 1.0, well above 0.92
    result = await cache.get(vec)
    assert result is not None
    assert result.answer == "cached answer"


@pytest.mark.asyncio
async def test_cache_get_miss_below_threshold():
    vec_a = _unit_vec(seed=1)
    vec_b = _unit_vec(seed=99)   # different random vector, similarity will be low

    redis = AsyncMock()
    redis.keys = AsyncMock(return_value=[b"rag:cache:abc"])
    redis.hget = AsyncMock(return_value=json.dumps(vec_a).encode())

    with patch("app.services.cache.cfg") as mock_cfg:
        mock_cfg.CACHE_SIMILARITY_THRESHOLD = 0.92
        cache = SemanticCache(redis)
        result = await cache.get(vec_b)

    assert result is None


# ── SemanticCache.set ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_set_writes_correct_fields():
    redis = AsyncMock()
    redis.hset = AsyncMock()
    redis.expire = AsyncMock()

    with patch("app.services.cache.cfg") as mock_cfg:
        mock_cfg.CACHE_TTL = 3600
        cache = SemanticCache(redis)
        vec = _unit_vec()
        cost_meta = _make_cost_meta()
        await cache.set(vec, "test answer", cost_meta)

    redis.hset.assert_called_once()
    call_kwargs = redis.hset.call_args[1]["mapping"]
    assert call_kwargs["answer"] == "test answer"
    assert json.loads(call_kwargs["embedding"]) == vec
    redis.expire.assert_called_once()
