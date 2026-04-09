import json
from datetime import datetime, timezone

import numpy as np
import redis.asyncio as aioredis

from app.config import cfg
from app.models.schemas import CacheResult, CostMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "rag:cache:"


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


class SemanticCache:
    """Redis-backed semantic cache keyed by embedding similarity."""

    def __init__(self, client: aioredis.Redis) -> None:
        self._r = client

    async def get(self, query_embedding: list[float]) -> CacheResult | None:
        """Return cached answer if any stored embedding exceeds similarity threshold."""
        keys = await self._r.keys(f"{_KEY_PREFIX}*")
        if not keys:
            return None

        best_score = 0.0
        best_key: bytes | None = None

        for key in keys:
            raw = await self._r.hget(key, "embedding")  # type: ignore[misc]
            if raw is None:
                continue
            stored_embedding: list[float] = json.loads(raw)
            score = _cosine(query_embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_key = key

        if best_score < cfg.CACHE_SIMILARITY_THRESHOLD or best_key is None:
            return None

        fields = await self._r.hgetall(best_key.decode())  # type: ignore[misc]
        logger.info(
            "semantic cache hit",
            extra={"score": round(best_score, 4), "key": best_key.decode()},
        )
        return CacheResult(
            answer=fields[b"answer"].decode(),
            cost_metadata=CostMetadata(**json.loads(fields[b"cost_metadata"])),
        )

    async def set(
        self,
        query_embedding: list[float],
        answer: str,
        cost_metadata: CostMetadata,
    ) -> None:
        """Store answer + embedding in Redis with TTL."""
        key = f"{_KEY_PREFIX}{id(query_embedding)}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        payload = {
            "answer": answer,
            "cost_metadata": cost_metadata.model_dump_json(),
            "embedding": json.dumps(query_embedding),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._r.hset(key, mapping=payload)  # type: ignore[misc]
        await self._r.expire(key, cfg.CACHE_TTL)  # type: ignore[misc]
        logger.info("semantic cache set", extra={"key": key, "ttl": cfg.CACHE_TTL})
