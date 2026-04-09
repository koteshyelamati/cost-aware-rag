from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import cfg
from app.models.schemas import HealthResponse, MetricsResponse
from app.routers import ingest, query
from app.services.embedder import Embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    motor_client: AsyncIOMotorClient = AsyncIOMotorClient(cfg.MONGODB_URI)
    await motor_client.admin.command({"ping": 1})
    app.state.db = motor_client[cfg.DB_NAME]
    logger.info("mongodb connected", extra={"db": cfg.DB_NAME})

    redis_client = await aioredis.from_url(cfg.REDIS_URL, decode_responses=False)
    await redis_client.ping()
    app.state.redis = redis_client
    logger.info("redis connected", extra={"url": cfg.REDIS_URL})

    app.state.embedder = Embedder()
    logger.info("embedder initialised")

    yield

    motor_client.close()
    await redis_client.aclose()
    logger.info("connections closed")


app = FastAPI(title="Cost-Aware RAG", version="1.0.0", lifespan=lifespan)

app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Live connectivity check for MongoDB and Redis."""
    mongo_status = "ok"
    redis_status = "ok"

    try:
        await app.state.db.client.admin.command({"ping": 1})
    except Exception as exc:
        mongo_status = f"error: {exc}"

    try:
        await app.state.redis.ping()
    except Exception as exc:
        redis_status = f"error: {exc}"

    overall = "ok" if mongo_status == "ok" and redis_status == "ok" else "degraded"
    return HealthResponse(status=overall, mongodb=mongo_status, redis=redis_status)


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Aggregate cost records from MongoDB and return usage stats."""
    pipeline = [
        {
            "$facet": {
                "totals": [
                    {
                        "$group": {
                            "_id": None,
                            "total_queries": {"$sum": 1},
                            "cache_hit_rate": {"$avg": {"$cond": ["$cache_hit", 1, 0]}},
                            "avg_cost_usd": {"$avg": "$estimated_cost_usd"},
                            "total_cost_usd": {"$sum": "$estimated_cost_usd"},
                        }
                    }
                ],
                "simple_count": [
                    {"$match": {"model_used": {"$regex": "flash"}}},
                    {"$count": "n"},
                ],
                "complex_count": [
                    {"$match": {"model_used": {"$regex": "pro"}}},
                    {"$count": "n"},
                ],
            }
        }
    ]

    cursor = app.state.db[cfg.COSTS_COLLECTION].aggregate(pipeline)
    rows = await cursor.to_list(length=1)
    row = rows[0] if rows else {}

    totals = (row.get("totals") or [{}])[0]
    simple_n = (row.get("simple_count") or [{"n": 0}])[0].get("n", 0)
    complex_n = (row.get("complex_count") or [{"n": 0}])[0].get("n", 0)

    return MetricsResponse(
        total_queries=totals.get("total_queries", 0),
        cache_hit_rate=totals.get("cache_hit_rate") or 0.0,
        avg_cost_usd=totals.get("avg_cost_usd") or 0.0,
        total_cost_usd=totals.get("total_cost_usd") or 0.0,
        simple_query_count=simple_n,
        complex_query_count=complex_n,
    )
