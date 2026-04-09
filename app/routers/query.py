import asyncio
import time
import uuid

from fastapi import APIRouter, Depends

from app.config import cfg
from app.dependencies import get_db, get_embedder, get_redis, verify_api_key
from app.graph.rag_graph import RAGState, build_graph
from app.models.schemas import CostMetadata, QueryRequest, QueryResponse
from app.services.cache import SemanticCache
from app.services.cost_tracker import calculate_cost, write_cost
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query(
    body: QueryRequest,
    db=Depends(get_db),
    embedder=Depends(get_embedder),
    redis=Depends(get_redis),
) -> QueryResponse:
    """Embed query, check semantic cache, run RAG graph, return answer with cost metadata."""
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    embedding = (await embedder.embed([body.query]))[0]

    cache = SemanticCache(redis)
    cached = None
    try:
        cached = await cache.get(embedding)
    except Exception as exc:
        logger.warning("Redis unavailable, bypassing cache", extra={"error": str(exc)})

    if cached:
        latency_ms = (time.perf_counter() - start) * 1000
        cached.cost_metadata.cache_hit = True
        cached.cost_metadata.latency_ms = latency_ms
        logger.info(
            "query served from cache",
            extra={"request_id": request_id, "latency_ms": round(latency_ms, 2)},
        )
        return QueryResponse(
            request_id=request_id,
            answer=cached.answer,
            cited_chunks=[],
            cost_metadata=cached.cost_metadata,
            latency_ms=latency_ms,
        )

    compiled = build_graph(db=db, collection_name=cfg.COLLECTION_NAME)

    initial_state: RAGState = {
        "query": body.query,
        "query_embedding": embedding,
        "complexity": None,
        "retrieved_chunks": [],
        "answer": "",
        "cost_metadata": None,
        "cache_hit": False,
    }

    final_state: RAGState = await compiled.ainvoke(initial_state)

    latency_ms = (time.perf_counter() - start) * 1000

    raw_meta = final_state["cost_metadata"]
    cost_meta = CostMetadata(
        model_used=raw_meta.model_used,
        tokens_in=raw_meta.tokens_in,
        tokens_out=raw_meta.tokens_out,
        estimated_cost_usd=calculate_cost(
            raw_meta.model_used, raw_meta.tokens_in, raw_meta.tokens_out
        ),
        cache_hit=False,
        latency_ms=latency_ms,
    )

    asyncio.create_task(write_cost(cost_meta, db))
    try:
        await cache.set(embedding, final_state["answer"], cost_meta)
    except Exception as exc:
        logger.warning("Redis unavailable, skipping cache write", extra={"error": str(exc)})

    logger.info(
        "query complete",
        extra={
            "request_id": request_id,
            "tier": final_state["complexity"].tier if final_state["complexity"] else "unknown",
            "latency_ms": round(latency_ms, 2),
            "cost_usd": cost_meta.estimated_cost_usd,
        },
    )

    return QueryResponse(
        request_id=request_id,
        answer=final_state["answer"],
        cited_chunks=final_state["retrieved_chunks"],
        cost_metadata=cost_meta,
        latency_ms=latency_ms,
    )
