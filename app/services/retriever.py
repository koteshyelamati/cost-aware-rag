import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import OperationFailure

from app.models.schemas import RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def retrieve(
    embedding: list[float],
    top_k: int,
    db: AsyncIOMotorDatabase,
    collection_name: str,
) -> list[RetrievedChunk]:
    """Run MongoDB $vectorSearch and return scored chunks; falls back to in-process cosine similarity when Atlas is unavailable."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 50,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "doc_id": 1,
                "chunk_index": 1,
                "text": 1,
                "source_file": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        cursor = db[collection_name].aggregate(pipeline)
        chunks = [RetrievedChunk(**doc) async for doc in cursor]
        logger.info("vector search complete", extra={"top_k": top_k, "returned": len(chunks)})
        return chunks
    except OperationFailure as exc:
        if exc.code != 6047401:  # not the Atlas-only error
            raise
        logger.warning("$vectorSearch unavailable, falling back to in-process cosine similarity")
        return await _cosine_fallback(embedding, top_k, db, collection_name)


async def _cosine_fallback(
    query_vec: list[float],
    top_k: int,
    db: AsyncIOMotorDatabase,
    collection_name: str,
) -> list[RetrievedChunk]:
    """Fetch all docs and rank by cosine similarity. Dev-only path."""
    q = np.array(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)

    scored: list[tuple[float, dict]] = []
    async for doc in db[collection_name].find({}, {"embedding": 1, "doc_id": 1, "chunk_index": 1, "text": 1, "source_file": 1}):
        vec = np.array(doc["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        score = float(np.dot(q, vec) / (q_norm * norm)) if (q_norm and norm) else 0.0
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    chunks = [
        RetrievedChunk(
            doc_id=doc["doc_id"],
            chunk_index=doc["chunk_index"],
            text=doc["text"],
            source_file=doc.get("source_file", ""),
            score=score,
        )
        for score, doc in scored[:top_k]
    ]
    logger.info("cosine fallback complete", extra={"top_k": top_k, "returned": len(chunks)})
    return chunks
